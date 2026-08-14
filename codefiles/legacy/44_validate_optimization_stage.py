"""Final artifact and feasibility validation for the optimization stage."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "optimization" / "final_validation"
LOG_DIR = ROOT / "results" / "optimization" / "logs"


def setup_logger() -> logging.Logger:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("validate_optimization_stage")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "44_validate_optimization_stage.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def add_check(checks: list[dict[str, object]], name: str, passed: bool, evidence: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    logger = setup_logger()
    acceptance_path = ROOT / "results" / "optimization" / "dispatch_validation" / "selected_scenario_and_acceptance.json"
    schedule_path = ROOT / "results" / "optimization" / "dispatch_validation" / "selected_balanced_schedules.csv"
    prediction_manifest_path = ROOT / "results" / "visualizations" / "prediction" / "visualization_manifest.json"
    optimization_manifest_path = ROOT / "results" / "visualizations" / "optimization" / "visualization_manifest.json"
    report_paths = [
        ROOT / "docs" / "experimental_docs" / "forecasting" / "04_数据清洗再审计与预测优化实验报告.md",
        ROOT / "docs" / "experimental_docs" / "optimization" / "05_发电优化约束可辨识性审计报告.md",
        ROOT / "docs" / "experimental_docs" / "optimization" / "06_保守软约束发电优化与历史回放实验报告.md",
    ]
    checks: list[dict[str, object]] = []
    required = [acceptance_path, schedule_path, prediction_manifest_path, optimization_manifest_path, *report_paths]
    add_check(checks, "required_artifacts_exist", all(path.exists() for path in required), "; ".join(str(path.relative_to(ROOT)) for path in required))

    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    add_check(checks, "optimization_acceptance_pass", acceptance["acceptance"]["overall_pass"], json.dumps(acceptance["acceptance"], ensure_ascii=False))
    add_check(checks, "no_scoring_data_in_acceptance", acceptance["external_scoring_data_accessed"] is False, "external_scoring_data_accessed=false")

    schedules = pd.read_csv(schedule_path)
    required_columns = {
        "forecast_origin",
        "target_datetime",
        "opt_generator_use_blast_furnace_gas",
        "opt_generator_use_coke_gas",
        "opt_generator_use_converter_gas",
        "optimized_generator_1_mw",
        "optimized_generator_all_mw",
        "projected_bfg_holder_m3",
    }
    add_check(checks, "schedule_required_columns", required_columns.issubset(schedules.columns), ",".join(sorted(required_columns)))
    add_check(checks, "schedule_no_missing_values", not schedules[list(required_columns)].isna().any().any(), f"rows={len(schedules)}")
    duplicate_count = int(schedules.duplicated(["fold", "forecast_origin", "target_datetime"]).sum())
    add_check(checks, "schedule_composite_key_unique", duplicate_count == 0, f"duplicate_count={duplicate_count}")
    gas_columns = [
        "opt_generator_use_blast_furnace_gas",
        "opt_generator_use_coke_gas",
        "opt_generator_use_converter_gas",
    ]
    add_check(checks, "optimized_gas_nonnegative", bool((schedules[gas_columns] >= -1e-8).all().all()), f"minimum={schedules[gas_columns].min().min():.6g}")
    holder_ok = bool(schedules["projected_bfg_holder_m3"].between(30_000 - 1e-6, 180_000 + 1e-6).all())
    add_check(checks, "official_holder_bounds", holder_ok, f"range={schedules['projected_bfg_holder_m3'].min():.3f}..{schedules['projected_bfg_holder_m3'].max():.3f}")
    capacity_ok = bool(
        schedules["optimized_generator_1_mw"].between(-1e-6, 200 + 1e-6).all()
        and schedules["optimized_generator_120_group_mw"].between(-1e-6, 240 + 1e-6).all()
    )
    add_check(checks, "official_group_capacities", capacity_ok, "50MW group <=200MW; 120MW group <=240MW")

    figure_paths: list[Path] = []
    for manifest_path in (prediction_manifest_path, optimization_manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        add_check(checks, f"{manifest_path.parent.name}_no_scoring_data", manifest["external_scoring_data_accessed"] is False, str(manifest_path.relative_to(ROOT)))
        figure_paths.extend(ROOT / path for path in manifest["figures"])
    # Small vector charts can be below 10 KB while remaining complete; 5 KB
    # catches empty/corrupt exports without rejecting compact editable SVGs.
    add_check(checks, "all_figure_exports_exist", all(path.exists() and path.stat().st_size > 5_000 for path in figure_paths), f"figure_files={len(figure_paths)}; minimum_bytes={min(path.stat().st_size for path in figure_paths if path.exists())}")
    svg_paths = [path for path in figure_paths if path.suffix.lower() == ".svg"]
    editable = all("<text" in path.read_text(encoding="utf-8", errors="ignore") for path in svg_paths)
    add_check(checks, "svg_text_editable", editable, f"svg_files={len(svg_paths)}")

    code_files = [ROOT / "codefiles" / "legacy" / f"{number}_{name}.py" for number, name in [
        (38, "optimization_constraint_audit"),
        (39, "resource_mechanism_diagnostics"),
        (40, "prediction_visualizations"),
        (41, "conservative_dispatch_backtest"),
        (42, "dispatch_robustness_and_selection"),
        (43, "optimization_visualizations"),
    ]]
    add_check(checks, "stage_codefiles_exist", all(path.exists() for path in code_files), "; ".join(path.name for path in code_files))
    all_pass = all(bool(item["passed"]) for item in checks)
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(RESULT_DIR / "validation_checks.csv", index=False, encoding="utf-8-sig")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only optimization artifacts",
        "external_scoring_data_accessed": False,
        "checks": len(checks),
        "passed": int(checks_frame["passed"].sum()),
        "failed": int((~checks_frame["passed"]).sum()),
        "overall_pass": all_pass,
        "selected_scenario": acceptance["selected_scenario"],
        "selected_scenario_acceptance": acceptance["acceptance"],
    }
    (RESULT_DIR / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Validation checks: %d/%d passed; overall_pass=%s", summary["passed"], summary["checks"], all_pass)
    if not all_pass:
        failed = checks_frame.loc[~checks_frame["passed"], ["check", "evidence"]]
        logger.error("Failed checks:\n%s", failed.to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
