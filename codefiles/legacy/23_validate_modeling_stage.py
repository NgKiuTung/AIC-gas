"""Validate modeling artifacts, metrics, hashes, and training-only provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "validation"
MODEL_DIR = ROOT / "results" / "models" / "power_forecaster"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(condition), "detail": detail})

    required = [
        MODEL_DIR / "xgboost_hybrid_forecaster.json",
        MODEL_DIR / "model_manifest.json",
        MODEL_DIR / "feature_schema.csv",
        MODEL_DIR / "feature_importance.csv",
        MODEL_DIR / "training_feature_inference_check.csv",
        ROOT / "results" / "experiments" / "hybrid_target" / "hybrid_summary.json",
        ROOT / "results" / "experiments" / "final_hybrid_diagnostics" / "diagnostic_summary.json",
        ROOT / "results" / "experiments" / "temporal_effects" / "temporal_effect_summary.json",
    ]
    for path in required:
        check(f"exists::{path.name}", path.exists() and path.stat().st_size > 0, str(path.relative_to(ROOT)))

    manifest = json.loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    model_path = MODEL_DIR / manifest["model_file"]
    check("model_hash", sha256(model_path) == manifest["model_sha256"], manifest["model_sha256"])
    check("training_only_manifest", manifest["external_scoring_data_accessed"] is False, manifest["source_scope"])
    check("gpu_training", "4060" in manifest["gpu"], manifest["gpu"])
    check("feature_count", manifest["feature_count"] == 560, str(manifest["feature_count"]))
    check("horizon_count", manifest["horizon_minutes"] == list(range(15, 121, 15)), str(manifest["horizon_minutes"]))

    schema = pd.read_csv(MODEL_DIR / "feature_schema.csv", encoding="utf-8-sig")
    features = schema.sort_values("position")["feature"].tolist()
    leakage_names = [name for name in features if name.startswith("label_") or "future" in name.lower() or "lead" in name.lower()]
    check("schema_rows", len(schema) == manifest["feature_count"], f"rows={len(schema)}")
    check("schema_unique", schema["feature"].is_unique, f"unique={schema['feature'].nunique()}")
    check("feature_name_leakage", len(leakage_names) == 0, str(leakage_names[:10]))

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    check("model_load", True, xgb.__version__)
    inference = pd.read_csv(MODEL_DIR / "training_feature_inference_check.csv", encoding="utf-8-sig")
    numeric = inference.select_dtypes(include=[np.number])
    check("inference_rows", len(inference) == manifest["training_rows"], f"rows={len(inference)}")
    check("inference_finite", np.isfinite(numeric.to_numpy()).all(), f"numeric_columns={numeric.shape[1]}")
    hierarchy_ok = True
    for minutes in range(15, 121, 15):
        hierarchy_ok &= bool(
            (inference[f"generator_all_t_plus_{minutes}m"] >= inference[f"generator_1_t_plus_{minutes}m"]).all()
        )
    check("prediction_hierarchy", hierarchy_ok, "generator_all >= generator_1 for every row/horizon")

    hybrid = json.loads((ROOT / "results" / "experiments" / "hybrid_target" / "hybrid_summary.json").read_text(encoding="utf-8"))
    check("cv_improves_persistence", hybrid["overall_hybrid_mape"] < hybrid["overall_persistence_mape"], f"{hybrid['overall_hybrid_mape']:.6f} < {hybrid['overall_persistence_mape']:.6f}")
    check("bootstrap_positive", hybrid["daily_block_bootstrap"]["ci95_low"] > 0, str(hybrid["daily_block_bootstrap"]))
    fold = pd.read_csv(ROOT / "results" / "experiments" / "hybrid_target" / "fold_summary.csv", encoding="utf-8-sig")
    check("all_folds_improve", bool((fold["hybrid_mape"] < fold["persistence_mape"]).all()), fold.to_json(orient="records"))
    diagnostic = json.loads((ROOT / "results" / "experiments" / "final_hybrid_diagnostics" / "diagnostic_summary.json").read_text(encoding="utf-8"))
    check("all_operating_regimes_improve", diagnostic["all_regimes_improve"] is True, str(diagnostic["all_regimes_improve"]))
    check("all_time_sessions_improve", diagnostic["all_sessions_improve"] is True, str(diagnostic["all_sessions_improve"]))
    processed_test_derivative = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_test.csv"
    check("no_scoring_derivative", not processed_test_derivative.exists(), str(processed_test_derivative.relative_to(ROOT)))

    table = pd.DataFrame(checks)
    table.to_csv(RESULT_DIR / "modeling_validation_checks.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": len(table),
        "passed": int(table["passed"].sum()),
        "failed": int((~table["passed"]).sum()),
        "all_passed": bool(table["passed"].all()),
        "external_scoring_data_accessed": False,
    }
    (RESULT_DIR / "modeling_validation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
