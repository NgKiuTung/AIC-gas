"""Robustness analysis and scenario selection for dispatch backtests."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "results" / "optimization" / "dispatch_backtest"
OUT_DIR = ROOT / "results" / "optimization" / "dispatch_validation"
LOG_DIR = ROOT / "results" / "optimization" / "logs"
WINDOW_PATH = INPUT_DIR / "rolling_window_metrics.csv"
SCHEDULE_PATH = INPUT_DIR / "optimized_schedules.csv"
SCENARIO_PATH = INPUT_DIR / "scenario_summary.csv"
SELECTED_SCENARIO = "balanced"
BOOTSTRAP_REPLICATES = 4000
RANDOM_SEED = 20260806


def setup_logger() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dispatch_robustness")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "42_dispatch_robustness_and_selection.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def bootstrap_by_window(values: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(RANDOM_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True).mean(axis=1)
    frame = pd.DataFrame({"replicate": np.arange(1, BOOTSTRAP_REPLICATES + 1), "mean_revenue_uplift": samples})
    summary = {
        "point_estimate": float(values.mean()),
        "bootstrap_mean": float(samples.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": RANDOM_SEED,
        "sampling_unit": "two-hour forecast window",
    }
    return frame, summary


def main() -> None:
    logger = setup_logger()
    for path in (WINDOW_PATH, SCHEDULE_PATH, SCENARIO_PATH):
        if "评分所用测试集" in str(path):
            raise RuntimeError("Forbidden external scoring dataset path")
    windows = pd.read_csv(WINDOW_PATH, parse_dates=["forecast_origin"])
    schedules = pd.read_csv(SCHEDULE_PATH, parse_dates=["forecast_origin", "target_datetime"])
    scenarios = pd.read_csv(SCENARIO_PATH)
    selected_windows = windows[(windows["scenario"] == SELECTED_SCENARIO) & windows["solver_success"]].copy()
    selected_schedule = schedules[schedules["scenario"] == SELECTED_SCENARIO].copy()

    by_fold = selected_windows.groupby("fold", as_index=False).agg(
        windows=("forecast_origin", "size"),
        price_varying_windows=("price_levels", lambda values: int((values > 1).sum())),
        mean_uplift_all=("forecast_revenue_uplift", "mean"),
        mean_dispatch_change_mw=("mean_absolute_dispatch_change_mw", "mean"),
        max_energy_slack_fraction=("energy_slack_fraction", "max"),
        holder_violations=("official_holder_violations", "sum"),
        capacity_violations=("official_capacity_violations", "sum"),
        ramp_violations=("official_ramp_violations", "sum"),
        transition_fraction=("normal_minload_transition_fraction", "mean"),
    )
    varying_by_fold = selected_windows[selected_windows["price_levels"] > 1].groupby("fold", as_index=False).agg(
        mean_uplift_price_varying=("forecast_revenue_uplift", "mean"),
        median_uplift_price_varying=("forecast_revenue_uplift", "median"),
        minimum_uplift_price_varying=("forecast_revenue_uplift", "min"),
        positive_fraction=("forecast_revenue_uplift", lambda values: float((values > 1e-10).mean())),
    )
    by_fold = by_fold.merge(varying_by_fold, on="fold", validate="one_to_one")
    by_fold.to_csv(OUT_DIR / "selected_scenario_by_fold.csv", index=False, encoding="utf-8-sig")

    varying = selected_windows[selected_windows["price_levels"] > 1].copy()
    bootstrap_frame, bootstrap_summary = bootstrap_by_window(varying["forecast_revenue_uplift"].to_numpy(float))
    bootstrap_frame.to_csv(OUT_DIR / "price_varying_uplift_bootstrap.csv", index=False, encoding="utf-8-sig")

    # Exact resource-conservation checks use the submitted gas quantities.
    gas_pairs = [
        ("generator_use_blast_furnace_gas", "BFG"),
        ("generator_use_coke_gas", "COG"),
        ("generator_use_converter_gas", "LDG"),
    ]
    grouped = selected_schedule.groupby(["fold", "forecast_origin"])
    check_records: list[dict[str, object]] = []
    for (fold, origin), group in grouped:
        record: dict[str, object] = {
            "fold": fold,
            "forecast_origin": origin,
            "rows": int(len(group)),
            "energy_relative_difference": float(
                (group["optimized_generator_all_mw"].sum() - group["baseline_generator_all_mw"].sum())
                / max(abs(group["baseline_generator_all_mw"].sum()), 1e-9)
            ),
            "holder_min_m3": float(group["projected_bfg_holder_m3"].min()),
            "holder_max_m3": float(group["projected_bfg_holder_m3"].max()),
            "capacity_violation_steps": int(
                ((group["optimized_generator_1_mw"] < -1e-6) | (group["optimized_generator_1_mw"] > 200 + 1e-6)).sum()
                + ((group["optimized_generator_120_group_mw"] < -1e-6) | (group["optimized_generator_120_group_mw"] > 240 + 1e-6)).sum()
            ),
        }
        for gas, short in gas_pairs:
            base = float(group[f"baseline_{gas}"].sum())
            opt = float(group[f"opt_{gas}"].sum())
            record[f"{short.lower()}_relative_difference"] = (opt - base) / max(abs(base), 1e-9)
        check_records.append(record)
    checks = pd.DataFrame(check_records)
    checks.to_csv(OUT_DIR / "window_resource_and_constraint_checks.csv", index=False, encoding="utf-8-sig")

    # Select a non-cherry-picked price-varying window closest to median uplift.
    median_uplift = float(varying["forecast_revenue_uplift"].median())
    representative = varying.iloc[(varying["forecast_revenue_uplift"] - median_uplift).abs().argmin()]
    representative_schedule = selected_schedule[
        (selected_schedule["fold"] == representative["fold"])
        & (selected_schedule["forecast_origin"] == representative["forecast_origin"])
    ].copy()
    representative_schedule.to_csv(OUT_DIR / "representative_balanced_schedule.csv", index=False, encoding="utf-8-sig")
    selected_schedule.to_csv(OUT_DIR / "selected_balanced_schedules.csv", index=False, encoding="utf-8-sig")

    selected_scenario_row = scenarios.loc[scenarios["scenario"] == SELECTED_SCENARIO].iloc[0]
    max_abs_conservation_error = float(
        checks[["energy_relative_difference", "bfg_relative_difference", "cog_relative_difference", "ldg_relative_difference"]]
        .abs()
        .to_numpy()
        .max()
    )
    hard_violation_total = int(
        selected_windows["official_holder_violations"].sum()
        + selected_windows["official_capacity_violations"].sum()
        + selected_windows["official_ramp_violations"].sum()
    )
    acceptance = {
        "solver_success_rate": float(selected_scenario_row["solver_success_rate"]),
        "solver_success_rate_pass": bool(selected_scenario_row["solver_success_rate"] >= 0.99),
        "hard_violation_total": hard_violation_total,
        "hard_constraints_pass": hard_violation_total == 0,
        "maximum_absolute_energy_or_gas_conservation_error": max_abs_conservation_error,
        "resource_conservation_pass": max_abs_conservation_error <= 1e-8,
        "maximum_energy_slack_fraction": float(selected_windows["energy_slack_fraction"].max()),
        "zero_energy_slack_pass": bool(selected_windows["energy_slack_fraction"].max() <= 1e-10),
        "bootstrap_uplift_ci95_lower": bootstrap_summary["ci95_lower"],
        "positive_uplift_ci_pass": bootstrap_summary["ci95_lower"] > 0,
        "all_fold_price_varying_uplift_positive": bool((by_fold["mean_uplift_price_varying"] > 0).all()),
        "transition_step_fraction": float(selected_windows["normal_minload_transition_fraction"].mean()),
        "transition_step_fraction_pass": bool(selected_windows["normal_minload_transition_fraction"].mean() <= 0.03),
    }
    acceptance["overall_pass"] = bool(
        acceptance["solver_success_rate_pass"]
        and acceptance["hard_constraints_pass"]
        and acceptance["resource_conservation_pass"]
        and acceptance["zero_energy_slack_pass"]
        and acceptance["positive_uplift_ci_pass"]
        and acceptance["all_fold_price_varying_uplift_positive"]
        and acceptance["transition_step_fraction_pass"]
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only chronological OOF dispatch backtest",
        "external_scoring_data_accessed": False,
        "selected_scenario": SELECTED_SCENARIO,
        "selection_reason": (
            "Balanced retains a 5,000 m3 internal holder margin and moderate dispatch movement while capturing "
            "most of the price-varying-window uplift. Aggressive is retained only as an upside sensitivity case."
        ),
        "bootstrap_price_varying_uplift": bootstrap_summary,
        "representative_window": {
            "fold": representative["fold"],
            "forecast_origin": representative["forecast_origin"].isoformat(),
            "revenue_uplift": float(representative["forecast_revenue_uplift"]),
            "selection_rule": "closest to median uplift among price-varying balanced windows",
        },
        "acceptance": acceptance,
        "limitations": [
            "Confidence interval resamples windows and does not fully model serial dependence between nearby operating periods.",
            "The backtest evaluates dispatch against OOF forecast resource boundaries, not a plant intervention counterfactual.",
            "Transition slack is explicit and auditable because single-unit startup states are not supplied in preliminary data.",
        ],
    }
    (OUT_DIR / "selected_scenario_and_acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Selected scenario: %s", SELECTED_SCENARIO)
    logger.info("Price-varying uplift: %.3f%% (95%% bootstrap CI %.3f%% to %.3f%%)", bootstrap_summary["point_estimate"] * 100, bootstrap_summary["ci95_lower"] * 100, bootstrap_summary["ci95_upper"] * 100)
    logger.info("Acceptance overall pass=%s; max conservation error=%.3e", acceptance["overall_pass"], max_abs_conservation_error)
    logger.info("Wrote validation artifacts to %s", OUT_DIR)


if __name__ == "__main__":
    main()
