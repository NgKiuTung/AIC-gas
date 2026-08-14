"""Robustness and operating-regime analysis for selected OOF forecasts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "results" / "experiments" / "xgboost_direct" / "raw_oof_predictions_long.csv"
FEATURE_PATH = ROOT / "results" / "features" / "train_supervised_features.pkl"
RESULT_DIR = ROOT / "results" / "experiments" / "robustness"
PARAMETERS = {
    "generator_1": {"beta_h15": 0.10, "beta_h120": 0.15},
    "generator_all": {"beta_h15": 0.65, "beta_h120": 0.60},
}
BOOTSTRAP_REPEATS = 5000
SEED = 20260803


def build_rows() -> pd.DataFrame:
    raw = pd.read_csv(OOF_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    raw = raw[raw["model"] == "direct_d5"].copy()
    features = pd.read_pickle(FEATURE_PATH)
    features["datetime"] = pd.to_datetime(features["datetime"], errors="raise")
    regime_features = features[
        ["datetime", "feat_diff4_p50_current", "feat_diff4_generator_all_filled"]
    ].copy()
    raw = raw.merge(regime_features, on="datetime", how="left", validate="many_to_one")
    if raw[["feat_diff4_p50_current", "feat_diff4_generator_all_filled"]].isna().any().any():
        raise ValueError("Regime features failed to join")

    beta_start = raw["target"].map({key: value["beta_h15"] for key, value in PARAMETERS.items()})
    beta_end = raw["target"].map({key: value["beta_h120"] for key, value in PARAMETERS.items()})
    raw["beta"] = beta_start + (beta_end - beta_start) * (raw["horizon_step"] - 1) / 7
    raw["prediction_selected"] = np.maximum(raw["current"] + raw["beta"] * raw["raw_correction"], 0.0)
    raw["ape_persistence"] = np.abs(raw["current"] - raw["actual"]) / np.maximum(np.abs(raw["actual"]), 1e-6)
    raw["ape_selected"] = np.abs(raw["prediction_selected"] - raw["actual"]) / np.maximum(np.abs(raw["actual"]), 1e-6)
    raw["ape_improvement"] = raw["ape_persistence"] - raw["ape_selected"]
    raw["origin_date"] = raw["datetime"].dt.date.astype(str)

    diff = np.where(
        raw["target"].eq("generator_1"),
        np.abs(raw["feat_diff4_p50_current"]),
        np.abs(raw["feat_diff4_generator_all_filled"]),
    )
    raw["observed_change_ratio_1h"] = diff / np.maximum(np.abs(raw["current"]), 1e-6)
    raw["operating_regime"] = ""
    for target in PARAMETERS:
        mask = raw["target"].eq(target)
        origin_values = raw.loc[mask].drop_duplicates("datetime")["observed_change_ratio_1h"]
        q50, q80 = origin_values.quantile([0.50, 0.80]).to_numpy()
        raw.loc[mask, "operating_regime"] = pd.cut(
            raw.loc[mask, "observed_change_ratio_1h"],
            bins=[-np.inf, q50, q80, np.inf],
            labels=["stable_0_50pct", "changing_50_80pct", "volatile_80_100pct"],
            include_lowest=True,
        ).astype(str)
    return raw


def paired_daily_bootstrap(rows: pd.DataFrame) -> dict[str, float]:
    daily = rows.groupby("origin_date", as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"),
        selected_mape=("ape_selected", "mean"),
    )
    daily["improvement"] = daily["persistence_mape"] - daily["selected_mape"]
    values = daily["improvement"].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "days": int(len(values)),
        "daily_block_bootstrap_repeats": BOOTSTRAP_REPEATS,
        "mean_absolute_mape_reduction": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_improvement_positive": float(np.mean(samples > 0)),
        "fraction_days_selected_wins": float(np.mean(values > 0)),
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    regime = (
        rows.groupby(["target", "operating_regime"], as_index=False)
        .agg(
            samples=("actual", "size"),
            persistence_mape=("ape_persistence", "mean"),
            selected_mape=("ape_selected", "mean"),
            mean_absolute_mape_reduction=("ape_improvement", "mean"),
        )
    )
    regime["relative_mape_reduction"] = (
        regime["persistence_mape"] - regime["selected_mape"]
    ) / regime["persistence_mape"]
    pair = (
        rows.groupby(["fold", "target", "horizon_minutes"], as_index=False)
        .agg(
            persistence_mape=("ape_persistence", "mean"),
            selected_mape=("ape_selected", "mean"),
            mean_absolute_mape_reduction=("ape_improvement", "mean"),
        )
    )
    pair["selected_wins"] = pair["mean_absolute_mape_reduction"] > 0
    fold = (
        rows.groupby("fold", as_index=False)
        .agg(persistence_mape=("ape_persistence", "mean"), selected_mape=("ape_selected", "mean"))
    )
    fold["relative_mape_reduction"] = (fold["persistence_mape"] - fold["selected_mape"]) / fold["persistence_mape"]
    bootstrap = paired_daily_bootstrap(rows)
    overall_persistence = float(rows.groupby(["fold", "target", "horizon_step"])["ape_persistence"].mean().mean())
    overall_selected = float(rows.groupby(["fold", "target", "horizon_step"])["ape_selected"].mean().mean())

    keep_cols = [
        "datetime", "origin_date", "fold", "target", "horizon_step", "horizon_minutes",
        "actual", "current", "prediction_selected", "beta", "observed_change_ratio_1h",
        "operating_regime", "ape_persistence", "ape_selected", "ape_improvement",
    ]
    rows[keep_cols].to_csv(RESULT_DIR / "row_level_errors.csv", index=False, encoding="utf-8-sig")
    regime.to_csv(RESULT_DIR / "operating_regime_metrics.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(RESULT_DIR / "paired_fold_target_horizon.csv", index=False, encoding="utf-8-sig")
    fold.to_csv(RESULT_DIR / "fold_comparison.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only out-of-fold predictions and causal origin-time features",
        "external_scoring_data_accessed": False,
        "selected_policy": PARAMETERS,
        "overall_persistence_mape": overall_persistence,
        "overall_selected_mape": overall_selected,
        "relative_mape_reduction": (overall_persistence - overall_selected) / overall_persistence,
        "fold_target_horizon_comparisons": int(len(pair)),
        "fold_target_horizon_wins": int(pair["selected_wins"].sum()),
        "bootstrap": bootstrap,
    }
    (RESULT_DIR / "robustness_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
