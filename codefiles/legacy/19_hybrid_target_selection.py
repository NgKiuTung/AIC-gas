"""Select a low-dimensional absolute/relative-residual hybrid per target."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ABS_PATH = ROOT / "results" / "experiments" / "xgboost_direct" / "raw_oof_predictions_long.csv"
REL_PATH = ROOT / "results" / "experiments" / "relative_residual" / "raw_oof_predictions_long.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "hybrid_target"
WEIGHTS_REL = np.round(np.arange(0.0, 1.0001, 0.25), 2)
BETA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
BOOTSTRAP_REPEATS = 5000
SEED = 20260803


def load() -> pd.DataFrame:
    absolute = pd.read_csv(ABS_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    absolute = absolute[absolute["model"] == "direct_d5"].copy()
    relative = pd.read_csv(REL_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    keys = ["datetime", "fold", "target", "horizon_step", "horizon_minutes", "actual", "current"]
    joined = absolute[keys + ["raw_correction"]].merge(
        relative[keys + ["raw_correction"]], on=keys, validate="one_to_one",
        suffixes=("_absolute", "_relative"),
    )
    if len(joined) != len(absolute):
        raise ValueError("Absolute and relative OOF predictions do not align")
    return joined


def equal_group_mape(frame: pd.DataFrame, ape: np.ndarray) -> float:
    detail = frame[["fold", "horizon_step"]].copy()
    detail["ape"] = ape
    return float(detail.groupby(["fold", "horizon_step"])["ape"].mean().mean())


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    selected_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    search_rows: list[dict[str, object]] = []

    for target, group in data.groupby("target"):
        h = (group["horizon_step"].to_numpy(dtype=float) - 1.0) / 7.0
        truth = group["actual"].to_numpy(dtype=float)
        current = group["current"].to_numpy(dtype=float)
        best: dict[str, object] | None = None
        for weight_rel in WEIGHTS_REL:
            correction = (
                weight_rel * group["raw_correction_relative"].to_numpy(dtype=float)
                + (1.0 - weight_rel) * group["raw_correction_absolute"].to_numpy(dtype=float)
            )
            for beta_start in BETA_GRID:
                for beta_end in BETA_GRID:
                    beta = beta_start + (beta_end - beta_start) * h
                    pred = np.maximum(current + beta * correction, 0.0)
                    ape = np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-6)
                    value = equal_group_mape(group, ape)
                    row = {
                        "target": target, "weight_relative": float(weight_rel),
                        "weight_absolute": float(1.0 - weight_rel),
                        "beta_h15": float(beta_start), "beta_h120": float(beta_end),
                        "mean_mape": value,
                    }
                    search_rows.append(row)
                    if best is None or value < float(best["mean_mape"]):
                        best = row
        assert best is not None
        selected_rows.append(best)
        correction = (
            float(best["weight_relative"]) * group["raw_correction_relative"].to_numpy(dtype=float)
            + float(best["weight_absolute"]) * group["raw_correction_absolute"].to_numpy(dtype=float)
        )
        beta = float(best["beta_h15"]) + (float(best["beta_h120"]) - float(best["beta_h15"])) * h
        output = group[["datetime", "fold", "target", "horizon_step", "horizon_minutes", "actual", "current"]].copy()
        output["beta"] = beta
        output["prediction"] = np.maximum(current + beta * correction, 0.0)
        output["ape_persistence"] = np.abs(current - truth) / np.maximum(np.abs(truth), 1e-6)
        output["ape_hybrid"] = np.abs(output["prediction"].to_numpy() - truth) / np.maximum(np.abs(truth), 1e-6)
        prediction_parts.append(output)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = predictions.groupby(["fold", "target", "horizon_step", "horizon_minutes"], as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), hybrid_mape=("ape_hybrid", "mean")
    )
    metrics["absolute_mape_reduction"] = metrics["persistence_mape"] - metrics["hybrid_mape"]
    metrics["hybrid_wins"] = metrics["absolute_mape_reduction"] > 0
    fold_summary = predictions.groupby("fold", as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), hybrid_mape=("ape_hybrid", "mean")
    )
    overall_persistence = float(metrics["persistence_mape"].mean())
    overall_hybrid = float(metrics["hybrid_mape"].mean())

    daily = predictions.assign(origin_date=predictions["datetime"].dt.date.astype(str)).groupby("origin_date", as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), hybrid_mape=("ape_hybrid", "mean")
    )
    daily_improvement = (daily["persistence_mape"] - daily["hybrid_mape"]).to_numpy()
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(daily_improvement), size=(BOOTSTRAP_REPEATS, len(daily_improvement)))
    bootstrap = daily_improvement[indices].mean(axis=1)

    pd.DataFrame(search_rows).to_csv(RESULT_DIR / "parameter_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(RESULT_DIR / "selected_parameters.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULT_DIR / "hybrid_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "metrics_by_fold_target_horizon.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(RESULT_DIR / "fold_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only out-of-fold predictions",
        "external_scoring_data_accessed": False,
        "overall_persistence_mape": overall_persistence,
        "overall_hybrid_mape": overall_hybrid,
        "overall_hybrid_score": 1.0 - overall_hybrid,
        "relative_mape_reduction": (overall_persistence - overall_hybrid) / overall_persistence,
        "fold_target_horizon_wins": int(metrics["hybrid_wins"].sum()),
        "fold_target_horizon_comparisons": int(len(metrics)),
        "selected_parameters": selected_rows,
        "daily_block_bootstrap": {
            "days": int(len(daily_improvement)), "repeats": BOOTSTRAP_REPEATS,
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "probability_improvement_positive": float(np.mean(bootstrap > 0)),
            "fraction_days_hybrid_wins": float(np.mean(daily_improvement > 0)),
        },
    }
    (RESULT_DIR / "hybrid_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
