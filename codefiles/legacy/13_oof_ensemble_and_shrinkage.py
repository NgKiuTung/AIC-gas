"""Low-dimensional OOF ensemble and horizon-shrinkage search.

Uses only predictions already produced by training-only rolling validation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "experiments" / "xgboost_direct" / "raw_oof_predictions_long.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "oof_ensemble"
WEIGHTS_D5 = np.round(np.arange(0.0, 1.0001, 0.25), 2)
BETA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def prepare() -> pd.DataFrame:
    raw = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    keys = ["datetime", "fold", "target", "horizon_step", "horizon_minutes", "actual", "current"]
    wide = raw.pivot_table(index=keys, columns="model", values="raw_correction", aggfunc="first").reset_index()
    if wide[["direct_d3", "direct_d5"]].isna().any().any():
        raise ValueError("OOF model predictions are not aligned")
    return wide


def score(frame: pd.DataFrame, correction: np.ndarray, beta: np.ndarray) -> tuple[float, pd.DataFrame]:
    prediction = np.maximum(frame["current"].to_numpy() + beta * correction, 0.0)
    errors = np.abs(prediction - frame["actual"].to_numpy()) / np.maximum(np.abs(frame["actual"].to_numpy()), 1e-6)
    detail = frame[["fold", "target", "horizon_step", "horizon_minutes"]].copy()
    detail["mape_row"] = errors
    grouped = detail.groupby(["fold", "target", "horizon_step", "horizon_minutes"], as_index=False)["mape_row"].mean()
    return float(grouped["mape_row"].mean()), grouped


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare()
    search_rows: list[dict[str, object]] = []
    chosen_rows: list[dict[str, object]] = []
    metric_frames: list[pd.DataFrame] = []

    for target in ("generator_1", "generator_all"):
        part = data[data["target"] == target].copy()
        normalized_h = (part["horizon_step"].to_numpy(dtype=float) - 1.0) / 7.0
        best: dict[str, object] | None = None
        for weight_d5 in WEIGHTS_D5:
            correction = weight_d5 * part["direct_d5"].to_numpy() + (1.0 - weight_d5) * part["direct_d3"].to_numpy()
            for beta_start in BETA_GRID:
                for beta_end in BETA_GRID:
                    beta = beta_start + (beta_end - beta_start) * normalized_h
                    mean_mape, _ = score(part, correction, beta)
                    row = {
                        "target": target, "weight_d5": float(weight_d5),
                        "weight_d3": float(1.0 - weight_d5),
                        "beta_h15": float(beta_start), "beta_h120": float(beta_end),
                        "mean_mape": mean_mape,
                    }
                    search_rows.append(row)
                    if best is None or mean_mape < float(best["mean_mape"]):
                        best = row
        assert best is not None
        chosen_rows.append(best)
        correction = float(best["weight_d5"]) * part["direct_d5"].to_numpy() + float(best["weight_d3"]) * part["direct_d3"].to_numpy()
        beta = float(best["beta_h15"]) + (float(best["beta_h120"]) - float(best["beta_h15"])) * normalized_h
        _, detail = score(part, correction, beta)
        detail["model"] = "xgb_oof_ensemble_linear_shrinkage"
        detail["selected_weight_d5"] = float(best["weight_d5"])
        detail["selected_beta_h15"] = float(best["beta_h15"])
        detail["selected_beta_h120"] = float(best["beta_h120"])
        metric_frames.append(detail)

    chosen = pd.DataFrame(chosen_rows)
    metrics = pd.concat(metric_frames, ignore_index=True)
    fold_summary = metrics.groupby("fold", as_index=False).agg(mean_mape=("mape_row", "mean"))
    horizon_summary = metrics.groupby(["target", "horizon_minutes"], as_index=False).agg(mean_mape=("mape_row", "mean"))
    overall_mape = float(metrics["mape_row"].mean())

    pd.DataFrame(search_rows).to_csv(RESULT_DIR / "parameter_search.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(RESULT_DIR / "selected_parameters.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "metrics_by_fold_horizon.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(RESULT_DIR / "fold_summary.csv", index=False, encoding="utf-8-sig")
    horizon_summary.to_csv(RESULT_DIR / "horizon_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(INPUT_PATH.relative_to(ROOT)),
        "source_scope": "training-only out-of-fold predictions",
        "external_scoring_data_accessed": False,
        "policy_complexity": "per target: one d5 ensemble weight and two linear horizon endpoints",
        "overall_mape": overall_mape,
        "overall_score": 1.0 - overall_mape,
        "selected_parameters": chosen.to_dict(orient="records"),
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
