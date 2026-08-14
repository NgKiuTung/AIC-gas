"""OOF ensemble of base and cleaning-enhanced mixed-target models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ABS_PATH = ROOT / "results" / "experiments" / "xgboost_direct" / "raw_oof_predictions_long.csv"
REL_PATH = ROOT / "results" / "experiments" / "relative_residual" / "raw_oof_predictions_long.csv"
ENHANCED_PATH = ROOT / "results" / "experiments" / "cleaning_enhanced_hybrid" / "raw_oof_predictions.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "cleaning_oof_ensemble"
WEIGHTS_ENHANCED = np.round(np.arange(0.0, 1.0001, 0.10), 2)
BETA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    absolute = pd.read_csv(ABS_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    absolute = absolute[(absolute["model"] == "direct_d5") & (absolute["target"] == "generator_all")]
    relative = pd.read_csv(REL_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    relative = relative[(relative["model"] == "direct_d5_relative") & (relative["target"] == "generator_1")]
    base = pd.concat([relative, absolute], ignore_index=True)
    enhanced = pd.read_csv(ENHANCED_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    id_keys = ["datetime", "fold", "target", "horizon_step", "horizon_minutes"]
    joined = base[id_keys + ["actual", "current", "raw_correction"]].merge(
        enhanced[id_keys + ["actual", "current", "raw_correction"]], on=id_keys, how="inner", validate="one_to_one",
        suffixes=("_base", "_enhanced"),
    )
    if len(joined) != len(enhanced):
        raise ValueError("OOF predictions do not align")
    if not np.allclose(joined["actual_base"], joined["actual_enhanced"], rtol=1e-6, atol=1e-6):
        raise ValueError("OOF actual values differ")
    if not np.allclose(joined["current_base"], joined["current_enhanced"], rtol=1e-6, atol=1e-6):
        raise ValueError("OOF current values differ")
    joined["actual"] = joined["actual_enhanced"]
    joined["current"] = joined["current_enhanced"]
    keys = [*id_keys, "actual", "current"]

    search_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    for target, group in joined.groupby("target"):
        h = (group["horizon_step"].to_numpy(dtype=float) - 1.0) / 7.0
        truth = group["actual"].to_numpy(dtype=float)
        current = group["current"].to_numpy(dtype=float)
        best: dict[str, object] | None = None
        for weight_enhanced in WEIGHTS_ENHANCED:
            correction = (
                weight_enhanced * group["raw_correction_enhanced"].to_numpy(dtype=float)
                + (1.0 - weight_enhanced) * group["raw_correction_base"].to_numpy(dtype=float)
            )
            for start in BETA_GRID:
                for end in BETA_GRID:
                    beta = start + (end - start) * h
                    pred = np.maximum(current + beta * correction, 0.0)
                    ape = np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-6)
                    detail = group[["fold", "horizon_step"]].copy()
                    detail["ape"] = ape
                    value = float(detail.groupby(["fold", "horizon_step"])["ape"].mean().mean())
                    row = {
                        "target": target, "weight_enhanced": float(weight_enhanced),
                        "weight_base": float(1.0 - weight_enhanced),
                        "beta_h15": float(start), "beta_h120": float(end), "mean_mape": value,
                    }
                    search_rows.append(row)
                    if best is None or value < float(best["mean_mape"]):
                        best = row
        assert best is not None
        selected_rows.append(best)
        correction = (
            float(best["weight_enhanced"]) * group["raw_correction_enhanced"].to_numpy(dtype=float)
            + float(best["weight_base"]) * group["raw_correction_base"].to_numpy(dtype=float)
        )
        beta = float(best["beta_h15"]) + (float(best["beta_h120"]) - float(best["beta_h15"])) * h
        output = group[keys].copy()
        output["prediction"] = np.maximum(current + beta * correction, 0.0)
        output["ape_persistence"] = np.abs(current - truth) / np.maximum(np.abs(truth), 1e-6)
        output["ape_ensemble"] = np.abs(output["prediction"].to_numpy() - truth) / np.maximum(np.abs(truth), 1e-6)
        prediction_parts.append(output)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = predictions.groupby(["fold", "target", "horizon_step", "horizon_minutes"], as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), ensemble_mape=("ape_ensemble", "mean")
    )
    fold = predictions.groupby("fold", as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), ensemble_mape=("ape_ensemble", "mean")
    )
    overall = float(metrics["ensemble_mape"].mean())
    persistence = float(metrics["persistence_mape"].mean())
    pd.DataFrame(search_rows).to_csv(RESULT_DIR / "parameter_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(RESULT_DIR / "selected_parameters.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULT_DIR / "ensemble_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "metrics_by_fold_target_horizon.csv", index=False, encoding="utf-8-sig")
    fold.to_csv(RESULT_DIR / "fold_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only OOF predictions",
        "external_scoring_data_accessed": False,
        "selected_parameters": selected_rows, "persistence_mape": persistence,
        "ensemble_mape": overall, "score": 1.0 - overall,
        "relative_reduction_vs_persistence": (persistence - overall) / persistence,
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
