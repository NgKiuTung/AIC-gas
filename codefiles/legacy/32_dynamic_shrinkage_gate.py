"""Tune a low-dimensional confidence gate from OOF correction magnitude."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "experiments" / "cleaning_group_ablation" / "raw_oof_predictions.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "dynamic_shrinkage"
VARIANT = "core_smooth_transition_ratio_clean"
BETA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08)
GATE_FLOORS = (0.0, 0.25, 0.50, 0.75, 1.0)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    raw = raw[raw["variant"] == VARIANT].copy()
    selected_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    search_rows: list[dict[str, object]] = []
    for target, group in raw.groupby("target"):
        group = group.reset_index(drop=True)
        truth = group["actual"].to_numpy(dtype=float)
        current = group["current"].to_numpy(dtype=float)
        correction = group["raw_correction"].to_numpy(dtype=float)
        h = (group["horizon_step"].to_numpy(dtype=float) - 1.0) / 7.0
        predicted_relative_magnitude = np.abs(correction) / np.maximum(np.abs(current), 1e-6)
        group_size = group.groupby(["fold", "horizon_step"])["actual"].transform("size").to_numpy(dtype=float)
        equal_group_weight = 1.0 / group_size / group.groupby(["fold", "horizon_step"]).ngroups
        best: dict[str, object] | None = None
        for threshold in THRESHOLDS:
            magnitude_gate = np.ones(len(group)) if threshold == 0 else np.minimum(1.0, predicted_relative_magnitude / threshold)
            for floor in GATE_FLOORS:
                gate = floor + (1.0 - floor) * magnitude_gate
                for start in BETA_GRID:
                    for end in BETA_GRID:
                        beta = start + (end - start) * h
                        pred = np.maximum(current + beta * gate * correction, 0.0)
                        ape = np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-6)
                        value = float(np.sum(ape * equal_group_weight))
                        row = {
                            "target": target, "threshold": threshold, "gate_floor": floor,
                            "beta_h15": float(start), "beta_h120": float(end), "mean_mape": value,
                        }
                        search_rows.append(row)
                        if best is None or value < float(best["mean_mape"]):
                            best = row
        assert best is not None
        selected_rows.append(best)
        threshold = float(best["threshold"])
        magnitude_gate = np.ones(len(group)) if threshold == 0 else np.minimum(1.0, predicted_relative_magnitude / threshold)
        gate = float(best["gate_floor"]) + (1.0 - float(best["gate_floor"])) * magnitude_gate
        beta = float(best["beta_h15"]) + (float(best["beta_h120"]) - float(best["beta_h15"])) * h
        output = group.copy()
        output["gate"] = gate
        output["beta"] = beta
        output["prediction"] = np.maximum(current + beta * gate * correction, 0.0)
        output["ape_persistence"] = np.abs(current - truth) / np.maximum(np.abs(truth), 1e-6)
        output["ape_model"] = np.abs(output["prediction"].to_numpy() - truth) / np.maximum(np.abs(truth), 1e-6)
        prediction_parts.append(output)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = predictions.groupby(["fold", "target", "horizon_step", "horizon_minutes"], as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), model_mape=("ape_model", "mean")
    )
    fold = predictions.groupby("fold", as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), model_mape=("ape_model", "mean")
    )
    overall = float(metrics["model_mape"].mean())
    persistence = float(metrics["persistence_mape"].mean())
    pd.DataFrame(search_rows).to_csv(RESULT_DIR / "parameter_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(RESULT_DIR / "selected_parameters.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULT_DIR / "gated_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "metrics_by_fold_target_horizon.csv", index=False, encoding="utf-8-sig")
    fold.to_csv(RESULT_DIR / "fold_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only OOF corrections from selected cleaning feature model",
        "external_scoring_data_accessed": False,
        "selected_parameters": selected_rows, "persistence_mape": persistence,
        "model_mape": overall, "score": 1.0 - overall,
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
