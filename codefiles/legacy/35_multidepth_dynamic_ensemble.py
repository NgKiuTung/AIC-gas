"""Convex multi-depth OOF ensemble with correction-magnitude gating."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
D5_PATH = ROOT / "results" / "experiments" / "cleaning_group_ablation" / "raw_oof_predictions.csv"
CAPACITY_PATH = ROOT / "results" / "experiments" / "targeted_xgb_capacity" / "raw_oof_predictions.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "multidepth_ensemble"
BETA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
GATE_FLOORS = (0.0, 0.25, 0.50, 0.75, 1.0)
SIMPLEX_WEIGHTS = [
    (w5 / 4, w4 / 4, (4 - w5 - w4) / 4)
    for w5 in range(5) for w4 in range(5 - w5)
]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    d5 = pd.read_csv(D5_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    d5 = d5[d5["variant"] == "core_smooth_transition_ratio_clean"].copy()
    capacity = pd.read_csv(CAPACITY_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    id_keys = ["datetime", "fold", "target", "horizon_step", "horizon_minutes"]
    d4 = capacity[capacity["config"] == "d4_wide"]
    d6 = capacity[capacity["config"] == "d6_regularized"]
    joined = d5[id_keys + ["actual", "current", "raw_correction"]].merge(
        d4[id_keys + ["raw_correction"]], on=id_keys, validate="one_to_one",
        suffixes=("_d5", "_d4"),
    ).merge(
        d6[id_keys + ["raw_correction"]], on=id_keys, validate="one_to_one"
    ).rename(columns={"raw_correction": "raw_correction_d6"})

    constant_rows: list[dict[str, object]] = []
    gated_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    for target, group in joined.groupby("target"):
        group = group.reset_index(drop=True)
        truth = group["actual"].to_numpy(dtype=float)
        current = group["current"].to_numpy(dtype=float)
        h = (group["horizon_step"].to_numpy(dtype=float) - 1.0) / 7.0
        group_size = group.groupby(["fold", "horizon_step"])["actual"].transform("size").to_numpy(dtype=float)
        weights_equal = 1.0 / group_size / group.groupby(["fold", "horizon_step"]).ngroups
        correction_arrays = (
            group["raw_correction_d5"].to_numpy(dtype=float),
            group["raw_correction_d4"].to_numpy(dtype=float),
            group["raw_correction_d6"].to_numpy(dtype=float),
        )
        best_constant: dict[str, object] | None = None
        for weight_d5, weight_d4, weight_d6 in SIMPLEX_WEIGHTS:
            correction = weight_d5 * correction_arrays[0] + weight_d4 * correction_arrays[1] + weight_d6 * correction_arrays[2]
            for start in BETA_GRID:
                for end in BETA_GRID:
                    beta = start + (end - start) * h
                    pred = np.maximum(current + beta * correction, 0.0)
                    ape = np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-6)
                    value = float(np.sum(ape * weights_equal))
                    row = {
                        "target": target, "weight_d5": weight_d5, "weight_d4": weight_d4,
                        "weight_d6": weight_d6, "beta_h15": float(start),
                        "beta_h120": float(end), "mean_mape": value,
                    }
                    constant_rows.append(row)
                    if best_constant is None or value < float(best_constant["mean_mape"]):
                        best_constant = row
        assert best_constant is not None
        correction = (
            float(best_constant["weight_d5"]) * correction_arrays[0]
            + float(best_constant["weight_d4"]) * correction_arrays[1]
            + float(best_constant["weight_d6"]) * correction_arrays[2]
        )
        relative_magnitude = np.abs(correction) / np.maximum(np.abs(current), 1e-6)
        best_gate: dict[str, object] | None = None
        for threshold in THRESHOLDS:
            magnitude_gate = np.ones(len(group)) if threshold == 0 else np.minimum(1.0, relative_magnitude / threshold)
            for floor in GATE_FLOORS:
                gate = floor + (1.0 - floor) * magnitude_gate
                for start in BETA_GRID:
                    for end in BETA_GRID:
                        beta = start + (end - start) * h
                        pred = np.maximum(current + beta * gate * correction, 0.0)
                        ape = np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-6)
                        value = float(np.sum(ape * weights_equal))
                        row = {
                            "target": target, **{k: best_constant[k] for k in ("weight_d5", "weight_d4", "weight_d6")},
                            "threshold": threshold, "gate_floor": floor,
                            "beta_h15": float(start), "beta_h120": float(end), "mean_mape": value,
                        }
                        gated_rows.append(row)
                        if best_gate is None or value < float(best_gate["mean_mape"]):
                            best_gate = row
        assert best_gate is not None
        selected_rows.append(best_gate)
        threshold = float(best_gate["threshold"])
        magnitude_gate = np.ones(len(group)) if threshold == 0 else np.minimum(1.0, relative_magnitude / threshold)
        gate = float(best_gate["gate_floor"]) + (1.0 - float(best_gate["gate_floor"])) * magnitude_gate
        beta = float(best_gate["beta_h15"]) + (float(best_gate["beta_h120"]) - float(best_gate["beta_h15"])) * h
        output = group[id_keys + ["actual", "current"]].copy()
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
    pd.DataFrame(constant_rows).to_csv(RESULT_DIR / "constant_ensemble_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gated_rows).to_csv(RESULT_DIR / "gate_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(RESULT_DIR / "selected_parameters.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULT_DIR / "selected_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "metrics_by_fold_target_horizon.csv", index=False, encoding="utf-8-sig")
    fold.to_csv(RESULT_DIR / "fold_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only OOF predictions",
        "external_scoring_data_accessed": False,
        "selected_parameters": selected_rows, "persistence_mape": persistence,
        "model_mape": overall, "score": 1.0 - overall,
        "relative_reduction_vs_persistence": (persistence - overall) / persistence,
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
