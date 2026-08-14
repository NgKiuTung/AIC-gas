"""Production target construction and frozen multi-depth ensemble logic."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

HORIZONS = tuple(range(1, 9))


def build_mixed_targets(frame: pd.DataFrame) -> np.ndarray:
    generator_1 = frame[[f"label_p50_h{horizon}" for horizon in HORIZONS]].to_numpy(dtype=np.float32)
    generator_120 = frame[[f"label_p120_h{horizon}" for horizon in HORIZONS]].to_numpy(dtype=np.float32)
    generator_all = generator_1 + generator_120
    current_1 = frame["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
    current_all = (
        frame["feat_p50_current"].to_numpy(dtype=np.float32)
        + frame["feat_p120_current"].to_numpy(dtype=np.float32)
    )[:, None]
    relative_1 = (generator_1 - current_1) / np.maximum(np.abs(current_1), 1e-6)
    absolute_all = generator_all - current_all
    return np.concatenate([relative_1, absolute_all], axis=1)


def apply_frozen_ensemble(
    component_raw: Mapping[str, np.ndarray],
    current_1: np.ndarray,
    current_all: np.ndarray,
    ensemble_rows: list[dict[str, object]],
    *,
    enforce_hierarchy: bool = True,
) -> dict[str, np.ndarray]:
    current_by_target = {
        "generator_1": np.asarray(current_1, dtype=np.float64).reshape(-1, 1),
        "generator_all": np.asarray(current_all, dtype=np.float64).reshape(-1, 1),
    }
    output: dict[str, np.ndarray] = {}
    horizon_fraction = np.arange(8, dtype=np.float64) / 7.0
    for row in ensemble_rows:
        target = str(row["target"])
        current = current_by_target[target]
        correction = np.zeros((len(current), 8), dtype=np.float64)
        for component, weight_key in (("d5", "weight_d5"), ("d4", "weight_d4"), ("d6", "weight_d6")):
            weight = float(row[weight_key])
            if weight == 0.0:
                continue
            if component not in component_raw:
                raise KeyError(f"Missing required component prediction: {component}")
            raw = np.asarray(component_raw[component], dtype=np.float64)
            component_correction = current * raw[:, :8] if target == "generator_1" else raw[:, 8:]
            correction += weight * component_correction
        threshold = float(row["threshold"])
        if threshold == 0.0:
            gate = np.ones_like(correction)
        else:
            magnitude = np.abs(correction) / np.maximum(np.abs(current), 1e-6)
            magnitude_gate = np.minimum(1.0, magnitude / threshold)
            floor = float(row["gate_floor"])
            gate = floor + (1.0 - floor) * magnitude_gate
        beta = float(row["beta_h15"]) + (
            float(row["beta_h120"]) - float(row["beta_h15"])
        ) * horizon_fraction
        output[target] = np.maximum(current + beta.reshape(1, -1) * gate * correction, 0.0)
    if enforce_hierarchy and "generator_1" in output and "generator_all" in output:
        output["generator_all"] = np.maximum(output["generator_all"], output["generator_1"])
    return output
