from __future__ import annotations

import numpy as np
import pandas as pd
from gas_power.forecasting.production import apply_frozen_ensemble, build_mixed_targets


def test_build_mixed_targets_matches_target_design() -> None:
    frame = pd.DataFrame(
        {
            "feat_p50_current": [10.0],
            "feat_p120_current": [20.0],
            **{f"label_p50_h{horizon}": [12.0] for horizon in range(1, 9)},
            **{f"label_p120_h{horizon}": [23.0] for horizon in range(1, 9)},
        }
    )
    mixed = build_mixed_targets(frame)
    assert mixed.shape == (1, 16)
    np.testing.assert_allclose(mixed[:, :8], 0.2)
    np.testing.assert_allclose(mixed[:, 8:], 5.0)


def test_frozen_ensemble_uses_only_nonzero_components() -> None:
    raw_d6 = np.concatenate([np.full((2, 8), 0.1), np.full((2, 8), 5.0)], axis=1)
    ensemble_rows = [
        {
            "target": "generator_1",
            "weight_d5": 0.0,
            "weight_d4": 0.0,
            "weight_d6": 1.0,
            "threshold": 0.0,
            "gate_floor": 1.0,
            "beta_h15": 1.0,
            "beta_h120": 1.0,
        },
        {
            "target": "generator_all",
            "weight_d5": 0.0,
            "weight_d4": 0.0,
            "weight_d6": 1.0,
            "threshold": 0.0,
            "gate_floor": 1.0,
            "beta_h15": 1.0,
            "beta_h120": 1.0,
        },
    ]
    prediction = apply_frozen_ensemble(
        {"d6": raw_d6}, np.array([10.0, 20.0]), np.array([30.0, 40.0]), ensemble_rows
    )
    np.testing.assert_allclose(prediction["generator_1"], [[11.0] * 8, [22.0] * 8])
    np.testing.assert_allclose(prediction["generator_all"], [[35.0] * 8, [45.0] * 8])


def test_frozen_ensemble_projects_generator_hierarchy() -> None:
    raw = np.concatenate([np.full((1, 8), 0.5), np.zeros((1, 8))], axis=1)
    rows = [
        {
            "target": target,
            "weight_d5": 0.0,
            "weight_d4": 0.0,
            "weight_d6": 1.0,
            "threshold": 0.0,
            "gate_floor": 1.0,
            "beta_h15": 1.0,
            "beta_h120": 1.0,
        }
        for target in ("generator_1", "generator_all")
    ]
    prediction = apply_frozen_ensemble({"d6": raw}, np.array([20.0]), np.array([21.0]), rows)
    np.testing.assert_allclose(prediction["generator_1"], [[30.0] * 8])
    np.testing.assert_allclose(prediction["generator_all"], [[30.0] * 8])
