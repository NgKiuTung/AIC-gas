from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gas_power.submission.input_schema import validate_input_frame


def test_input_frame_accepts_official_raw_and_prefixed_engineered_features() -> None:
    timestamps = pd.date_range("2025-05-01", periods=3, freq="15min")
    frame = pd.DataFrame(
        {
            "datetime": timestamps,
            "blast_furnace_1": np.full(3, 100.0),
            "feat_hour": timestamps.hour.astype(float),
        }
    )
    validate_input_frame(
        frame,
        expected_datetimes=timestamps,
        expected_feature_schema=["blast_furnace_1", "feat_hour"],
    )


def test_input_frame_rejects_unprefixed_engineered_feature() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-05-01", periods=2, freq="15min"),
            "rolling_mean": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="official raw fields"):
        validate_input_frame(frame)


def test_input_frame_rejects_nonfinite_values() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-05-01", periods=2, freq="15min"),
            "feat_hour": [0.0, np.nan],
        }
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_input_frame(frame)


def test_current_observed_loads_are_legal_raw_inputs() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-04-01", periods=2, freq="15min"),
            "generator_1": [120.0, 121.0],
            "generator_all": [310.0, 311.0],
        }
    )
    validate_input_frame(frame, expected_feature_schema=["generator_1", "generator_all"])


def test_future_target_column_is_not_a_current_observation() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-04-01", periods=2, freq="15min"),
            "generator_1_t+15_pred": [120.0, 121.0],
        }
    )
    with pytest.raises(ValueError, match="official raw fields"):
        validate_input_frame(frame)
