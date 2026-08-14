import numpy as np
import pandas as pd
import pytest
from gas_power.submission.schema import SUBMISSION_COLUMNS, build_submission_frame, validate_submission_frame


def test_build_submission_frame_has_exact_official_columns() -> None:
    timestamps = pd.date_range("2025-05-01", periods=2, freq="15min")
    predictions = {
        "generator_1": np.full((2, 8), 100.0),
        "generator_all": np.full((2, 8), 150.0),
    }
    frame = build_submission_frame(timestamps, predictions)
    assert tuple(frame.columns) == SUBMISSION_COLUMNS
    assert len(frame.columns) == 17


def test_submission_validator_rejects_duplicates_and_hierarchy_violation() -> None:
    timestamps = pd.date_range("2025-05-01", periods=2, freq="15min")
    predictions = {
        "generator_1": np.full((2, 8), 100.0),
        "generator_all": np.full((2, 8), 150.0),
    }
    frame = build_submission_frame(timestamps, predictions)
    duplicated = frame.copy()
    duplicated.loc[1, "datetime"] = duplicated.loc[0, "datetime"]
    with pytest.raises(ValueError):
        validate_submission_frame(duplicated)
    violated = frame.copy()
    violated.loc[0, "generator_all_t+15_pred"] = 99.0
    with pytest.raises(ValueError):
        validate_submission_frame(violated)
