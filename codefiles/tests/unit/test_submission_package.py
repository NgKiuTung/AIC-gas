import zipfile

import numpy as np
import pandas as pd
import pytest
from gas_power.submission.package import build_submission_zip, official_zip_name, validate_submission_zip
from gas_power.submission.schema import build_submission_frame


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2025-05-01", periods=3, freq="15min")
    return build_submission_frame(
        timestamps,
        {
            "generator_1": np.full((3, 8), 100.12345),
            "generator_all": np.full((3, 8), 150.98765),
        },
    )


def test_submission_zip_has_exact_member_and_round_trips(tmp_path) -> None:
    frame = _frame()
    path = build_submission_zip(frame, tmp_path / official_zip_name("synthetic_team"))
    result = validate_submission_zip(path, expected_datetimes=frame["datetime"])
    assert result["zip_members"] == ["result.csv"]
    assert result["rows"] == 3
    assert result["columns"] == 17


def test_submission_zip_validator_rejects_extra_member(tmp_path) -> None:
    path = build_submission_zip(_frame(), tmp_path / "bad.zip")
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("extra.txt", "not allowed")
    with pytest.raises(ValueError):
        validate_submission_zip(path)


def test_official_zip_name_sanitizes_team_name() -> None:
    assert official_zip_name("AIC Gas Team") == "AIC_Gas_Team_gas_predict_prelim.zip"
