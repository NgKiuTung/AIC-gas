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


def _input_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2025-05-01", periods=3, freq="15min")
    return pd.DataFrame(
        {
            "datetime": timestamps,
            "blast_furnace_1": np.full(3, 123.0),
            "feat_hour": timestamps.hour.astype(float),
        }
    )


def test_submission_zip_has_exact_member_and_round_trips(tmp_path) -> None:
    frame = _frame()
    path = build_submission_zip(frame, _input_frame(), tmp_path / official_zip_name("synthetic_team"))
    result = validate_submission_zip(
        path,
        expected_datetimes=frame["datetime"],
        expected_team_name="synthetic_team",
        expected_feature_schema=["blast_furnace_1", "feat_hour"],
    )
    assert result["zip_members"] == ["input.csv", "result.csv"]
    assert result["input_rows"] == 3
    assert result["input_columns"] == 3
    assert result["result_rows"] == 3
    assert result["result_columns"] == 17


def test_submission_zip_validator_rejects_extra_member(tmp_path) -> None:
    path = build_submission_zip(_frame(), _input_frame(), tmp_path / "bad.zip")
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("extra.txt", "not allowed")
    with pytest.raises(ValueError):
        validate_submission_zip(path)


def test_official_zip_name_sanitizes_team_name() -> None:
    assert official_zip_name("AIC Gas Team") == "AIC_Gas_Team_gas_predict_prelim.zip"


def test_submission_zip_validator_rejects_wrong_outer_filename(tmp_path) -> None:
    path = build_submission_zip(_frame(), _input_frame(), tmp_path / "wrong_name.zip")
    with pytest.raises(ValueError, match="ZIP name"):
        validate_submission_zip(path, expected_team_name="AIC-gas")


def test_submission_zip_validator_rejects_noncanonical_datetime_text(tmp_path) -> None:
    path = build_submission_zip(
        _frame(),
        _input_frame(),
        tmp_path / official_zip_name("AIC-gas"),
    )
    with zipfile.ZipFile(path, "r") as archive:
        input_text = archive.read("input.csv")
        text = archive.read("result.csv").decode("utf-8")
    text = text.replace("2025-05-01 00:00:00", "2025/05/01 00:00", 1)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("input.csv", input_text)
        archive.writestr("result.csv", text)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_submission_zip(path, expected_team_name="AIC-gas")
