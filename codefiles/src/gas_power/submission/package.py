"""Create and validate the exact preliminary-round ZIP contract."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from gas_power.submission.input_schema import validate_input_frame
from gas_power.submission.schema import SUBMISSION_COLUMNS, validate_submission_frame

ZIP_MEMBER_NAMES = ("input.csv", "result.csv")


def official_zip_name(team_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", team_name.strip()).strip("_")
    if not normalized:
        raise ValueError("Team name must contain at least one safe filename character")
    return f"{normalized}_gas_predict_prelim.zip"


def _result_csv_bytes(frame: pd.DataFrame) -> bytes:
    validate_submission_frame(frame)
    serializable = frame.copy()
    serializable["datetime"] = pd.to_datetime(serializable["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    buffer = io.StringIO(newline="")
    serializable.to_csv(buffer, index=False, lineterminator="\n", float_format="%.3f")
    return buffer.getvalue().encode("utf-8")


def _input_csv_bytes(frame: pd.DataFrame) -> bytes:
    validate_input_frame(frame)
    serializable = frame.copy()
    serializable["datetime"] = pd.to_datetime(serializable["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    buffer = io.StringIO(newline="")
    serializable.to_csv(buffer, index=False, lineterminator="\n", float_format="%.9f")
    return buffer.getvalue().encode("utf-8")


def _member_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2025, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def build_submission_zip(
    result_frame: pd.DataFrame,
    input_frame: pd.DataFrame,
    destination: str | Path,
) -> Path:
    """Write a deterministic ZIP containing exactly input.csv and result.csv."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        "input.csv": _input_csv_bytes(input_frame),
        "result.csv": _result_csv_bytes(result_frame),
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ZIP_MEMBER_NAMES:
            archive.writestr(_member_info(name), payloads[name])
    return output


def validate_submission_zip(
    path: str | Path,
    *,
    expected_datetimes: pd.Series | pd.DatetimeIndex | None = None,
    expected_team_name: str | None = None,
    expected_feature_schema: list[str] | None = None,
) -> dict[str, object]:
    archive_path = Path(path)
    if expected_team_name is not None and archive_path.name != official_zip_name(expected_team_name):
        raise ValueError(
            f"Submission ZIP name must be {official_zip_name(expected_team_name)!r}, "
            f"got {archive_path.name!r}"
        )
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(members) != 2 or set(names) != set(ZIP_MEMBER_NAMES):
            raise ValueError("Submission ZIP must contain exactly root-level input.csv and result.csv")
        if any(member.is_dir() or member.file_size == 0 for member in members):
            raise ValueError("Submission CSV members must be non-empty files")
        if any(member.flag_bits & 0x1 for member in members):
            raise ValueError("Submission CSV members must not be encrypted")
        payloads = {name: archive.read(name) for name in ZIP_MEMBER_NAMES}
        if archive.testzip() is not None:
            raise ValueError("Submission ZIP CRC validation failed")
    decoded: dict[str, str] = {}
    for name, payload in payloads.items():
        try:
            decoded[name] = payload.decode("utf-8").removeprefix("\ufeff")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} is not UTF-8") from error

    result_rows = list(csv.reader(io.StringIO(decoded["result.csv"])))
    if not result_rows or tuple(result_rows[0]) != SUBMISSION_COLUMNS:
        raise ValueError("result.csv header does not exactly match the official 17-column schema")
    if any(len(row) != len(result_rows[0]) for row in result_rows[1:]):
        raise ValueError("result.csv contains a malformed or incomplete row")
    datetime_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
    if any(not datetime_pattern.fullmatch(row[0]) for row in result_rows[1:]):
        raise ValueError("result.csv datetime must use the exact YYYY-MM-DD HH:MM:SS text format")
    decimal_pattern = re.compile(r"^[+-]?\d+\.\d{3,}$")
    if any(not decimal_pattern.fullmatch(token) for row in result_rows[1:] for token in row[1:]):
        raise ValueError("Prediction fields must use fixed-point text with at least three decimal places")
    result_frame = pd.read_csv(io.StringIO(decoded["result.csv"]))
    if len(result_rows) != len(result_frame) + 1:
        raise ValueError("result.csv contains blank or non-tabular records")

    input_rows = list(csv.reader(io.StringIO(decoded["input.csv"])))
    if not input_rows or not input_rows[0] or input_rows[0][0] != "datetime":
        raise ValueError("input.csv must begin with a datetime column")
    if any(len(row) != len(input_rows[0]) for row in input_rows[1:]):
        raise ValueError("input.csv contains a malformed or incomplete row")
    if any(not datetime_pattern.fullmatch(row[0]) for row in input_rows[1:]):
        raise ValueError("input.csv datetime must use the exact YYYY-MM-DD HH:MM:SS text format")
    input_frame = pd.read_csv(io.StringIO(decoded["input.csv"]))
    if len(input_rows) != len(input_frame) + 1:
        raise ValueError("input.csv contains blank or non-tabular records")

    validate_submission_frame(result_frame, expected_datetimes=expected_datetimes)
    validate_input_frame(
        input_frame,
        expected_datetimes=expected_datetimes,
        expected_feature_schema=expected_feature_schema,
    )
    if not pd.DatetimeIndex(pd.to_datetime(input_frame["datetime"])).equals(
        pd.DatetimeIndex(pd.to_datetime(result_frame["datetime"]))
    ):
        raise ValueError("input.csv and result.csv must use the same rolling-origin timestamps")
    prediction_columns = [column for column in result_frame if column != "datetime"]
    return {
        "zip_path": str(archive_path),
        "zip_filename": archive_path.name,
        "official_filename_match": None
        if expected_team_name is None
        else archive_path.name == official_zip_name(expected_team_name),
        "zip_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "zip_members": list(ZIP_MEMBER_NAMES),
        "root_level_exact_two_files": True,
        "unencrypted": True,
        "crc_pass": True,
        "input_csv_bytes": len(payloads["input.csv"]),
        "result_csv_bytes": len(payloads["result.csv"]),
        "input_rows": len(input_frame),
        "input_columns": len(input_frame.columns),
        "input_features": len(input_frame.columns) - 1,
        "result_rows": len(result_frame),
        "result_columns": len(result_frame.columns),
        "prediction_columns": len(prediction_columns),
        "expected_datetime_grid_match": None if expected_datetimes is None else True,
        "utf8": True,
        "datetime_text_format": "YYYY-MM-DD HH:MM:SS",
        "minimum_three_decimals": True,
        "all_checks_pass": True,
    }
