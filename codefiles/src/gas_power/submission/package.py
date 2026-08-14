"""Create and validate the exact preliminary-round ZIP contract."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from gas_power.submission.schema import validate_submission_frame

ZIP_MEMBER_NAME = "result.csv"


def official_zip_name(team_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", team_name.strip()).strip("_")
    if not normalized:
        raise ValueError("Team name must contain at least one safe filename character")
    return f"{normalized}_gas_predict_prelim.zip"


def _submission_csv_bytes(frame: pd.DataFrame) -> bytes:
    validate_submission_frame(frame)
    serializable = frame.copy()
    serializable["datetime"] = pd.to_datetime(serializable["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    buffer = io.StringIO(newline="")
    serializable.to_csv(buffer, index=False, lineterminator="\n", float_format="%.3f")
    return buffer.getvalue().encode("utf-8")


def build_submission_zip(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Write a deterministic ZIP containing exactly one UTF-8 result.csv."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _submission_csv_bytes(frame)
    info = zipfile.ZipInfo(ZIP_MEMBER_NAME, date_time=(2025, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(info, payload)
    return output


def validate_submission_zip(
    path: str | Path,
    *,
    expected_datetimes: pd.Series | pd.DatetimeIndex | None = None,
) -> dict[str, object]:
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != ZIP_MEMBER_NAME:
            raise ValueError("Submission ZIP must contain exactly one root-level result.csv")
        if members[0].is_dir() or members[0].file_size == 0:
            raise ValueError("result.csv is empty or is not a file")
        payload = archive.read(ZIP_MEMBER_NAME)
        if archive.testzip() is not None:
            raise ValueError("Submission ZIP CRC validation failed")
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("result.csv is not UTF-8") from error
    frame = pd.read_csv(io.StringIO(decoded))
    validate_submission_frame(frame, expected_datetimes=expected_datetimes)
    prediction_columns = [column for column in frame if column != "datetime"]
    decimal_contract = all(
        all("." in token and len(token.rsplit(".", 1)[1]) >= 3 for token in decoded.splitlines()[row].split(",")[1:])
        for row in range(1, len(frame) + 1)
    )
    if not decimal_contract:
        raise ValueError("Prediction fields do not retain at least three decimal places")
    return {
        "zip_path": str(archive_path),
        "zip_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "zip_members": [ZIP_MEMBER_NAME],
        "csv_bytes": len(payload),
        "rows": len(frame),
        "columns": len(frame.columns),
        "prediction_columns": len(prediction_columns),
        "utf8": True,
        "minimum_three_decimals": True,
        "all_checks_pass": True,
    }
