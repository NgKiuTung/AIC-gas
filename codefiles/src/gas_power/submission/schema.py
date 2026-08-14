"""Official preliminary-round 17-column prediction contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS_MINUTES = tuple(range(15, 121, 15))
TARGETS = ("generator_1", "generator_all")
SUBMISSION_COLUMNS = (
    "datetime",
    *(f"{target}_t+{minutes}_pred" for target in TARGETS for minutes in HORIZONS_MINUTES),
)


def build_submission_frame(
    reference_times: pd.Series | pd.DatetimeIndex | np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build, but do not write, an official-schema prediction table."""
    datetimes = pd.to_datetime(reference_times, errors="raise")
    frame = pd.DataFrame({"datetime": datetimes})
    for target in TARGETS:
        if target not in predictions:
            raise ValueError(f"Missing target prediction: {target}")
        values = np.asarray(predictions[target], dtype=float)
        if values.shape != (len(frame), len(HORIZONS_MINUTES)):
            raise ValueError(f"{target} prediction shape is {values.shape}, expected {(len(frame), 8)}")
        for index, minutes in enumerate(HORIZONS_MINUTES):
            frame[f"{target}_t+{minutes}_pred"] = values[:, index]
    validate_submission_frame(frame)
    return frame.loc[:, list(SUBMISSION_COLUMNS)]


def validate_submission_frame(
    frame: pd.DataFrame,
    *,
    expected_datetimes: pd.Series | pd.DatetimeIndex | np.ndarray | None = None,
) -> None:
    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError("Submission columns or order do not match the official 17-column schema")
    datetimes = pd.to_datetime(frame["datetime"], errors="raise")
    if datetimes.duplicated().any():
        raise ValueError("Submission contains duplicate reference timestamps")
    if not datetimes.is_monotonic_increasing:
        raise ValueError("Submission timestamps are not strictly increasing")
    if expected_datetimes is not None:
        expected = pd.DatetimeIndex(pd.to_datetime(expected_datetimes, errors="raise"))
        if not pd.DatetimeIndex(datetimes).equals(expected):
            raise ValueError("Submission timestamps do not exactly cover the expected reference times")
    values = frame.loc[:, list(SUBMISSION_COLUMNS[1:])].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Submission contains NaN or Inf")
    if (values < 0).any():
        raise ValueError("Submission contains negative power predictions")
    generator_1 = frame[[f"generator_1_t+{minutes}_pred" for minutes in HORIZONS_MINUTES]].to_numpy()
    generator_all = frame[[f"generator_all_t+{minutes}_pred" for minutes in HORIZONS_MINUTES]].to_numpy()
    if (generator_all < generator_1).any():
        raise ValueError("Submission violates generator_all >= generator_1")


def write_submission_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write UTF-8 with the official timestamp format and at least 3 decimals."""
    validate_submission_frame(frame)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = frame.copy()
    serializable["datetime"] = pd.to_datetime(serializable["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    serializable.to_csv(destination, index=False, encoding="utf-8", float_format="%.3f")
    return destination
