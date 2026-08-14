"""Explicit, fail-closed readers for training and final scoring raw tables."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from gas_power.data.io import SubmissionAccessGrant, assert_submission_input_path, assert_training_data_path

TRAINING_FILENAMES = {
    "gas": "Pre_gas.csv",
    "holder": "Pre_gas_holder.csv",
    "user": "Pre_gas_user.csv",
    "load": "Pre_load.csv",
}
SCORING_FILENAMES = {
    "gas": "Pre_test_gas.csv",
    "holder": "Pre_test_gas_holder.csv",
    "user": "Pre_test_gas_user.csv",
    "load": "Pre_test_load.csv",
}


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "datetime" not in frame:
        raise ValueError(f"Missing datetime in {path.name}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    if frame["datetime"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path.name}")
    return frame.sort_values("datetime").reset_index(drop=True)


def load_training_raw_tables(directory: str | Path) -> dict[str, pd.DataFrame]:
    root = assert_training_data_path(directory)
    return {source: _read_csv(root / filename) for source, filename in TRAINING_FILENAMES.items()}


def load_scoring_raw_tables(
    directory: str | Path,
    grant: SubmissionAccessGrant | None,
) -> dict[str, pd.DataFrame]:
    """Read only four exact files; never enumerate or mutate the input directory."""
    root = assert_submission_input_path(directory, grant)
    return {source: _read_csv(root / filename) for source, filename in SCORING_FILENAMES.items()}


def combine_history_and_scoring_tables(
    training: Mapping[str, pd.DataFrame],
    scoring: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    if set(training) != set(TRAINING_FILENAMES) or set(scoring) != set(SCORING_FILENAMES):
        raise ValueError("Training and scoring sources must both contain gas, holder, user, and load")
    reference_times = pd.DatetimeIndex(pd.to_datetime(scoring["load"]["datetime"], errors="raise"))
    if reference_times.empty or reference_times.duplicated().any() or not reference_times.is_monotonic_increasing:
        raise ValueError("Scoring load timestamps must be non-empty, unique, and increasing")
    if not reference_times.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Scoring load timestamps must form a complete 15-minute grid")
    scoring_start = reference_times.min()
    combined: dict[str, pd.DataFrame] = {}
    for source in TRAINING_FILENAMES:
        history = training[source].copy()
        future = scoring[source].copy()
        if set(history.columns) != set(future.columns):
            missing = sorted(set(history.columns) - set(future.columns))
            extra = sorted(set(future.columns) - set(history.columns))
            raise ValueError(f"{source} schema mismatch: missing={missing}, extra={extra}")
        if not pd.DatetimeIndex(future["datetime"]).isin(reference_times).all():
            raise ValueError(f"{source} contains timestamps outside scoring load reference grid")
        # Official training ends strictly before the first scoring reference time.
        history = history.loc[history["datetime"] < scoring_start, future.columns]
        merged = pd.concat([history, future], ignore_index=True).sort_values("datetime").reset_index(drop=True)
        if merged["datetime"].duplicated().any():
            raise ValueError(f"{source} overlaps at the training/scoring boundary")
        combined[source] = merged
    return combined, reference_times


def price_lookup_from_frame(frame: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Parse first-column half-hour intervals plus 12 positional month columns."""
    if frame.shape[1] < 13:
        raise ValueError("Price workbook requires one interval column and 12 month columns")
    month_columns = list(frame.columns[1:13])
    lookup: dict[tuple[int, int], float] = {}
    for _, row in frame.iterrows():
        match = re.search(r"(\d{1,2}):(\d{2})", str(row.iloc[0]))
        if match is None:
            raise ValueError(f"Unrecognized price interval: {row.iloc[0]}")
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour not in range(24) or minute not in (0, 30):
            raise ValueError(f"Price interval is not a half-hour boundary: {row.iloc[0]}")
        slot = hour * 2 + int(minute == 30)
        for month, column in enumerate(month_columns, start=1):
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(value):
                raise ValueError(f"Missing price for month={month}, slot={slot}")
            lookup[(month, slot)] = float(value)
    expected = {(month, slot) for month in range(1, 13) for slot in range(48)}
    if set(lookup) != expected:
        raise ValueError(f"Price lookup coverage mismatch: keys={len(lookup)}, expected={len(expected)}")
    return lookup


def load_price_lookup(path: str | Path) -> dict[tuple[int, int], float]:
    candidate = assert_training_data_path(path)
    return price_lookup_from_frame(pd.read_excel(candidate, sheet_name=0))
