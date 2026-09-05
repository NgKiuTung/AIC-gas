"""Validated model-input snapshot contract for the preliminary submission."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

OFFICIAL_RAW_FEATURES = frozenset(
    {
        "blast_furnace_1",
        "blast_furnace_2",
        "blast_furnace_4",
        "blast_furnace_5",
        "coke_oven_1",
        "converter_1",
        "air_heater_1",
        "air_heater_2",
        "air_heater_4",
        "air_heater_5",
        "into_gas_mixed_coke",
        "into_gas_mixed_blast_furnace",
        "into_gas_mixed_converter",
        "blast_furnace_gas_holder_2",
        "blast_furnace_user1",
        "blast_furnace_user2",
        "blast_furnace_user3",
        "blast_furnace_user4",
        "converter_user1",
        "converter_user2",
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
    }
)

# Current observed loads are legal inputs; future target columns remain forbidden.
CURRENT_LOAD_INPUT_FIELDS = frozenset({"generator_1", "generator_all"})


def validate_input_frame(
    frame: pd.DataFrame,
    *,
    expected_datetimes: pd.Series | pd.DatetimeIndex | np.ndarray | None = None,
    expected_feature_schema: Sequence[str] | None = None,
) -> None:
    if len(frame.columns) < 2 or frame.columns[0] != "datetime":
        raise ValueError("input.csv must begin with datetime followed by model features")
    if frame.columns.duplicated().any():
        raise ValueError("input.csv contains duplicate columns")
    feature_columns = list(frame.columns[1:])
    if expected_feature_schema is not None and feature_columns != list(expected_feature_schema):
        raise ValueError("input.csv feature names or order do not match the frozen model schema")
    invalid_raw = [
        column
        for column in feature_columns
        if not column.startswith("feat_")
        and column not in OFFICIAL_RAW_FEATURES | CURRENT_LOAD_INPUT_FIELDS
    ]
    if invalid_raw:
        raise ValueError(f"Non-engineered input columns are not official raw fields: {invalid_raw[:5]}")
    if any(column.startswith(("label_", "target_")) for column in feature_columns):
        raise ValueError("input.csv must not contain labels or prediction targets")
    datetimes = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], errors="raise"))
    if datetimes.duplicated().any() or not datetimes.is_monotonic_increasing:
        raise ValueError("input.csv timestamps must be unique and strictly increasing")
    if expected_datetimes is not None:
        expected = pd.DatetimeIndex(pd.to_datetime(expected_datetimes, errors="raise"))
        if not datetimes.equals(expected):
            raise ValueError("input.csv timestamps do not exactly cover the expected reference grid")
    values = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("input.csv contains NaN or Inf")


def write_input_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a UTF-8 model-input snapshot with stable numeric precision."""
    validate_input_frame(frame)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = frame.copy()
    serializable["datetime"] = pd.to_datetime(serializable["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    serializable.to_csv(destination, index=False, encoding="utf-8", float_format="%.9f")
    return destination
