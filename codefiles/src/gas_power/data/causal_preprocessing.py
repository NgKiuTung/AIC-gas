"""Reusable raw-table preprocessing with strict past/current-only semantics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

FREQUENCY = "15min"
SEASONAL_LAGS = (96, 192, 672)
TARGETS = ("generator_1", "generator_all")
EXPECTED_SOURCES = ("gas", "holder", "user", "load")
ALL_NULL_EXPECTED = (
    "blast_furnace_3",
    "air_heater_3",
    "blast_furnace_gas_holder_1",
    "converter_user3",
)

# The raw data currently contains measurements rather than a separate device-status
# table.  Name-based fallback keeps the policy usable if status columns are added.
HOLDER_FIELDS = {"blast_furnace_gas_holder_2"}
LOAD_FIELDS = set(TARGETS)
STATE_FIELDS: set[str] = set()
SHORT_GAP_STEPS = 2


def field_category(column: str) -> str:
    """Return the missing-value policy category for one raw measurement."""
    lowered = column.lower()
    if column in STATE_FIELDS or "status" in lowered or "state" in lowered:
        return "state"
    if column in HOLDER_FIELDS or "holder" in lowered:
        return "holder"
    if column in LOAD_FIELDS:
        return "load"
    return "flow"


def _missing_run_lengths(mask: pd.Series) -> tuple[int, int]:
    """Return total missing cells and the longest consecutive missing run."""
    current = 0
    longest = 0
    total = int(mask.sum())
    for missing in mask.astype(bool):
        current = current + 1 if missing else 0
        longest = max(longest, current)
    return total, longest


def _validate_raw_table(source: str, frame: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in frame:
        raise ValueError(f"{source}: missing datetime column")
    clean = frame.copy()
    clean["datetime"] = pd.to_datetime(clean["datetime"], errors="raise")
    if clean["datetime"].duplicated().any():
        raise ValueError(f"{source}: duplicate timestamps")
    return clean.sort_values("datetime").reset_index(drop=True)


def merge_raw_tables(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge four official-format raw tables onto a complete 15-minute grid."""
    if set(tables) != set(EXPECTED_SOURCES):
        raise ValueError(f"Expected sources {EXPECTED_SOURCES}, received {sorted(tables)}")
    frames = {source: _validate_raw_table(source, tables[source]) for source in EXPECTED_SOURCES}
    timestamps = pd.concat([frame[["datetime"]] for frame in frames.values()], ignore_index=True)["datetime"]
    if timestamps.empty:
        raise ValueError("Raw tables are empty")
    grid = pd.DataFrame({"datetime": pd.date_range(timestamps.min(), timestamps.max(), freq=FREQUENCY)})
    existing = {"datetime"}
    for source in EXPECTED_SOURCES:
        frame = frames[source]
        columns = [column for column in frame if column != "datetime"]
        collisions = existing.intersection(columns)
        if collisions:
            raise ValueError(f"Column collision: {sorted(collisions)}")
        presence = frame[["datetime"]].assign(**{f"feat_source_missing_{source}": 0})
        grid = grid.merge(frame, on="datetime", how="left", validate="one_to_one")
        grid = grid.merge(presence, on="datetime", how="left", validate="one_to_one")
        flag = f"feat_source_missing_{source}"
        grid[flag] = grid[flag].fillna(1).astype("int8")
        existing.update(columns)
    return grid


def causal_fill(
    series: pd.Series,
    timestamps: pd.Series,
    *,
    category: str = "flow",
) -> tuple[pd.Series, pd.Series]:
    """Fill with a category-specific policy using current/past values only."""
    if category not in {"flow", "holder", "load", "state"}:
        raise ValueError(f"Unsupported field category: {category}")
    raw = pd.to_numeric(series, errors="coerce")
    values: list[float] = []
    methods: list[str] = []
    history: dict[pd.Timestamp, float] = {}
    observed_history: list[float] = []
    missing_run = 0
    last = np.nan
    for timestamp, value in zip(timestamps, raw, strict=True):
        timestamp = pd.Timestamp(timestamp)
        if pd.notna(value) and np.isfinite(value):
            filled = float(value)
            method = "observed"
            missing_run = 0
            observed_history.append(filled)
        else:
            missing_run += 1
            if category == "state" and np.isfinite(last):
                filled = float(last)
                method = "causal_state_ffill"
            elif missing_run <= SHORT_GAP_STEPS and np.isfinite(last):
                filled = float(last)
                method = "causal_ffill"
            else:
                candidates = [history.get(timestamp - pd.Timedelta(minutes=15 * lag), np.nan) for lag in SEASONAL_LAGS]
                finite_candidates = [item for item in candidates if np.isfinite(item)]
                if finite_candidates:
                    filled = float(np.median(finite_candidates))
                    method = "past_seasonal_median"
                elif observed_history:
                    filled = float(np.median(observed_history))
                    method = "past_expanding_median"
                else:
                    filled = np.nan
                    method = "unresolved"
        values.append(filled)
        methods.append(method)
        if np.isfinite(filled):
            last = filled
            history[timestamp] = filled
    return pd.Series(values, index=series.index), pd.Series(methods, index=series.index)


def add_known_features(frame: pd.DataFrame, price_lookup: Mapping[tuple[int, int], float]) -> pd.DataFrame:
    out = frame.copy()
    dt = out["datetime"]
    slot = dt.dt.hour * 4 + dt.dt.minute // 15
    weekday = dt.dt.dayofweek
    half_hour = dt.dt.hour * 2 + dt.dt.minute // 30
    out["feat_month"] = dt.dt.month.astype("int8")
    out["feat_day"] = dt.dt.day.astype("int8")
    out["feat_weekday"] = weekday.astype("int8")
    out["feat_hour"] = dt.dt.hour.astype("int8")
    out["feat_quarter_hour_slot"] = slot.astype("int8")
    out["feat_is_weekend"] = (weekday >= 5).astype("int8")
    out["feat_time_of_day_sin"] = np.sin(2 * np.pi * slot / 96)
    out["feat_time_of_day_cos"] = np.cos(2 * np.pi * slot / 96)
    out["feat_week_sin"] = np.sin(2 * np.pi * (weekday * 96 + slot) / 672)
    out["feat_week_cos"] = np.cos(2 * np.pi * (weekday * 96 + slot) / 672)
    try:
        out["feat_known_price"] = [
            price_lookup[(int(month), int(price_slot))]
            for month, price_slot in zip(dt.dt.month, half_hour, strict=True)
        ]
    except KeyError as error:
        raise ValueError(f"Price lookup does not cover month/slot {error.args[0]}") from error
    ranks = {value: rank for rank, value in enumerate(sorted(set(price_lookup.values())))}
    out["feat_price_level"] = out["feat_known_price"].map(ranks).astype("int16")

    # Additional features
    out["feat_is_peak_hour"] = ((dt.dt.hour >= 8) & (dt.dt.hour < 22)).astype("int8")
    out["feat_price_time_interaction"] = out["feat_price_level"] * out["feat_time_of_day_sin"]
    out["feat_day_of_month_normalized"] = (dt.dt.day - 1) / (dt.dt.days_in_month - 1)

    return out


def add_observed_family_aggregates(
    frame: pd.DataFrame,
    observed_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add family aggregates from raw observations, not imputed values."""
    out = frame.copy()
    observed = frame if observed_frame is None else observed_frame
    families = {
        "blast_furnace": ["blast_furnace_1", "blast_furnace_2", "blast_furnace_4", "blast_furnace_5"],
        "air_heater": ["air_heater_1", "air_heater_2", "air_heater_4", "air_heater_5"],
        "blast_furnace_user": [f"blast_furnace_user{index}" for index in range(1, 5)],
        "converter_user": ["converter_user1", "converter_user2"],
    }
    for name, columns in families.items():
        missing = [column for column in columns if column not in out]
        if missing:
            raise ValueError(f"Missing columns for {name} aggregate: {missing}")
        out[f"feat_{name}_observed_sum"] = observed[columns].sum(axis=1)
        out[f"feat_{name}_observed_nonzero_count"] = (observed[columns] > 0).sum(axis=1).astype("int8")
    return out


def causal_outlier_flags(frame: pd.DataFrame, value_columns: list[str]) -> dict[str, pd.Series]:
    flags: dict[str, pd.Series] = {}
    for column in value_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        past = series.shift(1)
        median = past.rolling(96, min_periods=24).median()
        mad = past.rolling(96, min_periods=24).apply(
            lambda values: float(np.median(np.abs(values - np.median(values)))), raw=True
        )
        valid = mad > np.maximum(1e-9, median.abs() * 1e-6)
        robust_z = (series - median).abs() / (1.4826 * mad.where(valid))
        flags[column] = ((robust_z > 8) & valid).fillna(False).astype("int8")
    return flags


def preprocess_causal_raw_tables(
    tables: Mapping[str, pd.DataFrame],
    price_lookup: Mapping[tuple[int, int], float],
    *,
    split: str = "inference",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce the causal frame and a per-column imputation audit."""
    merged = merge_raw_tables(tables)
    raw_columns = [
        column for column in merged if column != "datetime" and not column.startswith("feat_source_missing_")
    ]
    for column in raw_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    all_null = [column for column in raw_columns if merged[column].isna().all()]
    if set(all_null) != set(ALL_NULL_EXPECTED):
        raise ValueError(f"Unexpected all-null columns: {all_null}")
    usable = [column for column in raw_columns if column not in all_null]
    causal = merged.drop(columns=all_null).copy()
    observed_values = merged.drop(columns=all_null).copy()
    method_rows: list[dict[str, object]] = []
    source_by_column = {
        column: source
        for source, table in tables.items()
        for column in table.columns
        if column != "datetime"
    }
    for column in all_null:
        method_rows.append(
            {
                "source": source_by_column.get(column, ""),
                "column": column,
                "category": field_category(column),
                "method": "structural_missing_not_filled",
                "count": int(len(merged)),
                "raw_missing_count": int(len(merged)),
                "max_missing_run_steps": int(len(merged)),
                "max_missing_run_minutes": int((len(merged) - 1) * 15),
                "source_row_missing_count": 0,
                "structural_missing": True,
            }
        )
    for column in usable:
        raw_missing = merged[column].isna()
        raw_missing_count, max_missing_run_steps = _missing_run_lengths(raw_missing)
        category = field_category(column)
        causal[f"feat_missing_{column}"] = raw_missing.astype("int8")
        filled, methods = causal_fill(merged[column], merged["datetime"], category=category)
        if filled.isna().any():
            raise ValueError(f"Causal fill unresolved for {column}; the first raw value must be observed")
        for method, count in methods.value_counts().items():
            method_rows.append(
                {
                    "source": source_by_column.get(column, ""),
                    "column": column,
                    "category": category,
                    "method": method,
                    "count": int(count),
                    "raw_missing_count": raw_missing_count,
                    "max_missing_run_steps": max_missing_run_steps,
                    "max_missing_run_minutes": int(max_missing_run_steps * 15),
                    "source_row_missing_count": int(
                        ((merged[f"feat_source_missing_{source_by_column[column]}"] == 1) & raw_missing).sum()
                    ),
                    "structural_missing": False,
                }
            )
        if column in TARGETS:
            causal[f"feat_{column}_filled"] = filled
        else:
            causal[column] = filled
    causal = add_observed_family_aggregates(
        add_known_features(causal, price_lookup),
        observed_frame=observed_values,
    )
    flag_values = [column for column in usable if column not in TARGETS]
    flag_values.extend(f"feat_{target}_filled" for target in TARGETS)
    for column, flag in causal_outlier_flags(causal, flag_values).items():
        raw_name = column.removeprefix("feat_").removesuffix("_filled") if column.startswith("feat_") else column
        causal[f"feat_outlier_{raw_name}"] = flag
    causal.insert(1, "split", split)
    return causal, pd.DataFrame(method_rows)
