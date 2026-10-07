"""Feature-architecture helpers grounded in the Phase7.1 feature dictionary."""
from __future__ import annotations

from collections import Counter
import re

import pandas as pd


OPERATOR_ORDER = [
    "state snapshot",
    "rolling statistics",
    "dynamics",
    "coverage QC",
    "known future",
]

SOURCE_ORDER = [
    "gas production",
    "air heaters",
    "mixed-gas inflow",
    "gas holder",
    "industrial users",
    "generator fuel use",
    "known future",
]


def feature_name_column(df: pd.DataFrame) -> str:
    """Return the feature-name column used by the archived dictionary."""
    for candidate in ("feature_name", "feature", "name", "column"):
        if candidate in df.columns:
            return candidate
    raise ValueError("No feature-name column found")


def operator_group(operator: str, source: str = "") -> str:
    """Map exact archived operators to report-level transformation families."""
    op = str(operator).strip().lower()
    src = str(source).strip().lower()
    if src == "deterministic_calendar_or_source_schedule" or op.startswith("feat_"):
        return "known future"
    if op in {"observed", "last", "age_minutes", "missing_now"}:
        return "state snapshot"
    if op.startswith("mean_") or op.startswith("std_"):
        return "rolling statistics"
    if "valid_count" in op or "scheduled_count" in op or "coverage" in op:
        return "coverage QC"
    if op.startswith("diff_") or "delta" in op or "change" in op:
        return "dynamics"
    return "state snapshot"


def source_group(source: str) -> str:
    """Map archived process signals to stable physical domains for reporting."""
    src = str(source).strip().lower()
    if src == "deterministic_calendar_or_source_schedule":
        return "known future"
    if src.startswith("generator_use_"):
        return "generator fuel use"
    if "gas_holder" in src:
        return "gas holder"
    if "_user" in src or src.startswith("blast_furnace_user") or src.startswith("converter_user"):
        return "industrial users"
    if src.startswith("into_gas_mixed_"):
        return "mixed-gas inflow"
    if src.startswith("air_heater_"):
        return "air heaters"
    if re.match(r"^(blast_furnace_[1-5]|coke_oven_1|converter_1)$", src):
        return "gas production"
    return "gas production"


def source_operator_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a physical-domain × transform-family feature-count matrix."""
    required = {"source", "operator"}
    if not required.issubset(df.columns):
        raise ValueError(f"Feature dictionary missing columns: {sorted(required - set(df.columns))}")
    rows = [
        (source_group(source), operator_group(operator, source))
        for source, operator in zip(df["source"], df["operator"])
    ]
    frame = pd.DataFrame(rows, columns=["source_group", "operator_group"])
    table = pd.crosstab(frame["source_group"], frame["operator_group"])
    return table.reindex(index=SOURCE_ORDER, columns=OPERATOR_ORDER, fill_value=0)


def operator_counts(df: pd.DataFrame) -> Counter:
    """Count features by transformation family using exact dictionary metadata."""
    return Counter(
        operator_group(operator, source)
        for source, operator in zip(df["source"], df["operator"])
    )


def availability_counts(df: pd.DataFrame) -> Counter:
    """Count causal versus known-before-origin feature availability classes."""
    if "availability" not in df.columns:
        raise ValueError("Feature dictionary has no availability column")
    return Counter(str(v) for v in df["availability"])


def target_feature_count(df: pd.DataFrame) -> int:
    """Return the number of target-derived features; expected to be zero."""
    if "uses_target" not in df.columns:
        return 0
    values = df["uses_target"].astype(str).str.lower().isin({"true", "1", "yes"})
    return int(values.sum())
