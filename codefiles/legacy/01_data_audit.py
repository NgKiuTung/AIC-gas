"""Audit official training data only.

All machine-readable outputs and the console log are written under
results/preprocessing/.  This script is read-only with respect to dataset/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "初赛-数据集"
RESULT_DIR = ROOT / "results" / "preprocessing"
AUDIT_DIR = RESULT_DIR / "audit"
SOURCE_DIR = RESULT_DIR / "source_reading"
LOG_DIR = RESULT_DIR / "logs"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("data_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "01_data_audit.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


LOGGER = setup_logging()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_robust(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"Cannot decode {path}: {'; '.join(errors)}")


def json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def datetime_column(df: pd.DataFrame) -> str | None:
    for candidate in ("datetime", "timestamp", "time", "date"):
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if "time" in str(col).lower() or "date" in str(col).lower():
            return str(col)
    return str(df.columns[0]) if len(df.columns) else None


def audit_csv(path: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    df, encoding = read_csv_robust(path)
    dt_col = datetime_column(df)
    parsed = pd.to_datetime(df[dt_col], errors="coerce") if dt_col else pd.Series(dtype="datetime64[ns]")
    valid_dt = parsed.dropna().sort_values()
    diffs = valid_dt.diff().dropna().dt.total_seconds().div(60)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [c for c in df.columns if c != dt_col and numeric[c].notna().any()]
    summary: dict[str, Any] = {
        "split": split,
        "file": path.name,
        "relative_path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "encoding": encoding,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "datetime_column": dt_col,
        "datetime_invalid": int(parsed.isna().sum()) if dt_col else None,
        "datetime_min": json_value(valid_dt.min()) if len(valid_dt) else None,
        "datetime_max": json_value(valid_dt.max()) if len(valid_dt) else None,
        "duplicate_timestamps": int(parsed.duplicated(keep=False).sum()) if dt_col else None,
        "duplicate_rows": int(df.duplicated(keep=False).sum()),
        "off_15min_grid": int(((parsed.dt.minute % 15 != 0) | (parsed.dt.second != 0)).fillna(False).sum()) if dt_col else None,
        "interval_minutes_counts": {str(json_value(k)): int(v) for k, v in Counter(diffs.tolist()).most_common(20)},
        "numeric_columns": [str(c) for c in numeric_cols],
    }
    column_rows: list[dict[str, Any]] = []
    for col in df.columns:
        ser = df[col]
        num = pd.to_numeric(ser, errors="coerce") if col != dt_col else pd.Series(np.nan, index=df.index)
        finite = num[np.isfinite(num)]
        is_numeric = col in numeric_cols
        row: dict[str, Any] = {
            "split": split,
            "file": path.name,
            "column": str(col),
            "dtype_read": str(ser.dtype),
            "is_datetime": bool(col == dt_col),
            "is_numeric": is_numeric,
            "rows": int(len(ser)),
            "missing_count": int(ser.isna().sum()),
            "missing_rate": float(ser.isna().mean()),
            "unique_count": int(ser.nunique(dropna=True)),
            "zero_count": int((num == 0).sum()) if is_numeric else None,
            "negative_count": int((num < 0).sum()) if is_numeric else None,
            "inf_count": int(np.isinf(num).sum()) if is_numeric else None,
            "min": json_value(finite.min()) if len(finite) else None,
            "q01": json_value(finite.quantile(0.01)) if len(finite) else None,
            "q25": json_value(finite.quantile(0.25)) if len(finite) else None,
            "median": json_value(finite.median()) if len(finite) else None,
            "q75": json_value(finite.quantile(0.75)) if len(finite) else None,
            "q99": json_value(finite.quantile(0.99)) if len(finite) else None,
            "max": json_value(finite.max()) if len(finite) else None,
        }
        column_rows.append(row)
    LOGGER.info(
        "%s/%s rows=%d cols=%d time=%s..%s invalid_dt=%s dup_ts=%s",
        split,
        path.name,
        len(df),
        len(df.columns),
        summary["datetime_min"],
        summary["datetime_max"],
        summary["datetime_invalid"],
        summary["duplicate_timestamps"],
    )
    normalized = df.copy()
    if dt_col:
        normalized[dt_col] = parsed
    return summary, column_rows, normalized


def audit_xlsx(path: Path) -> dict[str, Any]:
    book = pd.ExcelFile(path)
    output: dict[str, Any] = {
        "file": path.name,
        "relative_path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "sheets": [],
    }
    for sheet in book.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        nonempty = df.dropna(how="all").dropna(axis=1, how="all")
        sample = [[json_value(v) for v in row] for row in nonempty.head(20).to_numpy().tolist()]
        output["sheets"].append(
            {
                "sheet": sheet,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "nonempty_rows": int(nonempty.shape[0]),
                "nonempty_columns": int(nonempty.shape[1]),
                "sample_first_20_rows": sample,
            }
        )
        LOGGER.info("xlsx/%s/%s shape=%s", path.name, sheet, tuple(df.shape))
    return output


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(timezone.utc).isoformat()
    file_summaries: list[dict[str, Any]] = []
    column_summaries: list[dict[str, Any]] = []
    frames: dict[str, dict[str, pd.DataFrame]] = {"train": {}}

    for path in sorted(TRAIN_DIR.glob("*.csv")):
        summary, columns, frame = audit_csv(path, "train")
        file_summaries.append(summary)
        column_summaries.extend(columns)
        frames["train"][path.name] = frame

    workbook_summaries = [audit_xlsx(path) for path in sorted(TRAIN_DIR.glob("*.xlsx"))]

    cross_table: dict[str, Any] = {}
    for split in ("train",):
        timestamps: dict[str, set[pd.Timestamp]] = {}
        for name, frame in frames[split].items():
            col = datetime_column(frame)
            timestamps[name] = set(frame[col].dropna()) if col else set()
        sets = list(timestamps.values())
        union = set().union(*sets) if sets else set()
        intersection = set.intersection(*sets) if sets else set()
        cross_table[split] = {
            "union_timestamp_count": len(union),
            "intersection_timestamp_count": len(intersection),
            "union_min": json_value(min(union)) if union else None,
            "union_max": json_value(max(union)) if union else None,
            "missing_from_union_by_file": {name: len(union - values) for name, values in timestamps.items()},
        }

    split_isolation = {
        "policy": "Training-only audit. The official preliminary scoring test directory is not read.",
        "official_test_accessed": False,
    }

    audit = {
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "source_roots": {"train": str(TRAIN_DIR.relative_to(ROOT))},
        "files": file_summaries,
        "workbooks": workbook_summaries,
        "cross_table_alignment": cross_table,
        "split_isolation": split_isolation,
    }

    (AUDIT_DIR / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(file_summaries).drop(columns=["column_names", "numeric_columns", "interval_minutes_counts"]).to_csv(
        AUDIT_DIR / "file_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(column_summaries).to_csv(
        AUDIT_DIR / "column_summary.csv", index=False, encoding="utf-8-sig"
    )
    (SOURCE_DIR / "xlsx_structure_and_samples.json").write_text(
        json.dumps(workbook_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("cross_table=%s", json.dumps(cross_table, ensure_ascii=False))
    LOGGER.info("split_isolation=%s", json.dumps(split_isolation, ensure_ascii=False))
    LOGGER.info("Audit outputs written to %s", AUDIT_DIR)


if __name__ == "__main__":
    main()
