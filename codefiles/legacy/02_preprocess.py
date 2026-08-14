"""Complete training-only preprocessing for the preliminary-round dataset.

Outputs both an offline interpolation dataset for final fitting and a strictly
causal dataset for rolling validation. The official scoring test directory is
not referenced, read, transformed, or written by this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "dataset" / "初赛-数据集"
RESULT_DIR = ROOT / "results" / "preprocessing"
PROCESSED_DIR = RESULT_DIR / "processed"
AUDIT_DIR = RESULT_DIR / "audit"
LOG_DIR = RESULT_DIR / "logs"
TRAIN_FILES = {
    "gas": TRAIN_DIR / "Pre_gas.csv",
    "holder": TRAIN_DIR / "Pre_gas_holder.csv",
    "user": TRAIN_DIR / "Pre_gas_user.csv",
    "load": TRAIN_DIR / "Pre_load.csv",
}
TARGETS = ("generator_1", "generator_all")
FREQUENCY = "15min"
SEASONAL_LAGS = (96, 192, 672)
ALL_NULL_EXPECTED = (
    "blast_furnace_3",
    "air_heater_3",
    "blast_furnace_gas_holder_1",
    "converter_user3",
)


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocess_training_only")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "02_preprocess.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    if frame["datetime"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path.name}")
    return frame.sort_values("datetime").reset_index(drop=True)


def merge_training_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {name: read_csv(path) for name, path in TRAIN_FILES.items()}
    timestamps = pd.concat([frame[["datetime"]] for frame in frames.values()], ignore_index=True)["datetime"]
    merged = pd.DataFrame({"datetime": pd.date_range(timestamps.min(), timestamps.max(), freq=FREQUENCY)})
    lineage: list[dict[str, Any]] = []
    existing = {"datetime"}
    for source, frame in frames.items():
        cols = [c for c in frame.columns if c != "datetime"]
        collisions = existing.intersection(cols)
        if collisions:
            raise ValueError(f"Column collision: {sorted(collisions)}")
        presence = frame[["datetime"]].assign(**{f"feat_source_missing_{source}": 0})
        merged = merged.merge(frame, on="datetime", how="left", validate="one_to_one")
        merged = merged.merge(presence, on="datetime", how="left", validate="one_to_one")
        merged[f"feat_source_missing_{source}"] = merged[f"feat_source_missing_{source}"].fillna(1).astype("int8")
        for col in cols:
            lineage.append(
                {
                    "source_table": source,
                    "source_column": col,
                    "processed_column": col,
                    "role": "target" if col in TARGETS else "raw_feature",
                }
            )
        existing.update(cols)
    return merged, pd.DataFrame(lineage)


def load_method_selection() -> dict[str, dict[str, str]]:
    path = AUDIT_DIR / "interpolation_selection.json"
    if not path.exists():
        raise FileNotFoundError("Run codefiles/legacy/04_interpolation_benchmark.py before preprocessing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("official_test_accessed") is not False:
        raise ValueError("Interpolation selection provenance is not training-only")
    return payload["selection_by_column_and_gap_length"]


def missing_runs(series: pd.Series) -> list[tuple[int, int]]:
    missing = series.isna().to_numpy()
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(missing):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(series)))
    return runs


def nearest_anchor_indices(values: np.ndarray, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    left = np.where(np.isfinite(values[:start]))[0][-2:]
    right = np.where(np.isfinite(values[stop:]))[0][:2] + stop
    return left, right


def interpolate_run(values: np.ndarray, start: int, stop: int, method: str) -> tuple[np.ndarray, str]:
    target_x = np.arange(start, stop, dtype=float)
    left, right = nearest_anchor_indices(values, start, stop)
    if method == "causal_ffill" and len(left):
        return np.full(stop - start, values[left[-1]], dtype=float), method
    if len(left) and len(right):
        if method == "linear" or len(left) + len(right) < 4:
            anchor_x = np.asarray([left[-1], right[0]], dtype=float)
            return np.interp(target_x, anchor_x, values[anchor_x.astype(int)]), "linear"
        anchor_x = np.concatenate([left, right]).astype(float)
        anchor_y = values[anchor_x.astype(int)]
        if method == "pchip":
            return PchipInterpolator(anchor_x, anchor_y, extrapolate=False)(target_x), method
        if method == "lagrange4":
            coefficients = np.polyfit(anchor_x, anchor_y, deg=min(3, len(anchor_x) - 1))
            return np.polyval(coefficients, target_x), method
    if len(left):
        return np.full(stop - start, values[left[-1]], dtype=float), "fallback_causal_ffill"
    if len(right):
        return np.full(stop - start, values[right[0]], dtype=float), "fallback_backward_boundary"
    return np.full(stop - start, np.nan, dtype=float), "unresolved"


def offline_fill(series: pd.Series, selection: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    # Copy is essential: interpolation must never mutate the shared raw/base frame,
    # otherwise the independently generated causal dataset would inherit offline values.
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=True)
    methods = np.full(len(values), "observed", dtype=object)
    for start, stop in missing_runs(series):
        gap = stop - start
        method = selection.get(str(min(gap, 3)), "linear")
        pred, used = interpolate_run(values, start, stop, method)
        values[start:stop] = pred
        methods[start:stop] = used
    return pd.Series(values, index=series.index), pd.Series(methods, index=series.index)


def causal_fill(series: pd.Series, timestamps: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = pd.to_numeric(series, errors="coerce")
    values: list[float] = []
    methods: list[str] = []
    history: dict[pd.Timestamp, float] = {}
    observed_history: list[float] = []
    missing_run = 0
    last = np.nan
    for timestamp, value in zip(timestamps, raw):
        if pd.notna(value) and np.isfinite(value):
            filled = float(value)
            method = "observed"
            missing_run = 0
            observed_history.append(filled)
        else:
            missing_run += 1
            if missing_run <= 2 and pd.notna(last):
                filled = float(last)
                method = "causal_ffill"
            else:
                candidates = [history.get(timestamp - pd.Timedelta(minutes=15 * lag), np.nan) for lag in SEASONAL_LAGS]
                candidates = [v for v in candidates if pd.notna(v) and np.isfinite(v)]
                if candidates:
                    filled = float(np.median(candidates))
                    method = "past_seasonal_median"
                elif observed_history:
                    filled = float(np.median(observed_history))
                    method = "past_expanding_median"
                else:
                    filled = np.nan
                    method = "unresolved"
        values.append(filled)
        methods.append(method)
        if pd.notna(filled):
            last = filled
            history[pd.Timestamp(timestamp)] = filled
    return pd.Series(values, index=series.index), pd.Series(methods, index=series.index)


def load_price_lookup() -> dict[tuple[int, int], float]:
    raw = pd.read_excel(TRAIN_DIR / "price.xlsx", sheet_name=0)
    lookup: dict[tuple[int, int], float] = {}
    for _, row in raw.iterrows():
        start = str(row.iloc[0]).split("-")[0]
        hour, minute = [int(part) for part in start.split(":")]
        slot = hour * 2 + int(minute >= 30)
        for month in range(1, 13):
            lookup[(month, slot)] = float(row[f"{month}月"])
    return lookup


def add_known_features(frame: pd.DataFrame, price: dict[tuple[int, int], float]) -> pd.DataFrame:
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
    out["feat_known_price"] = [price[(int(m), int(s))] for m, s in zip(dt.dt.month, half_hour)]
    ranks = {value: rank for rank, value in enumerate(sorted(set(price.values())))}
    out["feat_price_level"] = out["feat_known_price"].map(ranks).astype("int8")
    return out


def add_observed_family_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    families = {
        "blast_furnace": ["blast_furnace_1", "blast_furnace_2", "blast_furnace_4", "blast_furnace_5"],
        "air_heater": ["air_heater_1", "air_heater_2", "air_heater_4", "air_heater_5"],
        "blast_furnace_user": [f"blast_furnace_user{i}" for i in range(1, 5)],
        "converter_user": ["converter_user1", "converter_user2"],
    }
    for name, columns in families.items():
        out[f"feat_{name}_observed_sum"] = out[columns].sum(axis=1)
        out[f"feat_{name}_observed_nonzero_count"] = (out[columns] > 0).sum(axis=1).astype("int8")
    return out


def causal_outlier_flags(frame: pd.DataFrame, value_columns: list[str]) -> dict[str, pd.Series]:
    flags: dict[str, pd.Series] = {}
    for col in value_columns:
        series = frame[col]
        past = series.shift(1)
        median = past.rolling(96, min_periods=24).median()
        mad = past.rolling(96, min_periods=24).apply(
            lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
        )
        valid = mad > np.maximum(1e-9, median.abs() * 1e-6)
        z = (series - median).abs() / (1.4826 * mad.where(valid))
        flags[col] = ((z > 8) & valid).fillna(False).astype("int8")
    return flags


def json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    merged, lineage = merge_training_grid()
    selection = load_method_selection()
    raw_columns = [c for c in merged.columns if c != "datetime" and not c.startswith("feat_source_missing_")]
    for col in raw_columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    all_null = [c for c in raw_columns if merged[c].isna().all()]
    if set(all_null) != set(ALL_NULL_EXPECTED):
        raise AssertionError(f"Unexpected all-null columns: {all_null}")
    usable = [c for c in raw_columns if c not in all_null]
    base = merged.drop(columns=all_null)
    offline = base.copy()
    causal = base.copy()
    method_rows: list[dict[str, Any]] = []
    for col in usable:
        missing_flag = base[col].isna().astype("int8")
        offline[f"feat_missing_{col}"] = missing_flag
        causal[f"feat_missing_{col}"] = missing_flag
        offline_filled, offline_method = offline_fill(base[col], selection.get(col, {}))
        causal_filled, causal_method = causal_fill(base[col], base["datetime"])
        for dataset, methods in (("offline", offline_method), ("causal", causal_method)):
            for method, count in methods.value_counts().items():
                method_rows.append({"dataset": dataset, "column": col, "method": method, "count": int(count)})
        if col in TARGETS:
            offline[f"feat_{col}_filled"] = offline_filled
            causal[f"feat_{col}_filled"] = causal_filled
        else:
            offline[col] = offline_filled
            causal[col] = causal_filled

    price = load_price_lookup()
    offline = add_observed_family_aggregates(add_known_features(offline, price))
    causal = add_observed_family_aggregates(add_known_features(causal, price))
    flag_values = [c for c in usable if c not in TARGETS] + [f"feat_{t}_filled" for t in TARGETS]
    outlier_rows: list[dict[str, Any]] = []
    for col, flag in causal_outlier_flags(causal, flag_values).items():
        raw_name = col.removeprefix("feat_").removesuffix("_filled") if col.startswith("feat_") else col
        offline[f"feat_outlier_{raw_name}"] = flag
        causal[f"feat_outlier_{raw_name}"] = flag
        outlier_rows.append(
            {
                "column": raw_name,
                "train_flag_count": int(flag.sum()),
                "train_flag_rate": float(flag.mean()),
                "rule": "past-only rolling 96-point median/MAD; robust_z>8; MAD=0 not flagged",
            }
        )

    offline.insert(1, "split", "train")
    causal.insert(1, "split", "train")
    for frame in (offline, causal):
        frame["datetime"] = frame["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    primary_path = PROCESSED_DIR / "preprocessed_train.csv"
    causal_path = PROCESSED_DIR / "preprocessed_train_causal.csv"
    offline.to_csv(primary_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    causal.to_csv(causal_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    pd.DataFrame(method_rows).to_csv(AUDIT_DIR / "imputation_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(outlier_rows).to_csv(
        AUDIT_DIR / "outlier_flag_summary.csv", index=False, encoding="utf-8-sig"
    )
    lineage.to_csv(AUDIT_DIR / "column_lineage.csv", index=False, encoding="utf-8-sig")

    scaler: dict[str, Any] = {}
    for col in usable:
        values = offline[f"feat_{col}_filled"] if col in TARGETS else offline[col]
        finite = values[np.isfinite(values)].dropna()
        q25, median, q75 = finite.quantile([0.25, 0.5, 0.75])
        scaler[col] = {
            "median": json_number(median),
            "q25": json_number(q25),
            "q75": json_number(q75),
            "iqr": json_number(q75 - q25),
            "fit_split": "official_training_only",
            "cv_requirement": "Refit inside every rolling-validation training fold.",
        }
    state = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "official training CSV and auxiliary XLSX files only",
        "official_test_accessed": False,
        "frequency": FREQUENCY,
        "primary_dataset": "preprocessed_train.csv (benchmark-selected offline interpolation)",
        "cv_safe_dataset": "preprocessed_train_causal.csv (past-only repair)",
        "original_targets_not_replaced": list(TARGETS),
        "target_continuous_input_columns": [f"feat_{target}_filled" for target in TARGETS],
        "all_null_columns_excluded": all_null,
        "all_null_policy": "Not identifiable; do not fabricate. Use observed-family aggregates instead.",
        "zero_policy": "Preserve all zeros.",
        "outlier_policy": "Past-only candidate flags; no deletion, clipping, or replacement.",
        "scaler_state_full_official_training": scaler,
    }
    state_path = PROCESSED_DIR / "preprocessing_state_full_train.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    output_paths = [
        primary_path,
        causal_path,
        state_path,
        AUDIT_DIR / "imputation_summary.csv",
        AUDIT_DIR / "outlier_flag_summary.csv",
        AUDIT_DIR / "column_lineage.csv",
        AUDIT_DIR / "interpolation_benchmark_metrics.csv",
        AUDIT_DIR / "interpolation_method_selection.csv",
        AUDIT_DIR / "interpolation_global_summary.csv",
        AUDIT_DIR / "interpolation_selection.json",
        AUDIT_DIR / "all_null_proxy_experiments.csv",
        AUDIT_DIR / "all_null_column_analysis.json",
    ]
    manifest = {
        "run_started_utc": started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "official_test_accessed": False,
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "role": "official_training"}
            for path in TRAIN_FILES.values()
        ]
        + [
            {"path": str((TRAIN_DIR / name).relative_to(ROOT)), "sha256": sha256(TRAIN_DIR / name), "role": "auxiliary"}
            for name in ("price.xlsx", "data_dictionary.xlsx")
        ],
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in output_paths
        ],
        "rows": len(offline),
        "columns": len(offline.columns),
    }
    (RESULT_DIR / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Training-only outputs: primary=%s causal=%s shape=%s", primary_path, causal_path, offline.shape)


if __name__ == "__main__":
    main()
