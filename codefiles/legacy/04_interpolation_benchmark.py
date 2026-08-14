"""Benchmark training-only interpolation methods using simulated missing blocks.

The official scoring test directory is deliberately not referenced or read.
"""

from __future__ import annotations

import json
import logging
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
AUDIT_DIR = RESULT_DIR / "audit"
LOG_DIR = RESULT_DIR / "logs"
TRAIN_FILES = sorted(TRAIN_DIR.glob("Pre_*.csv"))
FREQUENCY = "15min"
RNG_SEED = 20260803
MAX_BLOCKS_PER_COLUMN_AND_LENGTH = 80


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("interpolation_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "04_interpolation_benchmark.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def load_training_grid() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in TRAIN_FILES:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        frames.append(frame)
    timestamps = pd.concat([f[["datetime"]] for f in frames], ignore_index=True)["datetime"]
    merged = pd.DataFrame({"datetime": pd.date_range(timestamps.min(), timestamps.max(), freq=FREQUENCY)})
    used = {"datetime"}
    for frame in frames:
        cols = [c for c in frame.columns if c != "datetime"]
        overlap = used.intersection(cols)
        if overlap:
            raise ValueError(f"Duplicate fields across training tables: {sorted(overlap)}")
        merged = merged.merge(frame, on="datetime", how="left", validate="one_to_one")
        used.update(cols)
    for col in merged.columns[1:]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def method_ffill(x_anchor: np.ndarray, y_anchor: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    left = y_anchor[x_anchor < x_target.min()]
    return np.full(len(x_target), left[-1], dtype=float)


def method_linear(x_anchor: np.ndarray, y_anchor: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    return np.interp(x_target, x_anchor, y_anchor)


def method_lagrange4(x_anchor: np.ndarray, y_anchor: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    degree = min(3, len(x_anchor) - 1)
    coefficients = np.polyfit(x_anchor, y_anchor, deg=degree)
    return np.polyval(coefficients, x_target)


def method_pchip(x_anchor: np.ndarray, y_anchor: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    return PchipInterpolator(x_anchor, y_anchor, extrapolate=False)(x_target)


METHODS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "causal_ffill": method_ffill,
    "linear": method_linear,
    "lagrange4": method_lagrange4,
    "pchip": method_pchip,
}


def valid_block_starts(series: pd.Series, gap_length: int) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    candidates: list[int] = []
    for start in range(2, len(values) - gap_length - 2):
        indices = [start - 2, start - 1, *range(start, start + gap_length), start + gap_length, start + gap_length + 1]
        if np.isfinite(values[indices]).all():
            candidates.append(start)
    return np.asarray(candidates, dtype=int)


def choose_nonoverlapping(starts: np.ndarray, gap_length: int, rng: np.random.Generator) -> np.ndarray:
    if len(starts) == 0:
        return starts
    shuffled = starts.copy()
    rng.shuffle(shuffled)
    chosen: list[int] = []
    occupied: set[int] = set()
    for start in shuffled:
        footprint = set(range(int(start) - 2, int(start) + gap_length + 2))
        if footprint.isdisjoint(occupied):
            chosen.append(int(start))
            occupied.update(footprint)
        if len(chosen) >= MAX_BLOCKS_PER_COLUMN_AND_LENGTH:
            break
    return np.asarray(sorted(chosen), dtype=int)


def evaluate_column(series: pd.Series, column: str, gap_length: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    starts = choose_nonoverlapping(valid_block_starts(series, gap_length), gap_length, rng)
    records: list[dict[str, Any]] = []
    if len(starts) == 0:
        return records
    values = series.to_numpy(dtype=float)
    iqr = float(np.nanquantile(values, 0.75) - np.nanquantile(values, 0.25))
    scale = max(iqr, float(np.nanmedian(np.abs(values))) * 0.01, 1e-9)
    nonnegative = bool(np.nanmin(values) >= 0)
    for method_name, method in METHODS.items():
        truths: list[float] = []
        predictions: list[float] = []
        for start in starts:
            stop = int(start) + gap_length
            anchor_idx = np.asarray([start - 2, start - 1, stop, stop + 1], dtype=float)
            target_idx = np.arange(start, stop, dtype=float)
            pred = method(anchor_idx, values[anchor_idx.astype(int)], target_idx)
            truths.extend(values[int(start) : stop].tolist())
            predictions.extend(np.asarray(pred, dtype=float).tolist())
        true = np.asarray(truths, dtype=float)
        pred = np.asarray(predictions, dtype=float)
        error = pred - true
        denominator = np.abs(true) + np.abs(pred) + 1e-9
        invalid = (~np.isfinite(pred)) | ((pred < 0) if nonnegative else False)
        mae = float(np.nanmean(np.abs(error)))
        smape = float(np.nanmean(2 * np.abs(error) / denominator))
        invalid_rate = float(np.mean(invalid))
        records.append(
            {
                "column": column,
                "gap_length": gap_length,
                "method": method_name,
                "blocks": int(len(starts)),
                "points": int(len(true)),
                "mae": mae,
                "rmse": float(np.sqrt(np.nanmean(error**2))),
                "nmae_iqr": mae / scale,
                "smape": smape,
                "invalid_prediction_rate": invalid_rate,
                "selection_score": mae / scale + 0.25 * smape + 10 * invalid_rate,
            }
        )
    return records


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    grid = load_training_grid()
    rng = np.random.default_rng(RNG_SEED)
    numeric_columns = [c for c in grid.columns if c != "datetime" and grid[c].notna().any()]
    records: list[dict[str, Any]] = []
    for column in numeric_columns:
        for gap_length in (1, 2, 3):
            records.extend(evaluate_column(grid[column], column, gap_length, rng))
        LOGGER.info("Benchmarked %s", column)
    metrics = pd.DataFrame(records)
    if metrics.empty:
        raise RuntimeError("No interpolation benchmarks were generated")
    metrics.to_csv(AUDIT_DIR / "interpolation_benchmark_metrics.csv", index=False, encoding="utf-8-sig")

    best_rows = (
        metrics.sort_values(["column", "gap_length", "selection_score", "mae"])
        .groupby(["column", "gap_length"], as_index=False)
        .first()
    )
    best_rows.to_csv(AUDIT_DIR / "interpolation_method_selection.csv", index=False, encoding="utf-8-sig")
    selection: dict[str, dict[str, str]] = {}
    for _, row in best_rows.iterrows():
        selection.setdefault(str(row["column"]), {})[str(int(row["gap_length"]))] = str(row["method"])

    global_summary = (
        metrics.groupby(["gap_length", "method"], as_index=False)
        .agg(
            median_selection_score=("selection_score", "median"),
            median_nmae_iqr=("nmae_iqr", "median"),
            median_smape=("smape", "median"),
            mean_invalid_rate=("invalid_prediction_rate", "mean"),
        )
        .sort_values(["gap_length", "median_selection_score"])
    )
    global_summary.to_csv(AUDIT_DIR / "interpolation_global_summary.csv", index=False, encoding="utf-8-sig")
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "official training CSV files only",
        "official_test_accessed": False,
        "random_seed": RNG_SEED,
        "gap_lengths_evaluated": [1, 2, 3],
        "methods": list(METHODS),
        "selection_score": "NMAE_IQR + 0.25*sMAPE + 10*invalid_prediction_rate",
        "selection_by_column_and_gap_length": selection,
        "fallback": "linear when two-sided anchors exist; otherwise causal forward/seasonal fallback",
    }
    (AUDIT_DIR / "interpolation_selection.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Wrote %d metric rows and %d selections", len(metrics), len(best_rows))


if __name__ == "__main__":
    main()
