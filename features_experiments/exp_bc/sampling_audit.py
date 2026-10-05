"""Quantify sampling-rate bias for each common observed-sample rolling mean."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features_experiments.exp_bc.physical_features import (
    ROLLS,
    ROOT,
    SCREENING_END,
    TARGETS,
)


def audit(data_dir: Path, output: Path) -> pd.DataFrame:
    semi = (
        pd.read_parquet(data_dir / "semifinal_clean_1m.parquet")
        .loc[:SCREENING_END]
        .drop(columns=[*TARGETS, "dataset_phase"])
    )
    grid = semi.reindex(pd.date_range(semi.index.min(), semi.index.max(), freq="min"))
    quarter = grid.loc[grid.index.minute % 15 == 0]
    origins = quarter.index
    rows = []
    for col in grid.columns:
        full = grid[col]
        sparse = quarter[col]
        finite = np.abs(full.to_numpy(dtype="float64"))
        finite = finite[np.isfinite(finite)]
        scale = max(1e-6, float(np.median(finite)) * 0.01) if len(finite) else 1e-6
        for window in ROLLS:
            full_mean = (
                full.rolling(window, min_periods=max(1, int(window * 0.9)))
                .mean()
                .reindex(origins)
            )
            sparse_mean = sparse.rolling(
                window // 15, min_periods=max(1, int(window / 15 * 0.9))
            ).mean()
            denominator = np.maximum(np.abs(full_mean.to_numpy()), scale)
            difference = (
                np.abs(full_mean.to_numpy() - sparse_mean.to_numpy()) / denominator
            )
            valid = np.isfinite(difference)
            rows.append(
                {
                    "source_column": col,
                    "window_minutes": window,
                    "comparable_origins": int(valid.sum()),
                    "median_relative_difference": float(np.median(difference[valid]))
                    if valid.any()
                    else np.nan,
                    "p95_relative_difference": float(
                        np.quantile(difference[valid], 0.95)
                    )
                    if valid.any()
                    else np.nan,
                    "p99_relative_difference": float(
                        np.quantile(difference[valid], 0.99)
                    )
                    if valid.any()
                    else np.nan,
                }
            )
    report = pd.DataFrame(rows).sort_values("p95_relative_difference", ascending=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "features_experiments" / "exp_bc" / "data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "exp_bc" / "sampling_rate_audit.csv",
    )
    args = parser.parse_args()
    result = audit(args.data, args.output)
    print(result.head(12).to_string(index=False))
    print(
        f"p95 > 10%: {(result['p95_relative_difference'] > 0.1).sum()} / {len(result)}"
    )


if __name__ == "__main__":
    main()
