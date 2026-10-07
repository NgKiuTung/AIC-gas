"""Independent September replay of the released Stage2.2 proxy + week4 baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE22 = ROOT / "wjt" / "gas_stage22_rebuild"
sys.path.insert(0, str(STAGE22))
import gas22.bootstrap  # noqa: F401  # registers vendored gas2/gasstage paths
from gas2.data import DataBundle
from gasrefine.periodic import template
from gasrefine.snapshot import SnapshotModel, training_rows
from gasstage.baselines import reconcile
from gasstage.ingest import NativeDataset
from gasstage.targets import block_truth, future_truth

TARGETS = ("generator_1", "generator_all")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
END = pd.Timestamp("2025-09-30 23:45:00")


def load_bundle(data_dir: Path) -> DataBundle:
    """Recover only observed Jan-Sep rows; no hidden target or future process input."""
    pre = pd.read_parquet(data_dir / "preliminary_clean_15m.parquet")
    semi = pd.read_parquet(data_dir / "semifinal_clean_1m.parquet")
    pre_phase = pre.pop("dataset_phase")
    semi.pop("dataset_phase")
    if list(pre.columns) != list(semi.columns):
        raise ValueError("Incompatible cleaned columns")
    values = pd.concat([pre, semi]).sort_index()
    if values.index.has_duplicates:
        raise ValueError("Duplicate time in joined native data")
    phase_bits = np.concatenate(
        [
            np.where(pre_phase.eq("preliminary_released_eval"), 2, 1).astype("uint8"),
            np.full(len(semi), 4, dtype="uint8"),
        ]
    )
    sources = pd.DataFrame({"phase_bit": phase_bits}, index=values.index)
    flags = pd.DataFrame(0, index=values.index, columns=values.columns, dtype="uint8")
    native = NativeDataset(values, flags, sources, TARGETS)
    feature_path = STAGE22 / "features" / "history_process_features.npz"
    with np.load(feature_path, allow_pickle=False) as packed:
        features = pd.DataFrame(
            packed["features"],
            index=pd.DatetimeIndex(packed["origins"], name="datetime"),
            columns=packed["names"],
        )
    contract = json.loads(
        (
            STAGE22
            / "vendor"
            / "prior"
            / "vendor"
            / "stage1"
            / "configs"
            / "stage1.json"
        ).read_text(encoding="utf-8")
    )
    truth, _ = block_truth(native)
    return DataBundle(native, features, truth, contract, data_dir)


def score(predicted: np.ndarray, actual: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = []
    summary = {}
    for period, horizons in (("short", 8), ("long", 96)):
        parts = []
        for target_idx, name in enumerate(TARGETS):
            y = actual[:, :horizons, target_idx]
            p = predicted[:, :horizons, target_idx]
            valid = np.isfinite(y) & (y > 0) & np.isfinite(p)
            if not valid.any():
                raise ValueError(f"No valid truth: {period}, {name}")
            mae = float(np.mean(np.abs(y[valid] - p[valid])))
            mape = float(np.mean(np.abs(y[valid] - p[valid]) / y[valid]))
            parts.append(mape)
            rows.append(
                {
                    "period": period,
                    "target": name,
                    "count": int(valid.sum()),
                    "mae": mae,
                    "mape": mape,
                    "accuracy": 1 - mape,
                }
            )
        accuracy = 100 * (1 - float(np.mean(parts)))
        floor = 74.16 if period == "short" else 67.66
        estimated_score = 0.5 * (accuracy - floor) / (100 - floor) * 100
        summary[period] = {
            "accuracy_percent": accuracy,
            "estimated_score": estimated_score,
        }
    summary["estimated_total"] = (
        summary["short"]["estimated_score"] + summary["long"]["estimated_score"]
    )
    return pd.DataFrame(rows), summary


def score_by_horizon(predicted: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    """Keep target-specific horizon errors to detect offset and decay problems."""
    rows = []
    for horizon in range(96):
        for target_idx, name in enumerate(TARGETS):
            y = actual[:, horizon, target_idx]
            p = predicted[:, horizon, target_idx]
            valid = np.isfinite(y) & (y > 0) & np.isfinite(p)
            if not valid.any():
                continue
            error = np.abs(y[valid] - p[valid])
            rows.append(
                {
                    "horizon_minutes": (horizon + 1) * 15,
                    "target": name,
                    "count": int(valid.sum()),
                    "mae": float(error.mean()),
                    "mape": float(np.mean(error / y[valid])),
                    "accuracy": float(1 - np.mean(error / y[valid])),
                }
            )
    return pd.DataFrame(rows)


def run(data_dir: Path, result_dir: Path) -> dict:
    bundle = load_bundle(data_dir)
    settings = json.loads(
        (STAGE22 / "configs" / "production.json").read_text(encoding="utf-8")
    )
    candidate = settings["proxy"]
    release_meta = json.loads(
        (STAGE22 / "release" / "model" / "model.json").read_text(encoding="utf-8")
    )
    _, _, _, names, training_info = training_rows(
        bundle, pd.Timestamp("2025-10-01"), candidate
    )
    if (
        names != release_meta["features"]
        or training_info["pre_rows"] != release_meta["training"]["pre_rows"]
    ):
        raise ValueError(
            "Raw reconstruction differs from published pre-training cohort"
        )

    result_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = SnapshotModel.fit(
        bundle, CUTOFF, candidate, settings, result_dir / "model", time.time() + 3600
    )
    origins = bundle.features.loc[CUTOFF:END].index
    current = model.current(bundle.features.loc[origins])
    week = template(model.history, origins, bundle.contract, "week4")
    hours = (np.arange(96) * 15 + 7.5) / 60
    predicted = reconcile(
        week + (current - week[:, 0, :])[:, None, :] * np.exp(-hours / 6)[None, :, None]
    )
    actual = future_truth(bundle.truth, origins, bundle.contract)
    metrics, summary = score(predicted, actual)
    metrics.to_csv(result_dir / "metrics.csv", index=False)
    score_by_horizon(predicted, actual).to_csv(
        result_dir / "metrics_by_horizon.csv", index=False
    )
    summary.update(
        {
            "cutoff": str(CUTOFF),
            "validation_first": str(origins.min()),
            "validation_last": str(origins.max()),
            "origins": len(origins),
            "training": model.metadata["training"],
            "release_training_check": training_info,
            "runtime_seconds": time.monotonic() - started,
            "interpretation": (
                "September time-out validation only; estimated score is not "
                "an official result"
            ),
        }
    )
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "features_experiments" / "exp_bc" / "data"
    )
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results" / "exp_bc" / "b0_september"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
