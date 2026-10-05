"""Causal October preview for the independently trained EXP-B7/C candidate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE22 = ROOT / "wjt" / "gas_stage22_rebuild"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE22))
import gas22.bootstrap  # noqa: F401  # registers vendored gas2/gasstage paths

from features_experiments.exp_bc.b_variants import ProxyFit, predict_frozen
from features_experiments.exp_bc.data_builder import read_source
from features_experiments.exp_bc.oof_residual import model_matrix
from features_experiments.exp_bc.physical_features import (
    TARGETS,
    availability_grid,
    common_features,
    highfreq_features,
)

RATED_MW = np.array([200.0, 440.0])


def enforce_ramp(prediction: np.ndarray) -> np.ndarray:
    """Guard adjacent block means; this cannot certify minute-level unit ramps."""
    result = prediction.copy()
    limits = RATED_MW * 0.01 * 15
    result[:, 0, :] = np.clip(result[:, 0, :], 0, RATED_MW)
    result[:, 0, 1] = np.maximum(result[:, 0, 1], result[:, 0, 0])
    for horizon in range(1, result.shape[1]):
        lower = np.maximum(0, result[:, horizon - 1, :] - limits)
        upper = np.minimum(RATED_MW, result[:, horizon - 1, :] + limits)
        result[:, horizon, :] = np.clip(result[:, horizon, :], lower, upper)
        result[:, horizon, 1] = np.maximum(result[:, horizon, 1], result[:, horizon, 0])
    if (np.abs(np.diff(result, axis=1)) > limits[None, None, :] + 1e-9).any():
        raise AssertionError("Ramp guard failed")
    return result


def output_frame(
    origins: pd.DatetimeIndex, values: np.ndarray, horizons: int
) -> pd.DataFrame:
    columns = [
        f"{target}_t+{(h + 1) * 15}_pred" for target in TARGETS for h in range(horizons)
    ]
    matrix = np.concatenate([values[:, :horizons, 0], values[:, :horizons, 1]], axis=1)
    result = pd.DataFrame(matrix, columns=columns)
    result.insert(0, "datetime", origins.strftime("%Y-%m-%d %H:%M:%S"))
    return result


def run(data_dir: Path, model_dir: Path, output: Path) -> dict:
    started = time.monotonic()
    pre = pd.read_parquet(data_dir / "preliminary_clean_15m.parquet")
    semi = pd.read_parquet(data_dir / "semifinal_clean_1m.parquet")
    test, _ = read_source(ROOT, "复赛-评分所用测试集", "Semi_test_")
    if test[list(TARGETS)].notna().any().any():
        raise ValueError("Evaluation target observations must be hidden")
    test["dataset_phase"] = "semifinal_test"
    combined = pd.concat([semi, test]).sort_index()
    if combined.index.has_duplicates:
        raise ValueError("Duplicate semifinal train/test timestamp")
    manifest = json.loads(
        (data_dir / "feature_manifest.json").read_text(encoding="utf-8")
    )
    common_sources = [
        col
        for col in pre.columns
        if col
        not in (*TARGETS, "dataset_phase", *manifest["excluded_noncommon_sources"])
    ]
    excluded_rolls = set(manifest["moved_rolling_means_to_highfreq"])
    origins = pd.date_range(
        test.index.min(), test.index.max().floor("15min"), freq="15min", name="datetime"
    )
    if len(origins) != 960:
        raise ValueError(f"Unexpected October origin count: {len(origins)}")
    raw_grid, current = availability_grid(pre, combined)
    common, _ = common_features(
        raw_grid, current, origins, common_sources, excluded_rolls
    )
    if (
        list(common.columns)
        != json.loads(
            (model_dir / "base_model" / "fit.json").read_text(encoding="utf-8")
        )["features"]
    ):
        raise ValueError("Inference feature schema differs from training")
    highfreq, _ = highfreq_features(combined, origins, excluded_rolls)
    history = pd.read_csv(
        model_dir / "frozen_history.csv",
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    if (history.index + pd.Timedelta(minutes=15) > origins.min()).any():
        raise ValueError("Frozen history extends into evaluation")
    fit = ProxyFit.load(model_dir / "base_model", history)
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
    base = predict_frozen(fit, common, contract, day_weight=0.5)
    matrix = model_matrix(origins, base, highfreq)
    matrix[:, 6] = base[:, :8, 1].ravel()
    residual = lgb.Booster(
        model_file=str(model_dir / "c_residual_models" / "residual_t1.txt")
    )
    candidate = base.copy()
    candidate[:, :8, 1] += residual.predict(matrix, num_threads=4).reshape(
        len(origins), 8
    )
    candidate[:, :8, 1] = np.maximum(candidate[:, :8, 1], candidate[:, :8, 0])
    before_ramp = np.abs(np.diff(candidate, axis=1))
    guarded = enforce_ramp(candidate)
    if not np.isfinite(guarded).all():
        raise ValueError("Nonfinite October prediction")
    output.mkdir(parents=True, exist_ok=True)
    output_frame(origins, guarded, 8).to_csv(
        output / "s_result.csv", index=False, float_format="%.6f"
    )
    output_frame(origins, guarded, 96).to_csv(
        output / "l_result.csv", index=False, float_format="%.6f"
    )
    audit = {
        "origins": len(origins),
        "first": str(origins.min()),
        "last": str(origins.max()),
        "short_shape": [len(origins), 16],
        "long_shape": [len(origins), 192],
        "test_target_nonnull_cells": int(test[list(TARGETS)].notna().sum().sum()),
        "max_adjacent_block_change_before_guard_mw": before_ramp.max(
            axis=(0, 1)
        ).tolist(),
        "ramp_violations_before_guard": [
            int((before_ramp[:, :, i] > RATED_MW[i] * 0.15).sum()) for i in range(2)
        ],
        "ramp_violations_after_guard": 0,
        "generator_1_min_max": [
            float(guarded[:, :, 0].min()),
            float(guarded[:, :, 0].max()),
        ],
        "generator_all_min_max": [
            float(guarded[:, :, 1].min()),
            float(guarded[:, :, 1].max()),
        ],
        "ramp_interpretation": (
            "Adjacent 15min means constrained by 15 * 1% * aggregate rated MW "
            "(200/440 MW); not proof of minute-level or per-unit compliance"
        ),
        "runtime_seconds": time.monotonic() - started,
        "official_score": None,
    }
    (output / "inference_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "features_experiments" / "exp_bc" / "data"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "results" / "exp_bc_xgboost" / "final_jan_sep_b7",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results"
        / "exp_bc_xgboost"
        / "final_jan_sep_b7"
        / "candidate_results",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.data, args.model, args.output), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
