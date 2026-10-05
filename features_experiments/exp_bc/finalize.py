"""Train the selected Jan-Sep B7 base and optional generator_all-only C model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE22 = ROOT / "wjt" / "gas_stage22_rebuild"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE22))
import gas22.bootstrap  # noqa: F401  # registers vendored gas2/gasstage paths
from gasstage.targets import future_truth, history_only

from features_experiments.exp_bc.b_variants import (
    fit_proxy,
    load_feature_tables,
    predict_curve,
)
from features_experiments.exp_bc.baseline import load_bundle
from features_experiments.exp_bc.oof_residual import (
    BASE_LOOKBACK_DAYS,
    BASE_VARIANT,
    fit_residual,
    model_matrix,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    data_dir: Path,
    oof_dir: Path,
    output: Path,
    *,
    proxy_engine: str = "xgboost",
) -> dict:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(data_dir)
    pre, semi, pre_y, semi_y = load_feature_tables(data_dir)
    final_cutoff = pd.Timestamp("2025-10-01 00:00:00")
    base = fit_proxy(
        pre,
        semi,
        pre_y,
        semi_y,
        final_cutoff,
        output / "base_model",
        lookback_days=BASE_LOOKBACK_DAYS,
        engine=proxy_engine,
    )
    history = history_only(bundle.truth, final_cutoff)
    history.to_csv(output / "frozen_history.csv", float_format="%.17g")

    # September OOF: this proxy saw only data with availability < September 1.
    september_model = fit_proxy(
        pre,
        semi,
        pre_y,
        semi_y,
        pd.Timestamp("2025-09-01"),
        output / "september_oof_model",
        lookback_days=BASE_LOOKBACK_DAYS,
        engine=proxy_engine,
    )
    oof_metadata = json.loads(
        (oof_dir / "september_base_model" / "fit.json").read_text(encoding="utf-8")
    )
    if oof_metadata.get("engine", "lightgbm") != proxy_engine:
        raise ValueError("OOF proxy engine differs from final proxy engine")
    september_index = semi.loc["2025-09-01":"2025-09-30 23:45"].index
    september_base = predict_curve(
        september_model,
        semi.loc[september_index],
        bundle.truth,
        bundle.contract,
        day_weight=0.5,
    )
    september_truth = future_truth(bundle.truth, september_index, bundle.contract)
    previous = pd.read_parquet(oof_dir / "time_oof_short.parquet")
    previous_index = pd.DatetimeIndex(
        previous["reference_time"].drop_duplicates().to_numpy()
    )
    expected_horizons = np.tile(np.arange(1, 9) * 15, len(previous_index))
    if len(previous) != len(expected_horizons) or not np.array_equal(
        previous["horizon_minutes"].to_numpy(), expected_horizons
    ):
        raise ValueError("Unexpected OOF schema or horizon order")
    previous_base = previous[["base_g1", "base_gall"]].to_numpy().reshape(-1, 8, 2)
    previous_truth = previous[["truth_g1", "truth_gall"]].to_numpy().reshape(-1, 8, 2)
    oof_index = previous_index.append(september_index)
    oof_base = np.concatenate([previous_base, september_base[:, :8]], axis=0)
    oof_truth = np.concatenate([previous_truth, september_truth[:, :8]], axis=0)
    highfreq = pd.read_parquet(data_dir / "semifinal_highfreq_at_15m_origins.parquet")
    matrix = model_matrix(oof_index, oof_base, highfreq)
    fit_residual(matrix, oof_truth, oof_base, output / "c_residual_models")

    selection = json.loads(
        (oof_dir / "c_target_selection.json").read_text(encoding="utf-8")
    )
    if (
        selection["generator_1"]["retain_residual"]
        or not selection["generator_all"]["retain_residual"]
    ):
        raise ValueError(
            "This finalization is valid only for August-selected generator_all residual"
        )
    manifest = {
        "training_cutoff_exclusive": str(final_cutoff),
        "base_engine": proxy_engine,
        "base_variant": BASE_VARIANT,
        "base_lookback_days": BASE_LOOKBACK_DAYS,
        "base_template": "50% week4 / 50% day7 + 6h decay",
        "base_proxy_labels": (
            "preliminary same-record 15min sample and semifinal same-record "
            "1min load; process features available by origin"
        ),
        "base_rows": json.loads(
            (output / "base_model" / "fit.json").read_text(encoding="utf-8")
        )["rows"],
        "common_feature_count": len(base.columns),
        "residual_variant": (
            "EXP-C: pooled 8-horizon LightGBM; apply only generator_all"
        ),
        "residual_oof_folds": ["2025-06", "2025-07", "2025-08", "2025-09"],
        "residual_oof_origins": len(oof_index),
        "residual_selection_source": str(oof_dir / "c_target_selection.json"),
        "generator_1_residual_enabled": False,
        "generator_all_residual_enabled": True,
        "observed_feature_max_time": "reference_time",
        "pre_observation_delay_minutes": 15,
        "feature_manifest_sha256": digest(data_dir / "feature_manifest.json"),
        "files_sha256": {
            str(path.relative_to(output)): digest(path)
            for path in sorted([*output.rglob("*.txt"), *output.rglob("target_*.json")])
        },
        "runtime_seconds": time.monotonic() - started,
        "official_score": None,
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "features_experiments" / "exp_bc" / "data"
    )
    parser.add_argument(
        "--proxy-engine", choices=("lightgbm", "xgboost"), default="xgboost"
    )
    parser.add_argument("--oof", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result_root = (
        ROOT
        / "results"
        / ("exp_bc_xgboost" if args.proxy_engine == "xgboost" else "exp_bc")
    )
    print(
        json.dumps(
            run(
                args.data,
                args.oof or result_root / "oof_residual_b7",
                args.output or result_root / "final_jan_sep_b7",
                proxy_engine=args.proxy_engine,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
