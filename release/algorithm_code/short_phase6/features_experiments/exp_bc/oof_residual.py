"""Time-OOF target-calendar ablation and 1min short-horizon residual experiment."""

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
from gasstage.targets import future_truth

from features_experiments.exp_bc.b_variants import (
    fit_proxy,
    load_feature_tables,
    predict_curve,
)
from features_experiments.exp_bc.baseline import (
    CUTOFF,
    END,
    load_bundle,
    score,
    score_by_horizon,
)

FOLDS = (
    ("june", pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01")),
    ("july", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01")),
    ("august", pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01")),
)
BASE_LOOKBACK_DAYS = 180
BASE_VARIANT = "B7_full_semifinal_history"


def target_calendar(index: pd.DatetimeIndex, blocks: int = 8) -> np.ndarray:
    """Known future block-center calendar, with no target/process observation."""
    block_index = pd.DatetimeIndex(
        (
            index.to_numpy(dtype="datetime64[ns]")[:, None]
            + (np.arange(blocks)[None, :] * 15).astype("timedelta64[m]")
        ).ravel()
    )
    minute = block_index.hour.to_numpy() * 60 + block_index.minute.to_numpy() + 7.5
    week = block_index.dayofweek.to_numpy() + minute / 1440
    return np.column_stack(
        [
            np.tile((np.arange(blocks) + 1) / blocks, len(index)),
            np.sin(2 * np.pi * minute / 1440),
            np.cos(2 * np.pi * minute / 1440),
            np.sin(2 * np.pi * week / 7),
            np.cos(2 * np.pi * week / 7),
            block_index.month.to_numpy() / 12,
        ]
    ).astype("float32")


def model_matrix(
    index: pd.DatetimeIndex, base: np.ndarray, highfreq: pd.DataFrame | None
) -> np.ndarray:
    calendar = target_calendar(index)
    arrays = [calendar, base[:, :8, 0].reshape(-1, 1).astype("float32")]
    if highfreq is not None:
        hf = highfreq.reindex(index).to_numpy(dtype="float32")
        arrays.append(np.repeat(hf, 8, axis=0))
    return np.column_stack(arrays)


def fit_residual(
    matrix: np.ndarray,
    truth: np.ndarray,
    base: np.ndarray,
    output: Path,
    *,
    rounds: int = 120,
    leaves: int = 15,
) -> list[lgb.Booster]:
    output.mkdir(parents=True, exist_ok=True)
    models = []
    for target in range(2):
        y = truth[:, :8, target].ravel()
        baseline = base[:, :8, target].ravel()
        valid = np.isfinite(y) & (y > 0) & np.isfinite(baseline)
        x = matrix[valid].copy()
        x[:, 6] = baseline[valid]  # target-specific base prediction
        residual = y[valid] - baseline[valid]
        weights = 1 / y[valid]
        train = lgb.Dataset(x, label=residual, weight=weights / weights.mean())
        model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "None",
                "learning_rate": 0.04,
                "num_leaves": leaves,
                "min_data_in_leaf": 200,
                "max_bin": 127,
                "lambda_l2": 5.0,
                "feature_fraction": 0.85,
                "num_threads": 4,
                "seed": 20261003,
                "deterministic": True,
                "force_col_wise": True,
                "verbosity": -1,
            },
            train,
            num_boost_round=rounds,
        )
        model.save_model(str(output / f"residual_t{target}.txt"))
        models.append(model)
    return models


def apply_residual(
    models: list[lgb.Booster], matrix: np.ndarray, base: np.ndarray
) -> np.ndarray:
    updated = base.copy()
    for target, model in enumerate(models):
        x = matrix.copy()
        x[:, 6] = base[:, :8, target].ravel()
        correction = model.predict(x, num_threads=4).reshape(len(base), 8)
        updated[:, :8, target] += correction
    # Keep physical nonnegative/containment constraints without looking at truth.
    updated[:, :8, 0] = np.clip(updated[:, :8, 0], 0, 200)
    updated[:, :8, 1] = np.maximum(
        np.clip(updated[:, :8, 1], 0, 440), updated[:, :8, 0]
    )
    return updated


def run(
    data_dir: Path,
    result_dir: Path,
    *,
    lookback_days: int = BASE_LOOKBACK_DAYS,
    proxy_engine: str = "xgboost",
) -> dict:
    started = time.monotonic()
    if lookback_days not in (120, 180):
        raise ValueError("Only preregistered 120/180-day base windows are supported")
    base_variant = "B3_day_period" if lookback_days == 120 else BASE_VARIANT
    result_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(data_dir)
    pre, semi, pre_y, semi_y = load_feature_tables(data_dir)
    highfreq = pd.read_parquet(data_dir / "semifinal_highfreq_at_15m_origins.parquet")
    oof_index: list[pd.DatetimeIndex] = []
    oof_b1, oof_base, oof_truth = [], [], []
    fold_rows = []
    for name, cutoff, next_cutoff in FOLDS:
        fit = fit_proxy(
            pre,
            semi,
            pre_y,
            semi_y,
            cutoff,
            result_dir / f"oof_{name}_model",
            lookback_days=lookback_days,
            engine=proxy_engine,
        )
        index = semi.loc[(semi.index >= cutoff) & (semi.index < next_cutoff)].index
        frame = semi.loc[index]
        b1 = predict_curve(fit, frame, bundle.truth, bundle.contract)
        base = predict_curve(fit, frame, bundle.truth, bundle.contract, day_weight=0.5)
        actual = future_truth(bundle.truth, index, bundle.contract)
        oof_index.append(index)
        oof_b1.append(b1)
        oof_base.append(base)
        oof_truth.append(actual)
        fold_rows.append(
            {
                "fold": name,
                "train_cutoff": str(cutoff),
                "validation_origins": len(index),
                "b1_short_mape": float(
                    score(b1, actual)[0].query("period == 'short'")["mape"].mean()
                ),
                "base_short_mape": float(
                    score(base, actual)[0].query("period == 'short'")["mape"].mean()
                ),
            }
        )
    pd.DataFrame(fold_rows).to_csv(result_dir / "oof_fold_metrics.csv", index=False)
    index_oof = oof_index[0].append(oof_index[1:])
    b1_oof = np.concatenate(oof_b1)
    base_oof = np.concatenate(oof_base)
    truth_oof = np.concatenate(oof_truth)
    matrix_cal = model_matrix(index_oof, b1_oof, None)
    matrix_hf = model_matrix(index_oof, base_oof, highfreq)
    pd.DataFrame(
        {
            "reference_time": np.repeat(index_oof.to_numpy(), 8),
            "horizon_minutes": np.tile(np.arange(1, 9) * 15, len(index_oof)),
            "b1_g1": b1_oof[:, :8, 0].ravel(),
            "b1_gall": b1_oof[:, :8, 1].ravel(),
            "base_g1": base_oof[:, :8, 0].ravel(),
            "base_gall": base_oof[:, :8, 1].ravel(),
            "truth_g1": truth_oof[:, :8, 0].ravel(),
            "truth_gall": truth_oof[:, :8, 1].ravel(),
        }
    ).to_parquet(result_dir / "time_oof_short.parquet", index=False)

    # Select residual targets on August only; September remains untouched until
    # the selection rule is frozen. Earlier residual fitting sees June-July OOF.
    select_train = index_oof < pd.Timestamp("2025-08-01")
    select_valid = ~select_train
    selection_models = fit_residual(
        matrix_hf[np.repeat(select_train, 8)],
        truth_oof[select_train],
        base_oof[select_train],
        result_dir / "c_selection_june_july_models",
    )
    august_corrected = apply_residual(
        selection_models, matrix_hf[np.repeat(select_valid, 8)], base_oof[select_valid]
    )
    august_metrics_base, _ = score(base_oof[select_valid], truth_oof[select_valid])
    august_metrics_corrected, _ = score(august_corrected, truth_oof[select_valid])
    target_selection = {}
    for target in ("generator_1", "generator_all"):
        original = float(
            august_metrics_base.query("period == 'short' and target == @target")[
                "mape"
            ].iloc[0]
        )
        corrected = float(
            august_metrics_corrected.query("period == 'short' and target == @target")[
                "mape"
            ].iloc[0]
        )
        target_selection[target] = {
            "august_base_mape": original,
            "august_residual_mape": corrected,
            "retain_residual": corrected <= original - 0.001,
        }
    (result_dir / "c_target_selection.json").write_text(
        json.dumps(target_selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    b2_models = fit_residual(
        matrix_cal,
        truth_oof,
        b1_oof,
        result_dir / "b2_calendar_models",
        rounds=80,
        leaves=7,
    )
    c_models = fit_residual(
        matrix_hf, truth_oof, base_oof, result_dir / "c_highfreq_models"
    )

    final_fit = fit_proxy(
        pre,
        semi,
        pre_y,
        semi_y,
        CUTOFF,
        result_dir / "september_base_model",
        lookback_days=lookback_days,
        engine=proxy_engine,
    )
    index_sep = semi.loc[CUTOFF:END].index
    frame_sep = semi.loc[index_sep]
    b1_sep = predict_curve(final_fit, frame_sep, bundle.truth, bundle.contract)
    base_sep = predict_curve(
        final_fit, frame_sep, bundle.truth, bundle.contract, day_weight=0.5
    )
    actual_sep = future_truth(bundle.truth, index_sep, bundle.contract)
    b2_sep = apply_residual(b2_models, model_matrix(index_sep, b1_sep, None), b1_sep)
    c_sep = apply_residual(
        c_models, model_matrix(index_sep, base_sep, highfreq), base_sep
    )
    c_selected = base_sep.copy()
    for target_idx, target in enumerate(("generator_1", "generator_all")):
        if target_selection[target]["retain_residual"]:
            c_selected[:, :8, target_idx] = c_sep[:, :8, target_idx]
    c_selected[:, :8, 1] = np.maximum(c_selected[:, :8, 1], c_selected[:, :8, 0])
    rows = []
    for name, prediction in (
        ("B1_physical_time", b1_sep),
        ("B2_target_calendar", b2_sep),
        (base_variant, base_sep),
        (f"C_highfreq_residual_on_{base_variant[:2]}", c_sep),
        (f"C_august_selected_on_{base_variant[:2]}", c_selected),
    ):
        metrics, summary = score(prediction, actual_sep)
        metrics.insert(0, "variant", name)
        metrics.to_csv(result_dir / f"{name}_metrics.csv", index=False)
        horizons = score_by_horizon(prediction, actual_sep)
        horizons.insert(0, "variant", name)
        horizons.to_csv(result_dir / f"{name}_by_horizon.csv", index=False)
        rows.append(
            {
                "variant": name,
                **summary,
                "runtime_seconds_total": time.monotonic() - started,
            }
        )
    (result_dir / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "proxy_engine": proxy_engine,
        "oof_origins": len(index_oof),
        "folds": fold_rows,
        "september": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "features_experiments" / "exp_bc" / "data"
    )
    parser.add_argument(
        "--proxy-engine", choices=("lightgbm", "xgboost"), default="xgboost"
    )
    parser.add_argument("--results", type=Path)
    parser.add_argument("--lookback-days", type=int, default=BASE_LOOKBACK_DAYS)
    args = parser.parse_args()
    result_dir = args.results or (
        ROOT
        / "results"
        / ("exp_bc_xgboost" if args.proxy_engine == "xgboost" else "exp_bc")
        / "oof_residual_b7"
    )
    print(
        json.dumps(
            run(
                args.data,
                result_dir,
                lookback_days=args.lookback_days,
                proxy_engine=args.proxy_engine,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
