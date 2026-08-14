"""Run training-only rolling-validation baselines for 15-120 minute forecasts."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "results" / "features"
INPUT_PATH = FEATURE_DIR / "train_supervised_features.pkl"
CATALOG_PATH = FEATURE_DIR / "feature_catalog.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "baseline"
LOG_DIR = ROOT / "results" / "preprocessing" / "logs"
HORIZONS = tuple(range(1, 9))
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("baseline_experiments")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "08_baseline_experiments.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def clip_components(p50: np.ndarray, p120: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.clip(p50, 0, 200), np.clip(p120, 0, 240)


def metrics_for_predictions(
    fold: str,
    model: str,
    y50: np.ndarray,
    y120: np.ndarray,
    p50: np.ndarray,
    p120: np.ndarray,
    runtime_seconds: float,
) -> list[dict[str, Any]]:
    p50, p120 = clip_components(p50, p120)
    yall = y50 + y120
    pall = p50 + p120
    rows: list[dict[str, Any]] = []
    for h_index, horizon in enumerate(HORIZONS):
        for target, truth, pred in (
            ("generator_1", y50[:, h_index], p50[:, h_index]),
            ("generator_all", yall[:, h_index], pall[:, h_index]),
            ("generator_120_component", y120[:, h_index], p120[:, h_index]),
        ):
            error = pred - truth
            mape = float(np.mean(np.abs(error) / np.maximum(np.abs(truth), 1e-6)))
            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "target": target,
                    "horizon_step": horizon,
                    "horizon_minutes": horizon * 15,
                    "mape": mape,
                    "score_1_minus_mape": 1 - mape,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "samples": len(truth),
                    "runtime_seconds": runtime_seconds,
                }
            )
    return rows


def target_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y50 = frame[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y120 = frame[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    return y50, y120


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    feature_columns = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")["feature"].tolist()
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for fold_name, valid_start_text, valid_end_text in FOLDS:
        valid_start, valid_end = pd.Timestamp(valid_start_text), pd.Timestamp(valid_end_text)
        train_cutoff = valid_start - pd.Timedelta(minutes=120)
        train = data[data["datetime"] < train_cutoff]
        valid = data[(data["datetime"] >= valid_start) & (data["datetime"] <= valid_end)]
        if train.empty or valid.empty:
            raise ValueError(f"Empty fold {fold_name}")
        fold_rows.append(
            {
                "fold": fold_name,
                "train_rows": len(train),
                "train_max_reference": train["datetime"].max(),
                "gap_minutes": float((valid["datetime"].min() - train["datetime"].max()).total_seconds() / 60),
                "valid_rows": len(valid),
                "valid_min": valid["datetime"].min(),
                "valid_max": valid["datetime"].max(),
            }
        )
        y50_valid, y120_valid = target_arrays(valid)
        current50 = valid["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        current120 = valid["feat_p120_current"].to_numpy(dtype=np.float32)[:, None]

        start = time.perf_counter()
        p50 = np.repeat(current50, len(HORIZONS), axis=1)
        p120 = np.repeat(current120, len(HORIZONS), axis=1)
        metric_rows.extend(metrics_for_predictions(fold_name, "persistence", y50_valid, y120_valid, p50, p120, time.perf_counter() - start))

        start = time.perf_counter()
        delta50 = current50 - valid["feat_lag1_p50_current"].to_numpy(dtype=np.float32)[:, None]
        delta120 = current120 - valid["feat_lag1_p120_current"].to_numpy(dtype=np.float32)[:, None]
        steps = np.asarray(HORIZONS, dtype=np.float32)[None, :]
        p50, p120 = current50 + delta50 * steps, current120 + delta120 * steps
        metric_rows.extend(metrics_for_predictions(fold_name, "linear_trend", y50_valid, y120_valid, p50, p120, time.perf_counter() - start))

        for seasonal in ("day", "week"):
            start = time.perf_counter()
            p50 = valid[[f"feat_seasonal_{seasonal}_p50_current_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
            p120 = valid[[f"feat_seasonal_{seasonal}_p120_current_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
            metric_rows.extend(metrics_for_predictions(fold_name, f"seasonal_{seasonal}", y50_valid, y120_valid, p50, p120, time.perf_counter() - start))

        x_train = train[feature_columns].to_numpy(dtype=np.float32)
        x_valid = valid[feature_columns].to_numpy(dtype=np.float32)
        y50_train, y120_train = target_arrays(train)
        current50_train = train["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        current120_train = train["feat_p120_current"].to_numpy(dtype=np.float32)[:, None]
        residual_train = np.concatenate([y50_train - current50_train, y120_train - current120_train], axis=1)
        for alpha in RIDGE_ALPHAS:
            start = time.perf_counter()
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-4, max_iter=5000))
            model.fit(x_train, residual_train)
            pred = model.predict(x_valid)
            p50 = current50 + pred[:, : len(HORIZONS)]
            p120 = current120 + pred[:, len(HORIZONS) :]
            metric_rows.extend(
                metrics_for_predictions(
                    fold_name,
                    f"ridge_residual_alpha_{alpha:g}",
                    y50_valid,
                    y120_valid,
                    p50,
                    p120,
                    time.perf_counter() - start,
                )
            )
        LOGGER.info("Completed %s train=%d valid=%d", fold_name, len(train), len(valid))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULT_DIR / "baseline_metrics_by_fold_horizon.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(RESULT_DIR / "rolling_fold_definition.csv", index=False, encoding="utf-8-sig")
    official = metrics[metrics["target"].isin(["generator_1", "generator_all"])]
    summary = (
        official.groupby("model", as_index=False)
        .agg(
            mean_mape=("mape", "mean"),
            std_mape=("mape", "std"),
            worst_mape=("mape", "max"),
            mean_score=("score_1_minus_mape", "mean"),
            total_runtime_seconds=("runtime_seconds", "sum"),
        )
        .sort_values("mean_mape")
    )
    summary.to_csv(RESULT_DIR / "baseline_model_summary.csv", index=False, encoding="utf-8-sig")
    best = summary.iloc[0].to_dict()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(INPUT_PATH.relative_to(ROOT)),
        "official_test_accessed": False,
        "folds": [dict(zip(["name", "valid_start", "valid_end"], fold)) for fold in FOLDS],
        "training_label_gap_minutes": 120,
        "official_metric_aggregation": "mean MAPE over generator_1 and generator_all, 8 horizons, 3 folds",
        "best_model": best,
    }
    (RESULT_DIR / "baseline_experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Best baseline=%s mean_mape=%.6f", best["model"], best["mean_mape"])


if __name__ == "__main__":
    main()
