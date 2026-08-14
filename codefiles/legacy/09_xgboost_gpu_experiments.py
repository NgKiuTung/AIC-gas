"""GPU XGBoost residual forecasts with persistence-shrinkage blending.

Training-only rolling validation; no scoring-test path exists in this module.
"""

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
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "results" / "features"
INPUT_PATH = FEATURE_DIR / "train_supervised_features.pkl"
CATALOG_PATH = FEATURE_DIR / "feature_catalog.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "xgboost_gpu"
LOG_DIR = ROOT / "results" / "preprocessing" / "logs"
HORIZONS = tuple(range(1, 9))
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)
CONFIGS = {
    "xgb_d3": {
        "n_estimators": 450,
        "max_depth": 3,
        "learning_rate": 0.025,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 1.0,
        "reg_lambda": 20.0,
    },
    "xgb_d5": {
        "n_estimators": 350,
        "max_depth": 5,
        "learning_rate": 0.025,
        "min_child_weight": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.70,
        "reg_alpha": 2.0,
        "reg_lambda": 30.0,
    },
}
BLENDS = (0.25, 0.5, 0.75, 1.0)


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("xgboost_gpu_experiments")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "09_xgboost_gpu_experiments.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def target_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32),
        frame[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32),
    )


def metric_rows(
    fold: str,
    model: str,
    blend: float,
    y50: np.ndarray,
    y120: np.ndarray,
    p50: np.ndarray,
    p120: np.ndarray,
    runtime: float,
) -> list[dict[str, Any]]:
    p50 = np.clip(p50, 0, 200)
    p120 = np.clip(p120, 0, 240)
    rows: list[dict[str, Any]] = []
    for index, horizon in enumerate(HORIZONS):
        for target, truth, pred in (
            ("generator_1", y50[:, index], p50[:, index]),
            ("generator_all", y50[:, index] + y120[:, index], p50[:, index] + p120[:, index]),
            ("generator_120_component", y120[:, index], p120[:, index]),
        ):
            error = pred - truth
            mape = float(np.mean(np.abs(error) / np.maximum(np.abs(truth), 1e-6)))
            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "blend": blend,
                    "target": target,
                    "horizon_step": horizon,
                    "horizon_minutes": horizon * 15,
                    "mape": mape,
                    "score_1_minus_mape": 1 - mape,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "samples": len(truth),
                    "fit_runtime_seconds": runtime,
                }
            )
    return rows


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    features = catalog["feature"].tolist()
    metric_output: list[dict[str, Any]] = []
    runtime_output: list[dict[str, Any]] = []
    for fold_name, start_text, end_text in FOLDS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        train = data[data["datetime"] < start - pd.Timedelta(minutes=120)]
        valid = data[(data["datetime"] >= start) & (data["datetime"] <= end)]
        x_train = train[features].to_numpy(dtype=np.float32)
        x_valid = valid[features].to_numpy(dtype=np.float32)
        y50_train, y120_train = target_arrays(train)
        y50_valid, y120_valid = target_arrays(valid)
        current50_train = train["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        current120_train = train["feat_p120_current"].to_numpy(dtype=np.float32)[:, None]
        current50_valid = valid["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        current120_valid = valid["feat_p120_current"].to_numpy(dtype=np.float32)[:, None]
        residual = np.concatenate([y50_train - current50_train, y120_train - current120_train], axis=1)
        for config_name, parameters in CONFIGS.items():
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                device="cuda",
                random_state=20260803,
                n_jobs=1,
                max_bin=256,
                multi_strategy="one_output_per_tree",
                verbosity=0,
                **parameters,
            )
            begin = time.perf_counter()
            model.fit(x_train, residual, verbose=False)
            pred = model.predict(x_valid)
            runtime = time.perf_counter() - begin
            runtime_output.append(
                {
                    "fold": fold_name,
                    "model": config_name,
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "features": len(features),
                    "outputs": residual.shape[1],
                    "fit_predict_seconds": runtime,
                    "device_requested": "cuda",
                }
            )
            for blend in BLENDS:
                p50 = current50_valid + blend * pred[:, : len(HORIZONS)]
                p120 = current120_valid + blend * pred[:, len(HORIZONS) :]
                metric_output.extend(
                    metric_rows(fold_name, config_name, blend, y50_valid, y120_valid, p50, p120, runtime)
                )
            LOGGER.info("Completed %s/%s runtime=%.2fs", fold_name, config_name, runtime)

    metrics = pd.DataFrame(metric_output)
    metrics.to_csv(RESULT_DIR / "xgboost_metrics_by_fold_horizon.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_output).to_csv(RESULT_DIR / "xgboost_runtime.csv", index=False, encoding="utf-8-sig")
    official = metrics[metrics["target"].isin(["generator_1", "generator_all"])]
    summary = (
        official.groupby(["model", "blend"], as_index=False)
        .agg(
            mean_mape=("mape", "mean"),
            std_mape=("mape", "std"),
            worst_mape=("mape", "max"),
            mean_score=("score_1_minus_mape", "mean"),
        )
        .sort_values("mean_mape")
    )
    summary.to_csv(RESULT_DIR / "xgboost_model_summary.csv", index=False, encoding="utf-8-sig")
    best = summary.iloc[0].to_dict()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(INPUT_PATH.relative_to(ROOT)),
        "official_test_accessed": False,
        "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "xgboost_version": xgb.__version__,
        "configs": CONFIGS,
        "blends": list(BLENDS),
        "best": best,
    }
    (RESULT_DIR / "xgboost_experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    LOGGER.info("Best GPU XGBoost=%s blend=%s mape=%.6f", best["model"], best["blend"], best["mean_mape"])


if __name__ == "__main__":
    main()
