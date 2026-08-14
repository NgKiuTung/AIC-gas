"""GPU XGBoost ablation using the two scored generation targets directly.

The model predicts residuals from persistence for generator_1 and generator_all.
All rolling folds and features come only from causal training data.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features.pkl"
CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "xgboost_direct"
LOG_DIR = ROOT / "results" / "preprocessing" / "logs"
HORIZONS = tuple(range(1, 9))
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)
CONFIGS = {
    "direct_d3": {
        "n_estimators": 450, "max_depth": 3, "learning_rate": 0.025,
        "min_child_weight": 20, "subsample": 0.85, "colsample_bytree": 0.75,
        "reg_alpha": 1.0, "reg_lambda": 20.0,
    },
    "direct_d5": {
        "n_estimators": 350, "max_depth": 5, "learning_rate": 0.025,
        "min_child_weight": 30, "subsample": 0.85, "colsample_bytree": 0.70,
        "reg_alpha": 2.0, "reg_lambda": 30.0,
    },
}
BLENDS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
TARGET_MODE = "absolute"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("xgboost_direct_targets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "11_xgboost_direct_targets.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def target_matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y50 = frame[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y120 = frame[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    return y50, y50 + y120


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    features = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")["feature"].tolist()
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []

    for fold_name, start_text, end_text in FOLDS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        train = data[data["datetime"] < start - pd.Timedelta(minutes=120)]
        valid = data[(data["datetime"] >= start) & (data["datetime"] <= end)]
        x_train = train[features].to_numpy(dtype=np.float32)
        x_valid = valid[features].to_numpy(dtype=np.float32)
        y1_train, yall_train = target_matrices(train)
        y1_valid, yall_valid = target_matrices(valid)
        current1_train = train["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        currentall_train = (
            train["feat_p50_current"].to_numpy(dtype=np.float32)
            + train["feat_p120_current"].to_numpy(dtype=np.float32)
        )[:, None]
        current1_valid = valid["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        currentall_valid = (
            valid["feat_p50_current"].to_numpy(dtype=np.float32)
            + valid["feat_p120_current"].to_numpy(dtype=np.float32)
        )[:, None]
        if TARGET_MODE == "absolute":
            residuals = np.concatenate(
                [y1_train - current1_train, yall_train - currentall_train], axis=1
            )
        elif TARGET_MODE == "relative":
            residuals = np.concatenate(
                [
                    (y1_train - current1_train) / np.maximum(np.abs(current1_train), 1e-6),
                    (yall_train - currentall_train) / np.maximum(np.abs(currentall_train), 1e-6),
                ],
                axis=1,
            )
        else:
            raise ValueError(f"Unknown TARGET_MODE={TARGET_MODE}")

        for config_name, parameters in CONFIGS.items():
            model = xgb.XGBRegressor(
                objective="reg:squarederror", tree_method="hist", device="cuda",
                random_state=20260803, n_jobs=1, max_bin=256,
                multi_strategy="one_output_per_tree", verbosity=0, **parameters,
            )
            begin = time.perf_counter()
            model.fit(x_train, residuals, verbose=False)
            correction = model.predict(x_valid)
            if TARGET_MODE == "relative":
                correction = np.concatenate(
                    [current1_valid * correction[:, :8], currentall_valid * correction[:, 8:]], axis=1
                )
            runtime = time.perf_counter() - begin
            runtime_rows.append(
                {"fold": fold_name, "model": config_name, "train_rows": len(train),
                 "valid_rows": len(valid), "features": len(features),
                 "fit_predict_seconds": runtime, "device_requested": "cuda"}
            )

            for target, truth, current, delta in (
                ("generator_1", y1_valid, current1_valid, correction[:, :8]),
                ("generator_all", yall_valid, currentall_valid, correction[:, 8:]),
            ):
                for index, horizon in enumerate(HORIZONS):
                    long_frame = pd.DataFrame(
                        {
                            "datetime": valid["datetime"].to_numpy(), "fold": fold_name,
                            "model": config_name, "target": target,
                            "horizon_step": horizon, "horizon_minutes": horizon * 15,
                            "actual": truth[:, index], "current": current[:, 0],
                            "raw_correction": delta[:, index],
                        }
                    )
                    prediction_rows.append(long_frame)
                    for blend in BLENDS:
                        prediction = np.maximum(current[:, 0] + blend * delta[:, index], 0.0)
                        error = prediction - truth[:, index]
                        mape = float(np.mean(np.abs(error) / np.maximum(np.abs(truth[:, index]), 1e-6)))
                        metric_rows.append(
                            {"fold": fold_name, "model": config_name, "blend": blend,
                             "target": target, "horizon_step": horizon,
                             "horizon_minutes": horizon * 15, "mape": mape,
                             "score_1_minus_mape": 1.0 - mape,
                             "mae": float(np.mean(np.abs(error))),
                             "rmse": float(np.sqrt(np.mean(error**2))), "samples": len(valid)}
                        )
            LOGGER.info("Completed %s/%s in %.2fs", fold_name, config_name, runtime)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    target_summary = (
        metrics.groupby(["model", "target", "blend"], as_index=False)
        .agg(mean_mape=("mape", "mean"), std_mape=("mape", "std"), worst_mape=("mape", "max"))
        .sort_values(["target", "mean_mape"])
    )
    selected = target_summary.loc[target_summary.groupby(["model", "target"])["mean_mape"].idxmin()].copy()
    model_summary = (
        selected.groupby("model", as_index=False)
        .agg(mean_mape=("mean_mape", "mean"), worst_mape=("worst_mape", "max"))
        .sort_values("mean_mape")
    )
    model_summary["mean_score"] = 1.0 - model_summary["mean_mape"]

    metrics.to_csv(RESULT_DIR / "metrics_by_fold_horizon.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULT_DIR / "raw_oof_predictions_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(RESULT_DIR / "runtime.csv", index=False, encoding="utf-8-sig")
    target_summary.to_csv(RESULT_DIR / "target_blend_summary.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(RESULT_DIR / "selected_target_blends.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(RESULT_DIR / "model_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(INPUT_PATH.relative_to(ROOT)),
        "source_scope": "causal_preprocessed_training_data_only",
        "external_scoring_data_accessed": False,
        "target_design": "direct residual forecasts for generator_1 and generator_all",
        "target_mode": TARGET_MODE,
        "configs": CONFIGS,
        "blend_grid": list(BLENDS),
        "selection": selected.to_dict(orient="records"),
        "best": model_summary.iloc[0].to_dict(),
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    LOGGER.info("Best direct model: %s", report["best"])


if __name__ == "__main__":
    main()
