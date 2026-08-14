"""Screen recent-history windows for direct-target GPU XGBoost models."""

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
REFERENCE_METRICS = ROOT / "results" / "experiments" / "xgboost_direct" / "metrics_by_fold_horizon.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "recency_window"
LOG_DIR = ROOT / "results" / "preprocessing" / "logs"
HORIZONS = tuple(range(1, 9))
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)
WINDOW_DAYS = (21, 35, 49)
BLENDS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
PARAMETERS = {
    "n_estimators": 450, "max_depth": 3, "learning_rate": 0.025,
    "min_child_weight": 20, "subsample": 0.85, "colsample_bytree": 0.75,
    "reg_alpha": 1.0, "reg_lambda": 20.0,
}


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("recency_window_screen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "12_recency_window_screen.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y1 = frame[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y120 = frame[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    return y1, y1 + y120


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_summary = (
        metrics.groupby(["window", "target", "blend"], as_index=False)
        .agg(mean_mape=("mape", "mean"), std_mape=("mape", "std"), worst_mape=("mape", "max"))
    )
    selected = target_summary.loc[target_summary.groupby(["window", "target"])["mean_mape"].idxmin()].copy()
    window_summary = (
        selected.groupby("window", as_index=False)
        .agg(mean_mape=("mean_mape", "mean"), worst_mape=("worst_mape", "max"))
        .sort_values("mean_mape")
    )
    window_summary["mean_score"] = 1.0 - window_summary["mean_mape"]
    return selected, window_summary


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    features = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")["feature"].tolist()
    rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []

    reference = pd.read_csv(REFERENCE_METRICS, encoding="utf-8-sig")
    reference = reference[reference["model"] == "direct_d3"].copy()
    reference["window"] = "all_history"
    rows.extend(reference[["fold", "window", "blend", "target", "horizon_step", "horizon_minutes", "mape", "score_1_minus_mape", "mae", "rmse", "samples"]].to_dict(orient="records"))

    for fold_name, start_text, end_text in FOLDS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        cutoff = start - pd.Timedelta(minutes=120)
        valid = data[(data["datetime"] >= start) & (data["datetime"] <= end)]
        x_valid = valid[features].to_numpy(dtype=np.float32)
        y1_valid, yall_valid = targets(valid)
        current1_valid = valid["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        currentall_valid = (valid["feat_p50_current"] + valid["feat_p120_current"]).to_numpy(dtype=np.float32)[:, None]

        for days in WINDOW_DAYS:
            window_name = f"last_{days}d"
            train = data[(data["datetime"] < cutoff) & (data["datetime"] >= cutoff - pd.Timedelta(days=days))]
            x_train = train[features].to_numpy(dtype=np.float32)
            y1_train, yall_train = targets(train)
            current1_train = train["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
            currentall_train = (train["feat_p50_current"] + train["feat_p120_current"]).to_numpy(dtype=np.float32)[:, None]
            residual = np.concatenate([y1_train - current1_train, yall_train - currentall_train], axis=1)
            model = xgb.XGBRegressor(
                objective="reg:squarederror", tree_method="hist", device="cuda",
                random_state=20260803, n_jobs=1, max_bin=256,
                multi_strategy="one_output_per_tree", verbosity=0, **PARAMETERS,
            )
            begin = time.perf_counter()
            model.fit(x_train, residual, verbose=False)
            correction = model.predict(x_valid)
            runtime = time.perf_counter() - begin
            runtime_rows.append(
                {"fold": fold_name, "window": window_name, "train_rows": len(train),
                 "valid_rows": len(valid), "fit_predict_seconds": runtime, "device_requested": "cuda"}
            )
            for target, truth, current, delta in (
                ("generator_1", y1_valid, current1_valid, correction[:, :8]),
                ("generator_all", yall_valid, currentall_valid, correction[:, 8:]),
            ):
                for index, horizon in enumerate(HORIZONS):
                    for blend in BLENDS:
                        pred = np.maximum(current[:, 0] + blend * delta[:, index], 0.0)
                        error = pred - truth[:, index]
                        score_mape = float(np.mean(np.abs(error) / np.maximum(np.abs(truth[:, index]), 1e-6)))
                        rows.append(
                            {"fold": fold_name, "window": window_name, "blend": blend,
                             "target": target, "horizon_step": horizon, "horizon_minutes": horizon * 15,
                             "mape": score_mape, "score_1_minus_mape": 1.0 - score_mape,
                             "mae": float(np.mean(np.abs(error))),
                             "rmse": float(np.sqrt(np.mean(error**2))), "samples": len(valid)}
                        )
            LOGGER.info("Completed %s/%s rows=%d runtime=%.2fs", fold_name, window_name, len(train), runtime)

    metrics = pd.DataFrame(rows)
    selected, summary = summarize(metrics)
    metrics.to_csv(RESULT_DIR / "metrics_by_fold_horizon.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(RESULT_DIR / "selected_target_blends.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULT_DIR / "window_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(RESULT_DIR / "runtime.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "causal_preprocessed_training_data_only",
        "external_scoring_data_accessed": False,
        "screen_model": "direct-target GPU XGBoost depth=3",
        "parameters": PARAMETERS,
        "best": summary.iloc[0].to_dict(),
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    LOGGER.info("Best recency screen: %s", report["best"])


if __name__ == "__main__":
    main()
