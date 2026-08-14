"""Horizon-specific GPU models with objectives aligned to MAPE."""

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
INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_cleaning_enhanced.pkl"
CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog_cleaning_enhanced.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "mape_aligned"
LOG_DIR = ROOT / "results" / "preprocessing" / "logs"
HORIZONS = tuple(range(1, 9))
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)
CONFIGS = {
    "weighted_mae": {"objective": "reg:absoluteerror", "weight_power": 1.0},
    "weighted_mse": {"objective": "reg:squarederror", "weight_power": 2.0},
}
BETA_GRID = np.round(np.arange(0.0, 1.5001, 0.05), 2)
PARAMETERS = {
    "n_estimators": 450, "max_depth": 4, "learning_rate": 0.025,
    "min_child_weight": 20, "subsample": 0.85, "colsample_bytree": 0.70,
    "reg_alpha": 1.0, "reg_lambda": 20.0,
}


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mape_aligned_objectives")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "28_mape_aligned_objectives.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    features = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")["feature"].tolist()
    oof_parts: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []

    for fold_name, start_text, end_text in FOLDS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        train = data[data["datetime"] < start - pd.Timedelta(minutes=120)]
        valid = data[(data["datetime"] >= start) & (data["datetime"] <= end)]
        x_train = train[features].to_numpy(dtype=np.float32)
        x_valid = valid[features].to_numpy(dtype=np.float32)
        target_values = {
            "generator_1": (
                train[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32),
                valid[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32),
                valid["feat_p50_current"].to_numpy(dtype=np.float32),
            ),
            "generator_all": (
                train[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
                + train[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32),
                valid[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
                + valid[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32),
                (valid["feat_p50_current"] + valid["feat_p120_current"]).to_numpy(dtype=np.float32),
            ),
        }
        for config_name, config in CONFIGS.items():
            for target, (y_train, y_valid, current_valid) in target_values.items():
                for index, horizon in enumerate(HORIZONS):
                    labels = y_train[:, index]
                    weights = 1.0 / np.maximum(np.abs(labels), 1e-6) ** float(config["weight_power"])
                    weights = (weights / np.mean(weights)).astype(np.float32)
                    model = xgb.XGBRegressor(
                        objective=str(config["objective"]), tree_method="hist", device="cuda",
                        random_state=20260803 + horizon, n_jobs=1, max_bin=256,
                        verbosity=0, **PARAMETERS,
                    )
                    begin = time.perf_counter()
                    model.fit(x_train, labels, sample_weight=weights, verbose=False)
                    prediction = model.predict(x_valid)
                    runtime = time.perf_counter() - begin
                    oof_parts.append(
                        pd.DataFrame(
                            {"datetime": valid["datetime"].to_numpy(), "fold": fold_name,
                             "config": config_name, "target": target, "horizon_step": horizon,
                             "horizon_minutes": horizon * 15, "actual": y_valid[:, index],
                             "current": current_valid, "raw_prediction": prediction}
                        )
                    )
                    runtime_rows.append(
                        {"fold": fold_name, "config": config_name, "target": target,
                         "horizon_step": horizon, "train_rows": len(train), "features": len(features),
                         "runtime_seconds": runtime, "device_requested": "cuda"}
                    )
                LOGGER.info("Completed %s/%s/%s", fold_name, config_name, target)

    oof = pd.concat(oof_parts, ignore_index=True)
    selected_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    for (config, target), group in oof.groupby(["config", "target"]):
        h = (group["horizon_step"].to_numpy(dtype=float) - 1.0) / 7.0
        truth = group["actual"].to_numpy(dtype=float)
        current = group["current"].to_numpy(dtype=float)
        correction = group["raw_prediction"].to_numpy(dtype=float) - current
        best: tuple[float, float, float] | None = None
        for start in BETA_GRID:
            for end in BETA_GRID:
                beta = start + (end - start) * h
                pred = np.maximum(current + beta * correction, 0.0)
                ape = np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-6)
                detail = group[["fold", "horizon_step"]].copy()
                detail["ape"] = ape
                value = float(detail.groupby(["fold", "horizon_step"])["ape"].mean().mean())
                if best is None or value < best[2]:
                    best = (float(start), float(end), value)
        assert best is not None
        start, end, value = best
        selected_rows.append(
            {"config": config, "target": target, "beta_h15": start,
             "beta_h120": end, "mean_mape": value}
        )
        output = group.copy()
        output["beta"] = start + (end - start) * h
        output["prediction"] = np.maximum(current + output["beta"].to_numpy() * correction, 0.0)
        output["ape_persistence"] = np.abs(current - truth) / np.maximum(np.abs(truth), 1e-6)
        output["ape_model"] = np.abs(output["prediction"].to_numpy() - truth) / np.maximum(np.abs(truth), 1e-6)
        prediction_parts.append(output)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = predictions.groupby(["config", "fold", "target", "horizon_step", "horizon_minutes"], as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), model_mape=("ape_model", "mean")
    )
    summary = metrics.groupby("config", as_index=False).agg(
        mean_mape=("model_mape", "mean"), worst_mape=("model_mape", "max"),
        persistence_mape=("persistence_mape", "mean")
    ).sort_values("mean_mape")
    summary["score"] = 1.0 - summary["mean_mape"]
    oof.to_csv(RESULT_DIR / "raw_oof_predictions.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULT_DIR / "selected_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "metrics_by_fold_target_horizon.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(RESULT_DIR / "selected_parameters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(RESULT_DIR / "runtime.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULT_DIR / "model_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "cleaning-enhanced causal training features only",
        "external_scoring_data_accessed": False,
        "features": len(features), "parameters": PARAMETERS, "configs": CONFIGS,
        "best": summary.iloc[0].to_dict(), "selected_parameters": selected_rows,
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
