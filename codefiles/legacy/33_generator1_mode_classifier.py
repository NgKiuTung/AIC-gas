"""Forecast generator_1 operating modes and blend with continuous regression."""

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
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_cleaning_enhanced.pkl"
CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog_cleaning_enhanced.csv"
REGRESSION_OOF = ROOT / "results" / "experiments" / "cleaning_group_ablation" / "raw_oof_predictions.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "generator1_mode"
LOG_DIR = ROOT / "results" / "preprocessing" / "logs"
HORIZONS = tuple(range(1, 9))
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)
FEATURE_GROUPS = {"base", "lag", "difference", "rolling", "smooth", "transition", "mechanism_ratio", "clean_view"}
WEIGHT_MODE_GRID = np.round(np.arange(0.0, 1.0001, 0.10), 2)
BETA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
PARAMETERS = {
    "n_estimators": 350, "max_depth": 4, "learning_rate": 0.03,
    "min_child_weight": 15, "subsample": 0.85, "colsample_bytree": 0.70,
    "reg_alpha": 1.0, "reg_lambda": 20.0,
}


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("generator1_mode_classifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "33_generator1_mode_classifier.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def class_labels(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1).astype(np.int32)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    features = catalog.loc[catalog["group"].isin(FEATURE_GROUPS), "feature"].tolist()
    mode_parts: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    centers_rows: list[dict[str, object]] = []
    for fold_number, (fold_name, start_text, end_text) in enumerate(FOLDS):
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        train = data[data["datetime"] < start - pd.Timedelta(minutes=120)]
        valid = data[(data["datetime"] >= start) & (data["datetime"] <= end)]
        x_train = train[features].to_numpy(dtype=np.float32)
        x_valid = valid[features].to_numpy(dtype=np.float32)
        y_train = train[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
        y_valid = valid[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
        current = valid["feat_p50_current"].to_numpy(dtype=np.float32)
        clustering = KMeans(n_clusters=4, n_init=20, random_state=20260803 + fold_number)
        clustering.fit(y_train.reshape(-1, 1))
        centers = np.sort(clustering.cluster_centers_.ravel()).astype(np.float32)
        for index, center in enumerate(centers):
            centers_rows.append({"fold": fold_name, "mode": index, "center": float(center)})
        for index, horizon in enumerate(HORIZONS):
            labels = class_labels(y_train[:, index], centers)
            counts = np.bincount(labels, minlength=len(centers)).astype(float)
            class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
            sample_weight = class_weights[labels]
            sample_weight /= sample_weight.mean()
            classifier = xgb.XGBClassifier(
                objective="multi:softprob", num_class=len(centers), eval_metric="mlogloss",
                tree_method="hist", device="cuda", random_state=20260803 + horizon,
                n_jobs=1, max_bin=256, verbosity=0, **PARAMETERS,
            )
            begin = time.perf_counter()
            classifier.fit(x_train, labels, sample_weight=sample_weight, verbose=False)
            probability = classifier.predict_proba(x_valid)
            runtime = time.perf_counter() - begin
            expected_level = probability @ centers
            actual_class = class_labels(y_valid[:, index], centers)
            predicted_class = probability.argmax(axis=1)
            mode_parts.append(
                pd.DataFrame(
                    {"datetime": valid["datetime"].to_numpy(), "fold": fold_name,
                     "target": "generator_1", "horizon_step": horizon,
                     "horizon_minutes": horizon * 15, "actual": y_valid[:, index],
                     "current": current, "mode_prediction": expected_level,
                     "actual_mode_class": actual_class, "predicted_mode_class": predicted_class,
                     "mode_correct": actual_class == predicted_class}
                )
            )
            runtime_rows.append(
                {"fold": fold_name, "horizon_step": horizon, "features": len(features),
                 "runtime_seconds": runtime, "device_requested": "cuda"}
            )
        LOGGER.info("Completed %s centers=%s", fold_name, centers.tolist())

    mode = pd.concat(mode_parts, ignore_index=True)
    regression = pd.read_csv(REGRESSION_OOF, encoding="utf-8-sig", parse_dates=["datetime"])
    regression = regression[
        (regression["variant"] == "core_smooth_transition_ratio_clean")
        & (regression["target"] == "generator_1")
    ]
    id_keys = ["datetime", "fold", "target", "horizon_step", "horizon_minutes"]
    joined = mode.merge(
        regression[id_keys + ["actual", "current", "raw_correction"]],
        on=id_keys, how="inner", validate="one_to_one", suffixes=("_mode", "_regression")
    )
    if len(joined) != len(mode):
        raise ValueError("Mode and regression OOF predictions do not align")
    joined["actual"] = joined["actual_mode"]
    joined["current"] = joined["current_mode"]
    truth = joined["actual"].to_numpy(dtype=float)
    current = joined["current"].to_numpy(dtype=float)
    h = (joined["horizon_step"].to_numpy(dtype=float) - 1.0) / 7.0
    mode_correction = joined["mode_prediction"].to_numpy(dtype=float) - current
    regression_correction = joined["raw_correction"].to_numpy(dtype=float)
    group_size = joined.groupby(["fold", "horizon_step"])["actual"].transform("size").to_numpy(dtype=float)
    weight = 1.0 / group_size / joined.groupby(["fold", "horizon_step"]).ngroups
    search_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for mode_weight in WEIGHT_MODE_GRID:
        correction = mode_weight * mode_correction + (1.0 - mode_weight) * regression_correction
        for start in BETA_GRID:
            for end in BETA_GRID:
                beta = start + (end - start) * h
                pred = np.maximum(current + beta * correction, 0.0)
                ape = np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-6)
                value = float(np.sum(ape * weight))
                row = {
                    "mode_weight": float(mode_weight), "regression_weight": float(1.0 - mode_weight),
                    "beta_h15": float(start), "beta_h120": float(end), "mean_mape": value,
                }
                search_rows.append(row)
                if best is None or value < float(best["mean_mape"]):
                    best = row
    assert best is not None
    correction = float(best["mode_weight"]) * mode_correction + float(best["regression_weight"]) * regression_correction
    beta = float(best["beta_h15"]) + (float(best["beta_h120"]) - float(best["beta_h15"])) * h
    joined["prediction"] = np.maximum(current + beta * correction, 0.0)
    joined["ape_persistence"] = np.abs(current - truth) / np.maximum(np.abs(truth), 1e-6)
    joined["ape_model"] = np.abs(joined["prediction"].to_numpy() - truth) / np.maximum(np.abs(truth), 1e-6)
    metrics = joined.groupby(["fold", "horizon_step", "horizon_minutes"], as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), model_mape=("ape_model", "mean"),
        mode_accuracy=("mode_correct", "mean")
    )
    pd.DataFrame(centers_rows).to_csv(RESULT_DIR / "mode_centers.csv", index=False, encoding="utf-8-sig")
    mode.to_csv(RESULT_DIR / "mode_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(RESULT_DIR / "runtime.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(search_rows).to_csv(RESULT_DIR / "blend_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([best]).to_csv(RESULT_DIR / "selected_parameters.csv", index=False, encoding="utf-8-sig")
    joined.to_csv(RESULT_DIR / "selected_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "metrics_by_fold_horizon.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "cleaning-enhanced causal training features only",
        "external_scoring_data_accessed": False,
        "features": len(features), "parameters": PARAMETERS,
        "selected": best, "mean_mode_accuracy": float(mode["mode_correct"].mean()),
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
