"""Cross-fold calibration for the best GPU XGBoost residual model.

Every evaluated fold is calibrated exclusively with predictions and labels from
the other folds.  The source is the causal, training-only supervised table.
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
FEATURE_DIR = ROOT / "results" / "features"
INPUT_PATH = FEATURE_DIR / "train_supervised_features.pkl"
CATALOG_PATH = FEATURE_DIR / "feature_catalog.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "xgboost_calibrated"
LOG_DIR = ROOT / "results" / "preprocessing" / "logs"
HORIZONS = tuple(range(1, 9))
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)
BETA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
PARAMETERS = {
    "n_estimators": 350,
    "max_depth": 5,
    "learning_rate": 0.025,
    "min_child_weight": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.70,
    "reg_alpha": 2.0,
    "reg_lambda": 30.0,
}


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("xgboost_crossfold_calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "10_xgboost_crossfold_calibration.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def mape(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-6)))


def generate_oof(data: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    label50_cols = [f"label_p50_h{h}" for h in HORIZONS]
    label120_cols = [f"label_p120_h{h}" for h in HORIZONS]

    for fold_name, start_text, end_text in FOLDS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        train = data[data["datetime"] < start - pd.Timedelta(minutes=120)]
        valid = data[(data["datetime"] >= start) & (data["datetime"] <= end)].copy()
        x_train = train[features].to_numpy(dtype=np.float32)
        x_valid = valid[features].to_numpy(dtype=np.float32)
        y50_train = train[label50_cols].to_numpy(dtype=np.float32)
        y120_train = train[label120_cols].to_numpy(dtype=np.float32)
        current50_train = train["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        current120_train = train["feat_p120_current"].to_numpy(dtype=np.float32)[:, None]
        residual = np.concatenate(
            [y50_train - current50_train, y120_train - current120_train], axis=1
        )

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            random_state=20260803,
            n_jobs=1,
            max_bin=256,
            multi_strategy="one_output_per_tree",
            verbosity=0,
            **PARAMETERS,
        )
        begin = time.perf_counter()
        model.fit(x_train, residual, verbose=False)
        raw_residual = model.predict(x_valid)
        runtime = time.perf_counter() - begin

        output = pd.DataFrame({"datetime": valid["datetime"].to_numpy(), "fold": fold_name})
        output["current_generator_1"] = valid["feat_p50_current"].to_numpy(dtype=np.float32)
        output["current_generator_all"] = (
            valid["feat_p50_current"].to_numpy(dtype=np.float32)
            + valid["feat_p120_current"].to_numpy(dtype=np.float32)
        )
        for index, horizon in enumerate(HORIZONS):
            output[f"actual_generator_1_h{horizon}"] = valid[label50_cols[index]].to_numpy(dtype=np.float32)
            output[f"actual_generator_all_h{horizon}"] = (
                valid[label50_cols[index]].to_numpy(dtype=np.float32)
                + valid[label120_cols[index]].to_numpy(dtype=np.float32)
            )
            output[f"raw_correction_generator_1_h{horizon}"] = raw_residual[:, index]
            output[f"raw_correction_generator_all_h{horizon}"] = (
                raw_residual[:, index] + raw_residual[:, len(HORIZONS) + index]
            )
        prediction_frames.append(output)
        runtime_rows.append(
            {
                "fold": fold_name,
                "train_rows": len(train),
                "valid_rows": len(valid),
                "features": len(features),
                "fit_predict_seconds": runtime,
                "device_requested": "cuda",
            }
        )
        LOGGER.info("Generated OOF predictions for %s in %.2fs", fold_name, runtime)

    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(runtime_rows)


def evaluate_crossfold(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for eval_fold, *_ in FOLDS:
        calibration = oof[oof["fold"] != eval_fold]
        evaluation = oof[oof["fold"] == eval_fold]
        fold_predictions = evaluation[["datetime", "fold"]].copy()

        for target in ("generator_1", "generator_all"):
            current_col = f"current_{target}"
            for horizon in HORIZONS:
                actual_col = f"actual_{target}_h{horizon}"
                correction_col = f"raw_correction_{target}_h{horizon}"
                cal_truth = calibration[actual_col].to_numpy(dtype=np.float64)
                cal_current = calibration[current_col].to_numpy(dtype=np.float64)
                cal_correction = calibration[correction_col].to_numpy(dtype=np.float64)
                grid_errors = np.array(
                    [mape(cal_truth, cal_current + beta * cal_correction) for beta in BETA_GRID]
                )
                best_index = int(np.argmin(grid_errors))
                beta = float(BETA_GRID[best_index])

                truth = evaluation[actual_col].to_numpy(dtype=np.float64)
                current = evaluation[current_col].to_numpy(dtype=np.float64)
                correction = evaluation[correction_col].to_numpy(dtype=np.float64)
                calibrated = np.maximum(current + beta * correction, 0.0)
                fixed = np.maximum(current + 0.25 * correction, 0.0)

                coefficient_rows.append(
                    {
                        "evaluation_fold": eval_fold,
                        "calibration_folds": "+".join(f for f, *_ in FOLDS if f != eval_fold),
                        "target": target,
                        "horizon_step": horizon,
                        "horizon_minutes": horizon * 15,
                        "selected_beta": beta,
                        "calibration_mape": float(grid_errors[best_index]),
                        "calibration_persistence_mape": float(grid_errors[0]),
                        "calibration_samples": len(calibration),
                    }
                )
                for model_name, prediction in (
                    ("persistence", current),
                    ("xgb_fixed_beta_0.25", fixed),
                    ("xgb_crossfold_calibrated", calibrated),
                ):
                    error = prediction - truth
                    metric_rows.append(
                        {
                            "fold": eval_fold,
                            "model": model_name,
                            "target": target,
                            "horizon_step": horizon,
                            "horizon_minutes": horizon * 15,
                            "mape": mape(truth, prediction),
                            "score_1_minus_mape": 1.0 - mape(truth, prediction),
                            "mae": float(np.mean(np.abs(error))),
                            "rmse": float(np.sqrt(np.mean(error**2))),
                            "samples": len(truth),
                        }
                    )
                fold_predictions[actual_col] = truth
                fold_predictions[f"pred_persistence_{target}_h{horizon}"] = current
                fold_predictions[f"pred_xgb_fixed_{target}_h{horizon}"] = fixed
                fold_predictions[f"pred_xgb_calibrated_{target}_h{horizon}"] = calibrated

        prediction_rows.append(fold_predictions)

    return (
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    features = catalog["feature"].tolist()

    oof, runtime = generate_oof(data, features)
    coefficients, metrics, predictions = evaluate_crossfold(oof)
    official_summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            mean_mape=("mape", "mean"),
            std_mape=("mape", "std"),
            worst_mape=("mape", "max"),
            mean_score=("score_1_minus_mape", "mean"),
        )
        .sort_values("mean_mape")
    )

    runtime.to_csv(RESULT_DIR / "runtime.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(RESULT_DIR / "raw_oof_predictions.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(RESULT_DIR / "crossfold_coefficients.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "metrics_by_fold_horizon.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULT_DIR / "evaluated_predictions.csv", index=False, encoding="utf-8-sig")
    official_summary.to_csv(RESULT_DIR / "model_summary.csv", index=False, encoding="utf-8-sig")

    best = official_summary.iloc[0].to_dict()
    persistence_mape = float(
        official_summary.loc[official_summary["model"] == "persistence", "mean_mape"].iloc[0]
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(INPUT_PATH.relative_to(ROOT)),
        "source_scope": "causal_preprocessed_training_data_only",
        "external_scoring_data_accessed": False,
        "calibration_protocol": "for each evaluation fold, choose target/horizon beta using the other folds only",
        "beta_grid": BETA_GRID.tolist(),
        "parameters": PARAMETERS,
        "best": best,
        "persistence_mape": persistence_mape,
        "relative_mape_reduction_vs_persistence": (persistence_mape - float(best["mean_mape"])) / persistence_mape,
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    LOGGER.info(
        "Best=%s MAPE=%.6f; persistence=%.6f",
        best["model"],
        best["mean_mape"],
        persistence_mape,
    )


if __name__ == "__main__":
    main()
