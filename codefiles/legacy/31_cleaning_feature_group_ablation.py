"""GPU rolling ablation of cleaning-enhanced feature groups."""

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
RESULT_DIR = ROOT / "results" / "experiments" / "cleaning_group_ablation"
LOG_DIR = ROOT / "results" / "preprocessing" / "logs"
HORIZONS = tuple(range(1, 9))
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)
BASE_GROUPS = {"base", "lag", "difference", "rolling"}
VARIANTS = {
    "core_smooth_transition_ratio": BASE_GROUPS | {"smooth", "transition", "mechanism_ratio"},
    "core_smooth_transition_ratio_clean": BASE_GROUPS | {"smooth", "transition", "mechanism_ratio", "clean_view"},
}
BETA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
PARAMETERS = {
    "n_estimators": 350, "max_depth": 5, "learning_rate": 0.025,
    "min_child_weight": 30, "subsample": 0.85, "colsample_bytree": 0.70,
    "reg_alpha": 2.0, "reg_lambda": 30.0,
}


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cleaning_feature_group_ablation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "31_cleaning_feature_group_ablation.log", encoding="utf-8"),
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
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    feature_sets = {
        name: catalog.loc[catalog["group"].isin(groups), "feature"].tolist()
        for name, groups in VARIANTS.items()
    }
    oof_parts: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    for fold_name, start_text, end_text in FOLDS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        train = data[data["datetime"] < start - pd.Timedelta(minutes=120)]
        valid = data[(data["datetime"] >= start) & (data["datetime"] <= end)]
        y1_train = train[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
        y120_train = train[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
        y1_valid = valid[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
        y120_valid = valid[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
        yall_train, yall_valid = y1_train + y120_train, y1_valid + y120_valid
        current1_train = train["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        currentall_train = (train["feat_p50_current"] + train["feat_p120_current"]).to_numpy(dtype=np.float32)[:, None]
        current1_valid = valid["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        currentall_valid = (valid["feat_p50_current"] + valid["feat_p120_current"]).to_numpy(dtype=np.float32)[:, None]
        mixed = np.concatenate(
            [(y1_train - current1_train) / np.maximum(np.abs(current1_train), 1e-6),
             yall_train - currentall_train], axis=1
        )
        for variant, features in feature_sets.items():
            model = xgb.XGBRegressor(
                objective="reg:squarederror", tree_method="hist", device="cuda",
                random_state=20260803, n_jobs=1, max_bin=256,
                multi_strategy="one_output_per_tree", verbosity=0, **PARAMETERS,
            )
            begin = time.perf_counter()
            model.fit(train[features].to_numpy(dtype=np.float32), mixed, verbose=False)
            raw = model.predict(valid[features].to_numpy(dtype=np.float32))
            runtime = time.perf_counter() - begin
            runtime_rows.append(
                {"fold": fold_name, "variant": variant, "features": len(features),
                 "runtime_seconds": runtime, "device_requested": "cuda"}
            )
            for target, truth, current, correction in (
                ("generator_1", y1_valid, current1_valid, current1_valid * raw[:, :8]),
                ("generator_all", yall_valid, currentall_valid, raw[:, 8:]),
            ):
                for index, horizon in enumerate(HORIZONS):
                    oof_parts.append(
                        pd.DataFrame(
                            {"datetime": valid["datetime"].to_numpy(), "fold": fold_name,
                             "variant": variant, "target": target, "horizon_step": horizon,
                             "horizon_minutes": horizon * 15, "actual": truth[:, index],
                             "current": current[:, 0], "raw_correction": correction[:, index]}
                        )
                    )
            LOGGER.info("Completed %s/%s features=%d runtime=%.2fs", fold_name, variant, len(features), runtime)

    oof = pd.concat(oof_parts, ignore_index=True)
    selected_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    for (variant, target), group in oof.groupby(["variant", "target"]):
        h = (group["horizon_step"].to_numpy(dtype=float) - 1.0) / 7.0
        truth = group["actual"].to_numpy(dtype=float)
        current = group["current"].to_numpy(dtype=float)
        correction = group["raw_correction"].to_numpy(dtype=float)
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
            {"variant": variant, "target": target, "beta_h15": start,
             "beta_h120": end, "mean_mape": value}
        )
        output = group.copy()
        output["beta"] = start + (end - start) * h
        output["prediction"] = np.maximum(current + output["beta"].to_numpy() * correction, 0.0)
        output["ape_persistence"] = np.abs(current - truth) / np.maximum(np.abs(truth), 1e-6)
        output["ape_model"] = np.abs(output["prediction"].to_numpy() - truth) / np.maximum(np.abs(truth), 1e-6)
        prediction_parts.append(output)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = predictions.groupby(["variant", "fold", "target", "horizon_step"], as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), model_mape=("ape_model", "mean")
    )
    summary = metrics.groupby("variant", as_index=False).agg(
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
        "feature_counts": {k: len(v) for k, v in feature_sets.items()},
        "best": summary.iloc[0].to_dict(), "selected_parameters": selected_rows,
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
