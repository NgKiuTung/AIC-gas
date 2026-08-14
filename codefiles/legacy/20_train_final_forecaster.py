"""Fit and serialize the selected hybrid multi-horizon GPU forecaster."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features.pkl"
CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog.csv"
VALIDATION_PATH = ROOT / "results" / "experiments" / "hybrid_target" / "hybrid_summary.json"
MODEL_DIR = ROOT / "results" / "models" / "power_forecaster"
HORIZONS = tuple(range(1, 9))
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
BETA_GENERATOR_1 = np.linspace(0.35, 0.30, 8, dtype=np.float32)
BETA_GENERATOR_ALL = np.linspace(0.65, 0.60, 8, dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    features = catalog["feature"].tolist()
    x_train = data[features].to_numpy(dtype=np.float32)
    y1 = data[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y120 = data[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    yall = y1 + y120
    current1 = data["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
    currentall = (
        data["feat_p50_current"].to_numpy(dtype=np.float32)
        + data["feat_p120_current"].to_numpy(dtype=np.float32)
    )[:, None]
    mixed_target = np.concatenate(
        [
            (y1 - current1) / np.maximum(np.abs(current1), 1e-6),
            yall - currentall,
        ],
        axis=1,
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
    model.fit(x_train, mixed_target, verbose=False)
    runtime = time.perf_counter() - begin
    model_path = MODEL_DIR / "xgboost_hybrid_forecaster.json"
    model.save_model(model_path)

    smoke_data = data.tail(32)
    smoke_x = smoke_data[features].to_numpy(dtype=np.float32)
    raw = model.predict(smoke_x)
    smoke_current1 = smoke_data["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
    smoke_currentall = (
        smoke_data["feat_p50_current"].to_numpy(dtype=np.float32)
        + smoke_data["feat_p120_current"].to_numpy(dtype=np.float32)
    )[:, None]
    pred1 = np.maximum(smoke_current1 * (1.0 + raw[:, :8] * BETA_GENERATOR_1), 0.0)
    predall = np.maximum(smoke_currentall + raw[:, 8:] * BETA_GENERATOR_ALL, 0.0)
    if not np.isfinite(pred1).all() or not np.isfinite(predall).all():
        raise ValueError("Non-finite smoke-test predictions")
    if np.any(predall < pred1):
        raise ValueError("Smoke-test physical hierarchy violation: generator_all < generator_1")

    feature_schema = catalog.copy()
    feature_schema["position"] = np.arange(len(feature_schema))
    feature_schema.to_csv(MODEL_DIR / "feature_schema.csv", index=False, encoding="utf-8-sig")
    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_.astype(float)}
    ).sort_values("importance", ascending=False)
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    smoke_rows: list[dict[str, object]] = []
    for row_index, timestamp in enumerate(smoke_data["datetime"]):
        for index, horizon in enumerate(HORIZONS):
            smoke_rows.append(
                {
                    "origin_datetime": timestamp,
                    "horizon_step": horizon,
                    "horizon_minutes": horizon * 15,
                    "prediction_generator_1": float(pred1[row_index, index]),
                    "prediction_generator_all": float(predall[row_index, index]),
                }
            )
    pd.DataFrame(smoke_rows).to_csv(MODEL_DIR / "training_tail_smoke_predictions.csv", index=False, encoding="utf-8-sig")

    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "causal preprocessed training data only",
        "external_scoring_data_accessed": False,
        "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "xgboost_version": xgb.__version__,
        "model_file": model_path.name,
        "model_sha256": sha256(model_path),
        "input_matrix": str(INPUT_PATH.relative_to(ROOT)),
        "input_matrix_sha256": sha256(INPUT_PATH),
        "training_rows": len(data),
        "training_datetime_min": data["datetime"].min().isoformat(),
        "training_datetime_max": data["datetime"].max().isoformat(),
        "feature_count": len(features),
        "outputs": 16,
        "horizon_minutes": [15 * h for h in HORIZONS],
        "output_design": {
            "outputs_0_7": "relative residual of generator_1",
            "outputs_8_15": "absolute residual of generator_all",
        },
        "beta_generator_1": BETA_GENERATOR_1.tolist(),
        "beta_generator_all": BETA_GENERATOR_ALL.tolist(),
        "parameters": PARAMETERS,
        "fit_runtime_seconds": runtime,
        "validation": {
            "protocol": "three rolling-origin folds with 120-minute train/validation gap",
            "mape": validation["overall_hybrid_mape"],
            "score_1_minus_mape": validation["overall_hybrid_score"],
            "relative_mape_reduction_vs_persistence": validation["relative_mape_reduction"],
            "fold_target_horizon_wins": validation["fold_target_horizon_wins"],
            "fold_target_horizon_comparisons": validation["fold_target_horizon_comparisons"],
        },
        "smoke_test": {
            "rows": len(smoke_data),
            "finite_predictions": True,
            "generator_all_ge_generator_1": True,
        },
    }
    manifest_path = MODEL_DIR / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
