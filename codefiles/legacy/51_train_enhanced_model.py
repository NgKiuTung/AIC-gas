"""Train final model with enhanced features (future prices + interactions).

This script:
1. Loads/builds enhanced features (903 features)
2. Trains XGBoost model with adjusted hyperparameters
3. Evaluates on 3-fold cross-validation
4. Saves final model and manifest

Expected MAPE: 5.10% - 5.30% (vs 5.504% baseline)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

# Paths
INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_enhanced.pkl"
CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog_enhanced.csv"
MODEL_DIR = ROOT / "results" / "models" / "power_forecaster_enhanced"
RESULT_DIR = ROOT / "results" / "experiments" / "enhanced_full_pipeline"

HORIZONS = tuple(range(1, 9))

# Adjusted hyperparameters for more features (903 vs 869)
PARAMETERS = {
    "n_estimators": 400,         # Increased from 350
    "max_depth": 5,               # Keep same
    "learning_rate": 0.025,       # Keep same
    "min_child_weight": 30,       # Keep same
    "subsample": 0.85,            # Keep same
    "colsample_bytree": 0.65,     # Decreased from 0.70 (more features)
    "reg_alpha": 2.0,             # Keep same
    "reg_lambda": 30.0,           # Keep same
}

# Beta parameters (will tune if needed)
BETA_GENERATOR_1 = np.linspace(0.35, 0.30, 8, dtype=np.float32)
BETA_GENERATOR_ALL = np.linspace(0.65, 0.60, 8, dtype=np.float32)

# Cross-validation folds
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)


def sha256(path: Path) -> str:
    """Calculate SHA256 hash."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE."""
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) * 100.0)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Training Enhanced Model (Future Prices + Interactions)")
    print("=" * 80)

    # Load data
    print("\nLoading enhanced features...")
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    features = catalog["feature"].tolist()

    print(f"  Rows: {len(data):,}")
    print(f"  Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"  Features: {len(features)}")

    # Feature breakdown
    future_price_feats = [f for f in features if "future_price" in f]
    interaction_feats = [f for f in features if "interact" in f]
    base_feats = len(features) - len(future_price_feats) - len(interaction_feats)

    print(f"\n  Feature breakdown:")
    print(f"    - Base features: {base_feats}")
    print(f"    - Future price features: {len(future_price_feats)}")
    print(f"    - Interaction features: {len(interaction_feats)}")

    # Cross-validation
    print("\n" + "=" * 80)
    print("3-Fold Cross-Validation")
    print("=" * 80)

    fold_results = []
    total_train_time = 0

    for fold_name, start_text, end_text in FOLDS:
        print(f"\n[{fold_name}] Training...")

        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)

        # Split data
        train = data[data["datetime"] < start - pd.Timedelta(minutes=120)]
        valid = data[(data["datetime"] >= start) & (data["datetime"] <= end)]

        print(f"  Train: {len(train):,} rows, Valid: {len(valid):,} rows")

        # Prepare features and targets
        x_train = train[features].to_numpy(dtype=np.float32)
        x_valid = valid[features].to_numpy(dtype=np.float32)

        y1_train = train[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
        y120_train = train[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
        y1_valid = valid[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
        y120_valid = valid[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)

        yall_train = y1_train + y120_train
        yall_valid = y1_valid + y120_valid

        current1_train = train["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        currentall_train = (train["feat_p50_current"] + train["feat_p120_current"]).to_numpy(dtype=np.float32)[:, None]
        current1_valid = valid["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
        currentall_valid = (valid["feat_p50_current"] + valid["feat_p120_current"]).to_numpy(dtype=np.float32)[:, None]

        # Mixed target
        mixed_target = np.concatenate([
            (y1_train - current1_train) / np.maximum(np.abs(current1_train), 1e-6),
            yall_train - currentall_train,
        ], axis=1)

        # Train model
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

        t0 = time.perf_counter()
        model.fit(x_train, mixed_target, verbose=False)
        raw = model.predict(x_valid)
        train_time = time.perf_counter() - t0
        total_train_time += train_time

        # Predictions
        pred1 = np.maximum(current1_valid * (1.0 + raw[:, :8] * BETA_GENERATOR_1), 0.0)
        predall = np.maximum(currentall_valid + raw[:, 8:] * BETA_GENERATOR_ALL, 0.0)

        # Evaluate
        mape1 = mean_absolute_percentage_error(y1_valid, pred1)
        mapeall = mean_absolute_percentage_error(yall_valid, predall)
        mape_fold = (mape1 + mapeall) / 2.0

        print(f"  MAPE: {mape_fold:.4f}% (P50={mape1:.4f}%, All={mapeall:.4f}%)")
        print(f"  Training time: {train_time:.1f}s")

        fold_results.append({
            "fold": fold_name,
            "mape": float(mape_fold),
            "mape_p50": float(mape1),
            "mape_all": float(mapeall),
            "train_time": float(train_time),
        })

    # Overall results
    mean_mape = np.mean([r["mape"] for r in fold_results])
    mean_mape_p50 = np.mean([r["mape_p50"] for r in fold_results])
    mean_mape_all = np.mean([r["mape_all"] for r in fold_results])

    print("\n" + "=" * 80)
    print("Overall Results")
    print("=" * 80)
    print(f"\nMean MAPE: {mean_mape:.4f}%")
    print(f"  Generator 1: {mean_mape_p50:.4f}%")
    print(f"  Generator All: {mean_mape_all:.4f}%")
    print(f"\nScore: {1.0 - mean_mape/100:.6f}")
    print(f"Total training time: {total_train_time:.1f}s")

    # Compare with baseline
    baseline_mape = 5.504  # cleaning_enhanced result
    improvement = baseline_mape - mean_mape
    improvement_pct = (improvement / baseline_mape) * 100

    print("\n" + "=" * 80)
    print("Comparison with Baseline")
    print("=" * 80)
    print(f"Baseline (cleaning_enhanced): {baseline_mape:.4f}%")
    print(f"Enhanced (this model):         {mean_mape:.4f}%")
    print(f"Improvement:                   {improvement:+.4f} pp ({improvement_pct:+.2f}%)")

    if mean_mape < baseline_mape:
        print(f"\nRESULT: SUCCESS! Enhanced features improved MAPE")
    else:
        print(f"\nRESULT: No improvement. Consider hyperparameter tuning.")

    # Train final model on all data
    print("\n" + "=" * 80)
    print("Training Final Model on All Data")
    print("=" * 80)

    x_all = data[features].to_numpy(dtype=np.float32)
    y1_all = data[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y120_all = data[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    yall_all = y1_all + y120_all
    current1_all = data["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
    currentall_all = (data["feat_p50_current"] + data["feat_p120_current"]).to_numpy(dtype=np.float32)[:, None]

    mixed_all = np.concatenate([
        (y1_all - current1_all) / np.maximum(np.abs(current1_all), 1e-6),
        yall_all - currentall_all,
    ], axis=1)

    final_model = xgb.XGBRegressor(
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

    print("Training...")
    t0 = time.perf_counter()
    final_model.fit(x_all, mixed_all, verbose=False)
    final_train_time = time.perf_counter() - t0
    print(f"  Time: {final_train_time:.1f}s")

    # Save model
    model_path = MODEL_DIR / "xgboost_enhanced_forecaster.json"
    final_model.save_model(model_path)
    print(f"  Model saved: {model_path.name}")

    # Save feature schema and importance
    feature_schema = catalog.copy()
    feature_schema["position"] = np.arange(len(feature_schema))
    feature_schema.to_csv(MODEL_DIR / "feature_schema.csv", index=False, encoding="utf-8-sig")

    importance = pd.DataFrame({
        "feature": features,
        "importance": final_model.feature_importances_.astype(float)
    }).sort_values("importance", ascending=False)
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # Top features
    print("\n  Top 20 most important features:")
    for idx, row in importance.head(20).iterrows():
        feat_type = ""
        if "future_price" in row["feature"]:
            feat_type = "[FUTURE_PRICE]"
        elif "interact" in row["feature"]:
            feat_type = "[INTERACTION]"
        print(f"    {row['importance']:8.1f}  {feat_type:16s} {row['feature']}")

    # Count new feature usage
    top50 = importance.head(50)
    future_in_top50 = len([f for f in top50["feature"] if "future_price" in f])
    interact_in_top50 = len([f for f in top50["feature"] if "interact" in f])

    print(f"\n  New features in top 50:")
    print(f"    Future price: {future_in_top50}")
    print(f"    Interactions: {interact_in_top50}")
    print(f"    Total new:    {future_in_top50 + interact_in_top50}")

    # Save manifest
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "enhanced_features",
        "xgboost_version": xgb.__version__,
        "model_file": model_path.name,
        "model_sha256": sha256(model_path),
        "training_rows": len(data),
        "features": {
            "total": len(features),
            "base": base_feats,
            "future_price": len(future_price_feats),
            "interaction": len(interaction_feats),
        },
        "parameters": PARAMETERS,
        "beta": {
            "generator_1": BETA_GENERATOR_1.tolist(),
            "generator_all": BETA_GENERATOR_ALL.tolist(),
        },
        "cross_validation": {
            "folds": 3,
            "mean_mape": float(mean_mape),
            "fold_results": fold_results,
        },
        "baseline_comparison": {
            "baseline_mape": baseline_mape,
            "enhanced_mape": float(mean_mape),
            "improvement_pp": float(improvement),
            "improvement_pct": float(improvement_pct),
        },
        "feature_importance": {
            "future_price_in_top50": int(future_in_top50),
            "interaction_in_top50": int(interact_in_top50),
        },
        "training_time": {
            "cv_total_seconds": float(total_train_time),
            "final_model_seconds": float(final_train_time),
        },
    }

    manifest_path = MODEL_DIR / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  Manifest saved: {manifest_path.name}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
