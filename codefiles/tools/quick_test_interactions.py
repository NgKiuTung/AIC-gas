"""Quick test: Add interaction features to cleaning_enhanced and evaluate.

This script:
1. Loads existing cleaning_enhanced features (869 features)
2. Adds 22 interaction features
3. Trains model with same parameters
4. Compares MAPE before/after

Expected: MAPE improvement from 5.504% to ~5.35-5.45%
"""

from __future__ import annotations

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

# Import interaction features module
from gas_power.features.enhanced_interactions import add_interaction_features

# Paths
INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_cleaning_enhanced.pkl"
CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog_cleaning_enhanced.csv"
OUTPUT_DIR = ROOT / "results" / "experiments" / "quick_interaction_test"
HORIZONS = tuple(range(1, 9))

# Use same folds as cleaning_enhanced
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)

# Same parameters as cleaning_enhanced
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

# Best beta from cleaning_enhanced
BEST_BETA = {
    "generator_1": {"h15": 0.4, "h120": 0.4},
    "generator_all": {"h15": 0.7, "h120": 0.6},
}


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE."""
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) * 100.0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Quick Interaction Features Test")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")

    print(f"  Rows: {len(data):,}")
    print(f"  Original features: {len(catalog)}")

    # Add interaction features
    print("\nAdding interaction features...")
    start_time = time.perf_counter()
    data_enhanced = add_interaction_features(data)
    elapsed = time.perf_counter() - start_time

    # Identify new features
    new_features = [col for col in data_enhanced.columns if col not in data.columns]
    print(f"  Added {len(new_features)} interaction features")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Total features: {len(catalog) + len(new_features)}")

    # List new features
    print("\n  New interaction features:")
    for feat in sorted(new_features)[:10]:
        print(f"    - {feat}")
    if len(new_features) > 10:
        print(f"    ... and {len(new_features) - 10} more")

    # Prepare feature lists
    original_features = catalog["feature"].tolist()
    enhanced_features = original_features + new_features

    # Cross-validation
    print("\n" + "=" * 80)
    print("Running 3-fold cross-validation")
    print("=" * 80)

    results_original = []
    results_enhanced = []

    for fold_idx, (fold_name, start_text, end_text) in enumerate(FOLDS, 1):
        print(f"\n[{fold_name}] Processing...")

        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)

        # Split data
        train = data_enhanced[data_enhanced["datetime"] < start - pd.Timedelta(minutes=120)]
        valid = data_enhanced[(data_enhanced["datetime"] >= start) & (data_enhanced["datetime"] <= end)]

        print(f"  Train: {len(train):,} rows, Valid: {len(valid):,} rows")

        # Prepare targets
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

        # Test 1: Original features
        print(f"  Training with ORIGINAL features ({len(original_features)})...")
        x_train_orig = train[original_features].to_numpy(dtype=np.float32)
        x_valid_orig = valid[original_features].to_numpy(dtype=np.float32)

        model_orig = xgb.XGBRegressor(
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
        model_orig.fit(x_train_orig, mixed_target, verbose=False)
        raw_orig = model_orig.predict(x_valid_orig)
        time_orig = time.perf_counter() - t0

        # Predictions with original
        beta_1 = np.linspace(BEST_BETA["generator_1"]["h15"], BEST_BETA["generator_1"]["h120"], 8)
        beta_all = np.linspace(BEST_BETA["generator_all"]["h15"], BEST_BETA["generator_all"]["h120"], 8)

        pred1_orig = np.maximum(current1_valid * (1.0 + raw_orig[:, :8] * beta_1), 0.0)
        predall_orig = np.maximum(currentall_valid + raw_orig[:, 8:] * beta_all, 0.0)

        mape1_orig = mean_absolute_percentage_error(y1_valid, pred1_orig)
        mapeall_orig = mean_absolute_percentage_error(yall_valid, predall_orig)
        mape_orig = (mape1_orig + mapeall_orig) / 2.0

        print(f"    MAPE: {mape_orig:.4f}% (P50={mape1_orig:.4f}%, All={mapeall_orig:.4f}%)")
        print(f"    Time: {time_orig:.1f}s")

        # Test 2: Enhanced features
        print(f"  Training with ENHANCED features ({len(enhanced_features)})...")
        x_train_enh = train[enhanced_features].to_numpy(dtype=np.float32)
        x_valid_enh = valid[enhanced_features].to_numpy(dtype=np.float32)

        model_enh = xgb.XGBRegressor(
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
        model_enh.fit(x_train_enh, mixed_target, verbose=False)
        raw_enh = model_enh.predict(x_valid_enh)
        time_enh = time.perf_counter() - t0

        # Predictions with enhanced
        pred1_enh = np.maximum(current1_valid * (1.0 + raw_enh[:, :8] * beta_1), 0.0)
        predall_enh = np.maximum(currentall_valid + raw_enh[:, 8:] * beta_all, 0.0)

        mape1_enh = mean_absolute_percentage_error(y1_valid, pred1_enh)
        mapeall_enh = mean_absolute_percentage_error(yall_valid, predall_enh)
        mape_enh = (mape1_enh + mapeall_enh) / 2.0

        print(f"    MAPE: {mape_enh:.4f}% (P50={mape1_enh:.4f}%, All={mapeall_enh:.4f}%)")
        print(f"    Time: {time_enh:.1f}s")

        # Comparison
        improvement = ((mape_orig - mape_enh) / mape_orig) * 100
        delta = mape_orig - mape_enh

        print(f"\n  Improvement: {delta:+.4f} percentage points ({improvement:+.2f}% relative)")

        if mape_enh < mape_orig:
            print("  Result: BETTER with interactions!")
        elif mape_enh > mape_orig:
            print("  Result: Worse (may need hyperparameter tuning)")
        else:
            print("  Result: No change")

        results_original.append({
            "fold": fold_name,
            "mape": mape_orig,
            "mape_p50": mape1_orig,
            "mape_all": mapeall_orig,
        })

        results_enhanced.append({
            "fold": fold_name,
            "mape": mape_enh,
            "mape_p50": mape1_enh,
            "mape_all": mapeall_enh,
        })

    # Overall results
    print("\n" + "=" * 80)
    print("Overall Results")
    print("=" * 80)

    mean_mape_orig = np.mean([r["mape"] for r in results_original])
    mean_mape_enh = np.mean([r["mape"] for r in results_enhanced])

    print(f"\nOriginal features ({len(original_features)}):")
    print(f"  Mean MAPE: {mean_mape_orig:.4f}%")
    for r in results_original:
        print(f"    {r['fold']}: {r['mape']:.4f}%")

    print(f"\nEnhanced features ({len(enhanced_features)}):")
    print(f"  Mean MAPE: {mean_mape_enh:.4f}%")
    for r in results_enhanced:
        print(f"    {r['fold']}: {r['mape']:.4f}%")

    improvement = ((mean_mape_orig - mean_mape_enh) / mean_mape_orig) * 100
    delta = mean_mape_orig - mean_mape_enh

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Original MAPE:  {mean_mape_orig:.4f}%")
    print(f"Enhanced MAPE:  {mean_mape_enh:.4f}%")
    print(f"Improvement:    {delta:+.4f} pp ({improvement:+.2f}% relative)")
    print(f"Score change:   {-delta/100:+.6f}")

    if mean_mape_enh < mean_mape_orig:
        print("\nResult: SUCCESS! Interaction features improved MAPE")
        print("Recommendation: Use enhanced features in production")
    else:
        print("\nResult: No improvement yet")
        print("Recommendation: Try tuning hyperparameters (colsample_bytree, n_estimators)")

    # Save results
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_type": "quick_interaction_features",
        "original_features": len(original_features),
        "interaction_features": len(new_features),
        "total_features": len(enhanced_features),
        "original_mape": float(mean_mape_orig),
        "enhanced_mape": float(mean_mape_enh),
        "improvement_pp": float(delta),
        "improvement_pct": float(improvement),
        "fold_results_original": results_original,
        "fold_results_enhanced": results_enhanced,
        "new_features_list": new_features,
    }

    output_path = OUTPUT_DIR / "quick_test_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
