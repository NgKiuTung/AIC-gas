"""Test data processing optimizations: outlier detection, target transformation, label smoothing.

This script tests all 3 optimizations and compares to baseline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("DATA PROCESSING OPTIMIZATION TESTS")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

HORIZONS = tuple(range(1, 9))

PARAMS = {
    "n_estimators": 350,
    "max_depth": 5,
    "learning_rate": 0.025,
    "min_child_weight": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.70,
    "reg_alpha": 2.0,
    "reg_lambda": 30.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 20260803,
    "n_jobs": -1,
    "verbosity": 0,
}

BETA_1 = np.linspace(0.35, 0.30, 8, dtype=np.float32)
BETA_ALL = np.linspace(0.65, 0.60, 8, dtype=np.float32)

# ============================================================================
# Load Data
# ============================================================================

print("\n[1/6] Loading data...")

RAW_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")

tables = {
    "gas": pd.read_csv(RAW_DIR / "Pre_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(RAW_DIR / "Pre_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(RAW_DIR / "Pre_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(RAW_DIR / "Pre_load.csv", encoding="utf-8-sig"),
}

for df in tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

price_table = pd.read_excel(RAW_DIR / "price.xlsx")
price_lookup = {}
for row_idx, row in price_table.iterrows():
    for month in range(1, 13):
        price_lookup[(month, row_idx)] = float(row[f"{month}月"])

print(f"  Data loaded: {len(tables['gas'])} rows")

# ============================================================================
# Preprocess and build features
# ============================================================================

print("\n[2/6] Building base features...")

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame

causal, _ = preprocess_causal_raw_tables(tables, price_lookup, split="train")
base = build_inference_feature_frame(causal)

print(f"  Shape: {base.shape}")

# ============================================================================
# Optimization 1: Outlier Detection and Correction
# ============================================================================

print("\n[3/6] Applying Optimization 1: Outlier Detection...")

def detect_and_fix_outliers(df):
    """Detect and fix outliers."""
    df = df.copy()
    outliers_fixed = 0

    # Physical constraints
    power_cols = ['generator_1', 'generator_all']
    for col in power_cols:
        if col in df.columns:
            before = df[col].copy()
            if col == 'generator_1':
                df[col] = df[col].clip(0, 120)
            else:
                df[col] = df[col].clip(0, 200)
            outliers_fixed += (before != df[col]).sum()

    # Holder constraints
    if 'blast_furnace_gas_holder_2' in df.columns:
        before = df['blast_furnace_gas_holder_2'].copy()
        df['blast_furnace_gas_holder_2'] = df['blast_furnace_gas_holder_2'].clip(0, 100000)
        outliers_fixed += (before != df['blast_furnace_gas_holder_2']).sum()

    # Change rate outliers (for filled features)
    for col in ['feat_generator_1_filled', 'feat_generator_all_filled']:
        if col in df.columns:
            change_rate = df[col].pct_change(fill_method=None).abs()
            outliers = change_rate > 0.3

            if outliers.any():
                # Fix by interpolation
                df.loc[outliers, col] = np.nan
                df[col] = df[col].interpolate(method='linear', limit=4)
                outliers_fixed += outliers.sum()

    # Statistical outliers (IQR method)
    for col in ['blast_furnace_gas_holder_2', 'generator_use_blast_furnace_gas']:
        if col in df.columns:
            rolling = df[col].rolling(96, min_periods=24)
            Q1 = rolling.quantile(0.25)
            Q3 = rolling.quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR

            before = df[col].copy()
            df[col] = df[col].clip(lower.fillna(-np.inf), upper.fillna(np.inf))
            outliers_fixed += (before != df[col]).sum()

    print(f"  Fixed {outliers_fixed} outliers")
    return df

base_cleaned = detect_and_fix_outliers(base)

# ============================================================================
# Baseline model (fold 2 only for speed)
# ============================================================================

print("\n[4/6] Training baseline model (fold 2)...")

def train_fold2(features_df, use_target_transform=False, use_label_smoothing=False):
    """Train on fold 2 only for quick testing."""

    # Add labels
    df = features_df.copy()

    for h in HORIZONS:
        raw_label_p50 = df["feat_generator_1_filled"].shift(-h)
        raw_label_p120 = (df["feat_generator_all_filled"] - df["feat_generator_1_filled"]).shift(-h)

        # Optional: Label smoothing
        if use_label_smoothing:
            # Smooth labels with small window
            raw_label_p50 = raw_label_p50.rolling(3, center=True, min_periods=1).mean()
            raw_label_p120 = raw_label_p120.rolling(3, center=True, min_periods=1).mean()

        df[f"label_p50_h{h}"] = raw_label_p50
        df[f"label_p120_h{h}"] = raw_label_p120

    df = df.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])

    # Fold 2 split
    fold_start = pd.Timestamp("2025-04-01")
    fold_end = pd.Timestamp("2025-04-15 23:45:00")

    train = df[df["datetime"] < fold_start - pd.Timedelta(minutes=120)]
    valid = df[(df["datetime"] >= fold_start) & (df["datetime"] <= fold_end)]

    # Features
    feat_cols = [c for c in df.columns if c.startswith('feat_')]

    x_train = train[feat_cols].values.astype(np.float32)
    x_valid = valid[feat_cols].values.astype(np.float32)

    y1_train = train[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
    y120_train = train[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)
    y1_valid = valid[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
    y120_valid = valid[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)

    yall_train = y1_train + y120_train
    yall_valid = y1_valid + y120_valid

    current1_train = train["feat_p50_current"].values.astype(np.float32)[:, None]
    currentall_train = (train["feat_p50_current"].values + train["feat_p120_current"].values).astype(np.float32)[:, None]
    current1_valid = valid["feat_p50_current"].values.astype(np.float32)[:, None]
    currentall_valid = (valid["feat_p50_current"].values + valid["feat_p120_current"].values).astype(np.float32)[:, None]

    # Target transformation
    if use_target_transform:
        # Predict change instead of absolute value
        mixed_target = np.concatenate([
            y1_train - current1_train,  # Absolute change
            yall_train - currentall_train,
        ], axis=1)
    else:
        # Original: predict scaled change
        mixed_target = np.concatenate([
            (y1_train - current1_train) / np.maximum(np.abs(current1_train), 1e-6),
            yall_train - currentall_train,
        ], axis=1)

    # Train
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(x_train, mixed_target, verbose=False)

    # Predict
    raw = model.predict(x_valid)

    if use_target_transform:
        # Reverse transformation
        pred1 = np.maximum(current1_valid + raw[:, :8], 0.0)
        predall = np.maximum(currentall_valid + raw[:, 8:], 0.0)
    else:
        # Original prediction
        pred1 = np.maximum(current1_valid * (1.0 + raw[:, :8] * BETA_1), 0.0)
        predall = np.maximum(currentall_valid + raw[:, 8:] * BETA_ALL, 0.0)

    # MAPE
    mape1 = float(np.mean(np.abs(y1_valid - pred1) / np.maximum(np.abs(y1_valid), 1e-8)) * 100.0)
    mapeall = float(np.mean(np.abs(yall_valid - predall) / np.maximum(np.abs(yall_valid), 1e-8)) * 100.0)
    mape_avg = (mape1 + mapeall) / 2.0

    return mape_avg

# Baseline
baseline_mape = train_fold2(base)
print(f"  Baseline MAPE: {baseline_mape:.4f}%")

# ============================================================================
# Test each optimization
# ============================================================================

print("\n[5/6] Testing optimizations...")

# Test 1: Outlier detection only
opt1_mape = train_fold2(base_cleaned)
print(f"\n  Optimization 1 (Outlier Detection):")
print(f"    MAPE: {opt1_mape:.4f}%")
print(f"    vs Baseline: {baseline_mape - opt1_mape:+.4f}pp")

# Test 2: Target transformation only (on baseline data)
opt2_mape = train_fold2(base, use_target_transform=True)
print(f"\n  Optimization 2 (Target Transform):")
print(f"    MAPE: {opt2_mape:.4f}%")
print(f"    vs Baseline: {baseline_mape - opt2_mape:+.4f}pp")

# Test 3: Label smoothing only (on baseline data)
opt3_mape = train_fold2(base, use_label_smoothing=True)
print(f"\n  Optimization 3 (Label Smoothing):")
print(f"    MAPE: {opt3_mape:.4f}%")
print(f"    vs Baseline: {baseline_mape - opt3_mape:+.4f}pp")

# Test 4: All combined
opt_all_mape = train_fold2(base_cleaned, use_target_transform=True, use_label_smoothing=True)
print(f"\n  All Optimizations Combined:")
print(f"    MAPE: {opt_all_mape:.4f}%")
print(f"    vs Baseline: {baseline_mape - opt_all_mape:+.4f}pp")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("[6/6] SUMMARY")
print("="*80)

results = [
    ("Baseline", baseline_mape, 0.0),
    ("+ Outlier Detection", opt1_mape, baseline_mape - opt1_mape),
    ("+ Target Transform", opt2_mape, baseline_mape - opt2_mape),
    ("+ Label Smoothing", opt3_mape, baseline_mape - opt3_mape),
    ("+ All Combined", opt_all_mape, baseline_mape - opt_all_mape),
]

print("\nResults (Fold 2 only):")
print(f"{'Strategy':<30s} {'MAPE':<10s} {'Improvement':<12s}")
print("-" * 60)
for name, mape, improvement in results:
    print(f"{name:<30s} {mape:6.4f}%   {improvement:+.4f}pp")

# Find best
best_idx = np.argmin([r[1] for r in results])
best_name, best_mape, best_improvement = results[best_idx]

print(f"\nBest optimization: {best_name}")
print(f"  MAPE: {best_mape:.4f}%")
print(f"  Improvement: {best_improvement:+.4f}pp")

if best_improvement > 0.01:
    print(f"\n✅ SUCCESS! Found {best_improvement:.4f}pp improvement!")
    print(f"\nRecommendation: Implement '{best_name}' in full 3-fold training")
elif best_improvement > 0:
    print(f"\n⚠️ Marginal improvement ({best_improvement:.4f}pp)")
    print(f"\nRecommendation: May not be worth the complexity")
else:
    print(f"\n❌ No improvement found")
    print(f"\nRecommendation: Try other optimization directions (model ensemble, hyperparameter tuning)")

print("\n" + "="*80)
print("TESTING COMPLETE!")
print("="*80)
