"""Test multiscale temporal features impact on model performance.

Compares baseline (851 features) vs baseline + temporal features.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("TESTING MULTISCALE TEMPORAL FEATURES")
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

print("\n[1/5] Loading data...")

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
# Build Base Features
# ============================================================================

print("\n[2/5] Building base features...")

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame

causal, _ = preprocess_causal_raw_tables(tables, price_lookup, split="train")
base = build_inference_feature_frame(causal)

base_feat_cols = [c for c in base.columns if c.startswith('feat_')]
print(f"  Base features: {len(base_feat_cols)}")

# ============================================================================
# Add Multiscale Temporal Features
# ============================================================================

print("\n[3/5] Adding multiscale temporal features...")

from gas_power.features.multiscale_temporal_features import add_all_multiscale_temporal_features

t0 = time.perf_counter()
with_temporal = add_all_multiscale_temporal_features(
    base,
    add_rolling=True,
    add_ewma=True,
    add_trend=False,  # Skip expensive features for quick test
    add_fourier=True,
    add_autocorr=False,  # Skip expensive features
    add_changepoint=True
)
elapsed = time.perf_counter() - t0

temporal_feat_cols = [c for c in with_temporal.columns if c.startswith('feat_')]
new_temporal_features = [c for c in temporal_feat_cols if c not in base_feat_cols]

print(f"  Time: {elapsed:.1f}s")
print(f"  New temporal features: {len(new_temporal_features)}")
print(f"  Total features: {len(temporal_feat_cols)}")

# ============================================================================
# Test on Fold 2
# ============================================================================

print("\n[4/5] Training and comparing models (Fold 2 only)...")

def train_fold2(features_df, feat_cols):
    """Train on fold 2 and return MAPE."""

    # Add labels
    df = features_df.copy()

    for h in HORIZONS:
        # Use label smoothing
        raw_label_p50 = df["feat_generator_1_filled"].shift(-h)
        raw_label_p120 = (df["feat_generator_all_filled"] - df["feat_generator_1_filled"]).shift(-h)

        df[f"label_p50_h{h}"] = raw_label_p50.rolling(3, center=True, min_periods=1).mean()
        df[f"label_p120_h{h}"] = raw_label_p120.rolling(3, center=True, min_periods=1).mean()

    df = df.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])

    # Fold 2 split
    fold_start = pd.Timestamp("2025-04-01")
    fold_end = pd.Timestamp("2025-04-15 23:45:00")

    train = df[df["datetime"] < fold_start - pd.Timedelta(minutes=120)]
    valid = df[(df["datetime"] >= fold_start) & (df["datetime"] <= fold_end)]

    # Extract features
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

    # Target: predict absolute change
    mixed_target = np.concatenate([
        y1_train - current1_train,
        yall_train - currentall_train,
    ], axis=1)

    # Train
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(x_train, mixed_target, verbose=False)

    # Predict
    raw = model.predict(x_valid)
    pred1 = np.maximum(current1_valid + raw[:, :8], 0.0)
    predall = np.maximum(currentall_valid + raw[:, 8:], 0.0)

    # MAPE
    mape1 = float(np.mean(np.abs(y1_valid - pred1) / np.maximum(np.abs(y1_valid), 1e-8)) * 100.0)
    mapeall = float(np.mean(np.abs(yall_valid - predall) / np.maximum(np.abs(yall_valid), 1e-8)) * 100.0)
    mape_avg = (mape1 + mapeall) / 2.0

    return mape_avg

# Baseline
print("\n  Training baseline model (851 features)...")
baseline_mape = train_fold2(base, base_feat_cols)
print(f"    Baseline MAPE: {baseline_mape:.4f}%")

# With temporal features
print("\n  Training with temporal features ({} features)...".format(len(temporal_feat_cols)))
temporal_mape = train_fold2(with_temporal, temporal_feat_cols)
print(f"    Temporal MAPE: {temporal_mape:.4f}%")

# ============================================================================
# Results Summary
# ============================================================================

print("\n" + "="*80)
print("[5/5] RESULTS SUMMARY")
print("="*80)

improvement = baseline_mape - temporal_mape

print("\nFold 2 Results:")
print(f"  Baseline (851 features):     {baseline_mape:.4f}%")
print(f"  + Temporal ({len(temporal_feat_cols)} features): {temporal_mape:.4f}%")
print(f"  Improvement:                 {improvement:+.4f}pp")

if improvement > 0.05:
    print(f"\n✅ SUCCESS! Temporal features improved by {improvement:.4f}pp")
    print("\nRecommendation: Use temporal features in final model")
    print(f"  - New features added: {len(new_temporal_features)}")
    print(f"  - Computation time: {elapsed:.1f}s")
elif improvement > 0:
    print(f"\n⚠️ Marginal improvement ({improvement:.4f}pp)")
    print(f"\nRecommendation: Consider if {len(new_temporal_features)} extra features are worth {elapsed:.1f}s computation")
else:
    print(f"\n❌ No improvement ({improvement:.4f}pp)")
    print("\nRecommendation: Temporal features may not help, or need different configuration")

print("\n" + "="*80)
print("TESTING COMPLETE!")
print("="*80)
