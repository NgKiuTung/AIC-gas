"""Train model with selected features: 851 base + 20 new = 871 features.

This script uses only the most important features identified by importance analysis.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("TRAINING WITH SELECTED FEATURES (871 total)")
print("="*80)

# ============================================================================
# Load selected new features
# ============================================================================

selected_new_features_file = ROOT / "selected_new_features.txt"
with open(selected_new_features_file, 'r') as f:
    selected_new_features = [line.strip() for line in f if line.strip()]

print(f"\nSelected new features: {len(selected_new_features)}")
for feat in selected_new_features:
    print(f"  - {feat}")

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
    "max_bin": 256,
    "multi_strategy": "one_output_per_tree",
    "verbosity": 0,
}

BETA_1 = np.linspace(0.35, 0.30, 8, dtype=np.float32)
BETA_ALL = np.linspace(0.65, 0.60, 8, dtype=np.float32)

# ============================================================================
# Load Data
# ============================================================================

print("\n" + "="*80)
print("[1/5] Loading data...")
print("="*80)

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

print(f"Data loaded: {len(tables['gas'])} rows")

# ============================================================================
# Build Features
# ============================================================================

print("\n" + "="*80)
print("[2/5] Building features...")
print("="*80)

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame
from gas_power.features.enhanced_interactions import add_future_price_features
from gas_power.features.optimized_features import (
    add_target_lag_features,
    add_domain_knowledge_features,
    add_physical_constraint_features,
    add_adaptive_time_features
)

# Preprocess
print("  [2a] Preprocessing...")
t0 = time.perf_counter()
causal, _ = preprocess_causal_raw_tables(tables, price_lookup, split="train")
print(f"    Time: {time.perf_counter()-t0:.1f}s, Shape: {causal.shape}")

# Base features
print("  [2b] Base features...")
t0 = time.perf_counter()
base = build_inference_feature_frame(causal)
print(f"    Time: {time.perf_counter()-t0:.1f}s, Shape: {base.shape}")

# Add only selected new features
print("  [2c] Adding selected new features...")
t0 = time.perf_counter()

# Add future prices (we need these to extract selected ones)
with_prices = add_future_price_features(base, price_lookup)

# Add optimized features (we need these to extract selected ones)
with_optimized = add_target_lag_features(with_prices)
with_optimized = add_domain_knowledge_features(with_optimized)
with_optimized = add_physical_constraint_features(with_optimized)
with_optimized = add_adaptive_time_features(with_optimized)

print(f"    Time: {time.perf_counter()-t0:.1f}s, Shape: {with_optimized.shape}")

# Filter to keep only base + selected new features
base_feat_cols = [c for c in base.columns if c.startswith('feat_')]
all_feat_cols = base_feat_cols + selected_new_features

# Verify all selected features exist
missing_features = [f for f in selected_new_features if f not in with_optimized.columns]
if missing_features:
    print(f"\nWARNING: {len(missing_features)} selected features not found:")
    for f in missing_features:
        print(f"  - {f}")
    # Remove missing features
    selected_new_features = [f for f in selected_new_features if f in with_optimized.columns]
    all_feat_cols = base_feat_cols + selected_new_features

print("\nFinal feature count:")
print(f"  Base features: {len(base_feat_cols)}")
print(f"  Selected new features: {len(selected_new_features)}")
print(f"  Total: {len(all_feat_cols)}")

# Create final dataframe with only selected features
required_cols = ['datetime'] + all_feat_cols + [
    'feat_generator_1_filled', 'feat_generator_all_filled',
    'feat_p50_current', 'feat_p120_current'
]
features = with_optimized[required_cols].copy()

# Add labels
for h in HORIZONS:
    features[f"label_p50_h{h}"] = with_optimized["feat_generator_1_filled"].shift(-h)
    features[f"label_p120_h{h}"] = (
        with_optimized["feat_generator_all_filled"] - with_optimized["feat_generator_1_filled"]
    ).shift(-h)

features = features.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])
print(f"Final dataset shape: {features.shape}")

# ============================================================================
# Define Folds
# ============================================================================

print("\n" + "="*80)
print("[3/5] Defining folds...")
print("="*80)

folds = [
    ("fold_1", pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-15 23:45:00")),
    ("fold_2", pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-15 23:45:00")),
    ("fold_3", pd.Timestamp("2025-04-16"), pd.Timestamp("2025-04-30 23:45:00")),
]

print(f"Number of folds: {len(folds)}")
for fold_name, start, end in folds:
    print(f"  {fold_name}: {start.date()} to {end.date()}")

# ============================================================================
# Training
# ============================================================================

print("\n" + "="*80)
print("[4/5] Training models...")
print("="*80)

def train_and_evaluate(fold_name, fold_start, fold_end):
    """Train and evaluate on one fold."""

    # Split data
    train = features[features["datetime"] < fold_start - pd.Timedelta(minutes=120)]
    valid = features[(features["datetime"] >= fold_start) & (features["datetime"] <= fold_end)]

    print(f"\n  {fold_name}:")
    print(f"    Train: {len(train)}, Valid: {len(valid)}")

    # Prepare data
    x_train = train[all_feat_cols].to_numpy(dtype=np.float32)
    x_valid = valid[all_feat_cols].to_numpy(dtype=np.float32)

    y1_train = train[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y120_train = train[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y1_valid = valid[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y120_valid = valid[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)

    yall_train = y1_train + y120_train
    yall_valid = y1_valid + y120_valid

    current1_train = train["feat_p50_current"].values[:, None]
    currentall_train = (train["feat_p50_current"] + train["feat_p120_current"]).values[:, None]
    current1_valid = valid["feat_p50_current"].values[:, None]
    currentall_valid = (valid["feat_p50_current"] + valid["feat_p120_current"]).values[:, None]

    # Mixed target
    mixed_target = np.concatenate([
        (y1_train - current1_train) / np.maximum(np.abs(current1_train), 1e-6),
        yall_train - currentall_train,
    ], axis=1)

    # Train
    t0 = time.perf_counter()
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(x_train, mixed_target, verbose=False)

    # Predict
    raw = model.predict(x_valid)
    pred1 = np.maximum(current1_valid * (1.0 + raw[:, :8] * BETA_1), 0.0)
    predall = np.maximum(currentall_valid + raw[:, 8:] * BETA_ALL, 0.0)

    train_time = time.perf_counter() - t0

    # Calculate MAPE
    mape1 = float(np.mean(np.abs(y1_valid - pred1) / np.maximum(np.abs(y1_valid), 1e-8)) * 100.0)
    mapeall = float(np.mean(np.abs(yall_valid - predall) / np.maximum(np.abs(yall_valid), 1e-8)) * 100.0)
    mape_avg = (mape1 + mapeall) / 2.0

    print(f"    MAPE: {mape_avg:.4f}% (P50={mape1:.4f}%, All={mapeall:.4f}%)")
    print(f"    Time: {train_time:.1f}s")

    return mape_avg

# Train on all folds
fold_results = []
for fold_name, fold_start, fold_end in folds:
    mape = train_and_evaluate(fold_name, fold_start, fold_end)
    fold_results.append({'fold': fold_name, 'mape': mape})

# ============================================================================
# Results Summary
# ============================================================================

print("\n" + "="*80)
print("[5/5] RESULTS SUMMARY")
print("="*80)

avg_mape = np.mean([r['mape'] for r in fold_results])
std_mape = np.std([r['mape'] for r in fold_results])

print("\nFold Results:")
for r in fold_results:
    print(f"  {r['fold']}: {r['mape']:.4f}%")

print(f"\n3-Fold Average MAPE: {avg_mape:.4f}% (±{std_mape:.4f}%)")

# Compare to baseline
baseline_mape = 5.504
improvement = baseline_mape - avg_mape

print("\nComparison to Baseline (cleaning_enhanced):")
print(f"  Baseline MAPE: {baseline_mape:.4f}%")
print(f"  Selected Features MAPE: {avg_mape:.4f}%")
print(f"  Improvement: {improvement:+.4f}pp ({improvement/baseline_mape*100:+.2f}% relative)")

if improvement > 0.01:
    print(f"\n✅ SUCCESS! Model improved by {improvement:.4f} percentage points!")
elif improvement > -0.01:
    print(f"\n⚠️ Marginal change ({improvement:.4f}pp). Performance similar to baseline.")
else:
    print(f"\n❌ Performance degraded by {-improvement:.4f}pp. Consider adjusting strategy.")

print("\nFeature Summary:")
print(f"  Base features: {len(base_feat_cols)}")
print(f"  New features: {len(selected_new_features)}")
print(f"  Total: {len(all_feat_cols)}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
