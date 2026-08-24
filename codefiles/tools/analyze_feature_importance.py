"""Feature importance analysis and selection.

This script:
1. Trains a model on current features
2. Analyzes feature importance
3. Identifies low-importance features to remove
4. Retrains with selected features
5. Compares MAPE
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# ============================================================================
# Load and prepare data
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

# Preprocess
from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame

causal, _ = preprocess_causal_raw_tables(tables, price_lookup, split="train")
base = build_inference_feature_frame(causal)

print(f"  Features: {base.shape}")

# Add labels
HORIZONS = tuple(range(1, 9))
for h in HORIZONS:
    base[f"label_p50_h{h}"] = base["feat_generator_1_filled"].shift(-h)
    base[f"label_p120_h{h}"] = (
        base["feat_generator_all_filled"] - base["feat_generator_1_filled"]
    ).shift(-h)

base = base.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])

# Get feature columns
feature_cols = [col for col in base.columns if col.startswith("feat_")]
print(f"  Trainable features: {len(feature_cols)}")

# ============================================================================
# Train model and get feature importance
# ============================================================================

print("\n[2/5] Training model to analyze feature importance...")

# Use fold 2 for analysis
fold_start = pd.Timestamp("2025-04-01")
fold_end = pd.Timestamp("2025-04-15 23:45:00")

train = base[base["datetime"] < fold_start - pd.Timedelta(minutes=120)]
valid = base[(base["datetime"] >= fold_start) & (base["datetime"] <= fold_end)]

print(f"  Train: {len(train)}, Valid: {len(valid)}")

# Prepare data
x_train = train[feature_cols].to_numpy(dtype=np.float32)
x_valid = valid[feature_cols].to_numpy(dtype=np.float32)

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

BETA_1 = np.linspace(0.35, 0.30, 8, dtype=np.float32)
BETA_ALL = np.linspace(0.65, 0.60, 8, dtype=np.float32)

# Mixed target
mixed_target = np.concatenate([
    (y1_train - current1_train) / np.maximum(np.abs(current1_train), 1e-6),
    yall_train - currentall_train,
], axis=1)

# Train
params = {
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

model = xgb.XGBRegressor(**params)
model.fit(x_train, mixed_target, verbose=False)

# Predict
raw = model.predict(x_valid)
pred1 = np.maximum(current1_valid * (1.0 + raw[:, :8] * BETA_1), 0.0)
predall = np.maximum(currentall_valid + raw[:, 8:] * BETA_ALL, 0.0)

# Calculate MAPE
mape1 = float(np.mean(np.abs(y1_valid - pred1) / np.maximum(np.abs(y1_valid), 1e-8)) * 100.0)
mapeall = float(np.mean(np.abs(yall_valid - predall) / np.maximum(np.abs(yall_valid), 1e-8)) * 100.0)
baseline_mape = (mape1 + mapeall) / 2.0

print(f"  Baseline MAPE: {baseline_mape:.4f}% (P50={mape1:.4f}%, All={mapeall:.4f}%)")

# ============================================================================
# Analyze feature importance
# ============================================================================

print("\n[3/5] Analyzing feature importance...")

# Get importance scores
importance_dict = model.get_booster().get_score(importance_type='gain')

# Map to feature names
feature_importance = {}
for i, feat_name in enumerate(feature_cols):
    xgb_name = f'f{i}'
    feature_importance[feat_name] = importance_dict.get(xgb_name, 0.0)

# Sort by importance
sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])

print(f"\nTop 20 most important features:")
for i, (feat, score) in enumerate(sorted_features[:20], 1):
    print(f"  {i:2d}. {feat:50s} {score:10.1f}")

# Count zero-importance features
zero_importance = [f for f, s in feature_importance.items() if s == 0.0]
low_importance = [f for f, s in feature_importance.items() if s > 0.0 and s < np.percentile([s for s in feature_importance.values() if s > 0], 10)]

print(f"\nFeature importance summary:")
print(f"  Total features: {len(feature_cols)}")
print(f"  Zero importance: {len(zero_importance)} ({len(zero_importance)/len(feature_cols)*100:.1f}%)")
print(f"  Low importance (<10th percentile): {len(low_importance)} ({len(low_importance)/len(feature_cols)*100:.1f}%)")
print(f"  Can potentially remove: {len(zero_importance) + len(low_importance)}")

# ============================================================================
# Feature selection strategies
# ============================================================================

print("\n[4/5] Testing feature selection strategies...")

def test_feature_set(features_to_use, strategy_name):
    """Train and evaluate with selected features."""
    x_train_sel = train[features_to_use].to_numpy(dtype=np.float32)
    x_valid_sel = valid[features_to_use].to_numpy(dtype=np.float32)

    model_sel = xgb.XGBRegressor(**params)
    model_sel.fit(x_train_sel, mixed_target, verbose=False)

    raw = model_sel.predict(x_valid_sel)
    pred1 = np.maximum(current1_valid * (1.0 + raw[:, :8] * BETA_1), 0.0)
    predall = np.maximum(currentall_valid + raw[:, 8:] * BETA_ALL, 0.0)

    mape1 = float(np.mean(np.abs(y1_valid - pred1) / np.maximum(np.abs(y1_valid), 1e-8)) * 100.0)
    mapeall = float(np.mean(np.abs(yall_valid - predall) / np.maximum(np.abs(yall_valid), 1e-8)) * 100.0)
    mape = (mape1 + mapeall) / 2.0

    improvement = baseline_mape - mape

    print(f"\n  {strategy_name}:")
    print(f"    Features: {len(features_to_use)} (removed {len(feature_cols) - len(features_to_use)})")
    print(f"    MAPE: {mape:.4f}%")
    print(f"    vs Baseline: {improvement:+.4f}pp")

    return mape, features_to_use

# Strategy 1: Remove zero importance
features_nonzero = [f for f in feature_cols if feature_importance[f] > 0.0]
mape1, _ = test_feature_set(features_nonzero, "Strategy 1: Remove zero-importance")

# Strategy 2: Remove zero + low importance
features_medium = [f for f in feature_cols if feature_importance[f] >= np.percentile([s for s in feature_importance.values() if s > 0], 10)]
mape2, _ = test_feature_set(features_medium, "Strategy 2: Remove zero + low importance")

# Strategy 3: Keep top 50%
top_50pct = sorted_features[:len(sorted_features)//2]
features_top50 = [f for f, s in top_50pct]
mape3, _ = test_feature_set(features_top50, "Strategy 3: Keep top 50%")

# Strategy 4: Keep top 200 features
features_top200 = [f for f, s in sorted_features[:200]]
mape4, _ = test_feature_set(features_top200, "Strategy 4: Keep top 200")

# Strategy 5: Keep top 100 features
features_top100 = [f for f, s in sorted_features[:100]]
mape5, _ = test_feature_set(features_top100, "Strategy 5: Keep top 100")

# ============================================================================
# Summary and recommendation
# ============================================================================

print("\n" + "="*80)
print("[5/5] SUMMARY AND RECOMMENDATION")
print("="*80)

results = [
    ("Baseline (all 851 features)", baseline_mape, feature_cols),
    ("Remove zero-importance", mape1, features_nonzero),
    ("Remove zero + low", mape2, features_medium),
    ("Top 50%", mape3, features_top50),
    ("Top 200", mape4, features_top200),
    ("Top 100", mape5, features_top100),
]

print("\nComparison:")
for name, mape, feats in results:
    improvement = baseline_mape - mape
    print(f"  {name:30s}: {mape:.4f}% ({len(feats):3d} features, {improvement:+.4f}pp)")

# Find best strategy
best_idx = np.argmin([r[1] for r in results])
best_name, best_mape, best_features = results[best_idx]

print(f"\nBest strategy: {best_name}")
print(f"  MAPE: {best_mape:.4f}%")
print(f"  Improvement: {baseline_mape - best_mape:+.4f}pp")
print(f"  Features: {len(best_features)} (removed {len(feature_cols) - len(best_features)})")

# Save selected features
output_file = ROOT / "selected_features.txt"
with open(output_file, 'w') as f:
    for feat in best_features:
        f.write(f"{feat}\n")

print(f"\nSelected features saved to: {output_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
