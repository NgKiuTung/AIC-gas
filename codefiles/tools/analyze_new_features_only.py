"""Analyze importance of newly added features only.

Keep all 851 base features, only remove redundant new features.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("NEW FEATURES IMPORTANCE ANALYSIS")
print("="*80)

# ============================================================================
# Load data
# ============================================================================

print("\n[1/4] Loading data...")
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

# Build features
from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.enhanced_interactions import add_future_price_features
from gas_power.features.inference import build_inference_feature_frame
from gas_power.features.optimized_features import build_optimized_features

causal, _ = preprocess_causal_raw_tables(tables, price_lookup, split="train")
base = build_inference_feature_frame(causal)

print(f"  Base features: {base.shape}")

# Add new features
with_prices = add_future_price_features(base, price_lookup)
optimized = build_optimized_features(
    with_prices,
    add_target_lags=True,
    fix_outliers=False,
    add_domain=True,
    add_physical=True,
    add_time=True
)

print(f"  With new features: {optimized.shape}")

# Identify base vs new features
base_feat_cols = [c for c in base.columns if c.startswith('feat_')]
all_feat_cols = [c for c in optimized.columns if c.startswith('feat_')]
new_feat_cols = [c for c in all_feat_cols if c not in base_feat_cols]

print("\nFeature breakdown:")
print(f"  Base features: {len(base_feat_cols)}")
print(f"  New features: {len(new_feat_cols)}")
print(f"  Total: {len(all_feat_cols)}")

# Add labels
HORIZONS = tuple(range(1, 9))
for h in HORIZONS:
    optimized[f"label_p50_h{h}"] = optimized["feat_generator_1_filled"].shift(-h)
    optimized[f"label_p120_h{h}"] = (
        optimized["feat_generator_all_filled"] - optimized["feat_generator_1_filled"]
    ).shift(-h)

optimized = optimized.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])

# ============================================================================
# Train baseline (base features only)
# ============================================================================

print("\n[2/4] Training baseline model (base features only)...")

fold_start = pd.Timestamp("2025-04-01")
fold_end = pd.Timestamp("2025-04-15 23:45:00")

train = optimized[optimized["datetime"] < fold_start - pd.Timedelta(minutes=120)]
valid = optimized[(optimized["datetime"] >= fold_start) & (optimized["datetime"] <= fold_end)]

print(f"  Train: {len(train)}, Valid: {len(valid)}")

# Prepare training data
def prepare_data(df, feature_cols):
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y1 = df[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y120 = df[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    yall = y1 + y120
    current1 = df["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
    currentall = (df["feat_p50_current"] + df["feat_p120_current"]).to_numpy(dtype=np.float32)[:, None]

    BETA_1 = np.linspace(0.35, 0.30, 8, dtype=np.float32)
    BETA_ALL = np.linspace(0.65, 0.60, 8, dtype=np.float32)

    mixed_target = np.concatenate([
        (y1 - current1) / np.maximum(np.abs(current1), 1e-6),
        yall - currentall,
    ], axis=1)

    return x, mixed_target, y1, yall, current1, currentall, BETA_1, BETA_ALL

x_train_base, mixed_target, y1_train, yall_train, current1_train, currentall_train, BETA_1, BETA_ALL = prepare_data(train, base_feat_cols)
x_valid_base, _, y1_valid, yall_valid, current1_valid, currentall_valid, _, _ = prepare_data(valid, base_feat_cols)

# Train baseline
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
    "verbosity": 0,
}

model_base = xgb.XGBRegressor(**params)
model_base.fit(x_train_base, mixed_target, verbose=False)

raw = model_base.predict(x_valid_base)
pred1 = np.maximum(current1_valid * (1.0 + raw[:, :8] * BETA_1), 0.0)
predall = np.maximum(currentall_valid + raw[:, 8:] * BETA_ALL, 0.0)

mape1_base = float(np.mean(np.abs(y1_valid - pred1) / np.maximum(np.abs(y1_valid), 1e-8)) * 100.0)
mapeall_base = float(np.mean(np.abs(yall_valid - predall) / np.maximum(np.abs(yall_valid), 1e-8)) * 100.0)
baseline_mape = (mape1_base + mapeall_base) / 2.0

print(f"  Baseline MAPE: {baseline_mape:.4f}%")

# ============================================================================
# Train with all features and analyze new feature importance
# ============================================================================

print("\n[3/4] Training with all features and analyzing new feature importance...")

x_train_all, _, _, _, _, _, _, _ = prepare_data(train, all_feat_cols)
x_valid_all, _, _, _, _, _, _, _ = prepare_data(valid, all_feat_cols)

model_all = xgb.XGBRegressor(**params)
model_all.fit(x_train_all, mixed_target, verbose=False)

raw = model_all.predict(x_valid_all)
pred1 = np.maximum(current1_valid * (1.0 + raw[:, :8] * BETA_1), 0.0)
predall = np.maximum(currentall_valid + raw[:, 8:] * BETA_ALL, 0.0)

mape1_all = float(np.mean(np.abs(y1_valid - pred1) / np.maximum(np.abs(y1_valid), 1e-8)) * 100.0)
mapeall_all = float(np.mean(np.abs(yall_valid - predall) / np.maximum(np.abs(yall_valid), 1e-8)) * 100.0)
all_features_mape = (mape1_all + mapeall_all) / 2.0

print(f"  All features MAPE: {all_features_mape:.4f}%")
print(f"  Difference: {all_features_mape - baseline_mape:+.4f}pp")

# Get importance for new features
importance_dict = model_all.get_booster().get_score(importance_type='gain')

new_feature_importance = {}
for i, feat_name in enumerate(all_feat_cols):
    if feat_name in new_feat_cols:
        xgb_name = f'f{i}'
        new_feature_importance[feat_name] = importance_dict.get(xgb_name, 0.0)

sorted_new_features = sorted(new_feature_importance.items(), key=lambda x: -x[1])

print("\nNew features importance:")
print(f"  Total new features: {len(new_feat_cols)}")

zero_importance_new = [f for f, s in new_feature_importance.items() if s == 0.0]
print(f"  Zero importance: {len(zero_importance_new)} ({len(zero_importance_new)/len(new_feat_cols)*100:.1f}%)")

print("\nTop 20 new features by importance:")
for i, (feat, score) in enumerate(sorted_new_features[:20], 1):
    print(f"  {i:2d}. {feat:60s} {score:10.1f}")

print("\nBottom 20 new features (zero or low importance):")
for i, (feat, score) in enumerate(sorted_new_features[-20:], 1):
    print(f"  {i:2d}. {feat:60s} {score:10.1f}")

# ============================================================================
# Test feature selection strategies for new features
# ============================================================================

print("\n[4/4] Testing feature selection strategies...")

def test_with_selected_new_features(new_features_to_keep, strategy_name):
    """Test with base + selected new features."""
    selected_features = base_feat_cols + new_features_to_keep

    x_train_sel, _, _, _, _, _, _, _ = prepare_data(train, selected_features)
    x_valid_sel, _, _, _, _, _, _, _ = prepare_data(valid, selected_features)

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
    print(f"    Total features: {len(selected_features)} (base {len(base_feat_cols)} + new {len(new_features_to_keep)})")
    print(f"    MAPE: {mape:.4f}%")
    print(f"    vs Baseline: {improvement:+.4f}pp")

    return mape, new_features_to_keep

# Strategy 1: Remove zero-importance new features
new_nonzero = [f for f, s in new_feature_importance.items() if s > 0.0]
mape1, _ = test_with_selected_new_features(new_nonzero, "Remove zero-importance new features")

# Strategy 2: Keep top 50% of new features
top_50pct_new = [f for f, s in sorted_new_features[:len(sorted_new_features)//2]]
mape2, _ = test_with_selected_new_features(top_50pct_new, "Keep top 50% new features")

# Strategy 3: Keep top 30 new features
top_30_new = [f for f, s in sorted_new_features[:30]]
mape3, _ = test_with_selected_new_features(top_30_new, "Keep top 30 new features")

# Strategy 4: Keep top 20 new features
top_20_new = [f for f, s in sorted_new_features[:20]]
mape4, _ = test_with_selected_new_features(top_20_new, "Keep top 20 new features")

# Strategy 5: Keep top 10 new features
top_10_new = [f for f, s in sorted_new_features[:10]]
mape5, _ = test_with_selected_new_features(top_10_new, "Keep top 10 new features")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

results = [
    ("Baseline (base only)", baseline_mape, []),
    ("All features (base + all new)", all_features_mape, new_feat_cols),
    ("Base + non-zero new", mape1, new_nonzero),
    ("Base + top 50% new", mape2, top_50pct_new),
    ("Base + top 30 new", mape3, top_30_new),
    ("Base + top 20 new", mape4, top_20_new),
    ("Base + top 10 new", mape5, top_10_new),
]

print("\nComparison:")
print(f"{'Strategy':<35s} {'MAPE':<10s} {'Features':<15s} {'vs Baseline':<12s}")
print("-" * 80)
for name, mape, new_feats in results:
    total_feats = len(base_feat_cols) + len(new_feats)
    improvement = baseline_mape - mape
    print(f"{name:<35s} {mape:6.4f}%   {total_feats:4d} ({len(new_feats):3d} new)   {improvement:+.4f}pp")

# Find best
best_idx = np.argmin([r[1] for r in results])
best_name, best_mape, best_new_feats = results[best_idx]

print(f"\nBest strategy: {best_name}")
print(f"  MAPE: {best_mape:.4f}%")
print(f"  Improvement vs baseline: {baseline_mape - best_mape:+.4f}pp")
print(f"  Features: {len(base_feat_cols)} base + {len(best_new_feats)} new = {len(base_feat_cols) + len(best_new_feats)}")

if len(best_new_feats) > 0:
    print("\nSelected new features to keep:")
    for feat in best_new_feats:
        score = new_feature_importance.get(feat, 0.0)
        print(f"  {feat:60s} {score:10.1f}")

    # Save
    output_file = ROOT / "selected_new_features.txt"
    with open(output_file, 'w') as f:
        for feat in best_new_feats:
            f.write(f"{feat}\n")
    print(f"\nSelected new features saved to: {output_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
