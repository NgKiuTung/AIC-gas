"""Quick single-fold test to show MAPE improvement.

This runs 3 experiments on one fold:
1. Baseline (no enhancements)
2. + Future prices
3. + Interactions

Expected runtime: 10-15 minutes
"""

import sys
from pathlib import Path
import time

import pandas as pd
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame
from gas_power.features.enhanced_interactions import add_future_price_features, add_interaction_features

print("="*70)
print("QUICK MAPE IMPROVEMENT TEST")
print("="*70)

# Load data (reuse from previous test)
print("\n[1/4] Loading data...")
RAW_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")
tables = {
    "gas": pd.read_csv(RAW_DIR / "Pre_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(RAW_DIR / "Pre_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(RAW_DIR / "Pre_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(RAW_DIR / "Pre_load.csv", encoding="utf-8-sig"),
}
for name, df in tables.items():
    df["datetime"] = pd.to_datetime(df["datetime"])

price_table = pd.read_excel(RAW_DIR / "price.xlsx")
price_lookup = {}
for row_idx, row in price_table.iterrows():
    for month in range(1, 13):
        price_lookup[(month, row_idx)] = float(row[f"{month}月"])

print(f"  Data loaded: {len(tables['gas'])} rows")

# Preprocess
print("\n[2/4] Preprocessing...")
causal, _ = preprocess_causal_raw_tables(tables, price_lookup, split="train")
print(f"  Shape: {causal.shape}")

# Build features
print("\n[3/4] Building features...")
base_features = build_inference_feature_frame(causal)

# Add labels
HORIZONS = tuple(range(1, 9))
for h in HORIZONS:
    base_features[f"label_p50_h{h}"] = base_features["feat_generator_1_filled"].shift(-h)
    base_features[f"label_p120_h{h}"] = (
        base_features["feat_generator_all_filled"] - base_features["feat_generator_1_filled"]
    ).shift(-h)

base_features = base_features.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])
print(f"  Base features: {base_features.shape}")

# Enhanced features
enhanced_future = add_future_price_features(base_features, price_lookup)
print(f"  + Future prices: {enhanced_future.shape}")

enhanced_full = add_interaction_features(enhanced_future)
print(f"  + Interactions: {enhanced_full.shape}")

# Training setup
print("\n[4/4] Training models...")

fold_start = pd.Timestamp("2025-04-01 00:00:00")
fold_end = pd.Timestamp("2025-04-15 23:45:00")

BETA_1 = np.linspace(0.35, 0.30, 8, dtype=np.float32)
BETA_ALL = np.linspace(0.65, 0.60, 8, dtype=np.float32)

BASELINE_PARAMS = {
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
    "device": "cpu",  # Use CPU for compatibility
    "random_state": 20260803,
    "n_jobs": -1,
    "max_bin": 256,
    "multi_strategy": "one_output_per_tree",
    "verbosity": 0,
}

ENHANCED_PARAMS = BASELINE_PARAMS.copy()
ENHANCED_PARAMS["n_estimators"] = 400
ENHANCED_PARAMS["colsample_bytree"] = 0.65

def train_eval(features_df, feature_cols, params, name):
    print(f"\n  Training: {name}")
    print(f"    Features: {len(feature_cols)}")

    train = features_df[features_df["datetime"] < fold_start - pd.Timedelta(minutes=120)]
    valid = features_df[(features_df["datetime"] >= fold_start) & (features_df["datetime"] <= fold_end)]

    print(f"    Train: {len(train)}, Valid: {len(valid)}")

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

    mixed_target = np.concatenate([
        (y1_train - current1_train) / np.maximum(np.abs(current1_train), 1e-6),
        yall_train - currentall_train,
    ], axis=1)

    model = xgb.XGBRegressor(**params)

    t0 = time.perf_counter()
    model.fit(x_train, mixed_target, verbose=False)
    raw = model.predict(x_valid)
    train_time = time.perf_counter() - t0

    pred1 = np.maximum(current1_valid * (1.0 + raw[:, :8] * BETA_1), 0.0)
    predall = np.maximum(currentall_valid + raw[:, 8:] * BETA_ALL, 0.0)

    mape1 = float(np.mean(np.abs(y1_valid - pred1) / np.maximum(np.abs(y1_valid), 1e-8)) * 100.0)
    mapeall = float(np.mean(np.abs(yall_valid - predall) / np.maximum(np.abs(yall_valid), 1e-8)) * 100.0)
    mape_avg = (mape1 + mapeall) / 2.0

    print(f"    MAPE: {mape_avg:.4f}% (P50={mape1:.4f}%, All={mapeall:.4f}%)")
    print(f"    Time: {train_time:.1f}s")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return mape_avg, importance

# Experiment 1: Baseline
base_feature_cols = [col for col in base_features.columns if col.startswith("feat_")]
mape_baseline, imp_baseline = train_eval(base_features, base_feature_cols, BASELINE_PARAMS, "BASELINE")

# Experiment 2: + Future prices
future_cols = [col for col in enhanced_future.columns if col.startswith("feat_")]
mape_future, imp_future = train_eval(enhanced_future, future_cols, BASELINE_PARAMS, "+ FUTURE PRICES")

# Experiment 3: + Interactions
full_cols = [col for col in enhanced_full.columns if col.startswith("feat_")]
mape_full, imp_full = train_eval(enhanced_full, full_cols, ENHANCED_PARAMS, "+ INTERACTIONS (tuned)")

# Results
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print(f"\nExperiment                    Features    MAPE      vs Baseline")
print("-"*70)
print(f"1. Baseline                   {len(base_feature_cols):4d}     {mape_baseline:.4f}%   -")
print(f"2. + Future Prices            {len(future_cols):4d}     {mape_future:.4f}%   {mape_baseline-mape_future:+.4f}pp")
print(f"3. + Full Enhancement         {len(full_cols):4d}     {mape_full:.4f}%   {mape_baseline-mape_full:+.4f}pp")

improvement_future = mape_baseline - mape_future
improvement_full = mape_baseline - mape_full
improvement_interact = mape_future - mape_full

print("\n" + "="*70)
print("IMPROVEMENT BREAKDOWN")
print("="*70)
print(f"\nTotal improvement: {improvement_full:.4f} percentage points")
print(f"\nContribution:")
print(f"  Future prices:  {improvement_future:.4f}pp ({improvement_future/improvement_full*100:.1f}%)")
print(f"  Interactions:   {improvement_interact:.4f}pp ({improvement_interact/improvement_full*100:.1f}%)")
print(f"\nRelative: {improvement_full/mape_baseline*100:.2f}% reduction")

# Feature importance analysis
print("\n" + "="*70)
print("TOP 10 FEATURES (Final Model)")
print("="*70)
for idx, row in imp_full.head(10).iterrows():
    feat_type = ""
    if "future_price" in row["feature"]:
        feat_type = "[FUTURE]"
    elif "interact" in row["feature"]:
        feat_type = "[INTERACT]"
    print(f"{row['importance']:8.1f}  {feat_type:10s} {row['feature']}")

future_in_top20 = len([f for f in imp_full.head(20)["feature"] if "future_price" in f])
interact_in_top20 = len([f for f in imp_full.head(20)["feature"] if "interact" in f])

print(f"\nNew features in top 20:")
print(f"  Future prices: {future_in_top20}")
print(f"  Interactions:  {interact_in_top20}")

print("\n" + "="*70)
print("WHAT CAUSED THE IMPROVEMENT")
print("="*70)
print("""
1. FUTURE PRICE FEATURES:
   - Model can see known future electricity prices
   - Enables strategic decisions (save gas for high prices)
   - Contribution: ~{:.1f}% of total improvement

2. INTERACTION FEATURES:
   - Captures non-linear relationships
   - price x holder: High price + gas = full power
   - weekend x hour: Different patterns
   - Contribution: ~{:.1f}% of total improvement

3. HYPERPARAMETER TUNING:
   - Increased n_estimators for more features
   - Reduced colsample_bytree for better feature coverage
""".format(improvement_future/improvement_full*100, improvement_interact/improvement_full*100))

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
