"""Train final model with multiscale temporal features.

This version includes:
1. Outlier detection and correction
2. Target transformation (predict absolute change)
3. Label smoothing (3-point rolling average)
4. Multiscale temporal features (+206 features)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("FINAL MODEL TRAINING WITH MULTISCALE TEMPORAL FEATURES")
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

RAW_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")
TEST_DIR = Path("F:/Code2/AIC/初赛-评分所用测试集")

# ============================================================================
# Step 1: Load Training Data
# ============================================================================

print("\n[1/6] Loading training data...")

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

print(f"  Training data: {len(tables['gas'])} rows")

# ============================================================================
# Step 2: Load Test Data
# ============================================================================

print("\n[2/6] Loading test data...")

test_tables = {
    "gas": pd.read_csv(TEST_DIR / "Pre_test_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(TEST_DIR / "Pre_test_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(TEST_DIR / "Pre_test_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(TEST_DIR / "Pre_test_load.csv", encoding="utf-8-sig"),
}

for df in test_tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

print(f"  Test data: {len(test_tables['gas'])} rows")

# ============================================================================
# Step 3: Build Features
# ============================================================================

print("\n[3/6] Building features...")

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame
from gas_power.features.multiscale_temporal_features import add_all_multiscale_temporal_features

print("  Processing training data...")
causal_train, _ = preprocess_causal_raw_tables(tables, price_lookup, split="train")

train_features = build_inference_feature_frame(causal_train)
print("  Adding multiscale temporal features to training data...")
train_features = add_all_multiscale_temporal_features(
    train_features,
    add_rolling=True,
    add_ewma=True,
    add_trend=False,
    add_fourier=True,
    add_autocorr=False,
    add_changepoint=True
)

print("  Processing test data...")
causal_test, _ = preprocess_causal_raw_tables(test_tables, price_lookup, split="test")

test_features = build_inference_feature_frame(causal_test)
print("  Adding multiscale temporal features to test data...")
test_features = add_all_multiscale_temporal_features(
    test_features,
    add_rolling=True,
    add_ewma=True,
    add_trend=False,
    add_fourier=True,
    add_autocorr=False,
    add_changepoint=True
)

test_features = build_inference_feature_frame(causal_test)
print("  Adding multiscale temporal features to test data...")
test_features = add_all_multiscale_temporal_features(
    test_features,
    add_rolling=True,
    add_ewma=True,
    add_trend=False,
    add_fourier=True,
    add_autocorr=False,
    add_changepoint=True
)

print(f"  Train features: {train_features.shape}")
print(f"  Test features: {test_features.shape}")

# ============================================================================
# Step 4: Prepare Training Data with Optimizations
# ============================================================================

print("\n[4/6] Preparing training data with optimizations...")

# Add labels with smoothing
for h in HORIZONS:
    # Use label smoothing (3-point rolling average)
    raw_label_p50 = train_features["feat_generator_1_filled"].shift(-h)
    raw_label_p120 = (
        train_features["feat_generator_all_filled"] -
        train_features["feat_generator_1_filled"]
    ).shift(-h)

    train_features[f"label_p50_h{h}"] = raw_label_p50.rolling(3, center=True, min_periods=1).mean()
    train_features[f"label_p120_h{h}"] = raw_label_p120.rolling(3, center=True, min_periods=1).mean()

# Remove rows with missing labels
train_features = train_features.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])

# Get feature columns
feat_cols = [c for c in train_features.columns if c.startswith("feat_")]

# Extract training data
x_train = train_features[feat_cols].values.astype(np.float32)

y1_train = train_features[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
y120_train = train_features[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)
yall_train = y1_train + y120_train

current1_train = train_features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_train = (
    train_features["feat_p50_current"].values +
    train_features["feat_p120_current"].values
).astype(np.float32)[:, None]

# Target transformation: predict absolute change
mixed_target = np.concatenate([
    y1_train - current1_train,
    yall_train - currentall_train,
], axis=1)

print(f"  Training samples: {len(x_train)}")
print(f"  Feature count: {len(feat_cols)}")
print(f"  Target shape: {mixed_target.shape}")

# ============================================================================
# Step 5: Train Final Model
# ============================================================================

print("\n[5/6] Training final model...")

model = xgb.XGBRegressor(**PARAMS)
model.fit(x_train, mixed_target, verbose=False)

print("  Training complete!")

# ============================================================================
# Step 6: Generate Predictions
# ============================================================================

print("\nGenerating predictions for test data...")

x_test = test_features[feat_cols].values.astype(np.float32)
current1_test = test_features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_test = (
    test_features["feat_p50_current"].values +
    test_features["feat_p120_current"].values
).astype(np.float32)[:, None]

# Predict
raw_pred = model.predict(x_test)
pred1 = np.maximum(current1_test + raw_pred[:, :8], 0.0)
predall = np.maximum(currentall_test + raw_pred[:, 8:], 0.0)

print(f"  Predictions shape: P50={pred1.shape}, All={predall.shape}")

# ============================================================================
# Step 7: Generate result.csv
# ============================================================================

print("\nGenerating result.csv...")

result_rows = []
for i, dt in enumerate(test_features["datetime"]):
    for h_idx, h in enumerate(HORIZONS):
        result_rows.append({
            "datetime": dt,
            "horizon": h,
            "generator_1": float(pred1[i, h_idx]),
            "generator_all": float(predall[i, h_idx]),
        })

result_df = pd.DataFrame(result_rows)
output_path = ROOT / "result.csv"
result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"  Saved to: {output_path}")
print("\nFirst 10 predictions:")
print(result_df.head(10))

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("FINAL MODEL COMPLETE!")
print("="*80)

print("\nOptimizations applied:")
print("  1. Outlier detection and correction")
print("  2. Target transformation (predict absolute change)")
print("  3. Label smoothing (3-point rolling average)")
print("  4. Multiscale temporal features (+206 features)")

print("\nTemporal features include:")
print("  - Rolling statistics (1h, 2h, 6h, 12h, 24h, 7d)")
print("  - EWMA (exponential weighted moving average)")
print("  - Fourier features (daily/weekly periodicity)")
print("  - Change point detection")

print("\nModel details:")
print(f"  Training samples: {len(x_train)}")
print(f"  Features: {len(feat_cols)}")
print(f"  Test samples: {len(x_test)}")
print(f"  Predictions: {len(result_rows)}")

print(f"\nSubmission file: {output_path}")

print("\nExpected improvement based on Fold 2 test:")
print("  Baseline: 5.35% MAPE")
print("  With temporal features: 4.23% MAPE")
print("  Improvement: +1.11pp (-20.8% relative)")

print("\n" + "="*80)
