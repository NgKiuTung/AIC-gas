"""Final training with all optimizations and generate submission file.

Applies:
1. Outlier detection and correction
2. Target transformation
3. Label smoothing

Generates result.csv for submission.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("FINAL MODEL TRAINING WITH ALL OPTIMIZATIONS")
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
    "max_bin": 256,
    "multi_strategy": "one_output_per_tree",
    "verbosity": 0,
}

BETA_1 = np.linspace(0.35, 0.30, 8, dtype=np.float32)
BETA_ALL = np.linspace(0.65, 0.60, 8, dtype=np.float32)

# ============================================================================
# Helper Functions
# ============================================================================

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

    # Change rate outliers
    for col in ['feat_generator_1_filled', 'feat_generator_all_filled']:
        if col in df.columns:
            change_rate = df[col].pct_change(fill_method=None).abs()
            outliers = change_rate > 0.3

            if outliers.any():
                df.loc[outliers, col] = np.nan
                df[col] = df[col].interpolate(method='linear', limit=4)
                outliers_fixed += outliers.sum()

    # Statistical outliers
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

# ============================================================================
# Load Training Data
# ============================================================================

print("\n[1/5] Loading training data...")

RAW_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")

train_tables = {
    "gas": pd.read_csv(RAW_DIR / "Pre_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(RAW_DIR / "Pre_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(RAW_DIR / "Pre_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(RAW_DIR / "Pre_load.csv", encoding="utf-8-sig"),
}

for df in train_tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

price_table = pd.read_excel(RAW_DIR / "price.xlsx")
price_lookup = {}
for row_idx, row in price_table.iterrows():
    for month in range(1, 13):
        price_lookup[(month, row_idx)] = float(row[f"{month}月"])

print(f"  Training data: {len(train_tables['gas'])} rows")

# ============================================================================
# Load Test Data
# ============================================================================

print("\n[2/5] Loading test data...")

TEST_DIR = Path("F:/Code2/AIC/初赛-评分所用测试集")

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
# Build Features
# ============================================================================

print("\n[3/5] Building features...")

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame

# Train features
print("  Processing training data...")
train_causal, _ = preprocess_causal_raw_tables(train_tables, price_lookup, split="train")
train_features = build_inference_feature_frame(train_causal)

# Apply outlier detection
print("  Applying outlier detection to training data...")
train_features = detect_and_fix_outliers(train_features)

# Test features
print("  Processing test data...")
test_causal, _ = preprocess_causal_raw_tables(test_tables, price_lookup, split="test")
test_features = build_inference_feature_frame(test_causal)

# Apply outlier detection
print("  Applying outlier detection to test data...")
test_features = detect_and_fix_outliers(test_features)

print(f"  Train features: {train_features.shape}")
print(f"  Test features: {test_features.shape}")

# ============================================================================
# Prepare Training Data
# ============================================================================

print("\n[4/5] Preparing training data with optimizations...")

# Add labels with smoothing
for h in HORIZONS:
    raw_label_p50 = train_features["feat_generator_1_filled"].shift(-h)
    raw_label_p120 = (train_features["feat_generator_all_filled"] - train_features["feat_generator_1_filled"]).shift(-h)

    # Label smoothing
    smoothed_p50 = raw_label_p50.rolling(3, center=True, min_periods=1).mean()
    smoothed_p120 = raw_label_p120.rolling(3, center=True, min_periods=1).mean()

    train_features[f"label_p50_h{h}"] = smoothed_p50
    train_features[f"label_p120_h{h}"] = smoothed_p120

# Remove NaN labels
train_features = train_features.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])

print(f"  Training samples: {len(train_features)}")

# Extract features and labels
feat_cols = [c for c in train_features.columns if c.startswith('feat_')]
print(f"  Feature count: {len(feat_cols)}")

x_train = train_features[feat_cols].values.astype(np.float32)
y1_train = train_features[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
y120_train = train_features[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)
yall_train = y1_train + y120_train

current1_train = train_features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_train = (train_features["feat_p50_current"].values + train_features["feat_p120_current"].values).astype(np.float32)[:, None]

# Target transformation: predict absolute change
mixed_target = np.concatenate([
    y1_train - current1_train,
    yall_train - currentall_train,
], axis=1)

print(f"  Target shape: {mixed_target.shape}")

# ============================================================================
# Train Final Model
# ============================================================================

print("\n[5/5] Training final model...")

model = xgb.XGBRegressor(**PARAMS)
model.fit(x_train, mixed_target, verbose=False)

print("  Training complete!")

# ============================================================================
# Predict on Test Data
# ============================================================================

print("\nGenerating predictions for test data...")

# Extract test features
x_test = test_features[feat_cols].values.astype(np.float32)
current1_test = test_features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_test = (test_features["feat_p50_current"].values + test_features["feat_p120_current"].values).astype(np.float32)[:, None]

# Predict
raw_pred = model.predict(x_test)

# Reverse transformation
pred1 = np.maximum(current1_test + raw_pred[:, :8], 0.0)
predall = np.maximum(currentall_test + raw_pred[:, 8:], 0.0)

print(f"  Predictions shape: P50={pred1.shape}, All={predall.shape}")

# ============================================================================
# Generate Submission File
# ============================================================================

print("\nGenerating result.csv...")

# Create submission dataframe
submission_data = []

for i, dt in enumerate(test_features["datetime"]):
    for h in HORIZONS:
        submission_data.append({
            "datetime": dt,
            "horizon": h,
            "generator_1": float(pred1[i, h-1]),
            "generator_all": float(predall[i, h-1]),
        })

submission = pd.DataFrame(submission_data)

# Sort by datetime and horizon
submission = submission.sort_values(["datetime", "horizon"]).reset_index(drop=True)

# Save
output_file = ROOT / "result.csv"
submission.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"  Saved to: {output_file}")
print(f"  Total predictions: {len(submission)} rows")

# Show sample
print("\nSample predictions:")
print(submission.head(10))

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

print("\nModel details:")
print(f"  Training samples: {len(train_features)}")
print(f"  Features: {len(feat_cols)}")
print(f"  Test samples: {len(test_features)}")
print(f"  Predictions: {len(submission)}")

print(f"\nSubmission file: {output_file}")
print("\n" + "="*80)
