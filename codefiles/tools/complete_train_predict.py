"""Complete pipeline: Train on hybrid dataset and generate result.csv and input.csv."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("COMPLETE HYBRID TRAINING PIPELINE")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

HORIZONS = tuple(range(1, 9))
SHORT_GAP_THRESHOLD = 4

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

HYBRID_DIR = Path("F:/Code2/AIC/初赛-混合训练集")
TEST_DIR = Path("F:/Code2/AIC/初赛-评分所用测试集")
OUTPUT_DIR = Path("F:/Code2/AIC")

field_categories = {
    "气柜": ['blast_furnace_gas_holder_1', 'blast_furnace_gas_holder_2',
            'coke_oven_gas_holder', 'converter_gas_holder'],
    "流量": ['blast_furnace_gas_generation', 'coke_oven_gas_generation', 'converter_gas_generation',
            'hot_blast_stove_blast_furnace_gas', 'hot_blast_stove_coke_oven_gas',
            'hot_blast_stove_converter_gas', 'lime_kiln_blast_furnace_gas',
            'lime_kiln_coke_oven_gas', 'raw_material_blast_furnace_gas',
            'raw_material_coke_oven_gas', 'heating_furnace_blast_furnace_gas',
            'heating_furnace_coke_oven_gas', 'heating_furnace_converter_gas',
            'power_generation_blast_furnace_gas', 'power_generation_coke_oven_gas',
            'power_generation_converter_gas'],
    "发电负荷": ['generator_1', 'generator_all', 'load']
}

# ============================================================================
# Step 1: Load and Prepare Training Data
# ============================================================================

print("\n[1/6] Loading hybrid training data...")
train_tables = {
    "gas": pd.read_csv(HYBRID_DIR / "Pre_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(HYBRID_DIR / "Pre_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(HYBRID_DIR / "Pre_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(HYBRID_DIR / "Pre_load.csv", encoding="utf-8-sig"),
}

for df in train_tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

print(f"  Loaded {len(train_tables['gas'])} rows")

price_table = pd.read_excel(HYBRID_DIR / "price.xlsx")
price_lookup = {}
for row_idx, row in price_table.iterrows():
    for month in range(1, 13):
        price_lookup[(month, row_idx)] = float(row[f"{month}月"])

# Data cleaning
print("\n[2/6] Applying data cleaning...")
for _key, df in train_tables.items():
    for category, cols in field_categories.items():
        for col in cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue
            if category == "气柜":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both')
                df[col] = df[col].ffill().bfill()
            elif category == "流量":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD * 2, limit_direction='both')
                df[col] = df[col].ffill().bfill().fillna(0.0)
            elif category == "发电负荷":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both')
                df[col] = df[col].ffill().bfill()

print("  Done")

# Build training features
print("\n[3/6] Building training features...")

merged_train = train_tables['gas'].copy()
for key in ['holder', 'user', 'load']:
    merged_train = merged_train.merge(train_tables[key], on='datetime', how='left')

features = pd.DataFrame()
features['datetime'] = merged_train['datetime']
features['feat_generator_1_filled'] = merged_train['generator_1']
features['feat_generator_all_filled'] = merged_train['generator_all']
features['feat_p50_current'] = merged_train['generator_1']
features['feat_p120_current'] = merged_train['generator_all'] - merged_train['generator_1']
features['feat_pall_current'] = merged_train['generator_all']

key_cols = ['blast_furnace_gas_holder_2', 'blast_furnace_gas_generation',
            'coke_oven_gas_generation', 'converter_gas_generation',
            'power_generation_blast_furnace_gas', 'power_generation_coke_oven_gas',
            'power_generation_converter_gas', 'load']

for col in key_cols:
    if col in merged_train.columns:
        features[f'feat_{col}'] = merged_train[col]

features['feat_hour'] = features['datetime'].dt.hour
features['feat_dayofweek'] = features['datetime'].dt.dayofweek
features['feat_month'] = features['datetime'].dt.month
features['feat_is_weekend'] = (features['datetime'].dt.dayofweek >= 5).astype(float)
features['feat_known_price'] = features.apply(
    lambda row: price_lookup[(row['datetime'].month, row['datetime'].hour * 2)],
    axis=1
)

# Add labels
for h in HORIZONS:
    raw_label_p50 = features["feat_generator_1_filled"].shift(-h)
    raw_label_p120 = (features["feat_generator_all_filled"] - features["feat_generator_1_filled"]).shift(-h)
    features[f"label_p50_h{h}"] = raw_label_p50.rolling(3, center=True, min_periods=1).mean()
    features[f"label_p120_h{h}"] = raw_label_p120.rolling(3, center=True, min_periods=1).mean()

features = features.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])

feat_cols = [c for c in features.columns if c.startswith("feat_")]
print(f"  Features: {len(feat_cols)}, Training samples: {len(features)}")

# ============================================================================
# Step 2: Train Model
# ============================================================================

print("\n[4/6] Training model...")

x_full = features[feat_cols].values.astype(np.float32)
y1_full = features[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
y120_full = features[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)
yall_full = y1_full + y120_full

current1_full = features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_full = (features["feat_p50_current"].values + features["feat_p120_current"].values).astype(np.float32)[:, None]

mixed_target_full = np.concatenate([y1_full - current1_full, yall_full - currentall_full], axis=1)

model = xgb.XGBRegressor(**PARAMS)
model.fit(x_full, mixed_target_full, verbose=False)

print("  Model trained successfully")

# ============================================================================
# Step 3: Load and Process Test Data
# ============================================================================

print("\n[5/6] Loading and processing test data...")

test_tables = {
    "gas": pd.read_csv(TEST_DIR / "Pre_test_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(TEST_DIR / "Pre_test_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(TEST_DIR / "Pre_test_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(TEST_DIR / "Pre_test_load.csv", encoding="utf-8-sig"),
}

for df in test_tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

print(f"  Loaded {len(test_tables['gas'])} test rows")

# Clean test data
for _key, df in test_tables.items():
    for category, cols in field_categories.items():
        for col in cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue
            if category == "气柜":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both').ffill().bfill()
            elif category == "流量":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD * 2, limit_direction='both').ffill().bfill().fillna(0.0)
            elif category == "发电负荷":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both').ffill().bfill()

# Build test features
test_merged = test_tables['gas'].copy()
for key in ['holder', 'user', 'load']:
    test_merged = test_merged.merge(test_tables[key], on='datetime', how='left')

test_features = pd.DataFrame()
test_features['datetime'] = test_merged['datetime']
test_features['feat_generator_1_filled'] = test_merged['generator_1']
test_features['feat_generator_all_filled'] = test_merged['generator_all']
test_features['feat_p50_current'] = test_merged['generator_1']
test_features['feat_p120_current'] = test_merged['generator_all'] - test_merged['generator_1']
test_features['feat_pall_current'] = test_merged['generator_all']

for col in key_cols:
    if col in test_merged.columns:
        test_features[f'feat_{col}'] = test_merged[col]

test_features['feat_hour'] = test_features['datetime'].dt.hour
test_features['feat_dayofweek'] = test_features['datetime'].dt.dayofweek
test_features['feat_month'] = test_features['datetime'].dt.month
test_features['feat_is_weekend'] = (test_features['datetime'].dt.dayofweek >= 5).astype(float)
test_features['feat_known_price'] = test_features.apply(
    lambda row: price_lookup[(row['datetime'].month, row['datetime'].hour * 2)],
    axis=1
)

print(f"  Test features prepared: {len(test_features)} samples")

# ============================================================================
# Step 4: Generate Predictions
# ============================================================================

print("\n[6/6] Generating predictions and saving outputs...")

x_test = test_features[feat_cols].values.astype(np.float32)
current1_test = test_features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_test = (test_features["feat_p50_current"].values + test_features["feat_p120_current"].values).astype(np.float32)[:, None]

raw_pred_test = model.predict(x_test)
pred1_test = np.maximum(current1_test + raw_pred_test[:, :8], 0.0)
predall_test = np.maximum(currentall_test + raw_pred_test[:, 8:], 0.0)

# ============================================================================
# Generate result.csv (Submission format)
# ============================================================================

result_rows = []
for idx in range(len(test_features)):
    dt = test_features.iloc[idx]["datetime"]
    for h in HORIZONS:
        result_rows.append({
            "datetime": dt + pd.Timedelta(minutes=15 * h),
            "generator_1": pred1_test[idx, h - 1],
            "generator_all": predall_test[idx, h - 1],
        })

result_df = pd.DataFrame(result_rows)
result_df = result_df.sort_values("datetime").reset_index(drop=True)

result_file = OUTPUT_DIR / "result.csv"
result_df.to_csv(result_file, index=False, encoding="utf-8-sig")
print(f"  [1/2] result.csv saved: {result_file} ({len(result_df)} rows)")

# ============================================================================
# Generate input.csv (Test input data)
# ============================================================================

# Merge all test tables to create complete input
input_df = test_tables['gas'].copy()
for key in ['holder', 'user', 'load']:
    input_df = input_df.merge(test_tables[key], on='datetime', how='left', suffixes=('', f'_{key}'))

# Sort by datetime
input_df = input_df.sort_values('datetime').reset_index(drop=True)

input_file = OUTPUT_DIR / "input.csv"
input_df.to_csv(input_file, index=False, encoding="utf-8-sig")
print(f"  [2/2] input.csv saved: {input_file} ({len(input_df)} rows)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nTraining:")
print(f"  - Hybrid dataset: {len(train_tables['gas'])} rows")
print(f"  - Training samples (after feature engineering): {len(features)}")
print(f"  - Features used: {len(feat_cols)}")

print(f"\nTest:")
print(f"  - Test input: {len(test_tables['gas'])} rows")
print(f"  - Test features: {len(test_features)} samples")

print(f"\nOutputs:")
print(f"  - result.csv: {len(result_df)} predictions")
print(f"  - input.csv: {len(input_df)} test input rows")

print(f"\nFiles saved to: {OUTPUT_DIR}")
print("="*80)
