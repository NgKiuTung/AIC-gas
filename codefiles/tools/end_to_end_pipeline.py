"""Complete end-to-end pipeline: Process test data -> Feature engineering -> Model prediction -> Result.csv

Steps:
1. Process test data with field-specific missing value handling
2. Append training history for temporal features
3. Feature engineering
4. Extract test period
5. Model training
6. Prediction
7. Generate result.csv
8. Calculate MAPE (if ground truth available)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("END-TO-END PIPELINE: TEST DATA PROCESSING -> PREDICTION -> RESULT.CSV")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

HORIZONS = tuple(range(1, 9))
SHORT_GAP_THRESHOLD = 4
LONG_GAP_THRESHOLD = 16

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
# Step 1: Load and Process Test Data
# ============================================================================

print("\n[1/8] Loading and processing test data...")

# Load test data
test_files = {
    "gas": "Pre_test_gas.csv",
    "holder": "Pre_test_gas_holder.csv",
    "user": "Pre_test_gas_user.csv",
    "load": "Pre_test_load.csv"
}

test_data = {}
for key, filename in test_files.items():
    filepath = TEST_DIR / filename
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df['datetime'] = pd.to_datetime(df['datetime'])
    test_data[key] = df

print(f"  Loaded test data: {len(test_data['gas'])} rows")

# Field categories for specific handling
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

# Store missing indicators separately (will add after feature engineering)
missing_indicators = {}

# Apply field-specific missing value handling to each table
for key, df in test_data.items():
    missing_indicators[key] = {}

    for category, cols in field_categories.items():
        for col in cols:
            if col not in df.columns:
                continue

            original_missing = df[col].isna().sum()
            if original_missing == 0:
                continue

            # Store missing indicator (don't add to dataframe yet)
            missing_indicators[key][f'missing_flag_{col}'] = df[col].isna().astype(int)

            if category == "气柜":
                # Interpolate short gaps, forward fill others
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both')
                df[col] = df[col].ffill().bfill()

            elif category == "流量":
                # Interpolate short/medium, use 0 for long gaps
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD * 2, limit_direction='both')
                df[col] = df[col].ffill().bfill()
                df[col] = df[col].fillna(0.0)

            elif category == "发电负荷":
                # Forward fill, interpolate short gaps
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both')
                df[col] = df[col].ffill().bfill()

print("  Applied field-specific missing value handling")

# ============================================================================
# Step 2: Load Training Data
# ============================================================================

print("\n[2/8] Loading training data...")

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
# Step 3: Combine Train + Test for Feature Engineering
# ============================================================================

print("\n[3/8] Combining train and test data for feature engineering...")

# Remove any overlap between train and test
test_start = test_data['gas']['datetime'].min()
combined_tables = {}

for key in train_tables.keys():
    train_no_overlap = train_tables[key][train_tables[key]['datetime'] < test_start].copy()
    combined_tables[key] = pd.concat([train_no_overlap, test_data[key]], ignore_index=True)
    combined_tables[key] = combined_tables[key].sort_values('datetime').reset_index(drop=True)

print(f"  Combined data: {len(combined_tables['gas'])} rows")

# ============================================================================
# Step 4: Feature Engineering on Combined Data
# ============================================================================

print("\n[4/8] Building features on combined data...")

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame
from gas_power.features.multiscale_temporal_features import add_all_multiscale_temporal_features

causal_combined, _ = preprocess_causal_raw_tables(combined_tables, price_lookup, split="train")
features_combined = build_inference_feature_frame(causal_combined)

print("  Adding multiscale temporal features...")
features_combined = add_all_multiscale_temporal_features(
    features_combined,
    add_rolling=True,
    add_ewma=True,
    add_trend=False,
    add_fourier=True,
    add_autocorr=False,
    add_changepoint=True
)

feat_cols = [c for c in features_combined.columns if c.startswith("feat_")]
print(f"  Total features: {len(feat_cols)}")

# ============================================================================
# Step 5: Prepare Training Data
# ============================================================================

print("\n[5/8] Preparing training data...")

# Extract training period
train_end_time = test_start
train_mask = features_combined['datetime'] < train_end_time
train_features = features_combined[train_mask].copy()

# Add labels for training
for h in HORIZONS:
    raw_label_p50 = train_features["feat_generator_1_filled"].shift(-h)
    raw_label_p120 = (
        train_features["feat_generator_all_filled"] -
        train_features["feat_generator_1_filled"]
    ).shift(-h)

    train_features[f"label_p50_h{h}"] = raw_label_p50.rolling(3, center=True, min_periods=1).mean()
    train_features[f"label_p120_h{h}"] = raw_label_p120.rolling(3, center=True, min_periods=1).mean()

train_features = train_features.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])

x_train = train_features[feat_cols].values.astype(np.float32)
y1_train = train_features[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
y120_train = train_features[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)
yall_train = y1_train + y120_train

current1_train = train_features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_train = (
    train_features["feat_p50_current"].values +
    train_features["feat_p120_current"].values
).astype(np.float32)[:, None]

mixed_target = np.concatenate([
    y1_train - current1_train,
    yall_train - currentall_train,
], axis=1)

print(f"  Training samples: {len(x_train)}")

# ============================================================================
# Step 6: Train Model
# ============================================================================

print("\n[6/8] Training model...")

model = xgb.XGBRegressor(**PARAMS)
model.fit(x_train, mixed_target, verbose=False)

print("  Model training complete!")

# ============================================================================
# Step 7: Extract Test Period and Predict
# ============================================================================

print("\n[7/8] Extracting test period and making predictions...")

test_end_time = test_data['gas']['datetime'].max()
test_mask = (features_combined['datetime'] >= test_start) & (features_combined['datetime'] <= test_end_time)
test_features = features_combined[test_mask].copy().reset_index(drop=True)

print(f"  Test samples: {len(test_features)}")

# Handle missing values in test features
x_test_df = test_features[feat_cols].copy()
nan_count = x_test_df.isna().sum().sum()

if nan_count > 0:
    print(f"  Handling {nan_count} NaN values...")
    x_test_df = x_test_df.fillna(method='ffill').fillna(method='bfill')
    x_test_df = x_test_df.interpolate(method='linear', limit_direction='both')
    x_test_df = x_test_df.fillna(x_test_df.mean()).fillna(0.0)

x_test = x_test_df.values.astype(np.float32)

current1_test = test_features["feat_p50_current"].fillna(0).values.astype(np.float32)[:, None]
currentall_test = (
    test_features["feat_p50_current"].fillna(0).values +
    test_features["feat_p120_current"].fillna(0).values
).astype(np.float32)[:, None]

# Predict
print("  Predicting...")
raw_pred = model.predict(x_test)
pred1 = np.maximum(current1_test + raw_pred[:, :8], 0.0)
predall = np.maximum(currentall_test + raw_pred[:, 8:], 0.0)

print(f"  Predictions: P50={pred1.shape}, All={predall.shape}")

# ============================================================================
# Step 8: Generate result.csv
# ============================================================================

print("\n[8/8] Generating result.csv...")

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
print(f"  Rows: {len(result_df)}")

# ============================================================================
# Calculate MAPE if ground truth available
# ============================================================================

print("\n" + "="*80)
print("MAPE CALCULATION (if ground truth available)")
print("="*80)

# Check if we have ground truth in test data
if 'generator_1' in test_data['gas'].columns and 'generator_all' in test_data['gas'].columns:
    print("\nGround truth found in test data!")

    # Prepare ground truth
    test_gt = test_data['gas'][['datetime', 'generator_1', 'generator_all']].copy()

    # Merge predictions with ground truth
    eval_data = []
    for i, dt in enumerate(test_features["datetime"]):
        # Find ground truth for each horizon
        for h_idx, h in enumerate(HORIZONS):
            target_dt = dt + pd.Timedelta(minutes=15 * h)
            gt_row = test_gt[test_gt['datetime'] == target_dt]

            if len(gt_row) > 0:
                eval_data.append({
                    'datetime': dt,
                    'horizon': h,
                    'pred_p50': pred1[i, h_idx],
                    'pred_all': predall[i, h_idx],
                    'true_p50': gt_row.iloc[0]['generator_1'],
                    'true_all': gt_row.iloc[0]['generator_all']
                })

    if len(eval_data) > 0:
        eval_df = pd.DataFrame(eval_data)

        # Calculate MAPE
        def mape(y_true, y_pred):
            mask = y_true > 0
            if mask.sum() == 0:
                return np.nan
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        mape_p50 = mape(eval_df['true_p50'].values, eval_df['pred_p50'].values)
        mape_all = mape(eval_df['true_all'].values, eval_df['pred_all'].values)
        mape_avg = (mape_p50 + mape_all) / 2

        print("\nMAPE Results:")
        print(f"  P50 (generator_1):  {mape_p50:.4f}%")
        print(f"  All (generator_all): {mape_all:.4f}%")
        print(f"  Average:            {mape_avg:.4f}%")

        print(f"\nEvaluation samples: {len(eval_df)}")
    else:
        print("\n  Cannot calculate MAPE: No matching ground truth timestamps")
else:
    print("\n  Ground truth not available in test data (this is expected for competition)")
    print("  Predictions saved to result.csv for submission")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)

print("\nCompleted steps:")
print("  ✓ Load and process test data (field-specific strategies)")
print("  ✓ Load training data")
print("  ✓ Combine train + test for context")
print("  ✓ Feature engineering (1057 features)")
print("  ✓ Train XGBoost model")
print("  ✓ Make predictions on test set")
print("  ✓ Generate result.csv")
print("  ✓ Calculate MAPE (if available)")

print(f"\nOutput file: {output_path}")
print("Ready for submission!")

print("\n" + "="*80)
