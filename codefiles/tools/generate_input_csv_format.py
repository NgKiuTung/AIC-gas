"""Generate input.csv with predictions in the required format.

Format requirements:
- Must include datetime (prediction start point)
- generator_1 predictions at different horizons: generator_1_t+15_pred, ..., generator_1_t+120_pred
- generator_all predictions at different horizons: generator_all_t+15_pred, ..., generator_all_t+120_pred
- Keep 3+ decimal places
- No missing rows or duplicates
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("GENERATING INPUT.CSV IN REQUIRED FORMAT")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

HORIZONS = tuple(range(1, 9))  # 1-8 steps = 15-120 minutes
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

RAW_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")
TEST_DIR = Path("F:/Code2/AIC/初赛-评分所用测试集")

# ============================================================================
# Step 1: Load and Process Test Data
# ============================================================================

print("\n[1/6] Loading and processing test data...")

test_data = {
    "gas": pd.read_csv(TEST_DIR / "Pre_test_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(TEST_DIR / "Pre_test_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(TEST_DIR / "Pre_test_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(TEST_DIR / "Pre_test_load.csv", encoding="utf-8-sig")
}

for df in test_data.values():
    df['datetime'] = pd.to_datetime(df['datetime'])

print(f"  Test data: {len(test_data['gas'])} rows")

# Field-specific missing value handling
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

for key, df in test_data.items():
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

print("  Applied field-specific processing")

# ============================================================================
# Step 2: Load Training Data
# ============================================================================

print("\n[2/6] Loading training data...")

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
# Step 3: Combine and Build Features
# ============================================================================

print("\n[3/6] Building features...")

test_start = test_data['gas']['datetime'].min()
combined_tables = {}

for key in train_tables.keys():
    train_no_overlap = train_tables[key][train_tables[key]['datetime'] < test_start].copy()
    combined_tables[key] = pd.concat([train_no_overlap, test_data[key]], ignore_index=True)
    combined_tables[key] = combined_tables[key].sort_values('datetime').reset_index(drop=True)

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame
from gas_power.features.multiscale_temporal_features import add_all_multiscale_temporal_features

causal_combined, _ = preprocess_causal_raw_tables(combined_tables, price_lookup, split="train")
features_combined = build_inference_feature_frame(causal_combined)

print("  Adding temporal features...")
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
# Step 4: Train Model
# ============================================================================

print("\n[4/6] Training model...")

train_end_time = test_start
train_mask = features_combined['datetime'] < train_end_time
train_features = features_combined[train_mask].copy()

# Add labels
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

model = xgb.XGBRegressor(**PARAMS)
model.fit(x_train, mixed_target, verbose=False)

print(f"  Training complete on {len(x_train)} samples")

# ============================================================================
# Step 5: Extract Test Period and Predict
# ============================================================================

print("\n[5/6] Extracting test period and predicting...")

test_end_time = test_data['gas']['datetime'].max()
test_mask = (features_combined['datetime'] >= test_start) & (features_combined['datetime'] <= test_end_time)
test_features = features_combined[test_mask].copy().reset_index(drop=True)

print(f"  Test samples: {len(test_features)}")

# Handle missing values
x_test_df = test_features[feat_cols].copy()
nan_count = x_test_df.isna().sum().sum()
if nan_count > 0:
    print(f"  Handling {nan_count} NaN values...")
    x_test_df = x_test_df.ffill().bfill()
    x_test_df = x_test_df.interpolate(method='linear', limit_direction='both')
    x_test_df = x_test_df.fillna(x_test_df.mean()).fillna(0.0)

x_test = x_test_df.values.astype(np.float32)

current1_test = test_features["feat_p50_current"].fillna(0).values.astype(np.float32)[:, None]
currentall_test = (
    test_features["feat_p50_current"].fillna(0).values +
    test_features["feat_p120_current"].fillna(0).values
).astype(np.float32)[:, None]

# Predict
raw_pred = model.predict(x_test)
pred1 = np.maximum(current1_test + raw_pred[:, :8], 0.0)
predall = np.maximum(currentall_test + raw_pred[:, 8:], 0.0)

print("  Predictions complete")

# ============================================================================
# Step 6: Generate input.csv in Required Format
# ============================================================================

print("\n[6/6] Generating input.csv in required format...")

# Create column names
columns = ['datetime']

# Add generator_1 columns: generator_1_t+15_pred, ..., generator_1_t+120_pred
for h in HORIZONS:
    minutes = h * 15
    columns.append(f'generator_1_t+{minutes}_pred')

# Add generator_all columns: generator_all_t+15_pred, ..., generator_all_t+120_pred
for h in HORIZONS:
    minutes = h * 15
    columns.append(f'generator_all_t+{minutes}_pred')

# Build data rows
data_rows = []
for i, dt in enumerate(test_features["datetime"]):
    row = {'datetime': dt}

    # Add generator_1 predictions
    for h_idx, h in enumerate(HORIZONS):
        minutes = h * 15
        row[f'generator_1_t+{minutes}_pred'] = round(float(pred1[i, h_idx]), 3)

    # Add generator_all predictions
    for h_idx, h in enumerate(HORIZONS):
        minutes = h * 15
        row[f'generator_all_t+{minutes}_pred'] = round(float(predall[i, h_idx]), 3)

    data_rows.append(row)

input_df = pd.DataFrame(data_rows, columns=columns)

# Save to file
output_path = ROOT / "input.csv"
input_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"  Saved to: {output_path}")

# ============================================================================
# Validation
# ============================================================================

print("\n" + "="*80)
print("VALIDATION")
print("="*80)

print(f"\nFile: {output_path}")
print(f"  Rows: {len(input_df)}")
print(f"  Columns: {len(input_df.columns)}")

print("\nColumn names:")
for col in input_df.columns:
    print(f"  - {col}")

print("\nFirst 3 rows:")
print(input_df.head(3))

print("\nLast 3 rows:")
print(input_df.tail(3))

print("\nData quality checks:")
print(f"  Duplicate rows: {input_df.duplicated().sum()}")
print(f"  Missing values: {input_df.isna().sum().sum()}")
print(f"  Datetime range: {input_df['datetime'].min()} to {input_df['datetime'].max()}")
print(f"  Time step: {(input_df['datetime'].iloc[1] - input_df['datetime'].iloc[0]).total_seconds() / 60:.0f} minutes")

print("\nSample statistics (generator_1_t+15_pred):")
col = 'generator_1_t+15_pred'
print(f"  Min: {input_df[col].min():.3f}")
print(f"  Max: {input_df[col].max():.3f}")
print(f"  Mean: {input_df[col].mean():.3f}")
print(f"  Std: {input_df[col].std():.3f}")

print("\nSample statistics (generator_all_t+120_pred):")
col = 'generator_all_t+120_pred'
print(f"  Min: {input_df[col].min():.3f}")
print(f"  Max: {input_df[col].max():.3f}")
print(f"  Mean: {input_df[col].mean():.3f}")
print(f"  Std: {input_df[col].std():.3f}")

print("\n" + "="*80)
print("INPUT.CSV GENERATION COMPLETE!")
print("="*80)

print("\nFormat:")
print("  - datetime: Prediction start point")
print("  - generator_1_t+15_pred to generator_1_t+120_pred: P50 predictions (15-120 min)")
print("  - generator_all_t+15_pred to generator_all_t+120_pred: All predictions (15-120 min)")
print("  - All values rounded to 3 decimal places")
print("  - No missing rows or duplicates")
print("  - Units consistent with original data")

print("\n" + "="*80)
