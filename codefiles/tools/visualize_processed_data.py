"""Visualize what the processed data looks like at each stage.

Shows:
1. Raw test data
2. After missing value handling
3. After merging with training for context
4. After feature engineering
5. Final test features ready for model
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("VISUALIZING PROCESSED DATA AT EACH STAGE")
print("="*80)

TEST_DIR = Path("F:/Code2/AIC/初赛-评分所用测试集")
RAW_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")

SHORT_GAP_THRESHOLD = 4

# ============================================================================
# Stage 1: Raw Test Data
# ============================================================================

print("\n" + "="*80)
print("STAGE 1: RAW TEST DATA (ORIGINAL)")
print("="*80)

test_gas = pd.read_csv(TEST_DIR / "Pre_test_gas.csv", encoding="utf-8-sig")
test_gas['datetime'] = pd.to_datetime(test_gas['datetime'])

print(f"\nShape: {test_gas.shape}")
print(f"Columns: {list(test_gas.columns)}")

print("\nFirst 5 rows:")
print(test_gas.head())

print("\nMissing values:")
missing = test_gas.isna().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        print(f"  {col}: {count} / {len(test_gas)} ({count/len(test_gas)*100:.2f}%)")
else:
    print("  None")

print("\nData types:")
print(test_gas.dtypes)

# ============================================================================
# Stage 2: After Missing Value Handling
# ============================================================================

print("\n" + "="*80)
print("STAGE 2: AFTER FIELD-SPECIFIC MISSING VALUE HANDLING")
print("="*80)

# Load and process
test_data = {
    "gas": pd.read_csv(TEST_DIR / "Pre_test_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(TEST_DIR / "Pre_test_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(TEST_DIR / "Pre_test_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(TEST_DIR / "Pre_test_load.csv", encoding="utf-8-sig")
}

for df in test_data.values():
    df['datetime'] = pd.to_datetime(df['datetime'])

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
            if col not in df.columns:
                continue
            if df[col].isna().sum() == 0:
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

processed_gas = test_data['gas']

print(f"\nShape: {processed_gas.shape}")

print("\nMissing values after processing:")
missing = processed_gas.isna().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        print(f"  {col}: {count}")
else:
    print("  None - All missing values handled!")

print("\nSample rows (showing key columns):")
display_cols = ['datetime', 'generator_1', 'generator_all', 'blast_furnace_gas_generation']
print(processed_gas[display_cols].head(10))

# ============================================================================
# Stage 3: After Merging with Training Data
# ============================================================================

print("\n" + "="*80)
print("STAGE 3: AFTER MERGING WITH TRAINING DATA FOR CONTEXT")
print("="*80)

train_gas = pd.read_csv(RAW_DIR / "Pre_gas.csv", encoding="utf-8-sig")
train_gas['datetime'] = pd.to_datetime(train_gas['datetime'])

test_start = test_data['gas']['datetime'].min()
train_no_overlap = train_gas[train_gas['datetime'] < test_start].copy()

combined = pd.concat([train_no_overlap, test_data['gas']], ignore_index=True)
combined = combined.sort_values('datetime').reset_index(drop=True)

print(f"\nCombined shape: {combined.shape}")
print(f"  Training rows: {len(train_no_overlap)}")
print(f"  Test rows: {len(test_data['gas'])}")
print(f"  Total: {len(combined)}")

print("\nTime range:")
print(f"  Start: {combined['datetime'].min()}")
print(f"  End: {combined['datetime'].max()}")
print(f"  Training ends: {train_no_overlap['datetime'].max()}")
print(f"  Test starts: {test_data['gas']['datetime'].min()}")

print("\nLast 5 rows of training + first 5 rows of test:")
boundary_idx = len(train_no_overlap)
print(combined.iloc[boundary_idx-5:boundary_idx+5][['datetime', 'generator_1', 'generator_all']])

# ============================================================================
# Stage 4: After Feature Engineering
# ============================================================================

print("\n" + "="*80)
print("STAGE 4: AFTER FEATURE ENGINEERING")
print("="*80)

print("\nBuilding features (this may take a moment)...")

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame
from gas_power.features.multiscale_temporal_features import add_all_multiscale_temporal_features

# Load all tables
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

# Process test data
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

# Combine
test_start = test_data['gas']['datetime'].min()
combined_tables = {}
for key in train_tables.keys():
    train_no_overlap = train_tables[key][train_tables[key]['datetime'] < test_start].copy()
    combined_tables[key] = pd.concat([train_no_overlap, test_data[key]], ignore_index=True)
    combined_tables[key] = combined_tables[key].sort_values('datetime').reset_index(drop=True)

# Build features
causal_combined, _ = preprocess_causal_raw_tables(combined_tables, price_lookup, split="train")
features_combined = build_inference_feature_frame(causal_combined)
features_combined = add_all_multiscale_temporal_features(
    features_combined,
    add_rolling=True,
    add_ewma=True,
    add_trend=False,
    add_fourier=True,
    add_autocorr=False,
    add_changepoint=True
)

print(f"\nFeatures shape: {features_combined.shape}")

# Categorize columns
feat_cols = [c for c in features_combined.columns if c.startswith('feat_')]
original_cols = [c for c in features_combined.columns if not c.startswith('feat_') and c != 'datetime']

print("\nColumn breakdown:")
print("  datetime: 1")
print(f"  Original features: {len(original_cols)}")
print(f"  Engineered features (feat_*): {len(feat_cols)}")
print(f"  Total: {len(features_combined.columns)}")

print("\nSample original features:")
for col in original_cols[:10]:
    print(f"  - {col}")

print("\nSample engineered features (first 20):")
for col in feat_cols[:20]:
    print(f"  - {col}")

print("\nFeature categories:")
feat_categories = {
    'lag': len([c for c in feat_cols if 'lag' in c.lower()]),
    'rolling': len([c for c in feat_cols if any(x in c.lower() for x in ['roll', 'mean', 'std'])]),
    'temporal': len([c for c in feat_cols if 'temporal' in c.lower()]),
    'fourier': len([c for c in feat_cols if 'fourier' in c.lower()]),
    'ewma': len([c for c in feat_cols if 'ewma' in c.lower()]),
    'changepoint': len([c for c in feat_cols if 'changepoint' in c.lower() or 'change_detect' in c.lower()]),
}
for category, count in feat_categories.items():
    print(f"  {category}: {count}")

# ============================================================================
# Stage 5: Final Test Features
# ============================================================================

print("\n" + "="*80)
print("STAGE 5: FINAL TEST FEATURES (READY FOR MODEL)")
print("="*80)

test_end_time = test_data['gas']['datetime'].max()
test_mask = (features_combined['datetime'] >= test_start) & (features_combined['datetime'] <= test_end_time)
test_features = features_combined[test_mask].copy().reset_index(drop=True)

print(f"\nFinal test features shape: {test_features.shape}")
print(f"  Rows: {len(test_features)} (test period only)")
print(f"  Columns: {len(test_features.columns)}")

print("\nTime range:")
print(f"  Start: {test_features['datetime'].min()}")
print(f"  End: {test_features['datetime'].max()}")

print("\nMissing values in test features:")
missing = test_features.isna().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(f"  Total NaN: {missing.sum()}")
    print(f"  Columns with NaN: {len(missing_cols)}")
    for col, count in list(missing_cols.items())[:10]:
        print(f"    - {col}: {count} / {len(test_features)} ({count/len(test_features)*100:.2f}%)")
else:
    print("  None - All features complete!")

print("\nSample of final test features (first 5 rows, key columns):")
display_cols = ['datetime'] + feat_cols[:5]
print(test_features[display_cols].head())

print("\nStatistics of a sample feature:")
sample_feat = feat_cols[0]
print(f"  Feature: {sample_feat}")
print(f"  Min: {test_features[sample_feat].min():.4f}")
print(f"  Max: {test_features[sample_feat].max():.4f}")
print(f"  Mean: {test_features[sample_feat].mean():.4f}")
print(f"  Std: {test_features[sample_feat].std():.4f}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("DATA TRANSFORMATION SUMMARY")
print("="*80)

print("\nData journey:")
print(f"  1. Raw test data:        {test_gas.shape} -> {test_gas.isna().sum().sum()} NaN")
print(f"  2. After cleaning:       {processed_gas.shape} -> {processed_gas.isna().sum().sum()} NaN")
print(f"  3. Merged with train:    {combined.shape}")
print(f"  4. After features:       {features_combined.shape}")
print(f"  5. Final test features:  {test_features.shape} -> {test_features.isna().sum().sum()} NaN")

print("\nKey transformations:")
print("  - Field-specific missing value handling (interpolation, forward fill)")
print("  - Merged with training data to provide historical context")
print("  - Generated 1057 engineered features (lag, rolling, temporal, etc.)")
print("  - Extracted test period with complete features")
print("  - Result: Clean, feature-rich dataset ready for model prediction")

print("\n" + "="*80)
