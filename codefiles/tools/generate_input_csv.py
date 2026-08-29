"""Generate input.csv for model inference as required by competition rules.

The input.csv must contain:
1. datetime column
2. Original features (exact field names as specified)
3. Engineered features (with 'feat_' prefix)
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("GENERATING INPUT.CSV FOR MODEL INFERENCE")
print("="*80)

# ============================================================================
# Load Test Data
# ============================================================================

print("\n[1/4] Loading test data...")

TEST_DIR = Path("F:/Code2/AIC/初赛-评分所用测试集")

test_tables = {
    "gas": pd.read_csv(TEST_DIR / "Pre_test_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(TEST_DIR / "Pre_test_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(TEST_DIR / "Pre_test_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(TEST_DIR / "Pre_test_load.csv", encoding="utf-8-sig"),
}

for df in test_tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

RAW_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")
price_table = pd.read_excel(RAW_DIR / "price.xlsx")
price_lookup = {}
for row_idx, row in price_table.iterrows():
    for month in range(1, 13):
        price_lookup[(month, row_idx)] = float(row[f"{month}月"])

print(f"  Test data: {len(test_tables['gas'])} rows")

# ============================================================================
# Build Features
# ============================================================================

print("\n[2/4] Building features...")

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame
from gas_power.features.multiscale_temporal_features import add_all_multiscale_temporal_features

print("  Preprocessing test data...")
causal_test, _ = preprocess_causal_raw_tables(test_tables, price_lookup, split="test")

print("  Building inference features...")
test_features = build_inference_feature_frame(causal_test)

print("  Adding multiscale temporal features...")
test_features = add_all_multiscale_temporal_features(
    test_features,
    add_rolling=True,
    add_ewma=True,
    add_trend=False,
    add_fourier=True,
    add_autocorr=False,
    add_changepoint=True
)

# ============================================================================
# Prepare input.csv
# ============================================================================

print("\n[3/4] Preparing input.csv...")

# Get original feature columns (from raw data)
original_cols = []

# From gas table
gas_original = ['generator_1', 'generator_all', 'blast_furnace_gas_generation',
                'coke_oven_gas_generation', 'converter_gas_generation']

# From holder table
holder_original = ['blast_furnace_gas_holder_1', 'blast_furnace_gas_holder_2',
                   'coke_oven_gas_holder', 'converter_gas_holder']

# From user table
user_original = ['hot_blast_stove_blast_furnace_gas', 'hot_blast_stove_coke_oven_gas',
                 'hot_blast_stove_converter_gas', 'lime_kiln_blast_furnace_gas',
                 'lime_kiln_coke_oven_gas', 'raw_material_blast_furnace_gas',
                 'raw_material_coke_oven_gas', 'heating_furnace_blast_furnace_gas',
                 'heating_furnace_coke_oven_gas', 'heating_furnace_converter_gas',
                 'power_generation_blast_furnace_gas', 'power_generation_coke_oven_gas',
                 'power_generation_converter_gas']

# From load table
load_original = ['load']

# All original columns
original_cols = (['datetime'] + gas_original + holder_original +
                 user_original + load_original)

# Get engineered feature columns (with 'feat_' prefix)
feat_cols = [c for c in test_features.columns if c.startswith('feat_')]

print(f"  Original columns: {len(original_cols) - 1}")  # -1 for datetime
print(f"  Engineered features: {len(feat_cols)}")

# Merge original data with features
# Start with causal (which has original columns)
input_df = causal_test[['datetime'] + [c for c in original_cols[1:] if c in causal_test.columns]].copy()

# Add all engineered features
for col in feat_cols:
    if col in test_features.columns:
        input_df[col] = test_features[col].values

print(f"  Total columns in input.csv: {len(input_df.columns)}")
print("    - datetime: 1")
print(f"    - original features: {len([c for c in input_df.columns if not c.startswith('feat_') and c != 'datetime'])}")
print(f"    - feat_* features: {len([c for c in input_df.columns if c.startswith('feat_')])}")

# ============================================================================
# Save input.csv
# ============================================================================

print("\n[4/4] Saving input.csv...")

output_path = ROOT / "input.csv"
input_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"  Saved to: {output_path}")
print(f"  Rows: {len(input_df)}")
print(f"  Columns: {len(input_df.columns)}")

# Show sample
print("\n  First 5 rows (first 10 columns):")
print(input_df.iloc[:5, :10])

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("INPUT.CSV GENERATION COMPLETE!")
print("="*80)

print("\nFile structure:")
print(f"  Location: {output_path}")
print(f"  Rows: {len(input_df)}")
print(f"  Total columns: {len(input_df.columns)}")
print("    - datetime: 1")
print(f"    - Original features: {len([c for c in input_df.columns if not c.startswith('feat_') and c != 'datetime'])}")
print(f"    - Engineered features (feat_*): {len([c for c in input_df.columns if c.startswith('feat_')])}")

print("\nThis file can be used as model input according to competition rules.")
print("\n" + "="*80)
