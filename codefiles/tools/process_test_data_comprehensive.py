"""Comprehensive test data processing with field-specific missing value handling.

Analyzes missing values by:
1. Source (which field)
2. Length (short vs long gaps)
3. Field type (uses different strategies)
4. Adds missing indicators
"""

from pathlib import Path

import numpy as np
import pandas as pd

print("="*80)
print("TEST DATA PROCESSING WITH FIELD-SPECIFIC MISSING VALUE HANDLING")
print("="*80)

TEST_DIR = Path("F:/Code2/AIC/初赛-评分所用测试集")

# ============================================================================
# Step 1: Load Raw Test Data
# ============================================================================

print("\n[1] Loading raw test data...")

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
    print(f"  Loaded {key}: {len(df)} rows, {len(df.columns)} columns")

# ============================================================================
# Step 2: Merge Tables
# ============================================================================

print("\n[2] Merging tables on datetime...")

merged = test_data['gas'].copy()
for key in ['holder', 'user', 'load']:
    merged = merged.merge(test_data[key], on='datetime', how='outer', suffixes=('', f'_{key}'))

merged = merged.sort_values('datetime').reset_index(drop=True)
print(f"  Merged data: {len(merged)} rows, {len(merged.columns)} columns")

# ============================================================================
# Step 3: Analyze Missing Values by Field Type
# ============================================================================

print("\n" + "="*80)
print("[3] MISSING VALUE ANALYSIS BY FIELD TYPE")
print("="*80)

# Define field categories
field_categories = {
    "气柜柜位/压力": [
        'blast_furnace_gas_holder_1',
        'blast_furnace_gas_holder_2',
        'coke_oven_gas_holder',
        'converter_gas_holder'
    ],
    "流量": [
        'blast_furnace_gas_generation',
        'coke_oven_gas_generation',
        'converter_gas_generation',
        'hot_blast_stove_blast_furnace_gas',
        'hot_blast_stove_coke_oven_gas',
        'hot_blast_stove_converter_gas',
        'lime_kiln_blast_furnace_gas',
        'lime_kiln_coke_oven_gas',
        'raw_material_blast_furnace_gas',
        'raw_material_coke_oven_gas',
        'heating_furnace_blast_furnace_gas',
        'heating_furnace_coke_oven_gas',
        'heating_furnace_converter_gas',
        'power_generation_blast_furnace_gas',
        'power_generation_coke_oven_gas',
        'power_generation_converter_gas'
    ],
    "发电负荷": [
        'generator_1',
        'generator_all',
        'load'
    ],
    "其他": []
}

# Classify all columns
all_numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
classified_cols = set()
for category_cols in field_categories.values():
    classified_cols.update(category_cols)
field_categories["其他"] = [c for c in all_numeric_cols if c not in classified_cols]

print("\nField categorization:")
for category, cols in field_categories.items():
    print(f"  {category}: {len(cols)} fields")

# Analyze each field's missing pattern
missing_analysis = []

for category, cols in field_categories.items():
    for col in cols:
        if col not in merged.columns:
            continue

        missing_mask = merged[col].isna()
        total_missing = missing_mask.sum()

        if total_missing == 0:
            continue

        # Analyze missing gap lengths
        gaps = []
        current_gap = 0
        for is_missing in missing_mask:
            if is_missing:
                current_gap += 1
            else:
                if current_gap > 0:
                    gaps.append(current_gap)
                    current_gap = 0
        if current_gap > 0:
            gaps.append(current_gap)

        missing_analysis.append({
            'field': col,
            'category': category,
            'total_missing': total_missing,
            'missing_rate': total_missing / len(merged) * 100,
            'num_gaps': len(gaps),
            'max_gap': max(gaps) if gaps else 0,
            'avg_gap': np.mean(gaps) if gaps else 0,
            'gaps': gaps
        })

print(f"\n{len(missing_analysis)} fields have missing values:")
print(f"\n{'Field':<45} {'Category':<15} {'Missing':<10} {'Max Gap':<10} {'Avg Gap':<10}")
print("-" * 90)

for item in sorted(missing_analysis, key=lambda x: x['missing_rate'], reverse=True):
    print(f"{item['field']:<45} {item['category']:<15} "
          f"{item['total_missing']:>3} ({item['missing_rate']:>5.1f}%) "
          f"{item['max_gap']:>9.0f} {item['avg_gap']:>9.1f}")

# ============================================================================
# Step 4: Apply Field-Specific Handling Strategies
# ============================================================================

print("\n" + "="*80)
print("[4] APPLYING FIELD-SPECIFIC MISSING VALUE STRATEGIES")
print("="*80)

processed = merged.copy()

# Define thresholds
SHORT_GAP_THRESHOLD = 4  # <= 4 consecutive missing (1 hour)
LONG_GAP_THRESHOLD = 16   # > 16 consecutive missing (4 hours)

print("\nGap definitions:")
print(f"  Short gap: <= {SHORT_GAP_THRESHOLD} consecutive missing ({SHORT_GAP_THRESHOLD * 15} minutes)")
print(f"  Medium gap: {SHORT_GAP_THRESHOLD + 1} to {LONG_GAP_THRESHOLD} consecutive")
print(f"  Long gap: > {LONG_GAP_THRESHOLD} consecutive ({LONG_GAP_THRESHOLD * 15 / 60:.1f} hours)")

# Strategy 1: 气柜柜位/压力 - Linear interpolation for short gaps, forward fill for others
print("\n[Strategy 1] 气柜柜位/压力:")
print("  Short gaps (<= 1h): Linear interpolation")
print("  Longer gaps: Forward fill")

for col in field_categories["气柜柜位/压力"]:
    if col not in processed.columns:
        continue

    original_missing = processed[col].isna().sum()
    if original_missing == 0:
        continue

    # Add missing indicator
    processed[f'missing_flag_{col}'] = processed[col].isna().astype(int)

    # Interpolate short gaps
    processed[col] = processed[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both')

    # Forward fill remaining
    processed[col] = processed[col].fillna(method='ffill')

    # Backward fill if still missing at start
    processed[col] = processed[col].fillna(method='bfill')

    filled = original_missing - processed[col].isna().sum()
    print(f"  {col}: Filled {filled}/{original_missing}")

# Strategy 2: 流量 - Interpolation for short gaps, cautious for long gaps
print("\n[Strategy 2] 流量:")
print("  Short gaps (<= 1h): Linear interpolation")
print("  Medium gaps: Forward fill")
print("  Long gaps: Mark and use 0 or median")

for col in field_categories["流量"]:
    if col not in processed.columns:
        continue

    original_missing = processed[col].isna().sum()
    if original_missing == 0:
        continue

    # Add missing indicator
    processed[f'missing_flag_{col}'] = processed[col].isna().astype(int)

    # Identify long gaps
    missing_mask = processed[col].isna()
    long_gap_mask = pd.Series([False] * len(processed))

    current_gap = 0
    gap_start = 0
    for i, is_missing in enumerate(missing_mask):
        if is_missing:
            if current_gap == 0:
                gap_start = i
            current_gap += 1
        else:
            if current_gap > LONG_GAP_THRESHOLD:
                long_gap_mask.iloc[gap_start:i] = True
            current_gap = 0

    # Handle long gaps with 0 (flow could genuinely stop)
    if long_gap_mask.sum() > 0:
        processed.loc[long_gap_mask, col] = 0.0
        print(f"  {col}: Filled {long_gap_mask.sum()} long-gap values with 0")

    # Interpolate short/medium gaps
    processed[col] = processed[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD * 2, limit_direction='both')

    # Forward fill remaining
    processed[col] = processed[col].fillna(method='ffill')
    processed[col] = processed[col].fillna(method='bfill')

    filled = original_missing - processed[col].isna().sum()
    print(f"  {col}: Total filled {filled}/{original_missing}")

# Strategy 3: 发电负荷 - Forward fill for short gaps, could be 0 during shutdown
print("\n[Strategy 3] 发电负荷:")
print("  Short gaps: Forward fill or local interpolation")
print("  During potential shutdown: Keep as 0 or interpolate carefully")

for col in field_categories["发电负荷"]:
    if col not in processed.columns:
        continue

    original_missing = processed[col].isna().sum()
    if original_missing == 0:
        continue

    # Add missing indicator
    processed[f'missing_flag_{col}'] = processed[col].isna().astype(int)

    # Check if surrounding values are low (potential shutdown)
    for i in processed[processed[col].isna()].index:
        neighbors = processed[col].iloc[max(0, i-4):min(len(processed), i+5)]
        if neighbors.dropna().mean() < 10:  # Very low load
            processed.loc[i, col] = 0.0  # Likely shutdown

    # Interpolate remaining short gaps
    processed[col] = processed[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both')

    # Forward fill
    processed[col] = processed[col].fillna(method='ffill')
    processed[col] = processed[col].fillna(method='bfill')

    filled = original_missing - processed[col].isna().sum()
    print(f"  {col}: Filled {filled}/{original_missing}")

# Strategy 4: 其他 - Generic forward fill + interpolation
print("\n[Strategy 4] 其他字段:")
print("  Generic strategy: Interpolation + forward/backward fill")

for col in field_categories["其他"]:
    if col not in processed.columns:
        continue

    original_missing = processed[col].isna().sum()
    if original_missing == 0:
        continue

    # Add missing indicator
    processed[f'missing_flag_{col}'] = processed[col].isna().astype(int)

    # Interpolate
    processed[col] = processed[col].interpolate(method='linear', limit_direction='both')

    # Fill remaining
    processed[col] = processed[col].fillna(method='ffill')
    processed[col] = processed[col].fillna(method='bfill')

    # Last resort: median
    if processed[col].isna().sum() > 0:
        processed[col] = processed[col].fillna(processed[col].median())

    filled = original_missing - processed[col].isna().sum()
    if filled > 0:
        print(f"  {col}: Filled {filled}/{original_missing}")

# ============================================================================
# Step 5: Summary & Validation
# ============================================================================

print("\n" + "="*80)
print("[5] PROCESSING SUMMARY")
print("="*80)

print("\nBefore processing:")
print(f"  Total missing values: {merged.isna().sum().sum()}")
print(f"  Columns with missing: {(merged.isna().sum() > 0).sum()}")

print("\nAfter processing:")
print(f"  Total missing values: {processed.isna().sum().sum()}")
print(f"  Columns with missing: {(processed.isna().sum() > 0).sum()}")

# Count missing indicators
missing_flag_cols = [c for c in processed.columns if c.startswith('missing_flag_')]
print(f"\nMissing indicators added: {len(missing_flag_cols)} columns")

# Show remaining missing values
remaining_missing = processed.isna().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

if len(remaining_missing) > 0:
    print("\nRemaining missing values:")
    for col, count in remaining_missing.items():
        print(f"  {col}: {count}")
else:
    print("\n✓ All missing values handled!")

# Save processed data
output_path = Path("F:/Code2/AIC/AIC-gas/processed_test_data.csv")
processed.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\nProcessed data saved to: {output_path}")
print(f"  Rows: {len(processed)}")
print(f"  Columns: {len(processed.columns)} (original) + {len(missing_flag_cols)} (missing flags)")

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)

print("\nNext steps:")
print("  1. ✓ Raw test data loaded")
print("  2. ✓ Missing values analyzed by field type")
print("  3. ✓ Field-specific strategies applied")
print("  4. ✓ Missing indicators added")
print("  5. [ ] Append training history for lag/rolling features")
print("  6. [ ] Feature engineering")
print("  7. [ ] Extract test period")
print("  8. [ ] Ready for model inference")

print("\n" + "="*80)
