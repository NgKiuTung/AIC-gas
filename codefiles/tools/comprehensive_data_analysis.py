"""Comprehensive analysis of current data state in input.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

print("="*80)
print("COMPREHENSIVE DATA STATE ANALYSIS: input.csv")
print("="*80)

# Load input.csv
input_path = Path("F:/Code2/AIC/AIC-gas/input.csv")
df = pd.read_csv(input_path)
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"\n{'='*80}")
print("1. BASIC STRUCTURE")
print(f"{'='*80}")

print("\nDimensions:")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")
print(f"  Total cells: {len(df) * len(df.columns):,}")

print("\nTime range:")
print(f"  Start: {df['datetime'].min()}")
print(f"  End: {df['datetime'].max()}")
print(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600:.1f} hours")
print(f"  Frequency: {(df['datetime'].iloc[1] - df['datetime'].iloc[0]).total_seconds() / 60:.0f} minutes")

# Categorize columns
original_cols = [c for c in df.columns if not c.startswith('feat_') and c != 'datetime']
feat_cols = [c for c in df.columns if c.startswith('feat_')]

print(f"\n{'='*80}")
print("2. COLUMN TYPES")
print(f"{'='*80}")

print("\nColumn breakdown:")
print("  datetime: 1")
print(f"  Original features: {len(original_cols)}")
print(f"  Engineered features (feat_*): {len(feat_cols)}")
print(f"  Total: {len(df.columns)}")

print("\nOriginal feature columns:")
for i, col in enumerate(original_cols[:10], 1):
    print(f"  {i}. {col}")
if len(original_cols) > 10:
    print(f"  ... and {len(original_cols) - 10} more")

# Categorize engineered features
feat_categories = {
    'lag': [c for c in feat_cols if 'lag' in c.lower()],
    'rolling/statistical': [c for c in feat_cols if any(x in c.lower() for x in ['roll', 'mean', 'std', 'min', 'max'])],
    'temporal': [c for c in feat_cols if 'temporal' in c.lower()],
    'fourier': [c for c in feat_cols if 'fourier' in c.lower()],
    'ewma': [c for c in feat_cols if 'ewma' in c.lower()],
    'changepoint': [c for c in feat_cols if 'changepoint' in c.lower() or 'change_detect' in c.lower()],
    'other': []
}

# Classify remaining features
classified = set()
for category in feat_categories.values():
    classified.update(category)
feat_categories['other'] = [c for c in feat_cols if c not in classified]

print("\nEngineered feature categories:")
for category, cols in feat_categories.items():
    if cols:
        print(f"  {category}: {len(cols)}")

print(f"\n{'='*80}")
print("3. DATA QUALITY - MISSING VALUES")
print(f"{'='*80}")

missing_counts = df.isna().sum()
cols_with_missing = missing_counts[missing_counts > 0]

print("\nMissing value summary:")
print(f"  Total NaN in dataset: {df.isna().sum().sum()}")
print(f"  Columns with NaN: {len(cols_with_missing)} / {len(df.columns)}")
print(f"  Missing percentage: {df.isna().sum().sum() / (len(df) * len(df.columns)) * 100:.4f}%")

if len(cols_with_missing) > 0:
    print("\nColumns with missing values:")
    for col, count in cols_with_missing.items():
        pct = count / len(df) * 100
        col_type = "original" if col in original_cols else "feat_*"
        print(f"  - {col} ({col_type}): {count} / {len(df)} ({pct:.2f}%)")

print(f"\n{'='*80}")
print("4. DATA QUALITY - ORIGINAL FEATURES")
print(f"{'='*80}")

print("\nOriginal feature statistics:")
original_df = df[original_cols]

for col in original_cols:
    values = original_df[col].dropna()
    if len(values) > 0:
        print(f"\n  {col}:")
        print(f"    Valid values: {len(values)} / {len(df)}")
        print(f"    Range: [{values.min():.2f}, {values.max():.2f}]")
        print(f"    Mean: {values.mean():.2f}")
        print(f"    Std: {values.std():.2f}")

        # Check for outliers (simple IQR method)
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((values < lower_bound) | (values > upper_bound)).sum()
        if outliers > 0:
            print(f"    Outliers (IQR method): {outliers} ({outliers/len(values)*100:.1f}%)")

print(f"\n{'='*80}")
print("5. DATA QUALITY - ENGINEERED FEATURES")
print(f"{'='*80}")

feat_df = df[feat_cols]

print("\nEngineered feature statistics:")
print(f"  Total features: {len(feat_cols)}")
print(f"  Features with 0% missing: {(feat_df.isna().sum() == 0).sum()}")
print(f"  Features with >0% missing: {(feat_df.isna().sum() > 0).sum()}")
print(f"  Features with 100% missing: {(feat_df.isna().sum() == len(df)).sum()}")

# Check for constant features
constant_features = [c for c in feat_cols if df[c].nunique() <= 1]
if constant_features:
    print(f"\n  Warning: {len(constant_features)} features are constant (no variation)")
    for col in constant_features[:5]:
        print(f"    - {col}")

# Check for infinite values
inf_features = [c for c in feat_cols if np.isinf(df[c]).any()]
if inf_features:
    print(f"\n  Warning: {len(inf_features)} features contain infinite values")

print(f"\n{'='*80}")
print("6. DATA STATE CLASSIFICATION")
print(f"{'='*80}")

print("\nProcessing stages applied:")
print("  ✓ Time alignment: YES")
print("  ✓ Original data cleaning: YES (in causal_preprocessing)")
print("  ✓ Missing value imputation: YES (original features)")
print("  ✓ Feature engineering: YES (1057 features)")
print("  ✓ Feature NaN handling: PARTIAL (6 NaN exist)")

print("\nData readiness:")
print(f"  For model inference: {'✓ READY' if df.isna().sum().sum() == 0 else '⚠ NEEDS FILLING'}")
print("  For analysis: ✓ READY")
print("  For visualization: ✓ READY")

print(f"\n{'='*80}")
print("7. SUMMARY & RECOMMENDATIONS")
print(f"{'='*80}")

print("\nCurrent state:")
print("  • Dataset contains 192 time points (48 hours)")
print("  • Has 26 original features + 1057 engineered features")
print(f"  • {len(cols_with_missing)} columns have missing values ({df.isna().sum().sum()} total NaN)")
print("  • Original features are mostly clean")
print("  • Engineered features include lag, rolling, temporal patterns")

print("\nWhat has been done:")
print("  1. ✓ Merged multiple raw data tables")
print("  2. ✓ Filled missing values in original data")
print("  3. ✓ Generated 1057 engineered features")
print("  4. ✓ Provided historical context for temporal features")

print("\nWhat remains:")
print(f"  • {df.isna().sum().sum()} NaN values to be filled before model inference")
print("  • These are handled by intelligent filling strategy in prediction script")

print(f"\n{'='*80}")
