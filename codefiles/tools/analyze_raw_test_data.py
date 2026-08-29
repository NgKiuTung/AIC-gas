"""Analyze raw test set data and determine what data processing is needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

print("="*80)
print("RAW TEST SET DATA ANALYSIS")
print("="*80)

TEST_DIR = Path("F:/Code2/AIC/初赛-评分所用测试集")

# Load all test files
print("\n[1] Loading raw test files...")

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
    print(f"  {key}: {len(df)} rows, {len(df.columns)} columns")

print("\n" + "="*80)
print("[2] DATA STRUCTURE ANALYSIS")
print("="*80)

for key, df in test_data.items():
    print(f"\n{key.upper()} TABLE:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600:.1f} hours")
    print(f"  Time step: {(df['datetime'].iloc[1] - df['datetime'].iloc[0]).total_seconds() / 60:.0f} minutes")

print("\n" + "="*80)
print("[3] DATA QUALITY ISSUES")
print("="*80)

for key, df in test_data.items():
    print(f"\n{key.upper()} TABLE:")

    # Missing values
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]

    if len(missing_cols) > 0:
        print("  Missing values:")
        for col, count in missing_cols.items():
            if col != 'datetime':
                pct = count / len(df) * 100
                print(f"    - {col}: {count} / {len(df)} ({pct:.2f}%)")
    else:
        print("  Missing values: None")

    # Check for duplicates
    dup_count = df.duplicated(subset=['datetime']).sum()
    if dup_count > 0:
        print(f"  Duplicate timestamps: {dup_count}")
    else:
        print("  Duplicate timestamps: None")

    # Check numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("  Numeric columns statistics:")
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"    {col}:")
                print(f"      Range: [{values.min():.2f}, {values.max():.2f}]")
                print(f"      Mean: {values.mean():.2f}, Std: {values.std():.2f}")

                # Check for negative values (if shouldn't be negative)
                if (values < 0).any():
                    neg_count = (values < 0).sum()
                    print(f"      WARNING: {neg_count} negative values")

                # Check for zeros
                zero_count = (values == 0).sum()
                if zero_count > 0:
                    print(f"      Zeros: {zero_count} ({zero_count/len(values)*100:.1f}%)")

print("\n" + "="*80)
print("[4] REQUIRED DATA PROCESSING STEPS")
print("="*80)

print("\nBased on analysis, the following processing is needed:")

print("\n1. TIME ALIGNMENT:")
print("   - Merge all 4 tables on 'datetime'")
print("   - Ensure consistent time intervals (15 minutes)")
print("   - Handle any time gaps or overlaps")

print("\n2. MISSING VALUE HANDLING:")
print("   Strategy depends on missing patterns:")

for key, df in test_data.items():
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n   {key.upper()}:")
        for col, count in missing_cols.items():
            if col != 'datetime':
                pct = count / len(df) * 100
                if pct < 5:
                    print(f"     - {col} ({pct:.1f}%): Forward fill or interpolation")
                elif pct < 50:
                    print(f"     - {col} ({pct:.1f}%): Use historical mean or model-based imputation")
                else:
                    print(f"     - {col} ({pct:.1f}%): Consider dropping or special handling")

print("\n3. OUTLIER DETECTION & HANDLING:")
print("   - Check physical constraints (e.g., generator_1: 0-120 MW)")
print("   - Detect statistical outliers (IQR or Z-score method)")
print("   - Options: Cap, remove, or flag for review")

print("\n4. FEATURE ENGINEERING PREPARATION:")
print("   - For lag features: Need historical data from training set")
print("   - For rolling features: Need sufficient window (e.g., 7 days = 672 rows)")
print("   - For temporal features: Need to append training data as context")

print("\n5. DATA VALIDATION:")
print("   - Check value ranges are reasonable")
print("   - Ensure no data leakage")
print("   - Verify consistency across tables")

print("\n" + "="*80)
print("[5] PROPOSED PROCESSING PIPELINE")
print("="*80)

print("\nStep-by-step processing pipeline:")
print("\n  [Raw Test Data (192 rows)]")
print("        |")
print("        v")
print("  [1. Load & Parse]")
print("        - Read CSV files")
print("        - Convert datetime")
print("        - Basic validation")
print("        |")
print("        v")
print("  [2. Time Alignment]")
print("        - Merge 4 tables on datetime")
print("        - Check for gaps")
print("        |")
print("        v")
print("  [3. Missing Value Handling]")
print("        - Forward fill")
print("        - Linear interpolation")
print("        - Column mean (if needed)")
print("        |")
print("        v")
print("  [4. Outlier Detection]")
print("        - Physical constraints")
print("        - Statistical methods (IQR)")
print("        |")
print("        v")
print("  [5. Outlier Handling]")
print("        - Cap to valid ranges")
print("        - Or smooth with neighbors")
print("        |")
print("        v")
print("  [6. Append Training History]")
print("        - Add last N rows from training")
print("        - Provide context for features")
print("        |")
print("        v")
print("  [7. Feature Engineering]")
print("        - Lag features (1-672 steps)")
print("        - Rolling statistics (multiple windows)")
print("        - Temporal patterns (EWMA, Fourier, etc.)")
print("        |")
print("        v")
print("  [8. Extract Test Period]")
print("        - Remove training rows")
print("        - Keep only test 192 rows")
print("        |")
print("        v")
print("  [9. Final Validation]")
print("        - Check no NaN in critical features")
print("        - Verify feature alignment")
print("        |")
print("        v")
print("  [Clean Test Features (ready for model)]")

print("\n" + "="*80)
print("[6] SUMMARY")
print("="*80)

total_missing = sum(df.isna().sum().sum() for df in test_data.values())
total_cells = sum(len(df) * len(df.columns) for df in test_data.values())

print("\nCurrent raw test data state:")
print("  - Total rows: 192 (across 4 tables)")
print(f"  - Total missing values: {total_missing}")
print(f"  - Missing rate: {total_missing / total_cells * 100:.4f}%")

print("\nData processing complexity:")
print("  - Basic cleaning: SIMPLE (few missing values)")
print("  - Feature engineering: COMPLEX (needs historical context)")
print("  - Overall: MEDIUM")

print("\nKey challenge:")
print("  The test set is only 192 rows (48 hours), but many features")
print("  require weeks of historical data. Solution: Append training data")
print("  before feature engineering, then extract test period.")

print("\n" + "="*80)
