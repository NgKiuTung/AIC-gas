"""Convert input.csv to result.csv format.

Input format:
  datetime, generator_1_t+15_pred, ..., generator_1_t+120_pred, generator_all_t+15_pred, ..., generator_all_t+120_pred

Result format:
  datetime, horizon, generator_1, generator_all
"""

from pathlib import Path

import pandas as pd

print("="*80)
print("CONVERTING INPUT.CSV TO RESULT.CSV")
print("="*80)

# Load input.csv
input_path = Path("F:/Code2/AIC/AIC-gas/input.csv")
input_df = pd.read_csv(input_path)
input_df['datetime'] = pd.to_datetime(input_df['datetime'])

print(f"\nInput file: {input_path}")
print(f"  Rows: {len(input_df)}")
print(f"  Columns: {len(input_df.columns)}")

# Convert to result format
result_rows = []

for _, row in input_df.iterrows():
    datetime = row['datetime']

    # Extract predictions for each horizon
    for h in range(1, 9):  # horizons 1-8
        minutes = h * 15

        generator_1_pred = row[f'generator_1_t+{minutes}_pred']
        generator_all_pred = row[f'generator_all_t+{minutes}_pred']

        result_rows.append({
            'datetime': datetime,
            'horizon': h,
            'generator_1': generator_1_pred,
            'generator_all': generator_all_pred
        })

result_df = pd.DataFrame(result_rows)

# Save to result.csv
output_path = Path("F:/Code2/AIC/AIC-gas/result.csv")
result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nOutput file: {output_path}")
print(f"  Rows: {len(result_df)}")
print(f"  Columns: {len(result_df.columns)}")

# Validation
print("\n" + "="*80)
print("VALIDATION")
print("="*80)

print("\nFirst 10 rows of result.csv:")
print(result_df.head(10))

print("\nLast 10 rows of result.csv:")
print(result_df.tail(10))

print("\nData quality:")
print(f"  Expected rows: {len(input_df)} samples × 8 horizons = {len(input_df) * 8}")
print(f"  Actual rows: {len(result_df)}")
print(f"  Match: {'YES' if len(result_df) == len(input_df) * 8 else 'NO'}")

print("\nMissing values:")
print(f"  datetime: {result_df['datetime'].isna().sum()}")
print(f"  horizon: {result_df['horizon'].isna().sum()}")
print(f"  generator_1: {result_df['generator_1'].isna().sum()}")
print(f"  generator_all: {result_df['generator_all'].isna().sum()}")

print("\nHorizon distribution:")
print(result_df['horizon'].value_counts().sort_index())

print("\n" + "="*80)
print("CONVERSION COMPLETE!")
print("="*80)

print("\nYou can now submit result.csv!")

print("\n" + "="*80)
