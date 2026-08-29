"""Compare predictions in input.csv and result.csv to verify they match.
"""

from pathlib import Path

import pandas as pd

print("="*80)
print("COMPARING INPUT.CSV AND RESULT.CSV")
print("="*80)

# Load both files
input_path = Path("F:/Code2/AIC/AIC-gas/input.csv")
result_path = Path("F:/Code2/AIC/AIC-gas/result.csv")

input_df = pd.read_csv(input_path)
input_df['datetime'] = pd.to_datetime(input_df['datetime'])

result_df = pd.read_csv(result_path)
result_df['datetime'] = pd.to_datetime(result_df['datetime'])

print("\nInput.csv:")
print(f"  Shape: {input_df.shape}")
print(f"  Columns: {list(input_df.columns)}")

print("\nResult.csv:")
print(f"  Shape: {result_df.shape}")
print(f"  Columns: {list(result_df.columns)}")

# Convert input.csv to result.csv format for comparison
print("\n" + "="*80)
print("CONVERTING INPUT.CSV TO RESULT FORMAT FOR COMPARISON")
print("="*80)

input_converted = []

for _, row in input_df.iterrows():
    datetime = row['datetime']

    for h in range(1, 9):
        minutes = h * 15

        input_converted.append({
            'datetime': datetime,
            'horizon': h,
            'generator_1': row[f'generator_1_t+{minutes}_pred'],
            'generator_all': row[f'generator_all_t+{minutes}_pred']
        })

input_converted_df = pd.DataFrame(input_converted)

print(f"\nConverted input.csv shape: {input_converted_df.shape}")

# Sort both for comparison
input_converted_df = input_converted_df.sort_values(['datetime', 'horizon']).reset_index(drop=True)
result_df = result_df.sort_values(['datetime', 'horizon']).reset_index(drop=True)

# Compare
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print(f"\nRow count match: {len(input_converted_df)} vs {len(result_df)}")
print(f"  Match: {'YES' if len(input_converted_df) == len(result_df) else 'NO'}")

# Compare datetime and horizon
datetime_match = (input_converted_df['datetime'] == result_df['datetime']).all()
horizon_match = (input_converted_df['horizon'] == result_df['horizon']).all()

print(f"\nDatetime match: {'YES' if datetime_match else 'NO'}")
print(f"Horizon match: {'YES' if horizon_match else 'NO'}")

# Compare predictions
print("\n" + "-"*80)
print("PREDICTION VALUE COMPARISON")
print("-"*80)

# Generator_1
diff_g1 = input_converted_df['generator_1'] - result_df['generator_1']
max_diff_g1 = diff_g1.abs().max()
mean_diff_g1 = diff_g1.abs().mean()

print("\nGenerator_1:")
print(f"  Max absolute difference: {max_diff_g1:.6f}")
print(f"  Mean absolute difference: {mean_diff_g1:.6f}")
print(f"  Identical (exact match): {'YES' if max_diff_g1 == 0 else 'NO'}")

# Generator_all
diff_gall = input_converted_df['generator_all'] - result_df['generator_all']
max_diff_gall = diff_gall.abs().max()
mean_diff_gall = diff_gall.abs().mean()

print("\nGenerator_all:")
print(f"  Max absolute difference: {max_diff_gall:.6f}")
print(f"  Mean absolute difference: {mean_diff_gall:.6f}")
print(f"  Identical (exact match): {'YES' if max_diff_gall == 0 else 'NO'}")

# Show some examples where differences exist (if any)
if max_diff_g1 > 0 or max_diff_gall > 0:
    print("\n" + "-"*80)
    print("SAMPLE DIFFERENCES")
    print("-"*80)

    comparison_df = pd.DataFrame({
        'datetime': input_converted_df['datetime'],
        'horizon': input_converted_df['horizon'],
        'input_g1': input_converted_df['generator_1'],
        'result_g1': result_df['generator_1'],
        'diff_g1': diff_g1,
        'input_gall': input_converted_df['generator_all'],
        'result_gall': result_df['generator_all'],
        'diff_gall': diff_gall
    })

    # Show rows with largest differences
    comparison_df['abs_diff_total'] = diff_g1.abs() + diff_gall.abs()
    top_diffs = comparison_df.nlargest(10, 'abs_diff_total')

    print("\nTop 10 rows with largest differences:")
    for _, row in top_diffs.iterrows():
        print(f"\n  {row['datetime']} horizon={row['horizon']}")
        print(f"    Generator_1:  input={row['input_g1']:.3f}, result={row['result_g1']:.3f}, diff={row['diff_g1']:.3f}")
        print(f"    Generator_all: input={row['input_gall']:.3f}, result={row['result_gall']:.3f}, diff={row['diff_gall']:.3f}")

# Sample comparison
print("\n" + "-"*80)
print("SAMPLE ROWS (first 5)")
print("-"*80)

comparison_sample = pd.DataFrame({
    'datetime': input_converted_df['datetime'].head(),
    'horizon': input_converted_df['horizon'].head(),
    'input_g1': input_converted_df['generator_1'].head(),
    'result_g1': result_df['generator_1'].head(),
    'input_gall': input_converted_df['generator_all'].head(),
    'result_gall': result_df['generator_all'].head()
})

print(comparison_sample.to_string(index=False))

# Final verdict
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if max_diff_g1 == 0 and max_diff_gall == 0:
    print("\n✓ IDENTICAL: input.csv and result.csv contain EXACTLY the same predictions!")
elif max_diff_g1 < 0.01 and max_diff_gall < 0.01:
    print("\n≈ NEARLY IDENTICAL: Differences are negligible (< 0.01), likely due to rounding.")
else:
    print("\n✗ DIFFERENT: input.csv and result.csv contain DIFFERENT predictions!")
    print(f"  Max difference in generator_1: {max_diff_g1:.6f}")
    print(f"  Max difference in generator_all: {max_diff_gall:.6f}")

print("\n" + "="*80)
