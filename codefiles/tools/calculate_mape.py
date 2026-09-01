"""Calculate MAPE between predictions and actual test values."""

import numpy as np
import pandas as pd
from pathlib import Path

print("="*80)
print("MAPE EVALUATION")
print("="*80)

# Load predictions (result.csv)
print("\n[1/3] Loading predictions...")
result_file = Path("F:/Code2/AIC/result.csv")
predictions = pd.read_csv(result_file, encoding="utf-8-sig")
predictions['datetime'] = pd.to_datetime(predictions['datetime'])
print(f"  Loaded {len(predictions)} predictions")

# Load actual test values
print("\n[2/3] Loading actual test values...")
test_file = Path("F:/Code2/AIC/初赛-评分所用测试集/Pre_test_load.csv")
actuals = pd.read_csv(test_file, encoding="utf-8-sig")
actuals['datetime'] = pd.to_datetime(actuals['datetime'])
print(f"  Loaded {len(actuals)} actual values")

# Merge predictions with actuals
print("\n[3/3] Calculating MAPE...")
merged = predictions.merge(
    actuals[['datetime', 'generator_1', 'generator_all']],
    on='datetime',
    how='inner',
    suffixes=('_pred', '_actual')
)

print(f"  Matched {len(merged)} timestamps")

if len(merged) == 0:
    print("\n[ERROR] No matching timestamps found!")
    print("Predictions datetime range:", predictions['datetime'].min(), "to", predictions['datetime'].max())
    print("Actuals datetime range:", actuals['datetime'].min(), "to", actuals['datetime'].max())
else:
    # Calculate MAPE
    def mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true > 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Calculate for generator_1
    mape_gen1 = mape(merged['generator_1_actual'], merged['generator_1_pred'])

    # Calculate for generator_all
    mape_genall = mape(merged['generator_all_actual'], merged['generator_all_pred'])

    # Calculate average MAPE
    mape_avg = (mape_gen1 + mape_genall) / 2

    # Calculate MAE for reference
    mae_gen1 = np.mean(np.abs(merged['generator_1_actual'] - merged['generator_1_pred']))
    mae_genall = np.mean(np.abs(merged['generator_all_actual'] - merged['generator_all_pred']))

    # Print results
    print("\n" + "="*80)
    print("MAPE RESULTS")
    print("="*80)
    print(f"\nGenerator 1 (P50):")
    print(f"  MAPE: {mape_gen1:.4f}%")
    print(f"  MAE:  {mae_gen1:.4f}")

    print(f"\nGenerator All:")
    print(f"  MAPE: {mape_genall:.4f}%")
    print(f"  MAE:  {mae_genall:.4f}")

    print(f"\nAverage MAPE: {mape_avg:.4f}%")

    # Show some sample comparisons
    print("\n" + "="*80)
    print("SAMPLE COMPARISONS (First 10 predictions)")
    print("="*80)
    print(f"\n{'Datetime':<20} {'Gen1_Actual':>12} {'Gen1_Pred':>12} {'GenAll_Actual':>14} {'GenAll_Pred':>14}")
    print("-" * 84)

    for idx in range(min(10, len(merged))):
        row = merged.iloc[idx]
        print(f"{str(row['datetime']):<20} {row['generator_1_actual']:>12.2f} {row['generator_1_pred']:>12.2f} "
              f"{row['generator_all_actual']:>14.2f} {row['generator_all_pred']:>14.2f}")

    # Statistical summary
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)

    print(f"\nGenerator 1:")
    print(f"  Actual - Mean: {merged['generator_1_actual'].mean():.2f}, Std: {merged['generator_1_actual'].std():.2f}")
    print(f"  Predicted - Mean: {merged['generator_1_pred'].mean():.2f}, Std: {merged['generator_1_pred'].std():.2f}")

    print(f"\nGenerator All:")
    print(f"  Actual - Mean: {merged['generator_all_actual'].mean():.2f}, Std: {merged['generator_all_actual'].std():.2f}")
    print(f"  Predicted - Mean: {merged['generator_all_pred'].mean():.2f}, Std: {merged['generator_all_pred'].std():.2f}")

    # Save detailed comparison
    comparison_file = Path("F:/Code2/AIC/mape_comparison.csv")
    comparison_df = merged[['datetime', 'generator_1_actual', 'generator_1_pred',
                            'generator_all_actual', 'generator_all_pred']].copy()
    comparison_df['gen1_error'] = np.abs(comparison_df['generator_1_actual'] - comparison_df['generator_1_pred'])
    comparison_df['genall_error'] = np.abs(comparison_df['generator_all_actual'] - comparison_df['generator_all_pred'])
    comparison_df['gen1_pct_error'] = np.abs((comparison_df['generator_1_actual'] - comparison_df['generator_1_pred']) / comparison_df['generator_1_actual']) * 100
    comparison_df['genall_pct_error'] = np.abs((comparison_df['generator_all_actual'] - comparison_df['generator_all_pred']) / comparison_df['generator_all_actual']) * 100

    comparison_df.to_csv(comparison_file, index=False, encoding="utf-8-sig")
    print(f"\nDetailed comparison saved to: {comparison_file}")

print("\n" + "="*80)
