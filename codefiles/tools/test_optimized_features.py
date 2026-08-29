"""Simplified optimized training test."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("Testing optimized features...")

# Load data
RAW_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")
tables = {
    "gas": pd.read_csv(RAW_DIR / "Pre_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(RAW_DIR / "Pre_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(RAW_DIR / "Pre_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(RAW_DIR / "Pre_load.csv", encoding="utf-8-sig"),
}
for df in tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

price_table = pd.read_excel(RAW_DIR / "price.xlsx")
price_lookup = {}
for row_idx, row in price_table.iterrows():
    for month in range(1, 13):
        price_lookup[(month, row_idx)] = float(row[f"{month}月"])

print(f"Data loaded: {len(tables['gas'])} rows")

# Preprocess
from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables

causal, _ = preprocess_causal_raw_tables(tables, price_lookup, split="train")
print(f"Preprocessed: {causal.shape}")

# Build base features
from gas_power.features.inference import build_inference_feature_frame

base = build_inference_feature_frame(causal)
print(f"Base features: {base.shape}")

# Add future prices
from gas_power.features.enhanced_interactions import add_future_price_features

with_prices = add_future_price_features(base, price_lookup)
print(f"With future prices: {with_prices.shape}")

# Test optimized features
print("\nTesting optimized features...")
from gas_power.features.optimized_features import build_optimized_features

try:
    optimized = build_optimized_features(
        with_prices,
        add_target_lags=True,
        fix_outliers=False,  # Already in causal preprocessing
        add_domain=True,
        add_physical=True,
        add_time=True
    )
    print(f"Optimized features: {optimized.shape}")

    # Count new features
    new_cols = [c for c in optimized.columns if c not in with_prices.columns]
    print(f"New features added: {len(new_cols)}")

    # Show some examples
    print("\nExample new features:")
    for col in sorted(new_cols)[:10]:
        print(f"  {col}")

    print("\nSUCCESS! All optimizations working.")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
