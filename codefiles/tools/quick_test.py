"""Simplified version - test with minimal features first."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("Starting simplified test...")
print(f"Python: {sys.executable}")
print(f"Working directory: {Path.cwd()}")

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
    from gas_power.features.inference import build_inference_feature_frame
    from gas_power.features.enhanced_interactions import add_future_price_features, add_interaction_features
    print("  [OK] All imports successful")
except Exception as e:
    print(f"  [FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Load raw data
print("\n[2/5] Loading raw data...")
try:
    RAW_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")
    tables = {
        "gas": pd.read_csv(RAW_DIR / "Pre_gas.csv", encoding="utf-8-sig"),
        "holder": pd.read_csv(RAW_DIR / "Pre_gas_holder.csv", encoding="utf-8-sig"),
        "user": pd.read_csv(RAW_DIR / "Pre_gas_user.csv", encoding="utf-8-sig"),
        "load": pd.read_csv(RAW_DIR / "Pre_load.csv", encoding="utf-8-sig"),
    }
    for name, df in tables.items():
        df["datetime"] = pd.to_datetime(df["datetime"])
        print(f"  {name}: {len(df)} rows")

    price_table = pd.read_excel(RAW_DIR / "price.xlsx")
    price_lookup = {}
    for row_idx, row in price_table.iterrows():
        for month in range(1, 13):
            price_lookup[(month, row_idx)] = float(row[f"{month}月"])
    print(f"  price: {len(price_lookup)} entries")
    print("  [OK] Data loaded")
except Exception as e:
    print(f"  [FAIL] Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Preprocess
print("\n[3/5] Preprocessing...")
try:
    causal, audit = preprocess_causal_raw_tables(tables, price_lookup, split="train")
    print(f"  Shape: {causal.shape}")
    print(f"  Columns: {len(causal.columns)}")
    print("  [OK] Preprocessing successful")
except Exception as e:
    print(f"  [FAIL] Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Build features
print("\n[4/5] Building features...")
try:
    features = build_inference_feature_frame(causal)
    print(f"  Shape: {features.shape}")

    # Add labels
    HORIZONS = tuple(range(1, 9))
    for h in HORIZONS:
        features[f"label_p50_h{h}"] = features["feat_generator_1_filled"].shift(-h)

    features = features.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])
    print(f"  After adding labels: {features.shape}")
    print("  [OK] Features built")
except Exception as e:
    print(f"  [FAIL] Feature building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Add enhanced features
print("\n[5/5] Adding enhanced features...")
try:
    enhanced = add_future_price_features(features, price_lookup)
    print(f"  After future prices: {enhanced.shape}")

    enhanced = add_interaction_features(enhanced)
    print(f"  After interactions: {enhanced.shape}")
    print("  [OK] Enhanced features added")
except Exception as e:
    print(f"  [FAIL] Enhancement failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print(f"\nFinal feature count: {enhanced.shape[1]}")
print(f"Ready for training with {len(enhanced)} samples")
