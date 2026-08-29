"""Full 3-fold cross-validation with multiscale temporal features.

Tests the model on all 3 validation folds to get reliable MAPE estimate.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("3-FOLD CROSS-VALIDATION WITH MULTISCALE TEMPORAL FEATURES")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

HORIZONS = tuple(range(1, 9))

PARAMS = {
    "n_estimators": 350,
    "max_depth": 5,
    "learning_rate": 0.025,
    "min_child_weight": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.70,
    "reg_alpha": 2.0,
    "reg_lambda": 30.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 20260803,
    "n_jobs": -1,
    "verbosity": 0,
}

RAW_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")

# Define 3 folds
FOLDS = [
    ("Fold 1", pd.Timestamp("2025-03-15"), pd.Timestamp("2025-03-31 23:45:00")),
    ("Fold 2", pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-15 23:45:00")),
    ("Fold 3", pd.Timestamp("2025-04-16"), pd.Timestamp("2025-04-30 23:45:00")),
]

# ============================================================================
# Load Data
# ============================================================================

print("\n[1/4] Loading data...")

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

print(f"  Data loaded: {len(tables['gas'])} rows")

# ============================================================================
# Build Features
# ============================================================================

print("\n[2/4] Building features with multiscale temporal...")

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame
from gas_power.features.multiscale_temporal_features import add_all_multiscale_temporal_features

causal, _ = preprocess_causal_raw_tables(tables, price_lookup, split="train")
features = build_inference_feature_frame(causal)

print("  Adding multiscale temporal features...")
features = add_all_multiscale_temporal_features(
    features,
    add_rolling=True,
    add_ewma=True,
    add_trend=False,
    add_fourier=True,
    add_autocorr=False,
    add_changepoint=True
)

# Add labels with smoothing
for h in HORIZONS:
    raw_label_p50 = features["feat_generator_1_filled"].shift(-h)
    raw_label_p120 = (
        features["feat_generator_all_filled"] -
        features["feat_generator_1_filled"]
    ).shift(-h)

    features[f"label_p50_h{h}"] = raw_label_p50.rolling(3, center=True, min_periods=1).mean()
    features[f"label_p120_h{h}"] = raw_label_p120.rolling(3, center=True, min_periods=1).mean()

features = features.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])

feat_cols = [c for c in features.columns if c.startswith("feat_")]
print(f"  Total features: {len(feat_cols)}")

# ============================================================================
# Train and Validate on Each Fold
# ============================================================================

print("\n[3/4] Training and validating on 3 folds...")

fold_results = []

for fold_name, fold_start, fold_end in FOLDS:
    print(f"\n  {fold_name}: {fold_start.date()} to {fold_end.date()}")

    # Split data
    train = features[features["datetime"] < fold_start - pd.Timedelta(minutes=120)]
    valid = features[(features["datetime"] >= fold_start) & (features["datetime"] <= fold_end)]

    print(f"    Train samples: {len(train)}, Valid samples: {len(valid)}")

    # Extract features
    x_train = train[feat_cols].values.astype(np.float32)
    x_valid = valid[feat_cols].values.astype(np.float32)

    y1_train = train[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
    y120_train = train[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)
    y1_valid = valid[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
    y120_valid = valid[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)

    yall_train = y1_train + y120_train
    yall_valid = y1_valid + y120_valid

    current1_train = train["feat_p50_current"].values.astype(np.float32)[:, None]
    currentall_train = (train["feat_p50_current"].values + train["feat_p120_current"].values).astype(np.float32)[:, None]
    current1_valid = valid["feat_p50_current"].values.astype(np.float32)[:, None]
    currentall_valid = (valid["feat_p50_current"].values + valid["feat_p120_current"].values).astype(np.float32)[:, None]

    # Target transformation: predict absolute change
    mixed_target = np.concatenate([
        y1_train - current1_train,
        yall_train - currentall_train,
    ], axis=1)

    # Train model
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(x_train, mixed_target, verbose=False)

    # Predict
    raw_pred = model.predict(x_valid)
    pred1 = np.maximum(current1_valid + raw_pred[:, :8], 0.0)
    predall = np.maximum(currentall_valid + raw_pred[:, 8:], 0.0)

    # Calculate MAPE
    mape1 = float(np.mean(np.abs(y1_valid - pred1) / np.maximum(np.abs(y1_valid), 1e-8)) * 100.0)
    mapeall = float(np.mean(np.abs(yall_valid - predall) / np.maximum(np.abs(yall_valid), 1e-8)) * 100.0)
    mape_avg = (mape1 + mapeall) / 2.0

    print(f"    P50 MAPE:  {mape1:.4f}%")
    print(f"    All MAPE:  {mapeall:.4f}%")
    print(f"    Avg MAPE:  {mape_avg:.4f}%")

    fold_results.append({
        "fold": fold_name,
        "mape_p50": mape1,
        "mape_all": mapeall,
        "mape_avg": mape_avg
    })

# ============================================================================
# Results Summary
# ============================================================================

print("\n" + "="*80)
print("[4/4] FINAL RESULTS")
print("="*80)

results_df = pd.DataFrame(fold_results)

print("\nPer-Fold Results:")
print(results_df.to_string(index=False))

print("\n" + "-"*80)
print("Average Across 3 Folds:")
print(f"  P50 MAPE:  {results_df['mape_p50'].mean():.4f}% (±{results_df['mape_p50'].std():.4f})")
print(f"  All MAPE:  {results_df['mape_all'].mean():.4f}% (±{results_df['mape_all'].std():.4f})")
print(f"  Avg MAPE:  {results_df['mape_avg'].mean():.4f}% (±{results_df['mape_avg'].std():.4f})")

print("\n" + "="*80)
print("VALIDATION COMPLETE!")
print("="*80)

print(f"\nFinal estimated MAPE: {results_df['mape_avg'].mean():.4f}%")
print("\nNote: This is cross-validation on the training data.")
print("The actual test set MAPE may differ slightly.")
