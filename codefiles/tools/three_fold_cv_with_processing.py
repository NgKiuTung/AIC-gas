"""3-Fold Cross Validation to analyze prediction quality with field-specific data processing.

This script:
1. Applies field-specific missing value handling
2. Performs 3-fold time-series split
3. Trains model on each fold
4. Evaluates MAPE
5. Reports average performance
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("3-FOLD CROSS VALIDATION WITH FIELD-SPECIFIC DATA PROCESSING")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

HORIZONS = tuple(range(1, 9))
N_FOLDS = 3
SHORT_GAP_THRESHOLD = 4
LONG_GAP_THRESHOLD = 16

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

# ============================================================================
# Load and Process Data
# ============================================================================

print("\n[1/4] Loading training data...")

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

print(f"  Training data: {len(tables['gas'])} rows")

# Apply field-specific missing value handling
print("\n[2/4] Applying field-specific missing value handling...")

field_categories = {
    "气柜": ['blast_furnace_gas_holder_1', 'blast_furnace_gas_holder_2',
            'coke_oven_gas_holder', 'converter_gas_holder'],
    "流量": ['blast_furnace_gas_generation', 'coke_oven_gas_generation', 'converter_gas_generation',
            'hot_blast_stove_blast_furnace_gas', 'hot_blast_stove_coke_oven_gas',
            'hot_blast_stove_converter_gas', 'lime_kiln_blast_furnace_gas',
            'lime_kiln_coke_oven_gas', 'raw_material_blast_furnace_gas',
            'raw_material_coke_oven_gas', 'heating_furnace_blast_furnace_gas',
            'heating_furnace_coke_oven_gas', 'heating_furnace_converter_gas',
            'power_generation_blast_furnace_gas', 'power_generation_coke_oven_gas',
            'power_generation_converter_gas'],
    "发电负荷": ['generator_1', 'generator_all', 'load']
}

for _key, df in tables.items():
    for category, cols in field_categories.items():
        for col in cols:
            if col not in df.columns:
                continue

            original_missing = df[col].isna().sum()
            if original_missing == 0:
                continue

            if category == "气柜":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both')
                df[col] = df[col].ffill().bfill()
            elif category == "流量":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD * 2, limit_direction='both')
                df[col] = df[col].ffill().bfill().fillna(0.0)
            elif category == "发电负荷":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both')
                df[col] = df[col].ffill().bfill()

print("  Field-specific strategies applied")

# Build features
print("\n[3/4] Building features...")

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

# Add labels
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
print(f"  Total samples: {len(features)}")

# ============================================================================
# 3-Fold Cross Validation
# ============================================================================

print("\n[4/4] Running 3-fold cross validation...")

# Time-series split: divide into 3 consecutive folds
n_samples = len(features)
fold_size = n_samples // N_FOLDS

fold_results = []

for fold_idx in range(N_FOLDS):
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'='*80}")

    # Define train/val split
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < N_FOLDS - 1 else n_samples

    train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
    val_indices = list(range(val_start, val_end))

    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples: {len(val_indices)}")

    # Prepare training data
    train_data = features.iloc[train_indices]

    x_train = train_data[feat_cols].values.astype(np.float32)
    y1_train = train_data[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
    y120_train = train_data[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)
    yall_train = y1_train + y120_train

    current1_train = train_data["feat_p50_current"].values.astype(np.float32)[:, None]
    currentall_train = (
        train_data["feat_p50_current"].values +
        train_data["feat_p120_current"].values
    ).astype(np.float32)[:, None]

    mixed_target_train = np.concatenate([
        y1_train - current1_train,
        yall_train - currentall_train,
    ], axis=1)

    # Prepare validation data
    val_data = features.iloc[val_indices]

    x_val = val_data[feat_cols].values.astype(np.float32)
    y1_val = val_data[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
    y120_val = val_data[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)
    yall_val = y1_val + y120_val

    current1_val = val_data["feat_p50_current"].values.astype(np.float32)[:, None]
    currentall_val = (
        val_data["feat_p50_current"].values +
        val_data["feat_p120_current"].values
    ).astype(np.float32)[:, None]

    # Train model
    print("\n  Training model...")
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(x_train, mixed_target_train, verbose=False)

    # Predict
    print("  Predicting...")
    raw_pred = model.predict(x_val)
    pred1 = np.maximum(current1_val + raw_pred[:, :8], 0.0)
    predall = np.maximum(currentall_val + raw_pred[:, 8:], 0.0)

    # Calculate MAPE
    def mape(y_true, y_pred):
        mask = y_true > 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    mape_p50 = mape(y1_val.flatten(), pred1.flatten())
    mape_all = mape(yall_val.flatten(), predall.flatten())
    mape_avg = (mape_p50 + mape_all) / 2

    print("\n  MAPE Results:")
    print(f"    P50 (generator_1):  {mape_p50:.4f}%")
    print(f"    All (generator_all): {mape_all:.4f}%")
    print(f"    Average:            {mape_avg:.4f}%")

    fold_results.append({
        'fold': fold_idx + 1,
        'mape_p50': mape_p50,
        'mape_all': mape_all,
        'mape_avg': mape_avg,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices)
    })

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("CROSS VALIDATION SUMMARY")
print("="*80)

results_df = pd.DataFrame(fold_results)

print("\nPer-fold results:")
print(f"{'Fold':<8} {'P50 MAPE':<12} {'All MAPE':<12} {'Avg MAPE':<12} {'Train':<10} {'Val':<10}")
print("-" * 70)

for _, row in results_df.iterrows():
    print(f"{int(row['fold']):<8} {row['mape_p50']:<12.4f} {row['mape_all']:<12.4f} "
          f"{row['mape_avg']:<12.4f} {int(row['train_samples']):<10} {int(row['val_samples']):<10}")

print("-" * 70)
print(f"{'Mean':<8} {results_df['mape_p50'].mean():<12.4f} {results_df['mape_all'].mean():<12.4f} "
      f"{results_df['mape_avg'].mean():<12.4f}")
print(f"{'Std':<8} {results_df['mape_p50'].std():<12.4f} {results_df['mape_all'].std():<12.4f} "
      f"{results_df['mape_avg'].std():<12.4f}")

print("\n" + "="*80)
print("FINAL PERFORMANCE ESTIMATE")
print("="*80)

print("\nWith field-specific data processing:")
print(f"  Average MAPE: {results_df['mape_avg'].mean():.4f}% +/- {results_df['mape_avg'].std():.4f}%")
print(f"  P50 MAPE:     {results_df['mape_p50'].mean():.4f}% +/- {results_df['mape_p50'].std():.4f}%")
print(f"  All MAPE:     {results_df['mape_all'].mean():.4f}% +/- {results_df['mape_all'].std():.4f}%")

print("\nThis is the expected performance on unseen test data.")

print("\n" + "="*80)
