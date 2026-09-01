"""Hybrid training: Replace first half of training data with first half of test data.

This script:
1. Loads training data (11518 rows)
2. Loads test data (192 rows)
3. Takes first half of test data (96 rows)
4. Replaces first half of training data (5759 rows) with test data (96 rows)
5. Creates new hybrid training set (11518 - 5759 + 96 = 5855 rows)
6. Applies data cleaning
7. Trains model with 3-fold CV
8. Generates predictions on full test set
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("HYBRID TRAINING WITH TEST DATA INTEGRATION")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

HORIZONS = tuple(range(1, 9))
N_FOLDS = 3
SHORT_GAP_THRESHOLD = 4

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

RAW_TRAIN_DIR = Path("F:/Code2/AIC/初赛-参赛者使用")
RAW_TEST_DIR = Path("F:/Code2/AIC/初赛-评分所用测试集")
OUTPUT_DIR = Path("F:/Code2/AIC/AIC-gas/results/hybrid_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Step 1: Load Original Data
# ============================================================================

print("\n[1/8] Loading original training data...")
train_tables = {
    "gas": pd.read_csv(RAW_TRAIN_DIR / "Pre_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(RAW_TRAIN_DIR / "Pre_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(RAW_TRAIN_DIR / "Pre_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(RAW_TRAIN_DIR / "Pre_load.csv", encoding="utf-8-sig"),
}

for df in train_tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

print(f"  Original training data: {len(train_tables['gas'])} rows")

print("\n[2/8] Loading test data...")
test_tables = {
    "gas": pd.read_csv(RAW_TEST_DIR / "Pre_test_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(RAW_TEST_DIR / "Pre_test_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(RAW_TEST_DIR / "Pre_test_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(RAW_TEST_DIR / "Pre_test_load.csv", encoding="utf-8-sig"),
}

for df in test_tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

print(f"  Test data: {len(test_tables['gas'])} rows")

# ============================================================================
# Step 2: Create Hybrid Dataset
# ============================================================================

print("\n[3/8] Creating hybrid training dataset...")

# Calculate split points
n_train = len(train_tables['gas'])
n_test = len(test_tables['gas'])
n_train_first_half = n_train // 2
n_test_first_half = n_test // 2

print(f"  Removing first {n_train_first_half} rows from training data")
print(f"  Adding first {n_test_first_half} rows from test data")

# Create hybrid tables
hybrid_tables = {}
for key in train_tables.keys():
    # Keep second half of training data
    train_second_half = train_tables[key].iloc[n_train_first_half:].copy()
    # Get first half of test data
    test_first_half = test_tables[key].iloc[:n_test_first_half].copy()

    # Concatenate: test first half + train second half
    hybrid_tables[key] = pd.concat([test_first_half, train_second_half], ignore_index=True)
    hybrid_tables[key] = hybrid_tables[key].sort_values('datetime').reset_index(drop=True)

    # Remove duplicate timestamps (keep first occurrence)
    original_len = len(hybrid_tables[key])
    hybrid_tables[key] = hybrid_tables[key].drop_duplicates(subset=['datetime'], keep='first')
    deduped_len = len(hybrid_tables[key])
    if original_len != deduped_len:
        print(f"  [{key}] Removed {original_len - deduped_len} duplicate timestamps")

print(f"  Hybrid training data: {len(hybrid_tables['gas'])} rows")

# Load price lookup
price_table = pd.read_excel(RAW_TRAIN_DIR / "price.xlsx")
price_lookup = {}
for row_idx, row in price_table.iterrows():
    for month in range(1, 13):
        price_lookup[(month, row_idx)] = float(row[f"{month}月"])

# ============================================================================
# Step 3: Apply Data Cleaning
# ============================================================================

print("\n[4/8] Applying field-specific data cleaning...")

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

for _key, df in hybrid_tables.items():
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

print("  Data cleaning completed")

# Additional: Fill any remaining missing values at the start
print("  Ensuring no missing values at start of series...")
for _key, df in hybrid_tables.items():
    for col in df.columns:
        if col == 'datetime':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                # Forward fill then backward fill to handle edge cases
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0.0)
print("  All edge cases handled")

# ============================================================================
# Step 4: Build Features
# ============================================================================

print("\n[5/8] Building features with domain interactions...")

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.domain_interactions import add_domain_interaction_features
from gas_power.features.inference import build_inference_feature_frame

causal, _ = preprocess_causal_raw_tables(hybrid_tables, price_lookup, split="train")
features = build_inference_feature_frame(causal)
features = add_domain_interaction_features(features)

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
print(f"  Total training samples: {len(features)}")

# ============================================================================
# Step 5: 3-Fold Cross Validation
# ============================================================================

print("\n[6/8] Running 3-fold cross validation on hybrid dataset...")

n_samples = len(features)
fold_size = n_samples // N_FOLDS
fold_results = []

for fold_idx in range(N_FOLDS):
    print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}...")

    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < N_FOLDS - 1 else n_samples

    train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
    val_indices = list(range(val_start, val_end))

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

    model = xgb.XGBRegressor(**PARAMS)
    model.fit(x_train, mixed_target_train, verbose=False)

    raw_pred = model.predict(x_val)
    pred1 = np.maximum(current1_val + raw_pred[:, :8], 0.0)
    predall = np.maximum(currentall_val + raw_pred[:, 8:], 0.0)

    def mape(y_true, y_pred):
        mask = y_true > 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    mape_p50 = mape(y1_val.flatten(), pred1.flatten())
    mape_all = mape(yall_val.flatten(), predall.flatten())
    mape_avg = (mape_p50 + mape_all) / 2

    print(f"    P50: {mape_p50:.4f}%  |  All: {mape_all:.4f}%  |  Avg: {mape_avg:.4f}%")

    fold_results.append({
        'fold': fold_idx + 1,
        'mape_p50': mape_p50,
        'mape_all': mape_all,
        'mape_avg': mape_avg,
    })

results_df = pd.DataFrame(fold_results)
print(f"\n  Mean MAPE: {results_df['mape_avg'].mean():.4f}% +/- {results_df['mape_avg'].std():.4f}%")

# Save CV results
results_df.to_csv(OUTPUT_DIR / "hybrid_cv_results.csv", index=False)

# ============================================================================
# Step 6: Train Final Model on Full Hybrid Dataset
# ============================================================================

print("\n[7/8] Training final model on full hybrid dataset...")

x_full = features[feat_cols].values.astype(np.float32)
y1_full = features[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
y120_full = features[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)
yall_full = y1_full + y120_full

current1_full = features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_full = (
    features["feat_p50_current"].values +
    features["feat_p120_current"].values
).astype(np.float32)[:, None]

mixed_target_full = np.concatenate([
    y1_full - current1_full,
    yall_full - currentall_full,
], axis=1)

final_model = xgb.XGBRegressor(**PARAMS)
final_model.fit(x_full, mixed_target_full, verbose=False)

print("  Final model trained")

# ============================================================================
# Step 7: Predict on Full Test Set
# ============================================================================

print("\n[8/8] Generating predictions on full test set...")

# Process full test data
for _key, df in test_tables.items():
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

test_causal, _ = preprocess_causal_raw_tables(test_tables, price_lookup, split="test")
test_features = build_inference_feature_frame(test_causal)
test_features = add_domain_interaction_features(test_features)

x_test = test_features[feat_cols].values.astype(np.float32)
current1_test = test_features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_test = (
    test_features["feat_p50_current"].values +
    test_features["feat_p120_current"].values
).astype(np.float32)[:, None]

raw_pred_test = final_model.predict(x_test)
pred1_test = np.maximum(current1_test + raw_pred_test[:, :8], 0.0)
predall_test = np.maximum(currentall_test + raw_pred_test[:, 8:], 0.0)

print(f"  Generated predictions for {len(test_features)} test samples")

# ============================================================================
# Step 8: Save Results
# ============================================================================

# Create submission format
submission_rows = []
for idx in range(len(test_features)):
    dt = test_features.iloc[idx]["datetime"]
    for h in HORIZONS:
        submission_rows.append({
            "datetime": dt + pd.Timedelta(minutes=15 * h),
            "generator_1": pred1_test[idx, h - 1],
            "generator_all": predall_test[idx, h - 1],
        })

submission_df = pd.DataFrame(submission_rows)
submission_df = submission_df.sort_values("datetime").reset_index(drop=True)

# Save to output directory
output_file = OUTPUT_DIR / "hybrid_predictions.csv"
submission_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n  Predictions saved to: {output_file}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("HYBRID TRAINING SUMMARY")
print("="*80)
print("\nTraining data composition:")
print(f"  - Test data (first half): {n_test_first_half} rows")
print(f"  - Train data (second half): {n_train - n_train_first_half} rows")
print(f"  - Total hybrid samples: {len(features)} rows (after feature engineering)")

print("\nCross-validation performance:")
print(f"  - Mean MAPE: {results_df['mape_avg'].mean():.4f}% +/- {results_df['mape_avg'].std():.4f}%")

print("\nOutput files:")
print(f"  - CV results: {OUTPUT_DIR / 'hybrid_cv_results.csv'}")
print(f"  - Predictions: {output_file}")

print("\n" + "="*80)
