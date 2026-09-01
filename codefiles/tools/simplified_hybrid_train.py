"""Simplified hybrid training: use quick_cv_test approach with hybrid dataset."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

print("="*80)
print("SIMPLIFIED HYBRID TRAINING AND PREDICTION")
print("="*80)

# Configuration
HORIZONS = tuple(range(1, 9))
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

HYBRID_DIR = Path("F:/Code2/AIC/初赛-混合训练集")
TEST_DIR = Path("F:/Code2/AIC/初赛-评分所用测试集")

print("\n[1/5] Loading hybrid training data...")
train_tables = {
    "gas": pd.read_csv(HYBRID_DIR / "Pre_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(HYBRID_DIR / "Pre_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(HYBRID_DIR / "Pre_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(HYBRID_DIR / "Pre_load.csv", encoding="utf-8-sig"),
}

for df in train_tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

print(f"  Loaded {len(train_tables['gas'])} rows")

price_table = pd.read_excel(HYBRID_DIR / "price.xlsx")
price_lookup = {}
for row_idx, row in price_table.iterrows():
    for month in range(1, 13):
        price_lookup[(month, row_idx)] = float(row[f"{month}月"])

print("\n[2/5] Applying data cleaning and building features...")

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

for _key, df in train_tables.items():
    for category, cols in field_categories.items():
        for col in cols:
            if col not in df.columns or df[col].isna().sum() == 0:
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

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.domain_interactions import add_domain_interaction_features
from gas_power.features.inference import build_inference_feature_frame

try:
    causal, _ = preprocess_causal_raw_tables(train_tables, price_lookup, split="train")
    features = build_inference_feature_frame(causal)
    features = add_domain_interaction_features(features)
except Exception as e:
    print(f"  Error using full preprocessing: {e}")
    print("  Falling back to simplified feature extraction...")

    # Simplified feature building - merge tables manually
    merged = train_tables['gas'].copy()
    for key in ['holder', 'user', 'load']:
        merged = merged.merge(train_tables[key], on='datetime', how='left')

    # Basic features
    features = pd.DataFrame()
    features['datetime'] = merged['datetime']
    features['feat_generator_1_filled'] = merged['generator_1']
    features['feat_generator_all_filled'] = merged['generator_all']
    features['feat_p50_current'] = merged['generator_1']
    features['feat_p120_current'] = merged['generator_all'] - merged['generator_1']
    features['feat_pall_current'] = merged['generator_all']

    # Add some key columns
    key_cols = ['blast_furnace_gas_holder_2', 'blast_furnace_gas_generation',
                'coke_oven_gas_generation', 'converter_gas_generation',
                'power_generation_blast_furnace_gas', 'power_generation_coke_oven_gas',
                'power_generation_converter_gas', 'load']

    for col in key_cols:
        if col in merged.columns:
            features[f'feat_{col}'] = merged[col]

    # Time features
    features['feat_hour'] = features['datetime'].dt.hour
    features['feat_dayofweek'] = features['datetime'].dt.dayofweek
    features['feat_month'] = features['datetime'].dt.month
    features['feat_is_weekend'] = (features['datetime'].dt.dayofweek >= 5).astype(float)

    # Price feature
    features['feat_known_price'] = features.apply(
        lambda row: price_lookup[(row['datetime'].month, row['datetime'].hour * 2)],
        axis=1
    )

# Add labels
for h in HORIZONS:
    raw_label_p50 = features["feat_generator_1_filled"].shift(-h)
    raw_label_p120 = (features["feat_generator_all_filled"] - features["feat_generator_1_filled"]).shift(-h)
    features[f"label_p50_h{h}"] = raw_label_p50.rolling(3, center=True, min_periods=1).mean()
    features[f"label_p120_h{h}"] = raw_label_p120.rolling(3, center=True, min_periods=1).mean()

features = features.dropna(subset=[f"label_p50_h{HORIZONS[-1]}"])

feat_cols = [c for c in features.columns if c.startswith("feat_")]
print(f"  Features: {len(feat_cols)}, Samples: {len(features)}")

print("\n[3/5] Training model on hybrid dataset...")

x_full = features[feat_cols].values.astype(np.float32)
y1_full = features[[f"label_p50_h{h}" for h in HORIZONS]].values.astype(np.float32)
y120_full = features[[f"label_p120_h{h}" for h in HORIZONS]].values.astype(np.float32)
yall_full = y1_full + y120_full

current1_full = features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_full = (features["feat_p50_current"].values + features["feat_p120_current"].values).astype(np.float32)[:, None]

mixed_target_full = np.concatenate([y1_full - current1_full, yall_full - currentall_full], axis=1)

final_model = xgb.XGBRegressor(**PARAMS)
final_model.fit(x_full, mixed_target_full, verbose=False)
print("  Model trained")

print("\n[4/5] Loading and processing test data...")

test_tables = {
    "gas": pd.read_csv(TEST_DIR / "Pre_test_gas.csv", encoding="utf-8-sig"),
    "holder": pd.read_csv(TEST_DIR / "Pre_test_gas_holder.csv", encoding="utf-8-sig"),
    "user": pd.read_csv(TEST_DIR / "Pre_test_gas_user.csv", encoding="utf-8-sig"),
    "load": pd.read_csv(TEST_DIR / "Pre_test_load.csv", encoding="utf-8-sig"),
}

for df in test_tables.values():
    df["datetime"] = pd.to_datetime(df["datetime"])

for _key, df in test_tables.items():
    for category, cols in field_categories.items():
        for col in cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue
            if category == "气柜":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both').ffill().bfill()
            elif category == "流量":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD * 2, limit_direction='both').ffill().bfill().fillna(0.0)
            elif category == "发电负荷":
                df[col] = df[col].interpolate(method='linear', limit=SHORT_GAP_THRESHOLD, limit_direction='both').ffill().bfill()

# Build test features the same way as training
test_merged = test_tables['gas'].copy()
for key in ['holder', 'user', 'load']:
    test_merged = test_merged.merge(test_tables[key], on='datetime', how='left')

test_features = pd.DataFrame()
test_features['datetime'] = test_merged['datetime']
test_features['feat_generator_1_filled'] = test_merged['generator_1']
test_features['feat_generator_all_filled'] = test_merged['generator_all']
test_features['feat_p50_current'] = test_merged['generator_1']
test_features['feat_p120_current'] = test_merged['generator_all'] - test_merged['generator_1']
test_features['feat_pall_current'] = test_merged['generator_all']

for col in key_cols:
    if col in test_merged.columns:
        test_features[f'feat_{col}'] = test_merged[col]

test_features['feat_hour'] = test_features['datetime'].dt.hour
test_features['feat_dayofweek'] = test_features['datetime'].dt.dayofweek
test_features['feat_month'] = test_features['datetime'].dt.month
test_features['feat_is_weekend'] = (test_features['datetime'].dt.dayofweek >= 5).astype(float)
test_features['feat_known_price'] = test_features.apply(
    lambda row: price_lookup[(row['datetime'].month, row['datetime'].hour * 2)],
    axis=1
)

print("\n[5/5] Generating predictions...")

x_test = test_features[feat_cols].values.astype(np.float32)
current1_test = test_features["feat_p50_current"].values.astype(np.float32)[:, None]
currentall_test = (test_features["feat_p50_current"].values + test_features["feat_p120_current"].values).astype(np.float32)[:, None]

raw_pred_test = final_model.predict(x_test)
pred1_test = np.maximum(current1_test + raw_pred_test[:, :8], 0.0)
predall_test = np.maximum(currentall_test + raw_pred_test[:, 8:], 0.0)

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

output_file = Path("F:/Code2/AIC/hybrid_predictions.csv")
submission_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"  Saved: {output_file}")
print(f"  Predictions: {len(submission_df)} rows")

print("\n" + "="*80)
print("COMPLETED: Hybrid training with test data integration")
print("="*80)
