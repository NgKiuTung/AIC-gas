"""Fit the enhanced mixed-target model once to audit feature-group importance."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_cleaning_enhanced.pkl"
CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog_cleaning_enhanced.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "cleaning_feature_importance"
HORIZONS = tuple(range(1, 9))


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(INPUT_PATH)
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    features = catalog["feature"].tolist()
    y1 = data[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y120 = data[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    yall = y1 + y120
    current1 = data["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
    currentall = (data["feat_p50_current"] + data["feat_p120_current"]).to_numpy(dtype=np.float32)[:, None]
    mixed = np.concatenate(
        [(y1 - current1) / np.maximum(np.abs(current1), 1e-6), yall - currentall], axis=1
    )
    model = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist", device="cuda",
        random_state=20260803, n_jobs=1, max_bin=256, multi_strategy="one_output_per_tree",
        n_estimators=350, max_depth=5, learning_rate=0.025, min_child_weight=30,
        subsample=0.85, colsample_bytree=0.70, reg_alpha=2.0, reg_lambda=30.0,
        verbosity=0,
    )
    begin = time.perf_counter()
    model.fit(data[features].to_numpy(dtype=np.float32), mixed, verbose=False)
    runtime = time.perf_counter() - begin
    importance = catalog.copy()
    importance["importance"] = model.feature_importances_.astype(float)
    importance = importance.sort_values("importance", ascending=False)
    group = importance.groupby("group", as_index=False).agg(
        feature_count=("feature", "size"), importance_sum=("importance", "sum"),
        importance_mean=("importance", "mean"), importance_max=("importance", "max")
    ).sort_values("importance_sum", ascending=False)
    importance.to_csv(RESULT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    group.to_csv(RESULT_DIR / "group_importance.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "cleaning-enhanced causal training features only",
        "external_scoring_data_accessed": False,
        "features": len(features), "runtime_seconds": runtime,
        "top_groups": group.head(10).to_dict(orient="records"),
        "top_features": importance.head(30).to_dict(orient="records"),
    }
    (RESULT_DIR / "importance_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
