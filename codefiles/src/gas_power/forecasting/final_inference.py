"""Frozen-model raw-table inference used by synthetic rehearsal and final CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame, select_model_features
from gas_power.forecasting.production import apply_frozen_ensemble
from gas_power.submission.schema import build_submission_frame


def load_feature_schema(model_dir: str | Path) -> list[str]:
    schema = pd.read_csv(Path(model_dir) / "feature_schema.csv", encoding="utf-8-sig").sort_values("position")
    features = schema["feature"].tolist()
    if len(features) != 801 or len(features) != len(set(features)):
        raise ValueError("Frozen schema must contain 801 unique features")
    return features


def predict_from_raw_tables(
    combined_tables: dict[str, pd.DataFrame],
    reference_times: pd.DatetimeIndex,
    price_lookup: dict[tuple[int, int], float],
    model_dir: str | Path,
    ensemble_parameters: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    causal, imputation = preprocess_causal_raw_tables(combined_tables, price_lookup, split="final_inference")
    feature_frame = build_inference_feature_frame(causal)
    row_mask = feature_frame["datetime"].isin(reference_times)
    selected_rows = feature_frame.loc[row_mask].copy()
    selected_times = pd.DatetimeIndex(selected_rows["datetime"])
    if not selected_times.equals(reference_times):
        raise ValueError("Engineered inference rows do not exactly match scoring reference timestamps")
    schema = load_feature_schema(model_dir)
    matrix = select_model_features(selected_rows, schema).to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Final inference feature matrix contains NaN or Inf")
    component_raw: dict[str, np.ndarray] = {}
    for component in ("d5", "d6"):
        model = xgb.XGBRegressor()
        model.load_model(Path(model_dir) / f"{component}_xgboost.json")
        model.set_params(device="cpu")
        component_raw[component] = model.predict(matrix)
    current_1 = selected_rows["feat_p50_current"].to_numpy(dtype=float)
    current_all = current_1 + selected_rows["feat_p120_current"].to_numpy(dtype=float)
    predictions = apply_frozen_ensemble(component_raw, current_1, current_all, ensemble_parameters)
    submission = build_submission_frame(selected_times, predictions)
    audit = {
        "combined_grid_rows": len(causal),
        "reference_rows": len(reference_times),
        "feature_count": len(schema),
        "imputation_events": int(imputation.loc[imputation["method"] != "observed", "count"].sum()),
        "finite_matrix": bool(np.isfinite(matrix).all()),
        "finite_predictions": all(np.isfinite(values).all() for values in predictions.values()),
        "hierarchy_satisfied": bool((predictions["generator_all"] >= predictions["generator_1"]).all()),
    }
    return submission, audit
