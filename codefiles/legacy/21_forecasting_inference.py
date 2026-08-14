"""Inference interface for the serialized hybrid power forecaster.

The caller must provide an already constructed causal feature table with the
same schema as training. This module does not discover or read any dataset by
itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "results" / "models" / "power_forecaster"
MODEL_PATH = MODEL_DIR / "xgboost_hybrid_forecaster.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.csv"
BETA_GENERATOR_1 = np.linspace(0.35, 0.30, 8, dtype=np.float32)
BETA_GENERATOR_ALL = np.linspace(0.65, 0.60, 8, dtype=np.float32)


def load_feature_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    raise ValueError("Feature input must be .pkl/.pickle or .csv")


def predict_feature_rows(frame: pd.DataFrame) -> pd.DataFrame:
    schema = pd.read_csv(SCHEMA_PATH, encoding="utf-8-sig").sort_values("position")
    features = schema["feature"].tolist()
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing {len(missing)} model features; first entries: {missing[:10]}")
    x_values = frame[features].to_numpy(dtype=np.float32)
    if not np.isfinite(x_values).all():
        raise ValueError("Feature table contains non-finite values")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    raw = model.predict(x_values)
    current1 = frame["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
    currentall = (
        frame["feat_p50_current"].to_numpy(dtype=np.float32)
        + frame["feat_p120_current"].to_numpy(dtype=np.float32)
    )[:, None]
    pred1 = np.maximum(current1 * (1.0 + raw[:, :8] * BETA_GENERATOR_1), 0.0)
    predall = np.maximum(currentall + raw[:, 8:] * BETA_GENERATOR_ALL, pred1)
    output = pd.DataFrame(index=frame.index)
    if "datetime" in frame:
        output["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    for index, minutes in enumerate(range(15, 121, 15)):
        output[f"generator_1_t_plus_{minutes}m"] = pred1[:, index]
        output[f"generator_all_t_plus_{minutes}m"] = predall[:, index]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True, help="Causal model-feature table")
    parser.add_argument("--output", type=Path, required=True, help="Prediction CSV under results/")
    args = parser.parse_args()
    resolved_output = args.output.resolve()
    results_root = (ROOT / "results").resolve()
    if results_root not in resolved_output.parents:
        raise ValueError("Execution outputs must be written under results/")
    predictions = predict_feature_rows(load_feature_table(args.features))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(predictions)} forecast rows to {args.output}")


if __name__ == "__main__":
    main()
