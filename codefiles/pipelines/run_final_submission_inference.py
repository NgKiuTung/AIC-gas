"""Run the authorized final scoring inference and write both submission views."""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.data.io import SubmissionAccessGrant, assert_submission_input_path
from gas_power.features.inference import build_inference_feature_frame, select_model_features
from gas_power.forecasting.production import apply_frozen_ensemble
from gas_power.submission.schema import (
    build_submission_frame,
    validate_submission_frame,
    write_submission_csv,
)

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "dataset" / "初赛-数据集"
SCORING_DIR = Path(r"F:\Code2\AIC\初赛-评分所用测试集")
REFERENCE_INPUT = Path(r"F:\Code2\AIC\gas_predict_prelim\input.csv")
MODEL_DIR = ROOT / "results" / "models" / "production_forecaster"
MODEL_SPEC_PATH = ROOT / "configs" / "model_spec.yaml"
INPUT_OUTPUT = ROOT / "input.csv"
RESULT_OUTPUT = ROOT / "result.csv"
SUMMARY_OUTPUT = ROOT / "results" / "final_submission_inference_summary.json"
IMPUTATION_AUDIT_OUTPUT = ROOT / "results" / "final_submission_imputation_audit.csv"

SOURCE_FILES = {
    "gas": ("Pre_gas.csv", "Pre_test_gas.csv"),
    "holder": ("Pre_gas_holder.csv", "Pre_test_gas_holder.csv"),
    "user": ("Pre_gas_user.csv", "Pre_test_gas_user.csv"),
    "load": ("Pre_load.csv", "Pre_test_load.csv"),
}
COMPONENTS = ("d5", "d6")


def load_csv_tables(directory: Path, *, scoring: bool) -> dict[str, pd.DataFrame]:
    """Load the four raw tables while enforcing the scoring access boundary."""
    grant = SubmissionAccessGrant(
        purpose="final_submission_inference",
        read_only=True,
        allow_scoring_input=True,
    )
    tables: dict[str, pd.DataFrame] = {}
    for source, filenames in SOURCE_FILES.items():
        filename = filenames[1] if scoring else filenames[0]
        path = directory / filename
        if scoring:
            path = assert_submission_input_path(path, grant)
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if "datetime" not in frame or frame["datetime"].duplicated().any():
            raise ValueError(f"Invalid {source} table: datetime must exist and be unique")
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        tables[source] = frame.sort_values("datetime").reset_index(drop=True)
    return tables


def load_price_lookup() -> dict[tuple[int, int], float]:
    """Load the known price schedule from the training-only directory."""
    price_table = pd.read_excel(TRAINING_DIR / "price.xlsx")
    return {
        (month, row_index): float(row[f"{month}月"])
        for row_index, row in price_table.iterrows()
        for month in range(1, 13)
    }


def load_frozen_schema() -> list[str]:
    """Load and validate the feature order used when the production models were fit."""
    schema_path = MODEL_DIR / "feature_schema.csv"
    schema = pd.read_csv(schema_path, encoding="utf-8-sig").sort_values("position")
    features = schema["feature"].tolist()
    if len(features) != 801 or len(features) != len(set(features)):
        raise ValueError("Frozen production schema must contain 801 unique features")
    return features


def load_component_predictions(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Run the two frozen production components on the same finite matrix."""
    predictions: dict[str, np.ndarray] = {}
    for component in COMPONENTS:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_DIR / f"{component}_xgboost.json")
        model.set_params(device="cpu")
        predictions[component] = model.predict(matrix)
    return predictions


def build_combined_tables(
    training: dict[str, pd.DataFrame],
    scoring: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Append scoring rows after training history to preserve causal lookbacks."""
    scoring_times = pd.DatetimeIndex(scoring["gas"]["datetime"])
    test_start = scoring_times.min()
    combined: dict[str, pd.DataFrame] = {}
    for source in SOURCE_FILES:
        train_part = training[source].loc[training[source]["datetime"] < test_start]
        combined[source] = (
            pd.concat([train_part, scoring[source]], ignore_index=True)
            .sort_values("datetime")
            .reset_index(drop=True)
        )
    return combined, scoring_times


def build_scoring_missing_summary(scoring: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize missing cells in the scoring window before any filling."""
    rows: list[dict[str, object]] = []
    for source, frame in scoring.items():
        for column in frame.columns:
            if column == "datetime":
                continue
            missing = pd.to_numeric(frame[column], errors="coerce").isna()
            current = 0
            longest = 0
            for value in missing.astype(bool):
                current = current + 1 if value else 0
                longest = max(longest, current)
            rows.append(
                {
                    "source": source,
                    "column": column,
                    "scoring_missing_count": int(missing.sum()),
                    "scoring_max_missing_run_steps": longest,
                    "scoring_max_missing_run_minutes": int(longest * 15),
                }
            )
    return pd.DataFrame(rows)


def build_input_view(
    features: pd.DataFrame,
    reference_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Select the exact 802-column feature-table order used by the reference input."""
    reference_header = pd.read_csv(REFERENCE_INPUT, nrows=0, encoding="utf-8-sig").columns.tolist()
    if reference_header[0] != "datetime" or len(reference_header) != 802:
        raise ValueError("Reference input.csv must contain 802 columns with datetime first")
    missing = sorted(set(reference_header) - set(features.columns))
    if missing:
        raise ValueError(f"Reference input schema has missing features: {missing[:10]}")
    selected = features.loc[features["datetime"].isin(reference_times), reference_header].copy()
    if len(selected) != len(reference_times):
        raise ValueError("Input feature rows do not exactly cover scoring timestamps")
    selected = selected.sort_values("datetime").reset_index(drop=True)
    if selected["datetime"].duplicated().any():
        raise ValueError("Input feature view contains duplicate timestamps")
    values = selected.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Input feature view contains NaN or Inf")
    return selected


def write_input_csv(frame: pd.DataFrame) -> None:
    """Write the feature view with the reference input precision and encoding."""
    serializable = frame.copy()
    serializable["datetime"] = pd.to_datetime(serializable["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    serializable.to_csv(INPUT_OUTPUT, index=False, encoding="utf-8-sig", float_format="%.9f")


def main() -> None:
    training = load_csv_tables(TRAINING_DIR, scoring=False)
    scoring = load_csv_tables(SCORING_DIR, scoring=True)
    combined, reference_times = build_combined_tables(training, scoring)
    causal, imputation = preprocess_causal_raw_tables(
        combined,
        load_price_lookup(),
        split="final_submission_inference",
    )
    imputation = imputation.merge(
        build_scoring_missing_summary(scoring),
        on=["source", "column"],
        how="left",
        validate="many_to_one",
    )
    features = build_inference_feature_frame(causal)
    schema = load_frozen_schema()
    input_view = build_input_view(features, reference_times)
    write_input_csv(input_view)

    test_features = features.loc[features["datetime"].isin(reference_times)].sort_values("datetime")
    matrix_frame = select_model_features(test_features, schema)
    matrix = matrix_frame.to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Model feature matrix contains NaN or Inf")
    component_raw = load_component_predictions(matrix)
    current_1 = test_features["feat_p50_current"].to_numpy(dtype=float)
    current_all = (
        test_features["feat_p50_current"].to_numpy(dtype=float)
        + test_features["feat_p120_current"].to_numpy(dtype=float)
    )
    spec = yaml.safe_load(MODEL_SPEC_PATH.read_text(encoding="utf-8"))
    prediction = apply_frozen_ensemble(
        component_raw,
        current_1,
        current_all,
        spec["ensemble_parameters"],
    )
    result = build_submission_frame(test_features["datetime"].to_numpy(), prediction)
    validate_submission_frame(result, expected_datetimes=reference_times)
    write_submission_csv(result, RESULT_OUTPUT)
    imputation.to_csv(IMPUTATION_AUDIT_OUTPUT, index=False, encoding="utf-8-sig")

    audit = {
        "official_scoring_data_accessed": True,
        "scoring_access_purpose": "final_submission_inference",
        "training_rows_used_for_context": len(combined["gas"]) - len(scoring["gas"]),
        "scoring_rows": len(scoring["gas"]),
        "input_rows": len(input_view),
        "input_columns": len(input_view.columns),
        "model_feature_count": len(schema),
        "result_rows": len(result),
        "result_columns": len(result.columns),
        "finite_model_matrix": bool(np.isfinite(matrix).all()),
        "finite_predictions": bool(all(np.isfinite(values).all() for values in prediction.values())),
        "generator_hierarchy": bool((prediction["generator_all"] >= prediction["generator_1"]).all()),
        "imputation_rows": len(imputation),
        "imputation_audit_output": str(IMPUTATION_AUDIT_OUTPUT.relative_to(ROOT)),
        "input_output": str(INPUT_OUTPUT.relative_to(ROOT)),
        "result_output": str(RESULT_OUTPUT.relative_to(ROOT)),
    }
    SUMMARY_OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
