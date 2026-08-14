"""Validate the causal raw-table inference chain without accessing scoring data."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from gas_power.data.causal_preprocessing import ALL_NULL_EXPECTED, preprocess_causal_raw_tables
from gas_power.features.inference import build_inference_feature_frame, select_model_features
from gas_power.forecasting.production import apply_frozen_ensemble
from gas_power.submission.schema import build_submission_frame, write_submission_csv

ROOT = Path(__file__).resolve().parents[2]
CAUSAL_TRAIN_PATH = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train_causal.csv"
TRAIN_MATRIX_PATH = ROOT / "results" / "features" / "train_supervised_features_cleaning_enhanced.pkl"
MODEL_DIR = ROOT / "results" / "models" / "production_forecaster"
MODEL_SPEC_PATH = ROOT / "configs" / "model_spec.yaml"
RESULT_DIR = ROOT / "results" / "phase3_inference_validation"
FIGURE_DIR = ROOT / "results" / "figures_safe" / "phase3"
LOG_PATH = RESULT_DIR / "phase3_inference_validation.log"
REPLAY_TOLERANCE = 1e-9

SOURCE_COLUMNS = {
    "gas": [
        "blast_furnace_1", "blast_furnace_2", "blast_furnace_3", "blast_furnace_4",
        "blast_furnace_5", "coke_oven_1", "converter_1", "air_heater_1", "air_heater_2",
        "air_heater_3", "air_heater_4", "air_heater_5", "into_gas_mixed_coke",
        "into_gas_mixed_blast_furnace", "into_gas_mixed_converter",
    ],
    "holder": ["blast_furnace_gas_holder_1", "blast_furnace_gas_holder_2"],
    "user": [
        "blast_furnace_user1", "blast_furnace_user2", "blast_furnace_user3",
        "blast_furnace_user4", "converter_user1", "converter_user2", "converter_user3",
    ],
    "load": [
        "generator_all", "generator_1", "generator_use_blast_furnace_gas",
        "generator_use_coke_gas", "generator_use_converter_gas",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_logging() -> logging.Logger:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase3_inference_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def synthetic_raw_tables(periods: int = 800) -> dict[str, pd.DataFrame]:
    """Create deterministic, structurally official-like tables with no competition rows."""
    timestamps = pd.date_range("2025-03-01 00:00:00", periods=periods, freq="15min")
    step = np.arange(periods, dtype=float)
    daily = np.sin(2.0 * np.pi * step / 96.0)
    weekly = np.sin(2.0 * np.pi * step / 672.0)
    tables: dict[str, pd.DataFrame] = {}
    offset = 0
    for source, columns in SOURCE_COLUMNS.items():
        frame = pd.DataFrame({"datetime": timestamps})
        for index, column in enumerate(columns):
            if column in ALL_NULL_EXPECTED:
                frame[column] = np.nan
                continue
            baseline = 80.0 + 7.0 * offset + 2.0 * index
            frame[column] = baseline + (4.0 + index % 3) * daily + 1.5 * weekly + 0.002 * step
        tables[source] = frame
        offset += 1

    load = tables["load"]
    load["generator_1"] = 210.0 + 18.0 * daily + 5.0 * weekly + 0.01 * step
    load["generator_all"] = load["generator_1"] + 125.0 + 10.0 * np.cos(2.0 * np.pi * step / 96.0)
    for column in (
        "generator_use_blast_furnace_gas", "generator_use_coke_gas", "generator_use_converter_gas"
    ):
        load[column] = load[column].clip(lower=1.0)

    # Realistic zeros are operating states, not missing values.
    zero_columns = {
        "gas": ["blast_furnace_5", "air_heater_2", "air_heater_4", "air_heater_5",
                "into_gas_mixed_blast_furnace", "into_gas_mixed_converter"],
        "user": ["blast_furnace_user2", "blast_furnace_user3", "blast_furnace_user4",
                 "converter_user1", "converter_user2"],
        "load": ["generator_use_coke_gas", "generator_use_converter_gas"],
    }
    for source, columns in zero_columns.items():
        for column in columns:
            tables[source].loc[(step.astype(int) % 160) < 12, column] = 0.0

    # Inject missing cells and one absent source timestamp after a fully observed first row.
    tables["gas"].loc[[130, 131, 132], "blast_furnace_1"] = np.nan
    tables["load"].loc[[310, 311, 312, 313], "generator_1"] = np.nan
    tables["user"] = tables["user"].drop(index=420).reset_index(drop=True)
    return tables


def synthetic_price_lookup() -> dict[tuple[int, int], float]:
    return {
        (month, slot): float(220 + 25 * (slot in range(16, 22)) + 45 * (slot in range(36, 44)))
        for month in range(1, 13)
        for slot in range(48)
    }


def validate_training_replay(schema: list[str]) -> dict[str, object]:
    logger = logging.getLogger("phase3_inference_validation")
    causal = pd.read_csv(CAUSAL_TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    expected = pd.read_pickle(TRAIN_MATRIX_PATH)
    causal["datetime"] = pd.to_datetime(causal["datetime"], errors="raise")
    expected["datetime"] = pd.to_datetime(expected["datetime"], errors="raise")
    started = time.perf_counter()
    rebuilt = build_inference_feature_frame(causal)
    rebuilt = pd.concat([rebuilt[["datetime"]], select_model_features(rebuilt, schema)], axis=1)
    aligned = expected[["datetime", *schema]].merge(
        rebuilt, on="datetime", how="inner", validate="one_to_one", suffixes=("_expected", "_rebuilt")
    )
    differences: list[dict[str, object]] = []
    for feature in schema:
        maximum = float(
            np.max(
                np.abs(
                    aligned[f"{feature}_expected"].to_numpy(dtype=float)
                    - aligned[f"{feature}_rebuilt"].to_numpy(dtype=float)
                )
            )
        )
        differences.append({"feature": feature, "max_abs_difference": maximum})
    difference_table = pd.DataFrame(differences).sort_values("max_abs_difference", ascending=False)
    difference_table.to_csv(RESULT_DIR / "training_feature_replay_differences.csv", index=False, encoding="utf-8-sig")
    maximum = float(difference_table["max_abs_difference"].max())
    summary = {
        "rows_expected": len(expected),
        "rows_aligned": len(aligned),
        "feature_count": len(schema),
        "max_abs_difference": maximum,
        "tolerance": REPLAY_TOLERANCE,
        "all_features_match": bool(maximum <= REPLAY_TOLERANCE and len(aligned) == len(expected)),
        "seconds": time.perf_counter() - started,
    }
    logger.info("Training replay: %s", summary)
    return summary


def load_component_predictions(matrix: np.ndarray) -> dict[str, np.ndarray]:
    component_raw: dict[str, np.ndarray] = {}
    for component in ("d5", "d6"):
        model = xgb.XGBRegressor()
        model.load_model(MODEL_DIR / f"{component}_xgboost.json")
        model.set_params(device="cpu")
        component_raw[component] = model.predict(matrix)
    return component_raw


def validate_synthetic_end_to_end(schema: list[str], spec: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    logger = logging.getLogger("phase3_inference_validation")
    tables = synthetic_raw_tables()
    price_lookup = synthetic_price_lookup()
    started = time.perf_counter()
    causal, imputation = preprocess_causal_raw_tables(tables, price_lookup, split="synthetic_inference")
    features = build_inference_feature_frame(causal)
    matrix_frame = select_model_features(features, schema)
    origins = np.arange(len(features) - 16, len(features))
    matrix = matrix_frame.iloc[origins].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Synthetic model matrix contains NaN or Inf")
    component_raw = load_component_predictions(matrix)
    current_1 = features.iloc[origins]["feat_p50_current"].to_numpy(dtype=float)
    current_all = (
        features.iloc[origins]["feat_p50_current"].to_numpy(dtype=float)
        + features.iloc[origins]["feat_p120_current"].to_numpy(dtype=float)
    )
    prediction = apply_frozen_ensemble(component_raw, current_1, current_all, spec["ensemble_parameters"])

    output = build_submission_frame(features.iloc[origins]["datetime"].to_numpy(), prediction)
    write_submission_csv(output, RESULT_DIR / "synthetic_submission_preview.csv")
    imputation.to_csv(RESULT_DIR / "synthetic_imputation_summary.csv", index=False, encoding="utf-8-sig")

    # Causal invariance: values strictly after an origin must not change its feature vector.
    cutoff = 760
    mutated = {source: frame.copy() for source, frame in tables.items()}
    future_time = pd.Timestamp("2025-03-01") + pd.Timedelta(minutes=15 * (cutoff + 1))
    for source, frame in mutated.items():
        future_mask = pd.to_datetime(frame["datetime"]) >= future_time
        for column in SOURCE_COLUMNS[source]:
            if column not in ALL_NULL_EXPECTED:
                frame.loc[future_mask, column] = pd.to_numeric(frame.loc[future_mask, column]) + 1_000_000.0
    causal_mutated, _ = preprocess_causal_raw_tables(mutated, price_lookup, split="synthetic_inference")
    features_mutated = build_inference_feature_frame(causal_mutated)
    original_row = select_model_features(features, schema).iloc[cutoff].to_numpy(dtype=float)
    mutated_row = select_model_features(features_mutated, schema).iloc[cutoff].to_numpy(dtype=float)
    causal_difference = float(np.max(np.abs(original_row - mutated_row)))

    finite = all(np.isfinite(values).all() for values in prediction.values())
    hierarchy = bool((prediction["generator_all"] >= prediction["generator_1"]).all())
    summary = {
        "raw_rows": len(causal),
        "feature_count": len(schema),
        "prediction_origins": len(origins),
        "forecast_horizons_minutes": list(range(15, 121, 15)),
        "finite_features": bool(np.isfinite(matrix).all()),
        "finite_predictions": finite,
        "hierarchy_satisfied": hierarchy,
        "future_mutation_feature_max_abs_difference": causal_difference,
        "future_independence_pass": causal_difference == 0.0,
        "seconds": time.perf_counter() - started,
    }
    logger.info("Synthetic end-to-end: %s", summary)
    return summary, output


def plot_synthetic_predictions(output: pd.DataFrame) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    latest = output.iloc[-1]
    horizons = np.arange(15, 121, 15)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for axis, target, color in zip(
        axes, ("generator_1", "generator_all"), ("#1772b4", "#d95f02"), strict=True
    ):
        columns = [f"{target}_t+{minutes}_pred" for minutes in horizons]
        for _, row in output.iloc[:-1].iterrows():
            axis.plot(horizons, row[columns].to_numpy(dtype=float), color=color, alpha=0.12, linewidth=0.8)
        axis.plot(horizons, latest[columns].to_numpy(dtype=float), color=color, linewidth=2.4, marker="o")
        axis.set_title(f"Synthetic {target} forecasts")
        axis.set_xlabel("Forecast horizon (minutes)")
        axis.set_ylabel("Predicted power (synthetic units)")
        axis.grid(alpha=0.25)
    figure.suptitle("Phase 3 synthetic raw-table end-to-end smoke test", fontsize=13)
    paths = [
        FIGURE_DIR / "synthetic_end_to_end_forecasts.png",
        FIGURE_DIR / "synthetic_end_to_end_forecasts.pdf",
    ]
    for path in paths:
        figure.savefig(path, dpi=180 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> None:
    logger = setup_logging()
    spec = yaml.safe_load(MODEL_SPEC_PATH.read_text(encoding="utf-8"))
    schema_table = pd.read_csv(MODEL_DIR / "feature_schema.csv", encoding="utf-8-sig").sort_values("position")
    schema = schema_table["feature"].tolist()
    if len(schema) != 801 or len(schema) != len(set(schema)):
        raise ValueError("Frozen model schema must contain 801 unique features")
    replay = validate_training_replay(schema)
    synthetic, output = validate_synthetic_end_to_end(schema, spec)
    figure_paths = plot_synthetic_predictions(output)
    required_synthetic_checks = (
        "finite_features",
        "finite_predictions",
        "hierarchy_satisfied",
        "future_independence_pass",
    )
    all_passed = bool(
        replay["all_features_match"] and all(synthetic[key] for key in required_synthetic_checks)
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_scoring_data_accessed": False,
        "official_alignment": {
            "reference_time_information_only": True,
            "forecast_horizons_15_to_120_minutes": True,
            "targets": ["generator_1", "generator_all"],
            "submission_result_created": False,
        },
        "training_replay": replay,
        "synthetic_end_to_end": synthetic,
        "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
        "all_passed": all_passed,
    }
    summary_path = RESULT_DIR / "phase3_inference_validation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Phase 3 all_passed=%s", all_passed)
    manifest_paths = [
        summary_path,
        RESULT_DIR / "training_feature_replay_differences.csv",
        RESULT_DIR / "synthetic_submission_preview.csv",
        RESULT_DIR / "synthetic_imputation_summary.csv",
        LOG_PATH,
        *figure_paths,
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_scoring_data_accessed": False,
        "script": str(Path(__file__).relative_to(ROOT)),
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in manifest_paths
        ],
    }
    (RESULT_DIR / "phase3_artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
