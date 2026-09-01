"""Refit and serialize the production components from the frozen ModelSpec."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from gas_power.forecasting.production import apply_frozen_ensemble, build_mixed_targets

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_cleaning_enhanced.pkl"
CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog_cleaning_enhanced.csv"
SPEC_PATH = ROOT / "configs" / "model_spec.yaml"
MODEL_DIR = ROOT / "results" / "models" / "production_forecaster"
RESULT_DIR = ROOT / "results" / "production_validation"
LOG_DIR = ROOT / "results" / "training" / "logs"
SEED = 20260803
PARITY_ROWS = 256
PARITY_TOLERANCE = 1e-6
FROZEN_FEATURE_COUNT = 801
NEW_FEATURE_PREFIXES = (
    "feat_multivar_anomaly_score_",
    "feat_multivar_anomaly_",
    "feat_energy_balance_residual",
    "feat_energy_balance_abs",
    "feat_energy_balance_ratio",
    "feat_interact_holder_bfg_balance",
    "feat_generation_fuel_structure_hhi",
)
RETIRING_FEATURE_PREFIXES = ("feat_future_price_",)


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("production_refit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "production_refit.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_description() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _current_feature_importance(features: list[str]) -> dict[str, float] | None:
    """Read prior-model importance to replace the weakest engineered features."""
    importances: list[np.ndarray] = []
    for component in ("d5", "d6"):
        model_path = MODEL_DIR / f"{component}_xgboost.json"
        if not model_path.exists():
            return None
        model = xgb.XGBRegressor()
        try:
            model.load_model(model_path)
            values = np.asarray(model.feature_importances_, dtype=float)
        except (FileNotFoundError, ValueError, xgb.core.XGBoostError):
            return None
        if values.shape != (len(features),):
            return None
        importances.append(values)
    return dict(zip(features, np.mean(np.vstack(importances), axis=0), strict=True))


def augment_feature_schema(features: list[str], catalog_features: list[str]) -> list[str]:
    """Replace experimental candidates while preserving the 801-feature contract."""
    candidates = [
        feature
        for feature in catalog_features
        if feature not in features and feature.startswith(NEW_FEATURE_PREFIXES)
    ]
    if not candidates:
        return features
    retiring = [feature for feature in features if feature.startswith(RETIRING_FEATURE_PREFIXES)]
    if len(retiring) >= len(candidates):
        selected = [feature for feature in features if feature not in set(retiring[:len(candidates)])]
        selected.extend(candidates)
        LOGGER.info("Feature schema replaced: added=%d removed=%d", len(candidates), len(retiring[:len(candidates)]))
        return selected
    removable = [
        feature
        for feature in features
        if feature.startswith("feat_")
        and not feature.startswith(("feat_missing_", "feat_source_missing_"))
    ]
    if len(removable) < len(candidates):
        raise ValueError("Not enough engineered features to make room for anomaly candidates")
    importance = _current_feature_importance(features)
    if importance is None:
        drop = set(removable[-len(candidates):])
    else:
        positions = {feature: index for index, feature in enumerate(features)}
        weakest = sorted(removable, key=lambda feature: (importance[feature], positions[feature]))
        drop = set(weakest[:len(candidates)])
    selected = [feature for feature in features if feature not in drop]
    selected.extend(candidates)
    LOGGER.info("Feature schema augmented: added=%d removed=%d", len(candidates), len(drop))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    if spec["external_scoring_data_accessed"]:
        raise ValueError("Production ModelSpec must be training-only")
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    feature_groups = set(spec["feature_groups"])
    catalog_features = catalog.loc[catalog["group"].isin(feature_groups), "feature"].tolist()
    frozen_schema_path = MODEL_DIR / "feature_schema.csv"
    if frozen_schema_path.exists():
        frozen_schema = pd.read_csv(frozen_schema_path, encoding="utf-8-sig").sort_values("position")
        features = frozen_schema["feature"].tolist()
        if frozen_schema["position"].tolist() != list(range(len(features))):
            raise ValueError("Frozen feature schema positions are invalid")
        current_catalog = set(catalog_features)
        invalid = {
            feature
            for feature in features
            if feature not in current_catalog and not feature.startswith(RETIRING_FEATURE_PREFIXES)
        }
        if invalid:
            raise ValueError(
                f"Frozen feature schema contains features outside the current catalog: "
                f"{sorted(invalid)[:5]}"
            )
        features = augment_feature_schema(features, catalog_features)
    else:
        features = catalog_features
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature schema is empty or duplicated")
    if len(features) != FROZEN_FEATURE_COUNT:
        raise ValueError(f"Production feature schema must contain {FROZEN_FEATURE_COUNT} features")
    x_train = data[features].to_numpy(dtype=np.float32)
    y_train = build_mixed_targets(data)
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("Training matrix contains NaN or Inf")
    sample = x_train[-PARITY_ROWS:]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"position": range(len(features)), "feature": features}).to_csv(
        MODEL_DIR / "feature_schema.csv", index=False, encoding="utf-8-sig"
    )
    component_rows: list[dict[str, object]] = []
    raw_predictions: dict[str, np.ndarray] = {}
    for name, parameters in spec["component_models"].items():
        if not parameters["required_for_production"]:
            continue
        model_parameters = {key: value for key, value in parameters.items() if key != "required_for_production"}
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device=args.device,
            random_state=SEED,
            n_jobs=1,
            multi_strategy="one_output_per_tree",
            verbosity=0,
            **model_parameters,
        )
        LOGGER.info("Training %s on %s: rows=%d features=%d", name, args.device, len(data), len(features))
        started = time.perf_counter()
        model.fit(x_train, y_train, verbose=False)
        training_seconds = time.perf_counter() - started
        before = model.predict(sample)
        model_path = MODEL_DIR / f"{name}_xgboost.json"
        model.save_model(model_path)

        reloaded = xgb.XGBRegressor()
        reloaded.load_model(model_path)
        reloaded.set_params(device=args.device)
        after = reloaded.predict(sample)
        reload_max_abs_diff = float(np.max(np.abs(before - after)))

        cpu_model = xgb.XGBRegressor()
        cpu_model.load_model(model_path)
        cpu_model.set_params(device="cpu")
        cpu_prediction = cpu_model.predict(sample)
        cpu_max_abs_diff = float(np.max(np.abs(before - cpu_prediction)))
        if reload_max_abs_diff > PARITY_TOLERANCE or cpu_max_abs_diff > PARITY_TOLERANCE:
            raise ValueError(f"Prediction parity failed for {name}")
        raw_predictions[name] = cpu_prediction
        row = {
            "component": name,
            "model_file": model_path.name,
            "model_bytes": model_path.stat().st_size,
            "model_sha256": sha256(model_path),
            "training_seconds": training_seconds,
            "reload_max_abs_diff": reload_max_abs_diff,
            "cpu_max_abs_diff": cpu_max_abs_diff,
            "parity_pass": True,
            "parameters": model_parameters,
        }
        component_rows.append(row)
        LOGGER.info(
            "%s saved in %.2fs; reload diff=%.3g; CPU diff=%.3g",
            name,
            training_seconds,
            reload_max_abs_diff,
            cpu_max_abs_diff,
        )

    tail = data.iloc[-PARITY_ROWS:]
    current_1 = tail["feat_p50_current"].to_numpy(dtype=np.float64)
    current_all = (
        tail["feat_p50_current"].to_numpy(dtype=np.float64)
        + tail["feat_p120_current"].to_numpy(dtype=np.float64)
    )
    predictions = apply_frozen_ensemble(raw_predictions, current_1, current_all, spec["ensemble_parameters"])
    finite = all(np.isfinite(values).all() for values in predictions.values())
    hierarchy = bool((predictions["generator_all"] >= predictions["generator_1"]).all())
    if not finite or not hierarchy:
        raise ValueError("Production smoke prediction contract failed")
    smoke = pd.DataFrame({"datetime": tail["datetime"].to_numpy()})
    for target, values in predictions.items():
        for index, minutes in enumerate(range(15, 121, 15)):
            smoke[f"{target}_t_plus_{minutes}m"] = values[:, index]
    smoke.to_csv(MODEL_DIR / "training_tail_smoke_predictions.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_spec_id": spec["model_spec_id"],
        "protocol_frozen_commit": spec["protocol_frozen_commit"],
        "external_scoring_data_accessed": False,
        "training_rows": len(data),
        "feature_count": len(features),
        "output_count_per_component": y_train.shape[1],
        "device_requested": args.device,
        "gpu": gpu_description(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xgboost": xgb.__version__,
        "input_sha256": {"features": sha256(INPUT_PATH), "catalog": sha256(CATALOG_PATH)},
        "feature_schema_sha256": sha256(MODEL_DIR / "feature_schema.csv"),
        "components": component_rows,
        "smoke_rows": PARITY_ROWS,
        "finite_predictions": finite,
        "prediction_hierarchy": hierarchy,
        "reload_parity_tolerance": PARITY_TOLERANCE,
        "all_checks_pass": finite and hierarchy and all(row["parity_pass"] for row in component_rows),
    }
    (MODEL_DIR / "production_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "production_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(component_rows).drop(columns="parameters").to_csv(
        RESULT_DIR / "component_reload_parity.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "created_utc": manifest["created_utc"],
        "external_scoring_data_accessed": False,
        "required_components": [row["component"] for row in component_rows],
        "training_rows": len(data),
        "feature_count": len(features),
        "gpu_training": args.device == "cuda",
        "cpu_inference_verified": all(row["cpu_max_abs_diff"] <= PARITY_TOLERANCE for row in component_rows),
        "reload_parity_verified": all(row["reload_max_abs_diff"] <= PARITY_TOLERANCE for row in component_rows),
        "prediction_hierarchy": hierarchy,
        "all_checks_pass": manifest["all_checks_pass"],
    }
    (RESULT_DIR / "phase2_refit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
