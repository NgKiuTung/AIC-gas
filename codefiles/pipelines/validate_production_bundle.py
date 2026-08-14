"""Validate hashes, schema, full-row reload parity, and CPU inference for the production bundle."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from gas_power.forecasting.production import apply_frozen_ensemble

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_cleaning_enhanced.pkl"
CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog_cleaning_enhanced.csv"
SPEC_PATH = ROOT / "configs" / "model_spec.yaml"
MODEL_DIR = ROOT / "results" / "models" / "production_forecaster"
RESULT_DIR = ROOT / "results" / "production_validation"
TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((MODEL_DIR / "production_model_manifest.json").read_text(encoding="utf-8"))
    data = pd.read_pickle(INPUT_PATH)
    schema = pd.read_csv(MODEL_DIR / "feature_schema.csv", encoding="utf-8-sig").sort_values("position")
    features = schema["feature"].tolist()
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    expected_features = catalog.loc[catalog["group"].isin(set(spec["feature_groups"])), "feature"].tolist()
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("training_only_spec", spec["external_scoring_data_accessed"] is False, spec["model_spec_id"])
    check("training_only_manifest", manifest["external_scoring_data_accessed"] is False, manifest["model_spec_id"])
    check("feature_schema_exact", features == expected_features, f"features={len(features)}")
    check("feature_schema_unique", len(features) == len(set(features)), f"unique={len(set(features))}")
    check(
        "feature_schema_hash",
        sha256(MODEL_DIR / "feature_schema.csv") == manifest["feature_schema_sha256"],
        manifest["feature_schema_sha256"],
    )
    check(
        "input_feature_hash",
        sha256(INPUT_PATH) == manifest["input_sha256"]["features"],
        manifest["input_sha256"]["features"],
    )
    check(
        "input_catalog_hash",
        sha256(CATALOG_PATH) == manifest["input_sha256"]["catalog"],
        manifest["input_sha256"]["catalog"],
    )

    matrix = data[features].to_numpy(dtype=np.float32)
    check("full_matrix_finite", bool(np.isfinite(matrix).all()), f"shape={matrix.shape}")
    component_raw: dict[str, np.ndarray] = {}
    total_cpu_seconds = 0.0
    for component in manifest["components"]:
        model_path = MODEL_DIR / component["model_file"]
        check(
            f"{component['component']}_model_hash",
            sha256(model_path) == component["model_sha256"],
            component["model_sha256"],
        )
        gpu_model = xgb.XGBRegressor()
        gpu_model.load_model(model_path)
        gpu_model.set_params(device="cuda")
        gpu_prediction = gpu_model.predict(matrix)
        cpu_model = xgb.XGBRegressor()
        cpu_model.load_model(model_path)
        cpu_model.set_params(device="cpu")
        started = time.perf_counter()
        cpu_prediction = cpu_model.predict(matrix)
        cpu_seconds = time.perf_counter() - started
        total_cpu_seconds += cpu_seconds
        difference = float(np.max(np.abs(gpu_prediction - cpu_prediction)))
        name = str(component["component"])
        component_raw[name] = cpu_prediction
        check(f"{name}_output_shape", cpu_prediction.shape == (len(data), 16), str(cpu_prediction.shape))
        check(f"{name}_output_finite", bool(np.isfinite(cpu_prediction).all()), f"rows={len(cpu_prediction)}")
        check(f"{name}_full_gpu_cpu_parity", difference <= TOLERANCE, f"max_abs_diff={difference:.12g}")
        check(f"{name}_cpu_runtime", cpu_seconds < 1800.0, f"seconds={cpu_seconds:.6f}")

    current_1 = data["feat_p50_current"].to_numpy(dtype=np.float64)
    current_all = (
        data["feat_p50_current"].to_numpy(dtype=np.float64)
        + data["feat_p120_current"].to_numpy(dtype=np.float64)
    )
    started = time.perf_counter()
    prediction_before_projection = apply_frozen_ensemble(
        component_raw,
        current_1,
        current_all,
        spec["ensemble_parameters"],
        enforce_hierarchy=False,
    )
    prediction = apply_frozen_ensemble(component_raw, current_1, current_all, spec["ensemble_parameters"])
    ensemble_seconds = time.perf_counter() - started
    raw_violation = prediction_before_projection["generator_all"] < prediction_before_projection["generator_1"]
    raw_violation_count = int(raw_violation.sum())
    raw_violation_rate = float(raw_violation.mean())
    raw_maximum_gap = float(
        (
            prediction_before_projection["generator_1"][raw_violation]
            - prediction_before_projection["generator_all"][raw_violation]
        ).max()
        if raw_violation_count
        else 0.0
    )
    finite = all(np.isfinite(values).all() for values in prediction.values())
    hierarchy = bool((prediction["generator_all"] >= prediction["generator_1"]).all())
    check("full_ensemble_finite", finite, f"rows={len(data)}")
    check(
        "raw_hierarchy_violation_audited",
        True,
        f"count={raw_violation_count}; rate={raw_violation_rate:.8f}; max_gap={raw_maximum_gap:.8f}",
    )
    check("full_prediction_hierarchy", hierarchy, "generator_all >= generator_1")
    check(
        "feature_to_prediction_cpu_under_30_minutes",
        total_cpu_seconds + ensemble_seconds < 1800.0,
        f"seconds={total_cpu_seconds + ensemble_seconds:.6f}",
    )

    table = pd.DataFrame(checks)
    table.to_csv(RESULT_DIR / "production_bundle_checks.csv", index=False, encoding="utf-8-sig")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_scoring_data_accessed": False,
        "validation_rows": len(data),
        "checks": len(table),
        "passed": int(table["passed"].sum()),
        "failed": int((~table["passed"]).sum()),
        "cpu_feature_to_prediction_seconds": total_cpu_seconds + ensemble_seconds,
        "raw_hierarchy_violations_before_projection": raw_violation_count,
        "raw_hierarchy_violation_rate": raw_violation_rate,
        "raw_hierarchy_maximum_gap": raw_maximum_gap,
        "hierarchy_projection_applied": True,
        "all_passed": bool(table["passed"].all()),
    }
    (RESULT_DIR / "production_bundle_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if summary["all_passed"]:
        spec["status"] = "production_bundle_validated_pending_submission_pipeline"
        spec["production_bundle_available"] = True
        spec["production_bundle"] = {
            "storage": "local_only_gitignored",
            "path": "results/models/production_forecaster",
            "manifest": "results/production_validation/production_model_manifest.json",
            "training_rows": manifest["training_rows"],
            "feature_count": manifest["feature_count"],
            "components": [
                {
                    "component": component["component"],
                    "model_file": component["model_file"],
                    "model_sha256": component["model_sha256"],
                }
                for component in manifest["components"]
            ],
            "full_row_gpu_cpu_parity": True,
            "cpu_feature_to_prediction_seconds": total_cpu_seconds + ensemble_seconds,
        }
        spec["next_required_phase"] = "build causal raw-table inference and submission adapter"
        SPEC_PATH.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
