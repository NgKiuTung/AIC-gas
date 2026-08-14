"""Validate training-only preprocessing outputs and provenance."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "preprocessing"
PROCESSED_DIR = RESULT_DIR / "processed"
AUDIT_DIR = RESULT_DIR / "audit"
LOG_DIR = RESULT_DIR / "logs"
PRIMARY_PATH = PROCESSED_DIR / "preprocessed_train.csv"
CAUSAL_PATH = PROCESSED_DIR / "preprocessed_train_causal.csv"
STATE_PATH = PROCESSED_DIR / "preprocessing_state_full_train.json"
MANIFEST_PATH = RESULT_DIR / "preprocessing_manifest.json"
TARGETS = ("generator_1", "generator_all")


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("validate_training_preprocessing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "03_validate_preprocessing.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    LOGGER.info("%s | %s | %s", "PASS" if passed else "FAIL", name, detail)


def main() -> None:
    primary = pd.read_csv(PRIMARY_PATH, encoding="utf-8-sig", low_memory=False)
    causal = pd.read_csv(CAUSAL_PATH, encoding="utf-8-sig", low_memory=False)
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    selection = json.loads((AUDIT_DIR / "interpolation_selection.json").read_text(encoding="utf-8"))
    methods = pd.read_csv(AUDIT_DIR / "imputation_summary.csv", encoding="utf-8-sig")
    checks: list[dict[str, Any]] = []

    add(checks, "offline_and_causal_schema_equal", list(primary.columns) == list(causal.columns), str(primary.shape))
    for name, frame in (("offline", primary), ("causal", causal)):
        dt = pd.to_datetime(frame["datetime"], errors="coerce")
        intervals = dt.diff().dropna().dt.total_seconds().div(60)
        add(checks, f"{name}_datetime_valid_unique", dt.notna().all() and not dt.duplicated().any(), f"invalid={dt.isna().sum()}")
        add(checks, f"{name}_15min_continuous", bool((intervals == 15).all()), f"non15={int((intervals != 15).sum())}")
        add(checks, f"{name}_split_marker", set(frame["split"].unique()) == {"train"}, str(frame["split"].unique()))
        feature_cols = [c for c in frame.columns if c not in {"datetime", "split", *TARGETS}]
        numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
        add(checks, f"{name}_features_complete", int(numeric.isna().sum().sum()) == 0, f"missing={int(numeric.isna().sum().sum())}")
        add(checks, f"{name}_features_finite", bool(np.isfinite(numeric.to_numpy()).all()), "all feature cells checked")

    for target in TARGETS:
        filled = f"feat_{target}_filled"
        observed = primary[target].notna()
        add(
            checks,
            f"{target}_original_observations_preserved",
            bool(np.allclose(primary.loc[observed, target], primary.loc[observed, filled])),
            f"original_missing={int(primary[target].isna().sum())}",
        )
        add(
            checks,
            f"{target}_continuous_input_complete",
            primary[filled].notna().all() and causal[filled].notna().all(),
            f"offline_missing={int(primary[filled].isna().sum())}, causal_missing={int(causal[filled].isna().sum())}",
        )

    all_null = state["all_null_columns_excluded"]
    add(checks, "all_null_columns_excluded", all(c not in primary.columns for c in all_null), str(all_null))
    aggregate_cols = [
        "feat_blast_furnace_observed_sum",
        "feat_air_heater_observed_sum",
        "feat_blast_furnace_user_observed_sum",
        "feat_converter_user_observed_sum",
    ]
    add(checks, "observed_family_aggregates_present", all(c in primary.columns for c in aggregate_cols), str(aggregate_cols))
    add(checks, "no_unresolved_imputation", not methods["method"].str.contains("unresolved").any(), "offline and causal method logs checked")
    add(
        checks,
        "training_only_state",
        state.get("official_test_accessed") is False and state.get("source", "").startswith("official training"),
        state.get("source", ""),
    )
    manifest_paths = [item["path"] for item in manifest["inputs"]]
    add(
        checks,
        "training_only_manifest",
        manifest.get("official_test_accessed") is False and not any("测试集" in path or "test" in path.lower() for path in manifest_paths),
        str(manifest_paths),
    )
    add(
        checks,
        "training_only_interpolation_benchmark",
        selection.get("official_test_accessed") is False,
        selection.get("source", ""),
    )
    add(
        checks,
        "no_scoring_test_derived_csv",
        not (PROCESSED_DIR / "preprocessed_test.csv").exists(),
        "no processed scoring-test copy exists under results/preprocessing/processed",
    )
    hierarchy = primary[["generator_1", "generator_all"]].dropna()
    violations = int((hierarchy["generator_1"] > hierarchy["generator_all"]).sum())
    add(checks, "observed_target_hierarchy", violations == 0, f"violations={violations}")

    passed = all(item["passed"] for item in checks)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "official_test_accessed": False,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(item["passed"] for item in checks),
            "checks_failed": sum(not item["passed"] for item in checks),
            "shape": list(primary.shape),
        },
        "checks": checks,
    }
    path = AUDIT_DIR / "preprocessing_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Validation=%s passed=%s", path, passed)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
