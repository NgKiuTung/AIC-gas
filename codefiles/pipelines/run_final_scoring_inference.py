"""Read-only final scoring inference. Never trains, tunes, or evaluates labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml
from gas_power.data.io import SubmissionAccessGrant
from gas_power.data.raw_inputs import (
    SCORING_FILENAMES,
    combine_history_and_scoring_tables,
    load_price_lookup,
    load_scoring_raw_tables,
    load_training_raw_tables,
)
from gas_power.forecasting.final_inference import predict_from_raw_tables
from gas_power.submission.package import build_submission_zip, official_zip_name, validate_submission_zip

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "dataset" / "初赛-数据集"
MODEL_DIR = ROOT / "results" / "models" / "production_forecaster"
SPEC_PATH = ROOT / "configs" / "model_spec.yaml"
PRICE_PATH = TRAINING_DIR / "price.xlsx"
AUDIT_DIR = ROOT / "results" / "final_inference_audit"
AUTHORIZATION_TOKEN = "AUTHORIZED_FINAL_SCORING_READ_ONLY"
EXPECTED_START = pd.Timestamp("2025-05-01 00:00:00")
EXPECTED_END = pd.Timestamp("2025-05-02 23:45:00")
EXPECTED_ROWS = 192


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_logging() -> logging.Logger:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_scoring_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(AUDIT_DIR / "final_scoring_inference.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "submissions" / "final")
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--authorization-token", required=True)
    args = parser.parse_args()
    if args.authorization_token != AUTHORIZATION_TOKEN:
        raise PermissionError("Exact final scoring read-only authorization token is required")
    logger = setup_logging()
    grant = SubmissionAccessGrant(
        purpose="final_submission_inference",
        read_only=True,
        allow_scoring_input=True,
    )
    started = time.perf_counter()
    input_paths = [args.scoring_dir / filename for filename in SCORING_FILENAMES.values()]
    hashes_before = {path.name: sha256(path) for path in input_paths}
    training = load_training_raw_tables(TRAINING_DIR)
    scoring = load_scoring_raw_tables(args.scoring_dir, grant)
    combined, reference_times = combine_history_and_scoring_tables(training, scoring)
    if (
        len(reference_times) != EXPECTED_ROWS
        or reference_times.min() != EXPECTED_START
        or reference_times.max() != EXPECTED_END
    ):
        raise ValueError(
            f"Scoring reference grid mismatch: rows={len(reference_times)}, "
            f"start={reference_times.min()}, end={reference_times.max()}"
        )
    price_lookup = load_price_lookup(PRICE_PATH)
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    submission, inference_audit = predict_from_raw_tables(
        combined,
        reference_times,
        price_lookup,
        MODEL_DIR,
        spec["ensemble_parameters"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / official_zip_name(args.team_name)
    build_submission_zip(submission, zip_path)
    package_audit = validate_submission_zip(zip_path, expected_datetimes=reference_times)
    hashes_after = {path.name: sha256(path) for path in input_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("Scoring inputs changed during inference")
    elapsed = time.perf_counter() - started
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "explicit user authorization for final scoring read-only inference",
        "official_scoring_data_accessed": True,
        "training_or_tuning_performed": False,
        "label_metrics_computed": False,
        "reference_rows": len(reference_times),
        "reference_start": reference_times.min().isoformat(),
        "reference_end": reference_times.max().isoformat(),
        "input_sha256_before": hashes_before,
        "input_read_only_hash_match": hashes_before == hashes_after,
        "model_spec_id": spec["model_spec_id"],
        "model_spec_protocol_frozen_commit": spec["protocol_frozen_commit"],
        "inference_audit": inference_audit,
        "package_audit": package_audit,
        "zip_path": str(zip_path.relative_to(ROOT)),
        "zip_sha256": sha256(zip_path),
        "elapsed_seconds": elapsed,
        "under_30_minutes": elapsed < 1800.0,
        "all_passed": bool(
            hashes_before == hashes_after
            and inference_audit["finite_predictions"]
            and inference_audit["hierarchy_satisfied"]
            and package_audit["all_checks_pass"]
            and elapsed < 1800.0
        ),
    }
    manifest_path = AUDIT_DIR / "final_scoring_inference_manifest.json"
    logger.info("Final scoring inference manifest: %s", manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
