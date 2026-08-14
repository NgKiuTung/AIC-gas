"""Build and validate an official-shape ZIP from synthetic predictions only."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from gas_power.submission.package import build_submission_zip, official_zip_name, validate_submission_zip

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "phase3_inference_validation" / "synthetic_submission_preview.csv"
RESULT_DIR = ROOT / "results" / "phase4_runtime_validation"
LOG_PATH = RESULT_DIR / "synthetic_package_validation.log"


def setup_logging() -> logging.Logger:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("synthetic_package_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main() -> None:
    logger = setup_logging()
    frame = pd.read_csv(INPUT, encoding="utf-8")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    destination = RESULT_DIR / official_zip_name("synthetic_contract_test")
    build_submission_zip(frame, destination)
    result = validate_submission_zip(destination, expected_datetimes=frame["datetime"])
    result.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_scope": "synthetic predictions only",
            "official_scoring_data_accessed": False,
            "formal_submission": False,
        }
    )
    summary_path = RESULT_DIR / "synthetic_package_validation_summary.json"
    logger.info("Synthetic ZIP validation: %s", result)
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
