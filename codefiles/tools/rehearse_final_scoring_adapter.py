"""Rehearse the final adapter on a synthetic 192-row scoring directory."""

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
import yaml
from gas_power.data.io import SCORING_DIRECTORY_NAME, SubmissionAccessGrant
from gas_power.data.raw_inputs import (
    SCORING_FILENAMES,
    combine_history_and_scoring_tables,
    load_scoring_raw_tables,
    load_training_raw_tables,
)
from gas_power.forecasting.final_inference import predict_from_raw_tables
from gas_power.submission.package import build_submission_zip, official_zip_name, validate_submission_zip

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "dataset" / "初赛-数据集"
MODEL_DIR = ROOT / "results" / "models" / "production_forecaster"
SPEC_PATH = ROOT / "configs" / "model_spec.yaml"
RESULT_DIR = ROOT / "results" / "phase5_final_adapter_validation"
FIGURE_DIR = ROOT / "results" / "figures_safe" / "phase5"
LOG_PATH = RESULT_DIR / "synthetic_final_adapter_rehearsal.log"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_logging() -> logging.Logger:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("synthetic_final_adapter_rehearsal")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def build_synthetic_scoring(training: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    timestamps = pd.date_range("2025-05-01 00:00:00", periods=192, freq="15min")
    step = np.arange(len(timestamps), dtype=float)
    daily = np.sin(2 * np.pi * step / 96)
    output: dict[str, pd.DataFrame] = {}
    for source, history in training.items():
        frame = pd.DataFrame({"datetime": timestamps})
        for index, column in enumerate(item for item in history.columns if item != "datetime"):
            numeric = pd.to_numeric(history[column], errors="coerce")
            if numeric.notna().sum() == 0:
                frame[column] = np.nan
                continue
            baseline = float(numeric.dropna().iloc[-1])
            amplitude = max(abs(baseline) * 0.015, 0.1 + index * 0.01)
            frame[column] = baseline + amplitude * daily
        output[source] = frame
    p50 = 220.0 + 15.0 * daily
    output["load"]["generator_1"] = p50
    output["load"]["generator_all"] = p50 + 125.0 + 8.0 * np.cos(2 * np.pi * step / 96)
    return output


def synthetic_price_lookup() -> dict[tuple[int, int], float]:
    return {
        (month, slot): float(220 + 25 * (slot in range(16, 22)) + 45 * (slot in range(36, 44)))
        for month in range(1, 13)
        for slot in range(48)
    }


def plot_predictions(frame: pd.DataFrame) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    horizons = np.arange(15, 121, 15)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for axis, target, color in zip(
        axes, ("generator_1", "generator_all"), ("#1772b4", "#d95f02"), strict=True
    ):
        columns = [f"{target}_t+{minutes}_pred" for minutes in horizons]
        for row_index in (0, 48, 96, 144, 191):
            row = frame.iloc[row_index]
            axis.plot(
                horizons,
                row[columns].to_numpy(dtype=float),
                marker="o",
                linewidth=1.6,
                label=pd.Timestamp(row["datetime"]).strftime("%m-%d %H:%M"),
                color=color,
                alpha=0.35 + 0.12 * (row_index == 191),
            )
        axis.set_title(f"Synthetic scoring rehearsal: {target}")
        axis.set_xlabel("Forecast horizon (minutes)")
        axis.set_ylabel("Predicted power (synthetic units)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    paths = [
        FIGURE_DIR / "synthetic_192row_final_adapter.png",
        FIGURE_DIR / "synthetic_192row_final_adapter.pdf",
    ]
    for path in paths:
        figure.savefig(path, dpi=180 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> None:
    logger = setup_logging()
    training = load_training_raw_tables(TRAINING_DIR)
    synthetic = build_synthetic_scoring(training)
    cache_root = ROOT / "results" / "cache" / "phase5_synthetic" / SCORING_DIRECTORY_NAME
    cache_root.mkdir(parents=True, exist_ok=True)
    for source, frame in synthetic.items():
        frame.to_csv(cache_root / SCORING_FILENAMES[source], index=False, encoding="utf-8-sig")
    input_paths = [cache_root / filename for filename in SCORING_FILENAMES.values()]
    hashes_before = {path.name: sha256(path) for path in input_paths}
    grant = SubmissionAccessGrant(
        purpose="final_submission_inference",
        read_only=True,
        allow_scoring_input=True,
    )
    scoring = load_scoring_raw_tables(cache_root, grant)
    combined, reference_times = combine_history_and_scoring_tables(training, scoring)
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    started = time.perf_counter()
    submission, inference_audit = predict_from_raw_tables(
        combined,
        reference_times,
        synthetic_price_lookup(),
        MODEL_DIR,
        spec["ensemble_parameters"],
    )
    inference_seconds = time.perf_counter() - started
    zip_path = RESULT_DIR / official_zip_name("synthetic_192row_rehearsal")
    build_submission_zip(submission, zip_path)
    package_audit = validate_submission_zip(zip_path, expected_datetimes=reference_times)
    hashes_after = {path.name: sha256(path) for path in input_paths}
    input_unchanged = hashes_before == hashes_after
    figures = plot_predictions(submission)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic 192-row scoring rehearsal",
        "official_scoring_data_accessed": False,
        "formal_submission": False,
        "reference_start": reference_times.min().isoformat(),
        "reference_end": reference_times.max().isoformat(),
        "reference_rows": len(reference_times),
        "inference_seconds": inference_seconds,
        "input_read_only_hash_match": input_unchanged,
        "inference_audit": inference_audit,
        "package_audit": package_audit,
        "figures": [str(path.relative_to(ROOT)) for path in figures],
        "all_passed": bool(
            len(reference_times) == 192
            and input_unchanged
            and inference_audit["finite_predictions"]
            and inference_audit["hierarchy_satisfied"]
            and package_audit["all_checks_pass"]
        ),
    }
    summary_path = RESULT_DIR / "synthetic_final_adapter_rehearsal_summary.json"
    logger.info("Synthetic final adapter summary: %s", summary)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
