"""Independently revalidate and visualize a generated final submission ZIP."""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gas_power.forecasting.final_inference import load_feature_schema
from gas_power.submission.package import validate_submission_zip
from gas_power.submission.schema import HORIZONS_MINUTES

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "results" / "final_inference_audit"
DIAGNOSTIC_DIR = ROOT / "results" / "submissions" / "final" / "diagnostics"
EXPECTED_TIMES = pd.date_range("2025-05-01 00:00:00", periods=192, freq="15min")
MODEL_DIR = ROOT / "results" / "models" / "production_forecaster"


def setup_logging() -> logging.Logger:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_submission_independent_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(AUDIT_DIR / "final_submission_independent_audit.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_result(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as archive:
        payload = archive.read("result.csv")
    frame = pd.read_csv(io.BytesIO(payload))
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    return frame


def plot_diagnostics(frame: pd.DataFrame) -> Path:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    horizons = np.asarray(HORIZONS_MINUTES)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    selected_rows = (0, 48, 96, 144, 191)
    for axis, target, color in zip(
        axes[0], ("generator_1", "generator_all"), ("#1772b4", "#d95f02"), strict=True
    ):
        columns = [f"{target}_t+{minutes}_pred" for minutes in horizons]
        for row_index in selected_rows:
            row = frame.iloc[row_index]
            axis.plot(
                horizons,
                row[columns].to_numpy(dtype=float),
                marker="o",
                linewidth=1.5,
                alpha=0.72,
                color=color,
                label=row["datetime"].strftime("%m-%d %H:%M"),
            )
        axis.set_title(f"Forecast profiles: {target}")
        axis.set_xlabel("Forecast horizon (minutes)")
        axis.set_ylabel("Predicted power")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    for axis, target, color in zip(
        axes[1], ("generator_1", "generator_all"), ("#1772b4", "#d95f02"), strict=True
    ):
        columns = [f"{target}_t+{minutes}_pred" for minutes in horizons]
        matrix = frame[columns].to_numpy(dtype=float)
        axis.fill_between(
            horizons,
            np.quantile(matrix, 0.1, axis=0),
            np.quantile(matrix, 0.9, axis=0),
            color=color,
            alpha=0.18,
            label="10%–90% interval",
        )
        axis.plot(horizons, np.median(matrix, axis=0), color=color, marker="o", label="median")
        axis.set_title(f"Cross-origin distribution: {target}")
        axis.set_xlabel("Forecast horizon (minutes)")
        axis.set_ylabel("Predicted power")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    output = DIAGNOSTIC_DIR / "final_forecast_diagnostics.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--team-name")
    args = parser.parse_args()
    logger = setup_logging()
    package_audit = validate_submission_zip(
        args.zip_path,
        expected_datetimes=EXPECTED_TIMES,
        expected_team_name=args.team_name,
        expected_feature_schema=load_feature_schema(MODEL_DIR),
    )
    frame = load_result(args.zip_path)
    figure_path = plot_diagnostics(frame)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "independent post-generation structural and visual audit",
        "official_scoring_data_reused_for_training_or_tuning": False,
        "label_metrics_computed": False,
        "package_audit": package_audit,
        "diagnostic_figure": str(figure_path.relative_to(ROOT)),
        "diagnostic_figure_storage": "local_only_gitignored",
        "prediction_values_persisted_in_audit_json": False,
        "all_passed": True,
    }
    destination = AUDIT_DIR / "final_submission_independent_audit.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Independent final submission audit: %s", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
