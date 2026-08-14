"""Audit tracked Git content for competition-data exposure without reading dataset/."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "compliance" / "git_audit"
BLOCKED_PREFIXES = (
    "_audit_bundle/",
    "dataset/",
    "results/features/",
    "results/models/",
    "results/optimization/",
    "results/preprocessing/processed/",
    "results/releases/",
    "results/submissions/",
    "results/visualizations/",
)
BLOCKED_NAME_TOKENS = (
    "oof_predictions",
    "row_level",
    "selected_balanced_schedules",
    "training_feature_inference_check",
)
ROW_LEVEL_COLUMNS = {"actual", "current", "prediction"}


def tracked_paths() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )
    return [line.replace("\\", "/") for line in completed.stdout.splitlines() if line]


def csv_header(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return set(next(csv.reader(handle), []))


def findings() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in tracked_paths():
        reasons = []
        if relative.startswith(BLOCKED_PREFIXES):
            reasons.append("blocked_path")
        if any(token in relative.lower() for token in BLOCKED_NAME_TOKENS):
            reasons.append("blocked_name")
        path = ROOT / relative
        if path.suffix.lower() == ".csv" and path.is_file():
            header = csv_header(path)
            if "datetime" in header and ROW_LEVEL_COLUMNS.intersection(header):
                reasons.append("row_level_truth_or_prediction_columns")
        if reasons:
            rows.append({"path": relative, "reasons": ";".join(sorted(set(reasons)))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="current")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    rows = findings()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "dataset_directory_traversed": False,
        "external_scoring_data_accessed": False,
        "tracked_sensitive_findings": len(rows),
        "overall_pass": not rows,
        "findings": rows,
    }
    if not args.no_write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / f"{args.label}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Tracked sensitive-content audit: findings={len(rows)}, pass={not rows}")
    if args.check and rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
