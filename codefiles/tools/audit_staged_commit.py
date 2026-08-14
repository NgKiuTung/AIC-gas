"""Fail a local release commit if forbidden or oversized files are staged."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "repository_validation" / "staged_commit_audit.json"
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


def main() -> None:
    completed = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )
    paths = [line for line in completed.stdout.splitlines() if line]
    rows = []
    for relative in paths:
        path = ROOT / relative
        rows.append({"path": relative, "bytes": path.stat().st_size if path.is_file() else 0})
    forbidden = [row for row in rows if row["path"].startswith("dataset/")]
    official_pdfs = [
        row for row in rows if row["path"].startswith("docs/competition/official/") and row["path"].endswith(".pdf")
    ]
    blocked_results = [row for row in rows if row["path"].startswith(BLOCKED_PREFIXES)]
    oversized = [row for row in rows if int(row["bytes"]) > 10 * 1024 * 1024]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "staged_files_before_this_audit": len(rows),
        "staged_bytes_before_this_audit": sum(int(row["bytes"]) for row in rows),
        "dataset_files": forbidden,
        "official_competition_pdfs": official_pdfs,
        "blocked_competition_artifacts": blocked_results,
        "files_over_10_mib": oversized,
        "overall_pass": not forbidden and not official_pdfs and not blocked_results and not oversized,
    }
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Staged audit: files={len(rows)}, MiB={summary['staged_bytes_before_this_audit'] / 1024**2:.2f}, "
        f"overall_pass={summary['overall_pass']}"
    )
    if not summary["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
