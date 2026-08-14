"""Audit Git candidates using explicit non-dataset pathspecs."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "repository_validation"
PATHS = [
    ".github",
    "codefiles",
    "configs",
    "docs",
    "requirements",
    "results",
    ".env.example",
    ".gitattributes",
    ".gitignore",
    "pyproject.toml",
    "README.md",
]


def main() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--others", "--exclude-standard", "--", *PATHS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )
    rows = []
    for relative in completed.stdout.split("\0"):
        if not relative:
            continue
        path = ROOT / relative
        if path.is_file():
            rows.append({"path": relative.replace("\\", "/"), "bytes": path.stat().st_size})
    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "git_candidate_files.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_pathspec_used": False,
        "external_scoring_data_accessed": False,
        "candidate_files": len(rows),
        "candidate_bytes": sum(int(row["bytes"]) for row in rows),
        "largest_candidates": rows[:20],
        "files_over_10_mib": [row for row in rows if int(row["bytes"]) > 10 * 1024 * 1024],
    }
    (OUTPUT_DIR / "git_candidate_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Git candidate audit: files={summary['candidate_files']}, "
        f"MiB={summary['candidate_bytes'] / 1024**2:.2f}, over_10_MiB={len(summary['files_over_10_mib'])}"
    )


if __name__ == "__main__":
    main()
