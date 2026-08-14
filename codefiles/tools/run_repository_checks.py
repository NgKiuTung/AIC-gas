"""Run and persist repository-level compilation, test, layout, and archive checks."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "repository_validation"


def run(name: str, command: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    (OUTPUT_DIR / f"{name}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    print(f"{name}: returncode={completed.returncode}, elapsed={elapsed:.2f}s")
    return {"check": name, "returncode": completed.returncode, "elapsed_seconds": elapsed}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    current_pointer = ROOT / "results" / "releases" / "current" / "version.json"
    current_version = json.loads(current_pointer.read_text(encoding="utf-8"))["version"]
    checks = [
        run("compileall", [sys.executable, "-m", "compileall", "-q", "codefiles"]),
        run(
            "ruff",
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "codefiles/src",
                "codefiles/pipelines",
                "codefiles/tests",
                "codefiles/tools",
            ],
        ),
        run("pytest", [sys.executable, "-m", "pytest"]),
        run(
            "sensitive_git_content",
            [
                sys.executable,
                "codefiles/tools/audit_sensitive_git_content.py",
                "--check",
                "--no-write",
            ],
        ),
        run(
            "optimization_dry_run",
            [sys.executable, "codefiles/pipelines/run_optimization.py", "--dry-run"],
        ),
        run("layout", [sys.executable, "codefiles/tools/validate_repository_layout.py"]),
        run(
            "release_verification",
            [
                sys.executable,
                "codefiles/tools/verify_release_snapshot.py",
                "--version",
                current_version,
            ],
        ),
    ]
    overall = all(check["returncode"] == 0 for check in checks)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_scoring_data_accessed": False,
        "checks": checks,
        "overall_pass": overall,
    }
    (OUTPUT_DIR / "repository_checks.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
