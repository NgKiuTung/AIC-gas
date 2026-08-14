"""Subprocess runner for reproducible legacy experiment stages."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from gas_power.settings import PATHS


def run_legacy_scripts(names: list[str], stage: str, dry_run: bool = False) -> list[dict[str, object]]:
    log_dir = PATHS.results / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = PATHS.results / "pipeline_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for name in names:
        script = PATHS.legacy / name
        if not script.is_file():
            raise FileNotFoundError(script)
        if dry_run:
            records.append({"script": name, "status": "planned"})
            print(f"PLAN {script.relative_to(PATHS.root)}")
            continue
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=PATHS.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed = time.perf_counter() - started
        (log_dir / f"{script.stem}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        record = {"script": name, "returncode": completed.returncode, "elapsed_seconds": elapsed}
        records.append(record)
        print(f"{name}: returncode={completed.returncode}, elapsed={elapsed:.2f}s", flush=True)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, [sys.executable, str(script)])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "external_scoring_data_accessed": False,
        "dry_run": dry_run,
        "steps": records,
    }
    (manifest_dir / f"{stage}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return records

