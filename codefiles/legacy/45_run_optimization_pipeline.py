"""One-command reproducible runner for optimization audit, backtest and QA."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "codefiles" / "legacy"
LOG_PATH = ROOT / "results" / "optimization" / "logs" / "45_run_optimization_pipeline.log"
MANIFEST_PATH = ROOT / "results" / "optimization" / "pipeline_run_manifest.json"
SCRIPTS = [
    "38_optimization_constraint_audit.py",
    "39_resource_mechanism_diagnostics.py",
    "40_prediction_visualizations.py",
    "41_conservative_dispatch_backtest.py",
    "42_dispatch_robustness_and_selection.py",
    "43_optimization_visualizations.py",
    "44_validate_optimization_stage.py",
]


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    pipeline_start = time.perf_counter()
    with LOG_PATH.open("w", encoding="utf-8") as log:
        log.write(f"pipeline_started_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.write("data_policy=training-only; external scoring dataset forbidden\n")
        for script in SCRIPTS:
            if "评分所用测试集" in str(CODE_DIR / script):
                raise RuntimeError("Forbidden external scoring dataset path")
            start = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, str(CODE_DIR / script)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            elapsed = time.perf_counter() - start
            record = {
                "script": script,
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
            }
            records.append(record)
            log.write(f"\n===== {script} | returncode={completed.returncode} | elapsed={elapsed:.3f}s =====\n")
            log.write(completed.stdout)
            log.write(completed.stderr)
            log.flush()
            print(f"{script}: returncode={completed.returncode}, elapsed={elapsed:.2f}s", flush=True)
            if completed.returncode != 0:
                manifest = {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "external_scoring_data_accessed": False,
                    "success": False,
                    "failed_script": script,
                    "steps": records,
                }
                MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                raise SystemExit(completed.returncode)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only optimization audit/backtest/visualization pipeline",
        "external_scoring_data_accessed": False,
        "success": True,
        "elapsed_seconds": time.perf_counter() - pipeline_start,
        "steps": records,
        "final_validation": "results/optimization/final_validation/validation_summary.json",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Pipeline complete in {manifest['elapsed_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
