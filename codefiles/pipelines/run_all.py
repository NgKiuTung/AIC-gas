"""Top-level orchestration with explicit stage selection."""

import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["preprocessing", "training", "optimization", "validation"],
        required=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pipeline_dir = Path(__file__).resolve().parent
    for stage in args.stages:
        command = [sys.executable, str(pipeline_dir / f"run_{stage}.py")]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)

