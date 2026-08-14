"""Run the three stage-level artifact validators."""

import argparse

import _bootstrap  # noqa: F401
from gas_power.utils.pipeline import run_legacy_scripts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_legacy_scripts(
        ["03_validate_preprocessing.py", "23_validate_modeling_stage.py", "44_validate_optimization_stage.py"],
        "validation",
        args.dry_run,
    )

