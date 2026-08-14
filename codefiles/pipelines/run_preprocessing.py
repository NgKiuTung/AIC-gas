"""Stable training-data audit and preprocessing entrypoint."""

import argparse

import _bootstrap  # noqa: F401
from gas_power.utils.pipeline import run_legacy_scripts

SCRIPTS = [f"{number:02d}_{name}.py" for number, name in [
    (1, "data_audit"), (2, "preprocess"), (3, "validate_preprocessing"),
    (4, "interpolation_benchmark"), (5, "all_null_column_analysis"),
    (6, "environment_check"), (7, "build_supervised_features"),
]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_legacy_scripts(SCRIPTS, "preprocessing", args.dry_run)

