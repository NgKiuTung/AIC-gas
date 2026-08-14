"""Stable optimization audit, backtest, visualization, and validation entrypoint."""

import argparse

import _bootstrap  # noqa: F401
from gas_power.utils.pipeline import run_legacy_scripts

SCRIPTS = [
    "38_optimization_constraint_audit.py", "39_resource_mechanism_diagnostics.py",
    "40_prediction_visualizations.py", "41_conservative_dispatch_backtest.py",
    "42_dispatch_robustness_and_selection.py", "43_optimization_visualizations.py",
    "44_validate_optimization_stage.py",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_legacy_scripts(SCRIPTS, "optimization", args.dry_run)

