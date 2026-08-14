"""Run serialized forecasting inference on an explicitly supplied feature table."""

import argparse
import subprocess
import sys

import _bootstrap  # noqa: F401
from gas_power.data import assert_allowed_data_path
from gas_power.settings import PATHS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    features = assert_allowed_data_path(args.features)
    output_arg = assert_allowed_data_path(args.output)
    output = PATHS.root / output_arg if not output_arg.is_absolute() else output_arg
    subprocess.run(
        [
            sys.executable,
            str(PATHS.legacy / "21_forecasting_inference.py"),
            "--features",
            str(features),
            "--output",
            str(output),
        ],
        cwd=PATHS.root,
        check=True,
    )
