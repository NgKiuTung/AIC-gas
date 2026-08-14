"""Run direct-target XGBoost on relative residuals to align with MAPE."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "codefiles" / "legacy" / "11_xgboost_direct_targets.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("direct_target_relative", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RESULT_DIR = ROOT / "results" / "experiments" / "relative_residual"
    module.CONFIGS = {"direct_d5_relative": dict(module.CONFIGS["direct_d5"])}
    module.BLENDS = tuple(round(value * 0.05, 2) for value in range(21))
    module.TARGET_MODE = "relative"
    module.main()


if __name__ == "__main__":
    main()
