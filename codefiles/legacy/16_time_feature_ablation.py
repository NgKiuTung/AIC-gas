"""Run the direct-target depth-5 model with time-augmented features."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "codefiles" / "legacy" / "11_xgboost_direct_targets.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("direct_target_experiment", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_time_augmented.pkl"
    module.CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog_time_augmented.csv"
    module.RESULT_DIR = ROOT / "results" / "experiments" / "time_feature_ablation"
    module.CONFIGS = {"direct_d5_time_augmented": dict(module.CONFIGS["direct_d5"])}
    module.BLENDS = tuple(round(value * 0.05, 2) for value in range(21))
    module.main()


if __name__ == "__main__":
    main()
