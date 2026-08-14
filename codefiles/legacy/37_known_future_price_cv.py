"""Run one GPU validation of deterministic future-price features."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "codefiles" / "legacy" / "34_targeted_xgboost_capacity.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("known_future_price_cv", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_price_augmented.pkl"
    module.CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog_price_augmented.csv"
    module.RESULT_DIR = ROOT / "results" / "experiments" / "known_future_price"
    module.FEATURE_GROUPS = set(module.FEATURE_GROUPS) | {"known_future_price"}
    module.CONFIGS = {
        "d5_known_future_price": {
            "n_estimators": 350, "max_depth": 5, "learning_rate": 0.025,
            "min_child_weight": 30, "subsample": 0.85, "colsample_bytree": 0.70,
            "reg_alpha": 2.0, "reg_lambda": 30.0, "max_bin": 256,
        }
    }
    module.main()


if __name__ == "__main__":
    main()
