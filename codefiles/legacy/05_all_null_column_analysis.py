"""Assess whether training-all-null columns are identifiable from available data.

This is a training-only analysis. It does not read the official scoring test set.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "dataset" / "初赛-数据集"
RESULT_DIR = ROOT / "results" / "preprocessing"
AUDIT_DIR = RESULT_DIR / "audit"
LOG_DIR = RESULT_DIR / "logs"

FAMILIES = {
    "blast_furnace_3": ["blast_furnace_1", "blast_furnace_2", "blast_furnace_4", "blast_furnace_5"],
    "air_heater_3": ["air_heater_1", "air_heater_2", "air_heater_4", "air_heater_5"],
    "converter_user3": ["converter_user1", "converter_user2"],
    "blast_furnace_gas_holder_1": ["blast_furnace_gas_holder_2"],
}

SEMANTICS = {
    "blast_furnace_3": "3号高炉煤气发生量",
    "air_heater_3": "3号热风炉高炉煤气消耗量",
    "converter_user3": "转炉煤气用户3消耗量",
    "blast_furnace_gas_holder_1": "1号高炉煤气柜瞬时柜容",
}


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("all_null_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "05_all_null_column_analysis.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def load_all_training() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(TRAIN_DIR.glob("Pre_*.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="datetime", how="outer", validate="one_to_one")
    return merged.sort_values("datetime").reset_index(drop=True)


def proxy_experiment(df: pd.DataFrame, family: list[str]) -> list[dict[str, Any]]:
    if len(family) < 2:
        return []
    records: list[dict[str, Any]] = []
    numeric = df[family].apply(pd.to_numeric, errors="coerce").interpolate(method="linear", limit_direction="both")
    split = int(len(numeric) * 0.8)
    for held_out in family:
        predictors = [c for c in family if c != held_out]
        train = numeric.iloc[:split].dropna(subset=[held_out, *predictors])
        valid = numeric.iloc[split:].dropna(subset=[held_out, *predictors])
        if len(train) < 100 or len(valid) < 20:
            continue
        model = make_pipeline(RobustScaler(), Ridge(alpha=10.0))
        model.fit(train[predictors], train[held_out])
        pred = model.predict(valid[predictors])
        iqr = float(train[held_out].quantile(0.75) - train[held_out].quantile(0.25))
        mae = float(mean_absolute_error(valid[held_out], pred))
        records.append(
            {
                "held_out_analogue": held_out,
                "predictors": ";".join(predictors),
                "train_rows": len(train),
                "validation_rows": len(valid),
                "r2": float(r2_score(valid[held_out], pred)),
                "mae": mae,
                "nmae_iqr": mae / max(iqr, 1e-9),
            }
        )
    return records


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_training()
    proxy_rows: list[dict[str, Any]] = []
    conclusions: list[dict[str, Any]] = []
    for empty_col, family in FAMILIES.items():
        if empty_col not in df.columns or not df[empty_col].isna().all():
            raise AssertionError(f"Expected a training-all-null column: {empty_col}")
        experiment = proxy_experiment(df, family)
        for row in experiment:
            row["empty_column_under_review"] = empty_col
            proxy_rows.append(row)
        conclusion = {
            "column": empty_col,
            "semantic_meaning": SEMANTICS[empty_col],
            "training_observations": 0,
            "analogous_observed_columns": family,
            "direct_supervised_model_possible": False,
            "identifiable_from_official_training_data": False,
            "reason": (
                "No target observations and no independently observed family total or exact conservation residual. "
                "Analogue models can test redundancy but cannot determine this missing unit's scale or regime."
            ),
            "recommended_action": (
                "Do not fabricate the raw column. Exclude it from model inputs, retain an availability record, "
                "and use aggregates/counts over observed sibling sensors instead."
            ),
            "proxy_experiment_count": len(experiment),
        }
        conclusions.append(conclusion)
        LOGGER.info("%s: not identifiable; analogue experiments=%d", empty_col, len(experiment))
    pd.DataFrame(proxy_rows).to_csv(
        AUDIT_DIR / "all_null_proxy_experiments.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "official training CSV files only",
        "official_test_accessed": False,
        "conclusions": conclusions,
        "overall_decision": (
            "All four training-all-null columns remain excluded. Model-based filling is not statistically "
            "identifiable from the supplied training data and would create unverifiable synthetic signals."
        ),
    }
    (AUDIT_DIR / "all_null_column_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
