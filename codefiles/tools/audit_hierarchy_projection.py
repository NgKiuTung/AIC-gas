"""Quantify the physical generator hierarchy projection on frozen nested predictions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    ROOT
    / "results"
    / "experiments"
    / "nested_temporal_validation"
    / "nested_outer_predictions_row_level.csv"
)
OUTPUT_PATH = ROOT / "results" / "production_validation" / "hierarchy_projection_audit.json"
KEYS = ["outer_window", "datetime", "horizon_step", "horizon_minutes"]


def mape(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - actual) / np.maximum(np.abs(actual), 1e-6)))


def main() -> None:
    rows = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    generator_1 = rows[rows["target"] == "generator_1"][KEYS + ["actual", "prediction"]].rename(
        columns={"actual": "actual_1", "prediction": "prediction_1"}
    )
    generator_all = rows[rows["target"] == "generator_all"][KEYS + ["actual", "prediction"]].rename(
        columns={"actual": "actual_all", "prediction": "prediction_all"}
    )
    joined = generator_1.merge(generator_all, on=KEYS, how="inner", validate="one_to_one")
    if len(joined) != len(generator_1) or not (joined["actual_all"] >= joined["actual_1"]).all():
        raise ValueError("Actual target hierarchy or key alignment failed")
    violation = joined["prediction_all"] < joined["prediction_1"]
    projected_all = np.maximum(joined["prediction_all"], joined["prediction_1"])
    generator_1_mape = mape(joined["actual_1"].to_numpy(), joined["prediction_1"].to_numpy())
    all_before = mape(joined["actual_all"].to_numpy(), joined["prediction_all"].to_numpy())
    all_after = mape(joined["actual_all"].to_numpy(), projected_all.to_numpy())
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_scoring_data_accessed": False,
        "rows": len(joined),
        "hierarchy_violations": int(violation.sum()),
        "hierarchy_violation_rate": float(violation.mean()),
        "maximum_violation": float(
            (joined.loc[violation, "prediction_1"] - joined.loc[violation, "prediction_all"]).max()
            if violation.any()
            else 0.0
        ),
        "generator_1_mape": generator_1_mape,
        "generator_all_mape_before_projection": all_before,
        "generator_all_mape_after_projection": all_after,
        "overall_mape_before_projection": (generator_1_mape + all_before) / 2.0,
        "overall_mape_after_projection": (generator_1_mape + all_after) / 2.0,
        "projection_mape_delta": (all_after - all_before) / 2.0,
        "projection_rule": "generator_all = max(generator_all, generator_1)",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
