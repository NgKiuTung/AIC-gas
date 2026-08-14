"""Diagnose nonlinear gas-power conversion and holder-balance alignment.

Only the preprocessed preliminary training table is read.  The purpose is to
decide whether empirical relations are strong enough for optimization hard
constraints; this script does not optimize or access scoring data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train.csv"
OUT_DIR = ROOT / "results" / "optimization" / "mechanism_diagnostics"
LOG_DIR = ROOT / "results" / "optimization" / "logs"


def setup_logger() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("resource_mechanism_diagnostics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "39_resource_mechanism_diagnostics.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def evaluate(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_mw": float(mean_absolute_error(y_true, pred)),
        "mape": float(mean_absolute_percentage_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


def nonlinear_gas_power(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    gas = [
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
    ]
    work = frame[gas + ["generator_all"]].replace([np.inf, -np.inf], np.nan).dropna()
    split = int(len(work) * 0.8)
    train, valid = work.iloc[:split], work.iloc[split:]
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=30,
        subsample=0.85,
        colsample_bytree=1.0,
        reg_alpha=2.0,
        reg_lambda=30.0,
        tree_method="hist",
        device="cuda",
        random_state=20260803,
        n_jobs=4,
    )
    model.fit(train[gas], train["generator_all"], verbose=False)
    pred = model.predict(valid[gas])
    metrics = evaluate(valid["generator_all"], pred)
    metrics.update(
        {
            "train_rows": int(len(train)),
            "validation_rows": int(len(valid)),
            "validation_start": valid.index.min().isoformat(),
            "validation_end": valid.index.max().isoformat(),
            "model": "XGBoost nonlinear contemporaneous conversion",
        }
    )
    predictions = pd.DataFrame(
        {
            "datetime": valid.index,
            "actual_generator_all": valid["generator_all"].to_numpy(),
            "predicted_generator_all": pred,
        }
    )
    predictions["absolute_percentage_error"] = (
        (predictions["predicted_generator_all"] - predictions["actual_generator_all"]).abs()
        / predictions["actual_generator_all"].abs().clip(lower=1e-6)
    )
    importance = pd.DataFrame(
        {"feature": gas, "importance": model.feature_importances_.astype(float)}
    ).sort_values("importance", ascending=False)
    return predictions, importance, metrics


def gas_intensity_bins(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[
        [
            "generator_all",
            "generator_use_blast_furnace_gas",
            "generator_use_coke_gas",
            "generator_use_converter_gas",
        ]
    ].copy()
    work["load_bin"] = pd.cut(
        work["generator_all"],
        bins=[0, 120, 180, 240, 300, 360, 440, np.inf],
        right=False,
    )
    gas_cols = [column for column in work if column.startswith("generator_use_")]
    records: list[dict[str, object]] = []
    for load_bin, group in work.groupby("load_bin", observed=True):
        record: dict[str, object] = {
            "load_bin": str(load_bin),
            "rows": int(len(group)),
            "load_mean_mw": float(group["generator_all"].mean()),
            "load_q10_mw": float(group["generator_all"].quantile(0.10)),
            "load_q90_mw": float(group["generator_all"].quantile(0.90)),
        }
        for column in gas_cols:
            intensity = group[column] / group["generator_all"].clip(lower=1e-6)
            record[f"{column}_per_mw_median"] = float(intensity.median())
            record[f"{column}_per_mw_q10"] = float(intensity.quantile(0.10))
            record[f"{column}_per_mw_q90"] = float(intensity.quantile(0.90))
        records.append(record)
    return pd.DataFrame(records)


def holder_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    production = frame[
        ["blast_furnace_1", "blast_furnace_2", "blast_furnace_4", "blast_furnace_5"]
    ].sum(axis=1)
    heaters = frame[["air_heater_1", "air_heater_2", "air_heater_4", "air_heater_5"]].sum(axis=1)
    users = frame[
        ["blast_furnace_user1", "blast_furnace_user2", "blast_furnace_user3", "blast_furnace_user4"]
    ].sum(axis=1)
    mixed = frame["into_gas_mixed_blast_furnace"]
    generation = frame["generator_use_blast_furnace_gas"]
    return pd.DataFrame(
        {
            "full_known_balance": production - heaters - users - mixed - generation,
            "exclude_mixed_station": production - heaters - users - generation,
            "exclude_priority_users": production - heaters - mixed - generation,
            "production_minus_generation": production - generation,
            "all_known_consumption": heaters + users + mixed + generation,
        },
        index=frame.index,
    )


def holder_alignment_scan(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = holder_candidates(frame)
    holder = frame["blast_furnace_gas_holder_2"]
    records: list[dict[str, object]] = []
    # Positive lag means use a later flow row against the current holder change.
    for name in candidates.columns:
        for lag_steps in range(-8, 9):
            flow = candidates[name].shift(-lag_steps) * 0.25
            delta = holder.shift(-1) - holder
            pair = pd.concat([flow.rename("flow"), delta.rename("delta")], axis=1).dropna()
            records.append(
                {
                    "balance_definition": name,
                    "flow_lag_steps": lag_steps,
                    "flow_lag_minutes": lag_steps * 15,
                    "aggregation_steps": 1,
                    "rows": int(len(pair)),
                    "correlation": float(pair.corr().loc["flow", "delta"]),
                }
            )
    # Check whether integration suppresses timestamp noise.
    for name in candidates.columns:
        for steps in [2, 4, 8, 16, 32, 96]:
            net_volume = candidates[name].rolling(steps).sum() * 0.25
            holder_delta = holder - holder.shift(steps)
            pair = pd.concat(
                [net_volume.rename("flow"), holder_delta.rename("delta")], axis=1
            ).dropna()
            split = int(len(pair) * 0.8)
            train, valid = pair.iloc[:split], pair.iloc[split:]
            slope, intercept = np.polyfit(train["flow"], train["delta"], deg=1)
            pred = intercept + slope * valid["flow"]
            records.append(
                {
                    "balance_definition": name,
                    "flow_lag_steps": 0,
                    "flow_lag_minutes": 0,
                    "aggregation_steps": steps,
                    "rows": int(len(pair)),
                    "correlation": float(pair.corr().loc["flow", "delta"]),
                    "tail_r2": float(r2_score(valid["delta"], pred)),
                    "tail_mae_holder_units": float(mean_absolute_error(valid["delta"], pred)),
                    "slope": float(slope),
                    "intercept": float(intercept),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    logger = setup_logger()
    if "评分所用测试集" in str(TRAIN_PATH):
        raise RuntimeError("Forbidden external scoring dataset path")
    logger.info("Reading training-only table: %s", TRAIN_PATH)
    frame = pd.read_csv(TRAIN_PATH, parse_dates=["datetime"]).sort_values("datetime").set_index("datetime")

    predictions, importance, nonlinear_metrics = nonlinear_gas_power(frame)
    predictions.to_csv(OUT_DIR / "nonlinear_gas_power_tail_predictions.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(OUT_DIR / "nonlinear_gas_power_importance.csv", index=False, encoding="utf-8-sig")
    bins = gas_intensity_bins(frame)
    bins.to_csv(OUT_DIR / "gas_intensity_by_load_bin.csv", index=False, encoding="utf-8-sig")

    holder_scan = holder_alignment_scan(frame)
    holder_scan.to_csv(OUT_DIR / "holder_balance_alignment_scan.csv", index=False, encoding="utf-8-sig")
    best_single = holder_scan.loc[holder_scan["aggregation_steps"] == 1].iloc[
        holder_scan.loc[holder_scan["aggregation_steps"] == 1, "correlation"].abs().argmax()
    ]
    aggregated = holder_scan.loc[holder_scan["aggregation_steps"] > 1].copy()
    best_aggregate = aggregated.iloc[aggregated["tail_r2"].argmax()]

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "preprocessed preliminary training data only",
        "external_scoring_data_accessed": False,
        "gpu_training": "XGBoost device=cuda",
        "nonlinear_gas_power_tail_validation": nonlinear_metrics,
        "best_single_step_holder_alignment": best_single.replace({np.nan: None}).to_dict(),
        "best_aggregated_holder_balance": best_aggregate.replace({np.nan: None}).to_dict(),
        "hard_constraint_decision_rule": {
            "gas_power": "hard only if tail validation is sufficiently accurate and stable; otherwise use conservative envelope/soft surrogate",
            "holder_balance": "hard only if temporal tail R2 is positive and the balance definition is physically complete",
        },
    }
    (OUT_DIR / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "Nonlinear gas-power tail validation: MAPE=%.4f%%, R2=%.4f",
        nonlinear_metrics["mape"] * 100,
        nonlinear_metrics["r2"],
    )
    logger.info(
        "Best single-step holder alignment: definition=%s, lag=%s min, corr=%.4f",
        best_single["balance_definition"],
        best_single["flow_lag_minutes"],
        best_single["correlation"],
    )
    logger.info(
        "Best aggregated holder fit: definition=%s, steps=%s, tail R2=%.4f",
        best_aggregate["balance_definition"],
        best_aggregate["aggregation_steps"],
        best_aggregate["tail_r2"],
    )
    logger.info("Wrote diagnostics to %s", OUT_DIR)


if __name__ == "__main__":
    main()
