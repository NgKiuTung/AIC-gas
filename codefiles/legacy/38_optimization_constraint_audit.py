"""Training-only audit for generation-optimization constraints.

This script deliberately reads only the preprocessed training table.  It
separates official hard constraints from empirically calibratable quantities
and from quantities that are not identifiable in the preliminary data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train.csv"
OUT_DIR = ROOT / "results" / "optimization" / "constraint_audit"
LOG_DIR = ROOT / "results" / "optimization" / "logs"


def setup_logger() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("optimization_constraint_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "38_optimization_constraint_audit.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def quantile_record(name: str, values: pd.Series, limit: float) -> dict[str, float | str | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "power_group": name,
        "observations": int(clean.size),
        "abs_delta_q50_mw_per_15min": float(clean.quantile(0.50)),
        "abs_delta_q90_mw_per_15min": float(clean.quantile(0.90)),
        "abs_delta_q95_mw_per_15min": float(clean.quantile(0.95)),
        "abs_delta_q99_mw_per_15min": float(clean.quantile(0.99)),
        "abs_delta_max_mw_per_15min": float(clean.max()),
        "official_aggregate_ramp_limit_mw_per_15min": float(limit),
        "observed_limit_violations": int((clean > limit + 1e-9).sum()),
    }


def feasible_online_counts(power: float, unit_capacity: float, total_units: int) -> tuple[int, ...]:
    if not np.isfinite(power):
        return ()
    feasible: list[int] = []
    if abs(power) <= 1e-9:
        feasible.append(0)
    for count in range(1, total_units + 1):
        lower = 0.60 * unit_capacity * count
        upper = unit_capacity * count
        if lower - 1e-9 <= power <= upper + 1e-9:
            feasible.append(count)
    return tuple(feasible)


def count_identifiability(power: pd.Series, name: str, unit_capacity: float, total_units: int) -> dict[str, object]:
    sets = power.map(lambda value: feasible_online_counts(float(value), unit_capacity, total_units))
    sizes = sets.map(len)
    examples = power.loc[sizes == 0].head(10).round(6).tolist()
    return {
        "power_group": name,
        "unit_capacity_mw": unit_capacity,
        "total_units": total_units,
        "rows": int(len(power)),
        "unique_online_count_rows": int((sizes == 1).sum()),
        "ambiguous_online_count_rows": int((sizes > 1).sum()),
        "infeasible_under_60_to_100pct_rows": int((sizes == 0).sum()),
        "unique_online_count_fraction": float((sizes == 1).mean()),
        "ambiguous_online_count_fraction": float((sizes > 1).mean()),
        "infeasible_fraction": float((sizes == 0).mean()),
        "infeasible_power_examples_mw": json.dumps(examples, ensure_ascii=False),
    }


def chronological_positive_regression(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    work = frame[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    split = int(len(work) * 0.8)
    train = work.iloc[:split]
    valid = work.iloc[split:]
    model = LinearRegression(positive=True)
    model.fit(train[features], train[target])
    pred = model.predict(valid[features])
    coefficients = pd.DataFrame(
        {
            "target": target,
            "term": ["intercept", *features],
            "coefficient": [float(model.intercept_), *[float(value) for value in model.coef_]],
            "unit_interpretation": [
                "MW",
                *["MW per (reported gas-flow unit)" for _ in features],
            ],
        }
    )
    metrics: dict[str, object] = {
        "target": target,
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "validation_start": str(valid.index.min()),
        "validation_end": str(valid.index.max()),
        "validation_mae_mw": float(mean_absolute_error(valid[target], pred)),
        "validation_mape": float(mean_absolute_percentage_error(valid[target], pred)),
        "validation_r2": float(r2_score(valid[target], pred)),
        "prediction_min_mw": float(pred.min()),
        "prediction_max_mw": float(pred.max()),
    }
    return coefficients, metrics


def holder_balance_regression(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    bfg_production = frame[
        ["blast_furnace_1", "blast_furnace_2", "blast_furnace_4", "blast_furnace_5"]
    ].sum(axis=1)
    air_heaters = frame[
        ["air_heater_1", "air_heater_2", "air_heater_4", "air_heater_5"]
    ].sum(axis=1)
    priority_users = frame[
        ["blast_furnace_user1", "blast_furnace_user2", "blast_furnace_user3", "blast_furnace_user4"]
    ].sum(axis=1)
    known_net_flow = (
        bfg_production
        - air_heaters
        - priority_users
        - frame["into_gas_mixed_blast_furnace"]
        - frame["generator_use_blast_furnace_gas"]
    )
    work = pd.DataFrame(
        {
            "known_net_flow": known_net_flow,
            "known_net_volume_15min": known_net_flow * 0.25,
            "observed_holder_delta": frame["blast_furnace_gas_holder_2"].shift(-1)
            - frame["blast_furnace_gas_holder_2"],
        },
        index=frame.index,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    split = int(len(work) * 0.8)
    train = work.iloc[:split]
    valid = work.iloc[split:]
    model = LinearRegression()
    model.fit(train[["known_net_volume_15min"]], train["observed_holder_delta"])
    pred = model.predict(valid[["known_net_volume_15min"]])
    coefficients = pd.DataFrame(
        {
            "term": ["intercept", "known_net_volume_15min"],
            "coefficient": [float(model.intercept_), float(model.coef_[0])],
            "interpretation": [
                "unobserved balance/bias per 15-minute step",
                "holder-volume response per computed known net volume",
            ],
        }
    )
    metrics: dict[str, object] = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "validation_start": str(valid.index.min()),
        "validation_end": str(valid.index.max()),
        "validation_mae_holder_units": float(mean_absolute_error(valid["observed_holder_delta"], pred)),
        "validation_r2": float(r2_score(valid["observed_holder_delta"], pred)),
        "pearson_correlation_full": float(work.corr().loc["known_net_volume_15min", "observed_holder_delta"]),
        "observed_delta_q01": float(work["observed_holder_delta"].quantile(0.01)),
        "observed_delta_q50": float(work["observed_holder_delta"].quantile(0.50)),
        "observed_delta_q99": float(work["observed_holder_delta"].quantile(0.99)),
        "warning": (
            "Balance is incomplete because blast_furnace_3, air_heater_3 and other possible flows "
            "are not observed. Use as a soft empirical relation only unless fit is strong."
        ),
    }
    return coefficients, metrics


def main() -> None:
    logger = setup_logger()
    logger.info("Reading training-only processed table: %s", TRAIN_PATH)
    if "评分所用测试集" in str(TRAIN_PATH):
        raise RuntimeError("Forbidden external scoring dataset path")
    frame = pd.read_csv(TRAIN_PATH, parse_dates=["datetime"])
    frame = frame.sort_values("datetime").set_index("datetime")
    logger.info("Rows=%d, columns=%d, range=%s to %s", len(frame), len(frame.columns), frame.index.min(), frame.index.max())

    frame["generator_120_group"] = frame["generator_all"] - frame["generator_1"]

    official_constraints = pd.DataFrame(
        [
            ("holder_total_capacity", "hard", 200000.0, "m3", "Official PDF section 6.1"),
            ("holder_safe_lower", "hard", 30000.0, "m3", "15% of official capacity"),
            ("holder_safe_upper", "hard", 180000.0, "m3", "90% of official capacity"),
            ("holder_high_high", "hard", 190000.0, "m3", "95% of official capacity"),
            ("50mw_unit_count", "hard", 4.0, "units", "Official PDF section 6.1"),
            ("50mw_unit_capacity", "hard", 50.0, "MW", "Official PDF section 6.1"),
            ("120mw_unit_count", "hard", 2.0, "units", "Official PDF section 6.1"),
            ("120mw_unit_capacity", "hard", 120.0, "MW", "Official PDF section 6.1"),
            ("normal_load_lower_fraction", "hard", 0.60, "fraction", "Official PDF section 6.1"),
            ("normal_load_upper_fraction", "hard", 1.00, "fraction", "Official PDF section 6.1"),
            ("ramp_fraction_per_minute", "hard", 0.10, "rated capacity/min", "Official PDF section 6.1"),
            ("priority_user_shortage", "hard", 0.0, "shortage", "Official PDF: no gas-user shortage"),
        ],
        columns=["constraint", "status", "value", "unit", "source"],
    )
    official_constraints.to_csv(OUT_DIR / "official_constraints.csv", index=False, encoding="utf-8-sig")

    online = pd.DataFrame(
        [
            count_identifiability(frame["generator_1"], "four_50mw_units", 50.0, 4),
            count_identifiability(frame["generator_120_group"], "two_120mw_units", 120.0, 2),
        ]
    )
    online.to_csv(OUT_DIR / "online_count_identifiability.csv", index=False, encoding="utf-8-sig")

    ramp = pd.DataFrame(
        [
            quantile_record("four_50mw_units", frame["generator_1"].diff().abs(), 4 * 50 * 0.10 * 15),
            quantile_record("two_120mw_units", frame["generator_120_group"].diff().abs(), 2 * 120 * 0.10 * 15),
            quantile_record("all_six_units", frame["generator_all"].diff().abs(), (4 * 50 + 2 * 120) * 0.10 * 15),
        ]
    )
    ramp.to_csv(OUT_DIR / "observed_ramp_statistics.csv", index=False, encoding="utf-8-sig")

    gas_features = [
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
    ]
    gas_coefficients, gas_metrics = chronological_positive_regression(frame, gas_features, "generator_all")
    gas_coefficients.to_csv(OUT_DIR / "gas_power_calibration.csv", index=False, encoding="utf-8-sig")

    holder_coefficients, holder_metrics = holder_balance_regression(frame)
    holder_coefficients.to_csv(OUT_DIR / "holder_balance_calibration.csv", index=False, encoding="utf-8-sig")

    holder = frame["blast_furnace_gas_holder_2"]
    holder_stats = {
        "observations": int(holder.notna().sum()),
        "minimum": float(holder.min()),
        "q01": float(holder.quantile(0.01)),
        "median": float(holder.median()),
        "q99": float(holder.quantile(0.99)),
        "maximum": float(holder.max()),
        "below_official_safe_lower_rows": int((holder < 30000).sum()),
        "above_official_safe_upper_rows": int((holder > 180000).sum()),
        "above_official_high_high_rows": int((holder > 190000).sum()),
    }

    identifiability = pd.DataFrame(
        [
            ("holder hard bounds", "identified", "Official PDF provides 30,000 to 180,000 m3 safe interval."),
            ("unit capacities and normal load", "identified", "Official PDF provides counts, ratings and 60%-100% range."),
            ("15-minute aggregate ramp bound", "identified but nonbinding", "Official 10%/min becomes larger than full group range over 15 minutes."),
            ("individual unit on/off history", "not identified", "Only aggregate generator_1 and generator_all are observed."),
            ("minimum up/down time", "not identified", "No official value or individual-unit state is supplied."),
            ("startup/shutdown cost", "not identified", "No official economic coefficient is supplied."),
            ("coal-gas calorific values", "not identified", "No calorific value/unit conversion table is supplied."),
            ("gas-to-power conversion", "empirically calibratable", "Contemporaneous generation gas-use fields and total load are observed."),
            ("complete BFG holder mass balance", "partially identified", "BF3/AH3 and possible additional flows are unobserved; relation must remain soft."),
            ("coke/converter gas storage states", "not identified", "No observed coke/converter holder level in the preliminary table."),
            ("future gas production and priority demand", "requires forecasting/scenarios", "Optimization must not use future realized training values in deployment."),
            ("flare quantity and penalty", "not identified", "No explicit flare field or penalty coefficient is supplied."),
            ("electricity tariff", "identified", "Official monthly time-of-use price workbook is deterministic and available."),
        ],
        columns=["item", "status", "evidence_or_consequence"],
    )
    identifiability.to_csv(OUT_DIR / "constraint_identifiability.csv", index=False, encoding="utf-8-sig")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "preprocessed training data and official task statement only",
        "source_training_file": str(TRAIN_PATH.relative_to(ROOT)),
        "external_scoring_data_accessed": False,
        "official_constraint_source": "docs/competition/official/煤气发电量预测与发电优化-2.pdf, section 6.1",
        "rows": int(len(frame)),
        "date_start": frame.index.min().isoformat(),
        "date_end": frame.index.max().isoformat(),
        "holder_observed_statistics": holder_stats,
        "gas_power_calibration_validation": gas_metrics,
        "holder_balance_calibration_validation": holder_metrics,
        "ramp_note": (
            "At 15-minute resolution the official 10%-of-rated-capacity/minute aggregate bounds "
            "are 300 MW for the 50 MW group and 360 MW for the 120 MW group, exceeding their "
            "entire 200/240 MW operating ranges. A finer grid or explicit startup model is needed "
            "for ramping to constrain decisions materially."
        ),
        "next_decision_gate": (
            "Do not build a hard gas-balance optimizer until empirical gas-power and holder-balance "
            "fit quality are reviewed. Non-identifiable startup cost, minimum up/down time, flare "
            "penalty and missing storage states require either explicit user assumptions or robust soft constraints."
        ),
    }
    (OUT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("Gas-power tail validation: MAPE=%.4f%%, R2=%.4f", gas_metrics["validation_mape"] * 100, gas_metrics["validation_r2"])
    logger.info("Holder-balance tail validation: MAE=%.3f, R2=%.4f, corr=%.4f", holder_metrics["validation_mae_holder_units"], holder_metrics["validation_r2"], holder_metrics["pearson_correlation_full"])
    logger.info("Holder observed range: %.3f to %.3f", holder_stats["minimum"], holder_stats["maximum"])
    logger.info("Wrote audit artifacts to %s", OUT_DIR)


if __name__ == "__main__":
    main()
