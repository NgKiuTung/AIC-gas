"""Conservative two-hour rolling generation-dispatch backtest.

The optimizer never reads the scoring dataset.  It uses chronological OOF
power forecasts as the future resource boundary, known tariff values, and only
the gas composition observed at the forecast origin.  Unknown exogenous gas
flows are cancelled through a relative-to-baseline holder formulation rather
than replaced by fabricated hard balance equations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "results" / "experiments" / "multidepth_ensemble" / "selected_oof_predictions.csv"
TRAIN_PATH = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train.csv"
OUT_DIR = ROOT / "results" / "optimization" / "dispatch_backtest"
LOG_DIR = ROOT / "results" / "optimization" / "logs"
DT_HOURS = 0.25
HORIZON_STEPS = 8
HOLDER_OFFICIAL_LOW = 30_000.0
HOLDER_OFFICIAL_HIGH = 180_000.0


@dataclass(frozen=True)
class Scenario:
    name: str
    deviation_50_mw: float
    deviation_120_mw: float
    ramp_50_mw_per_step: float
    ramp_120_mw_per_step: float
    holder_margin_m3: float
    deviation_penalty: float
    smooth_penalty: float


SCENARIOS = (
    Scenario("conservative", 10.0, 20.0, 15.0, 30.0, 5_000.0, 0.0010, 0.0040),
    Scenario("balanced", 20.0, 35.0, 25.0, 50.0, 5_000.0, 0.0002, 0.0020),
    Scenario("aggressive", 35.0, 60.0, 40.0, 80.0, 0.0, 0.0000, 0.0005),
)


def setup_logger() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("conservative_dispatch_backtest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "41_conservative_dispatch_backtest.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class Layout:
    def __init__(self, steps: int):
        self.steps = steps
        offset = 0
        for name in (
            "p50", "p120", "n50", "n120", "dev50", "dev120", "tv50", "tv120",
            "minslack50", "minslack120", "rampslack50", "rampslack120",
        ):
            setattr(self, name, np.arange(offset, offset + steps))
            offset += steps
        self.energy_slack = np.arange(offset, offset + 4)
        self.size = offset + 4


def add_row(rows: list[np.ndarray], lower: list[float], upper: list[float], row: np.ndarray, lb: float, ub: float) -> None:
    rows.append(row)
    lower.append(lb)
    upper.append(ub)


def solve_window(
    baseline_50: np.ndarray,
    baseline_120: np.ndarray,
    current_50: float,
    current_120: float,
    prices: np.ndarray,
    holder_initial: float,
    bfg_per_mw: float,
    scenario: Scenario,
) -> dict[str, object]:
    steps = len(prices)
    layout = Layout(steps)
    c = np.zeros(layout.size)
    c[layout.p50] = -prices * DT_HOURS
    c[layout.p120] = -prices * DT_HOURS
    c[layout.dev50] = scenario.deviation_penalty
    c[layout.dev120] = scenario.deviation_penalty
    c[layout.tv50] = scenario.smooth_penalty
    c[layout.tv120] = scenario.smooth_penalty
    c[layout.minslack50] = 0.20
    c[layout.minslack120] = 0.20
    c[layout.rampslack50] = 0.10
    c[layout.rampslack120] = 0.10
    c[layout.energy_slack] = 100.0

    lb = np.zeros(layout.size)
    ub = np.full(layout.size, np.inf)
    lb[layout.p50] = np.maximum(0.0, baseline_50 - scenario.deviation_50_mw)
    ub[layout.p50] = np.minimum(200.0, baseline_50 + scenario.deviation_50_mw)
    lb[layout.p120] = np.maximum(0.0, baseline_120 - scenario.deviation_120_mw)
    ub[layout.p120] = np.minimum(240.0, baseline_120 + scenario.deviation_120_mw)
    ub[layout.n50] = 4.0
    ub[layout.n120] = 2.0
    # Resource energy is a hard equality.  Soft transition variables absorb
    # unidentified startup/minimum-load behavior instead of fabricating energy.
    ub[layout.energy_slack] = 0.0

    integrality = np.zeros(layout.size, dtype=int)
    integrality[layout.n50] = 1
    integrality[layout.n120] = 1

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for t in range(steps):
        row = np.zeros(layout.size)
        row[layout.p50[t]], row[layout.n50[t]], row[layout.minslack50[t]] = 1.0, -30.0, 1.0
        add_row(rows, lower, upper, row, 0.0, np.inf)
        row = np.zeros(layout.size)
        row[layout.p50[t]], row[layout.n50[t]] = 1.0, -50.0
        add_row(rows, lower, upper, row, -np.inf, 0.0)
        row = np.zeros(layout.size)
        row[layout.p120[t]], row[layout.n120[t]], row[layout.minslack120[t]] = 1.0, -72.0, 1.0
        add_row(rows, lower, upper, row, 0.0, np.inf)
        row = np.zeros(layout.size)
        row[layout.p120[t]], row[layout.n120[t]] = 1.0, -120.0
        add_row(rows, lower, upper, row, -np.inf, 0.0)

        for p_idx, d_idx, baseline in (
            (layout.p50[t], layout.dev50[t], baseline_50[t]),
            (layout.p120[t], layout.dev120[t], baseline_120[t]),
        ):
            row = np.zeros(layout.size)
            row[p_idx], row[d_idx] = 1.0, -1.0
            add_row(rows, lower, upper, row, -np.inf, float(baseline))
            row = np.zeros(layout.size)
            row[p_idx], row[d_idx] = -1.0, -1.0
            add_row(rows, lower, upper, row, -np.inf, -float(baseline))

        for p_indices, tv_idx, current in (
            (layout.p50, layout.tv50[t], current_50),
            (layout.p120, layout.tv120[t], current_120),
        ):
            row = np.zeros(layout.size)
            row[p_indices[t]], row[tv_idx] = 1.0, -1.0
            if t == 0:
                add_row(rows, lower, upper, row, -np.inf, current)
            else:
                row[p_indices[t - 1]] = -1.0
                add_row(rows, lower, upper, row, -np.inf, 0.0)
            row = np.zeros(layout.size)
            row[p_indices[t]], row[tv_idx] = -1.0, -1.0
            if t == 0:
                add_row(rows, lower, upper, row, -np.inf, -current)
            else:
                row[p_indices[t - 1]] = 1.0
                add_row(rows, lower, upper, row, -np.inf, 0.0)

        row = np.zeros(layout.size)
        row[layout.tv50[t]], row[layout.rampslack50[t]] = 1.0, -1.0
        add_row(rows, lower, upper, row, -np.inf, scenario.ramp_50_mw_per_step)
        row = np.zeros(layout.size)
        row[layout.tv120[t]], row[layout.rampslack120[t]] = 1.0, -1.0
        add_row(rows, lower, upper, row, -np.inf, scenario.ramp_120_mw_per_step)

    row = np.zeros(layout.size)
    row[layout.p50] = 1.0
    row[layout.energy_slack[0]], row[layout.energy_slack[1]] = -1.0, 1.0
    add_row(rows, lower, upper, row, float(baseline_50.sum()), float(baseline_50.sum()))
    row = np.zeros(layout.size)
    row[layout.p120] = 1.0
    row[layout.energy_slack[2]], row[layout.energy_slack[3]] = -1.0, 1.0
    add_row(rows, lower, upper, row, float(baseline_120.sum()), float(baseline_120.sum()))

    internal_low = HOLDER_OFFICIAL_LOW + scenario.holder_margin_m3
    internal_high = HOLDER_OFFICIAL_HIGH - scenario.holder_margin_m3
    if holder_initial < internal_low:
        internal_low = HOLDER_OFFICIAL_LOW
    if holder_initial > internal_high:
        internal_high = HOLDER_OFFICIAL_HIGH
    holder_factor = DT_HOURS * bfg_per_mw
    for t in range(steps):
        row = np.zeros(layout.size)
        row[layout.p50[: t + 1]] = holder_factor
        row[layout.p120[: t + 1]] = holder_factor
        baseline_cumulative = float((baseline_50[: t + 1] + baseline_120[: t + 1]).sum())
        add_row(
            rows,
            lower,
            upper,
            row,
            holder_initial + holder_factor * baseline_cumulative - internal_high,
            holder_initial + holder_factor * baseline_cumulative - internal_low,
        )

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True, "time_limit": 3.0, "mip_rel_gap": 1e-7},
    )
    if not result.success or result.x is None:
        return {"success": False, "status": int(result.status), "message": str(result.message)}
    values = result.x
    p50 = values[layout.p50]
    p120 = values[layout.p120]
    total = p50 + p120
    baseline_total = baseline_50 + baseline_120
    holder = holder_initial - holder_factor * np.cumsum(total - baseline_total)
    return {
        "success": True,
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "p50": p50,
        "p120": p120,
        "n50": np.rint(values[layout.n50]).astype(int),
        "n120": np.rint(values[layout.n120]).astype(int),
        "holder": holder,
        "energy_slack_50_step_mw": float(values[layout.energy_slack[0]] + values[layout.energy_slack[1]]),
        "energy_slack_120_step_mw": float(values[layout.energy_slack[2]] + values[layout.energy_slack[3]]),
        "minimum_load_slack_50": values[layout.minslack50],
        "minimum_load_slack_120": values[layout.minslack120],
        "preferred_ramp_slack_50": values[layout.rampslack50],
        "preferred_ramp_slack_120": values[layout.rampslack120],
    }


def prepare_windows(oof: pd.DataFrame) -> pd.DataFrame:
    work = oof.copy()
    work["forecast_origin"] = work["datetime"] - pd.to_timedelta(work["horizon_minutes"], unit="m")
    keys = ["fold", "forecast_origin", "datetime", "horizon_step", "horizon_minutes"]
    pred = work.pivot(index=keys, columns="target", values="prediction").reset_index()
    actual = work.pivot(index=keys, columns="target", values="actual").reset_index()
    current = work.pivot(index=keys, columns="target", values="current").reset_index()
    pred = pred.rename(columns={"generator_1": "pred_50", "generator_all": "pred_all"})
    actual = actual.rename(columns={"generator_1": "actual_50", "generator_all": "actual_all"})
    current = current.rename(columns={"generator_1": "current_50", "generator_all": "current_all"})
    joined = pred.merge(actual, on=keys, validate="one_to_one").merge(current, on=keys, validate="one_to_one")
    joined = joined[
        ((joined["forecast_origin"].dt.hour * 60 + joined["forecast_origin"].dt.minute) % 120 == 0)
    ].copy()
    counts = joined.groupby(["fold", "forecast_origin"])["horizon_step"].nunique()
    complete = counts[counts == HORIZON_STEPS].index
    joined = joined.set_index(["fold", "forecast_origin"]).loc[complete].reset_index()
    return joined.sort_values(["fold", "forecast_origin", "horizon_step"])


def summarize_results(windows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scenario, group in windows.groupby("scenario"):
        successful = group[group["solver_success"]]
        varying = successful[successful["price_levels"] > 1]
        records.append(
            {
                "scenario": scenario,
                "windows": int(len(group)),
                "solver_success_rate": float(group["solver_success"].mean()),
                "price_varying_windows": int(len(varying)),
                "mean_revenue_uplift_all": float(successful["forecast_revenue_uplift"].mean()),
                "median_revenue_uplift_all": float(successful["forecast_revenue_uplift"].median()),
                "mean_revenue_uplift_price_varying": float(varying["forecast_revenue_uplift"].mean()) if len(varying) else np.nan,
                "median_revenue_uplift_price_varying": float(varying["forecast_revenue_uplift"].median()) if len(varying) else np.nan,
                "positive_uplift_fraction_price_varying": float((varying["forecast_revenue_uplift"] > 1e-10).mean()) if len(varying) else np.nan,
                "mean_weighted_load_ratio_gain": float((successful["optimized_price_weighted_load_ratio"] - successful["baseline_price_weighted_load_ratio"]).mean()),
                "max_energy_slack_fraction": float(successful["energy_slack_fraction"].max()),
                "official_holder_violations": int(successful["official_holder_violations"].sum()),
                "official_capacity_violations": int(successful["official_capacity_violations"].sum()),
                "official_ramp_violations": int(successful["official_ramp_violations"].sum()),
                "normal_minload_transition_fraction": float(successful["normal_minload_transition_fraction"].mean()),
                "preferred_ramp_exceedance_fraction": float(successful["preferred_ramp_exceedance_fraction"].mean()),
                "mean_absolute_dispatch_change_mw": float(successful["mean_absolute_dispatch_change_mw"].mean()),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    logger = setup_logger()
    for path in (OOF_PATH, TRAIN_PATH):
        if "评分所用测试集" in str(path):
            raise RuntimeError("Forbidden external scoring dataset path")
    logger.info("Reading training-only OOF forecasts and processed training table")
    oof = pd.read_csv(OOF_PATH, parse_dates=["datetime"])
    train = pd.read_csv(TRAIN_PATH, parse_dates=["datetime"]).sort_values("datetime").set_index("datetime")
    forecast_rows = prepare_windows(oof)
    logger.info("Prepared %d complete non-overlapping two-hour forecast rows across %d windows", len(forecast_rows), forecast_rows.groupby(["fold", "forecast_origin"]).ngroups)

    gas_cols = [
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
    ]
    intensity = train[gas_cols].div(train["generator_all"].clip(lower=1e-6), axis=0)
    ratio_limits = {column: (float(intensity[column].quantile(0.05)), float(intensity[column].quantile(0.95))) for column in gas_cols}

    schedule_records: list[dict[str, object]] = []
    window_records: list[dict[str, object]] = []
    failure_records: list[dict[str, object]] = []
    grouped = forecast_rows.groupby(["fold", "forecast_origin"], sort=True)
    for window_number, ((fold, origin), group) in enumerate(grouped, start=1):
        group = group.sort_values("horizon_step")
        if origin not in train.index or not set(group["datetime"]).issubset(train.index):
            failure_records.append({"fold": fold, "forecast_origin": origin, "reason": "timestamp_not_in_training_table"})
            continue
        origin_row = train.loc[origin]
        holder_initial = float(origin_row["blast_furnace_gas_holder_2"])
        if not (HOLDER_OFFICIAL_LOW <= holder_initial <= HOLDER_OFFICIAL_HIGH):
            failure_records.append({"fold": fold, "forecast_origin": origin, "reason": "initial_holder_outside_official_bounds", "holder_initial": holder_initial})
            continue
        current_50 = float(np.clip(group["current_50"].iloc[0], 0.0, 200.0))
        current_all = float(np.clip(group["current_all"].iloc[0], current_50, 440.0))
        current_120 = float(np.clip(current_all - current_50, 0.0, 240.0))
        baseline_50 = np.clip(group["pred_50"].to_numpy(float), 0.0, 200.0)
        baseline_all = np.clip(group["pred_all"].to_numpy(float), baseline_50, 440.0)
        baseline_120 = np.clip(baseline_all - baseline_50, 0.0, 240.0)
        baseline_all = baseline_50 + baseline_120
        prices = train.loc[group["datetime"], "feat_known_price"].to_numpy(float)
        actual_50 = np.clip(group["actual_50"].to_numpy(float), 0.0, 200.0)
        actual_all = np.clip(group["actual_all"].to_numpy(float), actual_50, 440.0)
        ratios: dict[str, float] = {}
        for column in gas_cols:
            raw_ratio = float(origin_row[column] / max(float(origin_row["generator_all"]), 1e-6))
            ratios[column] = float(np.clip(raw_ratio, *ratio_limits[column]))

        finite_inputs = np.r_[baseline_50, baseline_120, prices, actual_50, actual_all, holder_initial, list(ratios.values())]
        if not np.isfinite(finite_inputs).all():
            failure_records.append({"fold": fold, "forecast_origin": origin, "reason": "nonfinite_training_or_oof_input"})
            continue

        for scenario in SCENARIOS:
            solved = solve_window(
                baseline_50=baseline_50,
                baseline_120=baseline_120,
                current_50=current_50,
                current_120=current_120,
                prices=prices,
                holder_initial=holder_initial,
                bfg_per_mw=ratios["generator_use_blast_furnace_gas"],
                scenario=scenario,
            )
            if not solved["success"]:
                failure_records.append(
                    {"fold": fold, "forecast_origin": origin, "scenario": scenario.name, "reason": "solver_failure", "status": solved["status"], "message": solved["message"]}
                )
                window_records.append({"fold": fold, "forecast_origin": origin, "scenario": scenario.name, "solver_success": False})
                continue
            opt_50 = np.asarray(solved["p50"], dtype=float)
            opt_120 = np.asarray(solved["p120"], dtype=float)
            opt_all = opt_50 + opt_120
            n50 = np.asarray(solved["n50"], dtype=int)
            n120 = np.asarray(solved["n120"], dtype=int)
            holder = np.asarray(solved["holder"], dtype=float)
            min_slack_50 = np.asarray(solved["minimum_load_slack_50"], dtype=float)
            min_slack_120 = np.asarray(solved["minimum_load_slack_120"], dtype=float)
            ramp_slack_50 = np.asarray(solved["preferred_ramp_slack_50"], dtype=float)
            ramp_slack_120 = np.asarray(solved["preferred_ramp_slack_120"], dtype=float)
            baseline_revenue = float(np.sum(prices * baseline_all * DT_HOURS * 1000.0))
            optimized_revenue = float(np.sum(prices * opt_all * DT_HOURS * 1000.0))
            historical_revenue = float(np.sum(prices * actual_all * DT_HOURS * 1000.0))
            mean_price = float(prices.mean())
            baseline_pwl = float(np.sum(prices * baseline_all) / max(mean_price * np.sum(baseline_all), 1e-9))
            optimized_pwl = float(np.sum(prices * opt_all) / max(mean_price * np.sum(opt_all), 1e-9))
            energy_slack = float(solved["energy_slack_50_step_mw"] + solved["energy_slack_120_step_mw"])
            capacity_violations = int(
                np.sum((opt_50 < -1e-5) | (opt_50 > 200.0 + 1e-5))
                + np.sum((opt_120 < -1e-5) | (opt_120 > 240.0 + 1e-5))
            )
            changes_50 = np.abs(np.diff(np.r_[current_50, opt_50]))
            changes_120 = np.abs(np.diff(np.r_[current_120, opt_120]))
            official_ramp_violations = int(
                np.sum(changes_50 > 300.0 + 1e-5)
                + np.sum(changes_120 > 360.0 + 1e-5)
            )
            holder_violations = int(np.sum((holder < HOLDER_OFFICIAL_LOW - 1e-5) | (holder > HOLDER_OFFICIAL_HIGH + 1e-5)))
            window_records.append(
                {
                    "fold": fold,
                    "forecast_origin": origin,
                    "scenario": scenario.name,
                    "solver_success": True,
                    "price_levels": int(pd.Series(prices).nunique()),
                    "baseline_forecast_revenue_cny": baseline_revenue,
                    "optimized_forecast_revenue_cny": optimized_revenue,
                    "historical_revenue_cny": historical_revenue,
                    "forecast_revenue_uplift": (optimized_revenue - baseline_revenue) / max(abs(baseline_revenue), 1e-9),
                    "optimized_vs_historical_revenue": (optimized_revenue - historical_revenue) / max(abs(historical_revenue), 1e-9),
                    "baseline_price_weighted_load_ratio": baseline_pwl,
                    "optimized_price_weighted_load_ratio": optimized_pwl,
                    "energy_slack_step_mw": energy_slack,
                    "energy_slack_fraction": energy_slack / max(float(np.sum(baseline_all)), 1e-9),
                    "official_holder_violations": holder_violations,
                    "official_capacity_violations": capacity_violations,
                    "official_ramp_violations": official_ramp_violations,
                    "normal_minload_transition_steps": int(np.sum(min_slack_50 > 1e-5) + np.sum(min_slack_120 > 1e-5)),
                    "normal_minload_transition_fraction": float((np.sum(min_slack_50 > 1e-5) + np.sum(min_slack_120 > 1e-5)) / (2 * HORIZON_STEPS)),
                    "minimum_load_shortfall_step_mw": float(min_slack_50.sum() + min_slack_120.sum()),
                    "preferred_ramp_exceedance_steps": int(np.sum(ramp_slack_50 > 1e-5) + np.sum(ramp_slack_120 > 1e-5)),
                    "preferred_ramp_exceedance_fraction": float((np.sum(ramp_slack_50 > 1e-5) + np.sum(ramp_slack_120 > 1e-5)) / (2 * HORIZON_STEPS)),
                    "preferred_ramp_excess_step_mw": float(ramp_slack_50.sum() + ramp_slack_120.sum()),
                    "holder_min_m3": float(holder.min()),
                    "holder_max_m3": float(holder.max()),
                    "mean_absolute_dispatch_change_mw": float(np.mean(np.abs(opt_all - baseline_all))),
                    "maximum_absolute_dispatch_change_mw": float(np.max(np.abs(opt_all - baseline_all))),
                }
            )
            for i, row in enumerate(group.itertuples(index=False)):
                record = {
                    "fold": fold,
                    "forecast_origin": origin,
                    "target_datetime": row.datetime,
                    "horizon_step": int(row.horizon_step),
                    "horizon_minutes": int(row.horizon_minutes),
                    "scenario": scenario.name,
                    "price_cny_per_kwh": float(prices[i]),
                    "baseline_generator_1_mw": float(baseline_50[i]),
                    "baseline_generator_120_group_mw": float(baseline_120[i]),
                    "baseline_generator_all_mw": float(baseline_all[i]),
                    "actual_generator_1_mw": float(actual_50[i]),
                    "actual_generator_all_mw": float(actual_all[i]),
                    "optimized_generator_1_mw": float(opt_50[i]),
                    "optimized_generator_120_group_mw": float(opt_120[i]),
                    "optimized_generator_all_mw": float(opt_all[i]),
                    "online_50mw_units": int(n50[i]),
                    "online_120mw_units": int(n120[i]),
                    "projected_bfg_holder_m3": float(holder[i]),
                    "minimum_load_transition_slack_50_mw": float(min_slack_50[i]),
                    "minimum_load_transition_slack_120_mw": float(min_slack_120[i]),
                    "preferred_ramp_excess_50_mw": float(ramp_slack_50[i]),
                    "preferred_ramp_excess_120_mw": float(ramp_slack_120[i]),
                }
                for gas_column in gas_cols:
                    record[f"baseline_{gas_column}"] = float(ratios[gas_column] * baseline_all[i])
                    record[f"opt_{gas_column}"] = float(ratios[gas_column] * opt_all[i])
                schedule_records.append(record)
        if window_number % 200 == 0:
            logger.info("Processed %d/%d forecast windows", window_number, grouped.ngroups)

    schedules = pd.DataFrame(schedule_records)
    windows = pd.DataFrame(window_records)
    failures = pd.DataFrame(failure_records)
    scenario_summary = summarize_results(windows)
    schedules.to_csv(OUT_DIR / "optimized_schedules.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(OUT_DIR / "rolling_window_metrics.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(OUT_DIR / "failures_and_exclusions.csv", index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(OUT_DIR / "scenario_summary.csv", index=False, encoding="utf-8-sig")

    acceptance = {}
    for row in scenario_summary.itertuples(index=False):
        acceptance[row.scenario] = {
            "solver_success_rate_at_least_99pct": bool(row.solver_success_rate >= 0.99),
            "official_hard_constraint_violations_zero": bool(
                row.official_holder_violations == 0 and row.official_capacity_violations == 0 and row.official_ramp_violations == 0
            ),
            "max_energy_slack_fraction_below_0_5pct": bool(row.max_energy_slack_fraction <= 0.005),
            "mean_price_varying_window_uplift_positive": bool(row.mean_revenue_uplift_price_varying > 0),
        }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only chronological OOF forecasts plus processed training covariates",
        "external_scoring_data_accessed": False,
        "optimization_horizon": "8 x 15 minutes",
        "forecast_window_sampling": "non-overlapping two-hour origins only",
        "solver": "scipy.optimize.milp (HiGHS)",
        "official_constraints": {
            "holder_m3": [HOLDER_OFFICIAL_LOW, HOLDER_OFFICIAL_HIGH],
            "50mw_units": 4,
            "120mw_units": 2,
            "online_loading_fraction": [0.6, 1.0],
        },
        "relative_holder_method": (
            "Projected holder equals origin holder minus cumulative incremental BFG use versus the forecast baseline. "
            "Unknown exogenous flows cancel by construction; this is a conservative scheduling-delta model, not a full plant mass balance."
        ),
        "gas_conversion_method": (
            "Each window uses the three gas-per-MW ratios observed at the forecast origin, clipped to training-only 5th-95th percentiles."
        ),
        "transition_treatment": (
            "The official 60%-100% range is enforced for normal operation through integer online counts. "
            "A separately reported, penalized minimum-load slack represents unidentified startup/shutdown or aggregate-meter transition states; resource energy remains a hard equality."
        ),
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "gas_ratio_training_limits": ratio_limits,
        "scenario_summary": scenario_summary.replace({np.nan: None}).to_dict(orient="records"),
        "acceptance_checks": acceptance,
        "important_limitations": [
            "This is an OOF historical backtest, not scoring-test evaluation.",
            "Forecast-relative uplift is the primary economic metric; optimized-vs-historical revenue is secondary because forecast energy differs from realized energy.",
            "No startup cost or minimum up/down time is invented because the official preliminary data do not identify them.",
            "The relative holder model checks incremental scheduling risk but cannot replace a complete plant gas balance when future-stage data provide all flows.",
        ],
    }
    (OUT_DIR / "optimization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Scenario summary:\n%s", scenario_summary.to_string(index=False))
    logger.info("Wrote dispatch backtest artifacts to %s", OUT_DIR)


if __name__ == "__main__":
    main()
