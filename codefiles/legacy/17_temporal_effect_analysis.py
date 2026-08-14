"""Audit daily peak/trough, weekday, monthly, and time-feature effects."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_OOF = ROOT / "results" / "experiments" / "xgboost_direct" / "raw_oof_predictions_long.csv"
TIME_OOF = ROOT / "results" / "experiments" / "time_feature_ablation" / "raw_oof_predictions_long.csv"
CAUSAL_TRAIN = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train_causal.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "temporal_effects"
BETA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def session(hour: pd.Series) -> pd.Categorical:
    return pd.cut(
        hour,
        bins=[-1, 5, 8, 16, 21, 23],
        labels=["night_00_06", "morning_06_09", "day_09_17", "evening_17_22", "late_22_24"],
    )


def load_joined() -> pd.DataFrame:
    base = pd.read_csv(BASE_OOF, encoding="utf-8-sig", parse_dates=["datetime"])
    base = base[base["model"] == "direct_d5"].copy()
    time = pd.read_csv(TIME_OOF, encoding="utf-8-sig", parse_dates=["datetime"])
    keys = ["datetime", "fold", "target", "horizon_step", "horizon_minutes", "actual", "current"]
    joined = base[keys + ["raw_correction"]].merge(
        time[keys + ["raw_correction"]], on=keys, how="inner", validate="one_to_one",
        suffixes=("_base", "_time"),
    )
    if len(joined) != len(base):
        raise ValueError("Base/time OOF predictions are not aligned")
    return joined


def tune_linear_beta(frame: pd.DataFrame, correction_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameter_rows: list[dict[str, object]] = []
    output_parts: list[pd.DataFrame] = []
    for target, group in frame.groupby("target"):
        h = (group["horizon_step"].to_numpy(dtype=float) - 1.0) / 7.0
        truth = group["actual"].to_numpy(dtype=float)
        current = group["current"].to_numpy(dtype=float)
        correction = group[correction_col].to_numpy(dtype=float)
        best: tuple[float, float, float] | None = None
        for start in BETA_GRID:
            for end in BETA_GRID:
                beta = start + (end - start) * h
                ape = np.abs(current + beta * correction - truth) / np.maximum(np.abs(truth), 1e-6)
                temp = group[["fold", "horizon_step"]].copy()
                temp["ape"] = ape
                score = float(temp.groupby(["fold", "horizon_step"])["ape"].mean().mean())
                if best is None or score < best[2]:
                    best = (float(start), float(end), score)
        assert best is not None
        start, end, score = best
        parameter_rows.append(
            {"target": target, "beta_h15": start, "beta_h120": end, "mean_mape": score}
        )
        selected = group.copy()
        selected["beta"] = start + (end - start) * h
        selected["prediction"] = np.maximum(current + selected["beta"].to_numpy() * correction, 0.0)
        selected["ape"] = np.abs(selected["prediction"] - truth) / np.maximum(np.abs(truth), 1e-6)
        output_parts.append(selected)
    return pd.DataFrame(parameter_rows), pd.concat(output_parts, ignore_index=True)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    joined = load_joined()
    models: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    for model, column in (("base_features", "raw_correction_base"), ("time_augmented", "raw_correction_time")):
        parameters, predictions = tune_linear_beta(joined, column)
        parameters["model"] = model
        predictions["model"] = model
        parameter_frames.append(parameters)
        models.append(predictions)
    rows = pd.concat(models, ignore_index=True)
    rows["forecast_datetime"] = rows["datetime"] + pd.to_timedelta(rows["horizon_minutes"], unit="m")
    rows["forecast_hour"] = rows["forecast_datetime"].dt.hour
    rows["forecast_session"] = session(rows["forecast_hour"])
    rows["day_night"] = np.where(rows["forecast_hour"].between(7, 18), "day_07_19", "night_19_07")
    rows["weekday_weekend"] = np.where(rows["forecast_datetime"].dt.dayofweek >= 5, "weekend", "weekday")
    rows["forecast_month"] = rows["forecast_datetime"].dt.to_period("M").astype(str)

    overall = (
        rows.groupby(["model", "fold", "target", "horizon_step"], as_index=False)["ape"].mean()
        .groupby("model", as_index=False)
        .agg(mean_mape=("ape", "mean"), std_fold_target_horizon_mape=("ape", "std"))
        .sort_values("mean_mape")
    )
    by_session = rows.groupby(["model", "target", "forecast_session"], observed=True, as_index=False).agg(
        samples=("ape", "size"), mape=("ape", "mean")
    )
    by_daynight = rows.groupby(["model", "target", "day_night"], as_index=False).agg(
        samples=("ape", "size"), mape=("ape", "mean")
    )
    by_weekpart = rows.groupby(["model", "target", "weekday_weekend"], as_index=False).agg(
        samples=("ape", "size"), mape=("ape", "mean")
    )
    by_month = rows.groupby(["model", "fold", "target", "forecast_month"], as_index=False).agg(
        samples=("ape", "size"), mape=("ape", "mean")
    )
    by_hour = rows.groupby(["model", "target", "forecast_hour"], as_index=False).agg(
        samples=("ape", "size"), mape=("ape", "mean")
    )

    causal = pd.read_csv(CAUSAL_TRAIN, encoding="utf-8-sig", low_memory=False)
    causal["datetime"] = pd.to_datetime(causal["datetime"], errors="raise")
    causal["hour"] = causal["datetime"].dt.hour
    causal["session"] = session(causal["hour"])
    causal["month"] = causal["datetime"].dt.to_period("M").astype(str)
    profiles: list[pd.DataFrame] = []
    for target in ("generator_1", "generator_all"):
        profile = causal.groupby(["hour", "session"], observed=True, as_index=False)[target].agg(["mean", "std", "median"])
        profile["target"] = target
        profiles.append(profile.reset_index())
    training_profile = pd.concat(profiles, ignore_index=True)
    monthly_level = causal.groupby("month", as_index=False).agg(
        generator_1_mean=("generator_1", "mean"),
        generator_all_mean=("generator_all", "mean"),
        rows=("datetime", "size"),
    )

    parameters = pd.concat(parameter_frames, ignore_index=True)
    parameters.to_csv(RESULT_DIR / "selected_linear_beta.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(RESULT_DIR / "time_feature_ablation_summary.csv", index=False, encoding="utf-8-sig")
    by_session.to_csv(RESULT_DIR / "forecast_metrics_by_session.csv", index=False, encoding="utf-8-sig")
    by_daynight.to_csv(RESULT_DIR / "forecast_metrics_by_day_night.csv", index=False, encoding="utf-8-sig")
    by_weekpart.to_csv(RESULT_DIR / "forecast_metrics_by_weekpart.csv", index=False, encoding="utf-8-sig")
    by_month.to_csv(RESULT_DIR / "forecast_metrics_by_month_fold.csv", index=False, encoding="utf-8-sig")
    by_hour.to_csv(RESULT_DIR / "forecast_metrics_by_hour.csv", index=False, encoding="utf-8-sig")
    training_profile.to_csv(RESULT_DIR / "training_generation_hourly_profile.csv", index=False, encoding="utf-8-sig")
    monthly_level.to_csv(RESULT_DIR / "training_monthly_generation_level.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only OOF predictions and causal training observations",
        "external_scoring_data_accessed": False,
        "available_training_months": sorted(causal["month"].unique().tolist()),
        "quarter_assessment": "not identifiable: Q1 has Jan-Mar, Q2 has Apr plus only one May timestamp, and validation month is confounded with fold/time",
        "time_feature_ablation": overall.to_dict(orient="records"),
        "selected_parameters": parameters.to_dict(orient="records"),
    }
    (RESULT_DIR / "temporal_effect_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
