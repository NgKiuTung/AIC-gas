"""Re-audit raw training data, current cleaning, and OOF error associations.

This script is read-only with respect to raw inputs and uses no scoring data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "dataset" / "初赛-数据集"
FILES = {
    "gas": TRAIN_DIR / "Pre_gas.csv",
    "holder": TRAIN_DIR / "Pre_gas_holder.csv",
    "user": TRAIN_DIR / "Pre_gas_user.csv",
    "load": TRAIN_DIR / "Pre_load.csv",
}
PROCESSED_PATH = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train_causal.csv"
OOF_PATH = ROOT / "results" / "experiments" / "hybrid_target" / "hybrid_oof_predictions.csv"
RESULT_DIR = ROOT / "results" / "cleaning_reaudit"
TARGETS = ("generator_1", "generator_all")


def longest_run(values: pd.Series, predicate: np.ndarray | None = None) -> int:
    array = values.to_numpy()
    if len(array) == 0:
        return 0
    if predicate is None:
        same = np.r_[False, array[1:] == array[:-1]]
        groups = np.cumsum(~same)
        return int(pd.Series(groups).value_counts().max())
    groups = np.cumsum(~predicate)
    counts = pd.Series(groups[predicate]).value_counts()
    return int(counts.max()) if len(counts) else 0


def robust_step_outliers(series: pd.Series) -> tuple[int, float]:
    step = series.diff().dropna().to_numpy(dtype=float)
    if not len(step):
        return 0, np.nan
    median = np.median(step)
    mad = np.median(np.abs(step - median))
    threshold = 8 * 1.4826 * mad
    if threshold <= 0:
        return 0, float(threshold)
    return int((np.abs(step - median) > threshold).sum()), float(threshold)


def read_sources() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    merged: pd.DataFrame | None = None
    for source, path in FILES.items():
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        frame = frame.sort_values("datetime").reset_index(drop=True)
        frames[source] = frame
        renamed = frame.rename(columns={c: c for c in frame.columns})
        merged = renamed if merged is None else merged.merge(renamed, on="datetime", how="outer", validate="one_to_one")
    assert merged is not None
    return frames, merged.sort_values("datetime").reset_index(drop=True)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frames, merged = read_sources()
    time_rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []
    column_source = {column: source for source, frame in frames.items() for column in frame.columns if column != "datetime"}

    for source, frame in frames.items():
        full = pd.date_range(frame["datetime"].min(), frame["datetime"].max(), freq="15min")
        duplicate_count = int(frame["datetime"].duplicated().sum())
        missing_grid = full.difference(frame["datetime"])
        time_rows.append(
            {
                "source": source, "rows": len(frame), "datetime_min": frame["datetime"].min(),
                "datetime_max": frame["datetime"].max(), "duplicate_timestamps": duplicate_count,
                "missing_grid_timestamps": len(missing_grid),
                "missing_grid_values": "|".join(ts.isoformat() for ts in missing_grid),
            }
        )

    for column in [c for c in merged.columns if c != "datetime"]:
        series = pd.to_numeric(merged[column], errors="coerce")
        finite = series[np.isfinite(series)]
        step = finite.diff().abs().dropna()
        outlier_count, threshold = robust_step_outliers(finite)
        zero_mask = series.fillna(np.nan).eq(0).to_numpy()
        quantiles = finite.quantile([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]) if len(finite) else pd.Series(dtype=float)
        column_rows.append(
            {
                "source": column_source[column], "column": column, "rows": len(series),
                "missing_count": int(series.isna().sum()), "missing_rate": float(series.isna().mean()),
                "zero_count": int(series.eq(0).sum()), "zero_rate": float(series.eq(0).mean()),
                "negative_count": int((series < 0).sum()), "unique_count": int(series.nunique(dropna=True)),
                "longest_constant_run": longest_run(series.dropna()),
                "longest_zero_run": longest_run(series, zero_mask),
                "q001": quantiles.get(0.001, np.nan), "q01": quantiles.get(0.01, np.nan),
                "q05": quantiles.get(0.05, np.nan), "median": quantiles.get(0.5, np.nan),
                "q95": quantiles.get(0.95, np.nan), "q99": quantiles.get(0.99, np.nan),
                "q999": quantiles.get(0.999, np.nan),
                "max_abs_step": float(step.max()) if len(step) else np.nan,
                "q999_abs_step": float(step.quantile(0.999)) if len(step) else np.nan,
                "robust_step_outlier_count": outlier_count,
                "robust_step_threshold": threshold,
            }
        )

    processed = pd.read_csv(PROCESSED_PATH, encoding="utf-8-sig", low_memory=False)
    processed["datetime"] = pd.to_datetime(processed["datetime"], errors="raise")
    comparison = merged.merge(processed, on="datetime", how="right", suffixes=("_raw", "_processed"), validate="one_to_one")
    change_rows: list[dict[str, object]] = []
    for column in column_source:
        raw_column = f"{column}_raw" if f"{column}_raw" in comparison else column
        if column in TARGETS:
            processed_column = f"feat_{column}_filled"
        else:
            processed_column = f"{column}_processed" if f"{column}_processed" in comparison else column
        if raw_column not in comparison or processed_column not in comparison:
            continue
        raw = pd.to_numeric(comparison[raw_column], errors="coerce")
        clean = pd.to_numeric(comparison[processed_column], errors="coerce")
        observed = raw.notna()
        changed_observed = observed & ~np.isclose(raw, clean, equal_nan=True)
        change_rows.append(
            {
                "column": column, "raw_missing": int(raw.isna().sum()),
                "filled_missing": int((raw.isna() & clean.notna()).sum()),
                "changed_observed_values": int(changed_observed.sum()),
                "max_abs_observed_change": float((raw[changed_observed] - clean[changed_observed]).abs().max()) if changed_observed.any() else 0.0,
            }
        )

    target_events: list[pd.DataFrame] = []
    load = frames["load"].set_index("datetime")
    for target in TARGETS:
        series = load[target].astype(float)
        event = pd.DataFrame(
            {
                "datetime": series.index, "target": target, "value": series.to_numpy(),
                "previous": series.shift(1).to_numpy(), "step": series.diff().to_numpy(),
                "relative_step": (series.diff().abs() / series.shift(1).abs().clip(lower=1e-6)).to_numpy(),
            }
        )
        target_events.append(event.nlargest(200, "relative_step"))
    target_event_table = pd.concat(target_events, ignore_index=True)

    oof = pd.read_csv(OOF_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    quality_columns = [c for c in processed.columns if c.startswith("feat_missing_") or c.startswith("feat_outlier_") or c.startswith("feat_source_missing_")]
    quality = processed[["datetime", *quality_columns]].copy()
    quality["origin_missing_count"] = quality[[c for c in quality_columns if "missing" in c]].sum(axis=1)
    quality["origin_outlier_count"] = quality[[c for c in quality_columns if "outlier" in c]].sum(axis=1)
    quality = quality[["datetime", "origin_missing_count", "origin_outlier_count"]]
    error = oof.merge(quality, on="datetime", how="left", validate="many_to_one")
    error["forecast_datetime"] = error["datetime"] + pd.to_timedelta(error["horizon_minutes"], unit="m")
    future = processed[["datetime", "feat_outlier_generator_1", "feat_outlier_generator_all"]].rename(columns={"datetime": "forecast_datetime"})
    error = error.merge(future, on="forecast_datetime", how="left", validate="many_to_one")
    error["origin_quality"] = np.select(
        [error["origin_missing_count"] > 0, error["origin_outlier_count"] > 0],
        ["origin_has_missing", "origin_has_outlier"], default="origin_clean",
    )
    error["future_target_outlier"] = pd.Series(
        np.where(
            error["target"].eq("generator_1"),
            error["feat_outlier_generator_1"],
            error["feat_outlier_generator_all"],
        ),
        index=error.index,
    ).fillna(0).astype(int)
    error_quality = error.groupby(["target", "origin_quality", "future_target_outlier"], as_index=False).agg(
        rows=("actual", "size"), persistence_mape=("ape_persistence", "mean"),
        hybrid_mape=("ape_hybrid", "mean"), mean_improvement=("ape_persistence", lambda x: 0.0),
    )
    # Compute paired improvement separately to keep the aggregation explicit.
    paired = error.assign(improvement=error["ape_persistence"] - error["ape_hybrid"]).groupby(
        ["target", "origin_quality", "future_target_outlier"], as_index=False
    )["improvement"].mean()
    error_quality = error_quality.drop(columns="mean_improvement").merge(
        paired, on=["target", "origin_quality", "future_target_outlier"], validate="one_to_one"
    )

    high_error = error.nlargest(1000, "ape_hybrid")[
        ["datetime", "forecast_datetime", "fold", "target", "horizon_minutes", "actual", "current", "prediction", "ape_hybrid", "origin_missing_count", "origin_outlier_count", "future_target_outlier"]
    ].copy()
    pd.DataFrame(time_rows).to_csv(RESULT_DIR / "timestamp_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(column_rows).to_csv(RESULT_DIR / "raw_column_quality.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(change_rows).to_csv(RESULT_DIR / "raw_vs_causal_changes.csv", index=False, encoding="utf-8-sig")
    target_event_table.to_csv(RESULT_DIR / "largest_target_steps.csv", index=False, encoding="utf-8-sig")
    error_quality.to_csv(RESULT_DIR / "oof_error_by_data_quality.csv", index=False, encoding="utf-8-sig")
    high_error.to_csv(RESULT_DIR / "largest_oof_errors.csv", index=False, encoding="utf-8-sig")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "raw and processed training data plus training-only OOF predictions",
        "external_scoring_data_accessed": False,
        "raw_rows_union": len(merged),
        "columns_audited": len(column_rows),
        "observed_values_changed_by_current_cleaning": int(pd.DataFrame(change_rows)["changed_observed_values"].sum()),
        "missing_values_filled": int(pd.DataFrame(change_rows)["filled_missing"].sum()),
        "target_hierarchy_violations": int((load["generator_all"] < load["generator_1"]).sum()),
        "target_zero_counts": {target: int(load[target].eq(0).sum()) for target in TARGETS},
        "quality_association_table": "oof_error_by_data_quality.csv",
    }
    (RESULT_DIR / "cleaning_reaudit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
