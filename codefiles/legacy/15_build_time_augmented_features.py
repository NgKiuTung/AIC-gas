"""Add explicit calendar, peak/trough profile, and slow-drift features.

All values are deterministic calendar attributes or are computed from the
current/past portion of the causal training series. No future observation is
used as a feature.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_MATRIX = ROOT / "results" / "features" / "train_supervised_features.pkl"
BASE_CATALOG = ROOT / "results" / "features" / "feature_catalog.csv"
CAUSAL_TRAIN = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train_causal.csv"
OUTPUT_MATRIX = ROOT / "results" / "features" / "train_supervised_features_time_augmented.pkl"
OUTPUT_CATALOG = ROOT / "results" / "features" / "feature_catalog_time_augmented.csv"
AUDIT_PATH = ROOT / "results" / "features" / "time_feature_audit.json"
HORIZONS = tuple(range(1, 9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calendar_features(datetimes: pd.Series) -> dict[str, np.ndarray]:
    minute = datetimes.dt.hour.to_numpy() * 60 + datetimes.dt.minute.to_numpy()
    weekday = datetimes.dt.dayofweek.to_numpy()
    day_of_year = datetimes.dt.dayofyear.to_numpy()
    month = datetimes.dt.month.to_numpy()
    output: dict[str, np.ndarray] = {}
    for harmonic in (1, 2, 3):
        angle = 2 * np.pi * harmonic * minute / 1440.0
        output[f"minute_sin_k{harmonic}"] = np.sin(angle)
        output[f"minute_cos_k{harmonic}"] = np.cos(angle)
    output["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    output["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    output["is_weekend"] = (weekday >= 5).astype(np.int8)
    output["dayofyear_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    output["dayofyear_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    output["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    output["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    hour = datetimes.dt.hour.to_numpy()
    sessions = {
        "night_00_06": (hour < 6),
        "morning_06_09": ((hour >= 6) & (hour < 9)),
        "day_09_17": ((hour >= 9) & (hour < 17)),
        "evening_17_22": ((hour >= 17) & (hour < 22)),
        "late_22_24": (hour >= 22),
    }
    for name, mask in sessions.items():
        output[f"session_{name}"] = mask.astype(np.int8)
    return output


def main() -> None:
    base = pd.read_pickle(BASE_MATRIX)
    base["datetime"] = pd.to_datetime(base["datetime"], errors="raise")
    causal = pd.read_csv(CAUSAL_TRAIN, encoding="utf-8-sig", low_memory=False)
    causal["datetime"] = pd.to_datetime(causal["datetime"], errors="raise")
    causal = causal.sort_values("datetime").reset_index(drop=True)
    if set(causal["split"].unique()) != {"train"}:
        raise ValueError("Input is not a training-only dataset")
    deltas = causal["datetime"].diff().dropna()
    if not deltas.eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Causal training grid is not complete at 15-minute intervals")

    added: dict[str, pd.Series | np.ndarray] = {}
    for horizon in HORIZONS:
        forecast_time = causal["datetime"] + pd.Timedelta(minutes=15 * horizon)
        for name, values in calendar_features(forecast_time).items():
            added[f"feat_calendar_h{horizon}_{name}"] = values

    target_sources = {
        "generator_1": causal["feat_generator_1_filled"].astype(float),
        "generator_all": causal["feat_generator_all_filled"].astype(float),
    }
    for target, series in target_sources.items():
        for horizon in HORIZONS:
            analogs = pd.concat(
                [series.shift(96 * day - horizon) for day in range(1, 8)], axis=1
            )
            weights = np.exp(-0.25 * np.arange(7, dtype=float))
            weights /= weights.sum()
            added[f"feat_daily_profile_h{horizon}_{target}_mean7"] = analogs.mean(axis=1)
            added[f"feat_daily_profile_h{horizon}_{target}_median7"] = analogs.median(axis=1)
            added[f"feat_daily_profile_h{horizon}_{target}_std7"] = analogs.std(axis=1, ddof=0)
            added[f"feat_daily_profile_h{horizon}_{target}_recent_weighted7"] = analogs.to_numpy() @ weights
            added[f"feat_daily_profile_h{horizon}_{target}_recent_minus_old"] = analogs.iloc[:, 0] - analogs.iloc[:, -1]

        for window, label in ((192, "2d"), (672, "7d")):
            rolling = series.rolling(window=window, min_periods=window)
            added[f"feat_long_{label}_{target}_mean"] = rolling.mean()
            added[f"feat_long_{label}_{target}_std"] = rolling.std(ddof=0)
            added[f"feat_long_{label}_{target}_min"] = rolling.min()
            added[f"feat_long_{label}_{target}_max"] = rolling.max()
        ewm = series.ewm(span=2688, adjust=False, min_periods=96)
        added[f"feat_long_28d_{target}_ewm_mean"] = ewm.mean()
        added[f"feat_long_28d_{target}_ewm_std"] = ewm.std(bias=True)
        added[f"feat_long_28d_{target}_level_gap"] = series - ewm.mean()

    augmented_full = pd.concat(
        [causal[["datetime"]], pd.DataFrame(added, index=causal.index)], axis=1
    )
    augmented = base.merge(augmented_full, on="datetime", how="left", validate="one_to_one")
    added_columns = [column for column in augmented_full.columns if column != "datetime"]
    if augmented[added_columns].isna().any().any():
        missing = augmented[added_columns].isna().sum()
        raise ValueError(f"Time augmentation created missing values: {missing[missing > 0].to_dict()}")
    if not np.isfinite(augmented[added_columns].to_numpy(dtype=float)).all():
        raise ValueError("Time augmentation created non-finite values")

    base_catalog = pd.read_csv(BASE_CATALOG, encoding="utf-8-sig")
    new_catalog = pd.DataFrame(
        {
            "feature": added_columns,
            "group": [
                "time_calendar" if "feat_calendar" in column
                else "time_profile" if "feat_daily_profile" in column
                else "slow_drift"
                for column in added_columns
            ],
        }
    )
    catalog = pd.concat([base_catalog, new_catalog], ignore_index=True)
    augmented.to_pickle(OUTPUT_MATRIX)
    catalog.to_csv(OUTPUT_CATALOG, index=False, encoding="utf-8-sig")
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "causal preprocessed training data only",
        "external_scoring_data_accessed": False,
        "base_matrix": str(BASE_MATRIX.relative_to(ROOT)),
        "base_matrix_sha256": sha256(BASE_MATRIX),
        "causal_train": str(CAUSAL_TRAIN.relative_to(ROOT)),
        "rows": len(augmented),
        "base_feature_count": int(len(base_catalog)),
        "added_feature_count": int(len(new_catalog)),
        "total_feature_count": int(len(catalog)),
        "added_group_counts": new_catalog["group"].value_counts().to_dict(),
        "calendar_design": "future timestamp is deterministic and contains no future observation",
        "daily_profile_design": "for horizon h, uses t+h-96*k for k=1..7; all offsets are at least 88 steps in the past",
        "quarter_limitation": "training covers only about four months, so annual/quarter effects are weak descriptors, not validated seasonal cycles",
        "output": str(OUTPUT_MATRIX.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT_MATRIX),
    }
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
