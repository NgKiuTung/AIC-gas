"""Causal physical-time feature tables for the preliminary/semifinal join."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("generator_1", "generator_all")
SCREENING_END = "2025-05-31 23:59:00"
LAGS = (15, 30, 60, 120, 360, 720, 1440)
ROLLS = (30, 60, 120, 360, 1440)
DIFFS = (15, 30, 60, 120)
HIGHFREQ_VARIABLES = (
    "blast_furnace_gas_holder_2",
    "generator_use_blast_furnace_gas",
    "generator_use_coke_gas",
    "generator_use_converter_gas",
    "blast_furnace_1",
    "blast_furnace_2",
    "blast_furnace_3",
    "blast_furnace_4",
    "blast_furnace_5",
    "air_heater_3",
    "blast_furnace_gas_holder_1",
)


def availability_grid(
    pre: pd.DataFrame, semi: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre observations become usable after 15m; Semi observations at timestamp t."""
    process_names = [c for c in pre.columns if c not in (*TARGETS, "dataset_phase")]
    pre_process = pre[process_names].copy()
    pre_process.index += pd.Timedelta(minutes=15)
    semi_process = semi[process_names]
    observed = pd.concat([pre_process, semi_process]).sort_index(kind="stable")
    # At May 3 00:00, the current Semi record supersedes the just-available
    # Pre_test record, matching Stage1 ProcessView's latest-source policy.
    observed = observed.loc[~observed.index.duplicated(keep="last")]
    minute_index = pd.date_range(
        pre.index.min(), semi.index.max(), freq="min", name="reference_time"
    )
    raw_grid = observed.reindex(minute_index)
    current = raw_grid.ffill(limit=360)
    return raw_grid, current


def origin_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    minute = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    week = index.dayofweek.to_numpy() + minute / 1440
    return pd.DataFrame(
        {
            "origin_day_sin": np.sin(2 * np.pi * minute / 1440),
            "origin_day_cos": np.cos(2 * np.pi * minute / 1440),
            "origin_week_sin": np.sin(2 * np.pi * week / 7),
            "origin_week_cos": np.cos(2 * np.pi * week / 7),
            "origin_month": index.month.astype("int8"),
            "origin_weekday": index.dayofweek.astype("int8"),
        },
        index=index,
    )


def common_features(
    raw_grid: pd.DataFrame,
    current: pd.DataFrame,
    origins: pd.DatetimeIndex,
    common_sources: list[str],
    excluded_rolls: set[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """All lag/rolling units are real minutes, not phase-dependent row counts."""
    columns: dict[str, np.ndarray] = {}
    manifest: list[dict] = []

    def add(
        name: str, values: np.ndarray, source: str, window: str, aggregation: str
    ) -> None:
        columns[name] = values.astype("float32")
        manifest.append(
            {
                "feature_name": name,
                "source_column": source,
                "physical_window": window,
                "aggregation": aggregation,
                "available_preliminary": True,
                "available_semifinal": True,
                "common_or_highfreq": "common",
                "max_information_time": "reference_time",
                "causal": True,
            }
        )

    for col in common_sources:
        now = current[col].reindex(origins)
        add(f"feat_{col}__current", now.to_numpy(), col, "0m", "latest_available")
        for lag in LAGS:
            past = current[col].reindex(origins - pd.Timedelta(minutes=lag))
            add(
                f"feat_{col}__lag_{lag}m",
                past.to_numpy(),
                col,
                f"-{lag}m",
                "latest_available",
            )
        for window in ROLLS:
            if f"feat_{col}__roll_mean_{window}m" in excluded_rolls:
                continue
            mean = raw_grid[col].rolling(window, min_periods=1).mean().reindex(origins)
            add(
                f"feat_{col}__roll_mean_{window}m",
                mean.to_numpy(),
                col,
                f"(-{window}m,0m]",
                "observed_sample_mean",
            )
        for lag in DIFFS:
            past = current[col].reindex(origins - pd.Timedelta(minutes=lag))
            add(
                f"feat_{col}__diff_{lag}m",
                (now.to_numpy() - past.to_numpy()),
                col,
                f"[-{lag}m,0m]",
                "endpoint_difference",
            )
    calendar = origin_calendar(origins)
    for name in calendar:
        add(name, calendar[name].to_numpy(), "reference_time", "0m", "known_calendar")
    return pd.DataFrame(columns, index=origins), manifest


def add_labels(
    pre: pd.DataFrame,
    semi: pd.DataFrame,
    pre_features: pd.DataFrame,
    semi_features: pd.DataFrame,
) -> pd.DataFrame:
    """Keep unverified preliminary samples explicitly separate from official means."""
    pre_train = pre_features.copy()
    semi_train = semi_features.copy()
    # h15 represents [t,t+15). Preliminary value at t+15 is only an endpoint
    # proxy; the organizer has not confirmed it is the interval mean.
    pre_endpoint = pre[list(TARGETS)].reindex(
        pre_train.index + pd.Timedelta(minutes=15)
    )
    pre_endpoint.index = pre_train.index
    pre_current = pre[list(TARGETS)].reindex(pre_train.index - pd.Timedelta(minutes=15))
    pre_current.index = pre_train.index
    semi_load = semi[list(TARGETS)].reindex(
        pd.date_range(semi.index.min(), semi.index.max(), freq="min")
    )
    semi_mean = semi_load.rolling(15, min_periods=15).mean().shift(-14)
    for target in TARGETS:
        pre_train[f"label_h15_{target}"] = pre_endpoint[target].to_numpy()
        semi_train[f"label_h15_{target}"] = (
            semi_mean[target].reindex(semi_train.index).to_numpy()
        )
        pre_train[f"proxy_current_{target}"] = pre_current[target].to_numpy()
        semi_train[f"proxy_current_{target}"] = (
            semi[target].reindex(semi_train.index).to_numpy()
        )
    pre_train["dataset_phase"] = "preliminary"
    semi_train["dataset_phase"] = "semifinal"
    pre_train["label_semantics"] = "endpoint_proxy_unverified"
    semi_train["label_semantics"] = "complete_15x1min_mean"
    return pd.concat([pre_train, semi_train]).sort_index(kind="stable")


def highfreq_features(
    semi: pd.DataFrame, origins: pd.DatetimeIndex, excluded_rolls: set[str]
) -> tuple[pd.DataFrame, list[dict]]:
    """One row per 15m origin; each statistic uses only <= origin 1m process."""
    minute = semi.drop(columns=[*TARGETS, "dataset_phase"]).reindex(
        pd.date_range(semi.index.min(), semi.index.max(), freq="min")
    )
    minute["observed_bf_production"] = minute[
        [f"blast_furnace_{i}" for i in range(1, 6)]
    ].sum(axis=1, min_count=5)
    columns: dict[str, np.ndarray] = {}
    manifest = []
    for col in (*HIGHFREQ_VARIABLES, "observed_bf_production"):
        series = minute[col]
        for lag in (1, 5, 15, 30, 60):
            name = f"hf_{col}__diff_{lag}m"
            columns[name] = (
                (series - series.shift(lag)).reindex(origins).to_numpy(dtype="float32")
            )
            manifest.append(
                {
                    "feature_name": name,
                    "source_column": col,
                    "physical_window": f"[-{lag}m,0m]",
                    "aggregation": "endpoint_difference",
                    "available_preliminary": False,
                    "available_semifinal": True,
                    "common_or_highfreq": "highfreq",
                    "max_information_time": "reference_time",
                    "causal": True,
                }
            )
        for window in (5, 15, 30, 60):
            name = f"hf_{col}__std_{window}m"
            columns[name] = (
                series.rolling(window, min_periods=2)
                .std(ddof=0)
                .reindex(origins)
                .to_numpy(dtype="float32")
            )
            manifest.append(
                {
                    "feature_name": name,
                    "source_column": col,
                    "physical_window": f"(-{window}m,0m]",
                    "aggregation": "observed_sample_std",
                    "available_preliminary": False,
                    "available_semifinal": True,
                    "common_or_highfreq": "highfreq",
                    "max_information_time": "reference_time",
                    "causal": True,
                }
            )
    for original_name in sorted(excluded_rolls):
        source, _, window_text = original_name.removeprefix("feat_").partition(
            "__roll_mean_"
        )
        window = int(window_text.removesuffix("m"))
        name = f"hf_{source}__roll_mean_{window}m"
        columns[name] = (
            minute[source]
            .rolling(window, min_periods=1)
            .mean()
            .reindex(origins)
            .to_numpy(dtype="float32")
        )
        manifest.append(
            {
                "feature_name": name,
                "source_column": source,
                "physical_window": f"(-{window}m,0m]",
                "aggregation": "semifinal_1min_observed_mean",
                "available_preliminary": False,
                "available_semifinal": True,
                "common_or_highfreq": "highfreq",
                "max_information_time": "reference_time",
                "causal": True,
            }
        )
    return pd.DataFrame(columns, index=origins), manifest


def build(data_dir: Path) -> dict:
    pre = pd.read_parquet(data_dir / "preliminary_clean_15m.parquet")
    semi = pd.read_parquet(data_dir / "semifinal_clean_1m.parquet")
    process_cols = [c for c in pre.columns if c not in (*TARGETS, "dataset_phase")]
    development_semi = semi.loc[:SCREENING_END]
    common_sources = [
        c
        for c in process_cols
        if pre[c].notna().any() and development_semi[c].notna().any()
    ]
    sampling_path = ROOT / "results" / "exp_bc" / "sampling_rate_audit.csv"
    if not sampling_path.exists():
        raise FileNotFoundError(
            "Run sampling_audit.py before building public common features"
        )
    sampling = pd.read_csv(sampling_path)
    excluded_rolls = {
        f"feat_{row.source_column}__roll_mean_{int(row.window_minutes)}m"
        for row in sampling.itertuples()
        if row.source_column in common_sources
        and (row.comparable_origins < 100 or row.p95_relative_difference > 0.1)
    }
    raw_grid, current = availability_grid(pre, semi)
    pre_origins = pd.date_range(
        pre.index.min() + pd.Timedelta(minutes=15),
        semi.index.min() - pd.Timedelta(minutes=15),
        freq="15min",
        name="reference_time",
    )
    semi_origins = pd.date_range(
        semi.index.min(),
        semi.index.max().floor("15min"),
        freq="15min",
        name="reference_time",
    )
    pre_common, manifest = common_features(
        raw_grid, current, pre_origins, common_sources, excluded_rolls
    )
    semi_common, semi_manifest = common_features(
        raw_grid, current, semi_origins, common_sources, excluded_rolls
    )
    if [x["feature_name"] for x in manifest] != [
        x["feature_name"] for x in semi_manifest
    ]:
        raise ValueError("Common schemas differ")
    pre_common.insert(0, "dataset_phase", "preliminary")
    semi_common.insert(0, "dataset_phase", "semifinal")
    pre_common.to_parquet(data_dir / "common_features_preliminary_15m.parquet")
    semi_common.to_parquet(data_dir / "common_features_semifinal_15m.parquet")
    training = add_labels(
        pre,
        semi,
        pre_common.drop(columns="dataset_phase"),
        semi_common.drop(columns="dataset_phase"),
    )
    if training.index.has_duplicates:
        raise ValueError("Joined 15min prediction origins must be unique")
    training.to_parquet(data_dir / "train_common_jan_sep_15m.parquet")
    highfreq, high_manifest = highfreq_features(semi, semi_origins, excluded_rolls)
    highfreq.to_parquet(data_dir / "semifinal_highfreq_at_15m_origins.parquet")
    all_manifest = {
        "reference_time": (
            "feature availability time, not raw preliminary measurement time"
        ),
        "pre_observation_delay_minutes": 15,
        "semi_observation_delay_minutes": 0,
        "preliminary_label": "t+15 endpoint sample proxy, unverified as 15min mean",
        "semifinal_label": "mean of 15 complete 1min values in [t,t+15)",
        "proxy_current_label": (
            "preliminary same-record t-15 sample / semifinal same-record t "
            "sample; used by B0-like LightGBM proxy"
        ),
        "pretest_included": True,
        "boundary_policy": (
            "May 1-2 released history retained; no interpolation; same-time "
            "May 3 latest Semi process wins"
        ),
        "sampling_audit": str(sampling_path),
        "sampling_audit_train_end": SCREENING_END,
        "excluded_noncommon_sources": sorted(set(process_cols) - set(common_sources)),
        "moved_rolling_means_to_highfreq": sorted(excluded_rolls),
        "feature_count_common": len(manifest),
        "feature_count_highfreq": len(high_manifest),
        "features": manifest + high_manifest,
    }
    (data_dir / "feature_manifest.json").write_text(
        json.dumps(all_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "pre_origins": len(pre_origins),
        "semi_origins": len(semi_origins),
        "common_features": len(manifest),
        "highfreq_features": len(high_manifest),
        "training_rows": len(training),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "features_experiments" / "exp_bc" / "data"
    )
    args = parser.parse_args()
    print(json.dumps(build(args.data), ensure_ascii=False))


if __name__ == "__main__":
    main()
