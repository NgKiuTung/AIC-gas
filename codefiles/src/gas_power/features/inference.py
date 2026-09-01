"""Build the frozen 801-feature production matrix without future labels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gas_power.features.domain_interactions import add_domain_interaction_features
from gas_power.features.multivariate_anomaly import (
    add_causal_multivariate_anomaly_features,
)
from gas_power.features.physical_balance import add_physical_balance_features

HORIZONS = tuple(range(1, 9))
LAGS = (1, 2, 3, 4, 8, 12, 16, 32, 96, 192, 672)
ROLLING_WINDOWS = (4, 8, 16, 32, 96)


def add_mechanism_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["feat_p50_current"] = out["feat_generator_1_filled"]
    out["feat_p120_current"] = out["feat_generator_all_filled"] - out["feat_generator_1_filled"]
    out["feat_bfg_balance_proxy"] = (
        out["feat_blast_furnace_observed_sum"]
        - out["feat_air_heater_observed_sum"]
        - out["feat_blast_furnace_user_observed_sum"]
        - out["into_gas_mixed_blast_furnace"]
        - out["generator_use_blast_furnace_gas"]
    )
    out["feat_converter_balance_proxy"] = (
        out["converter_1"]
        - out["feat_converter_user_observed_sum"]
        - out["into_gas_mixed_converter"]
        - out["generator_use_converter_gas"]
    )
    out["feat_generator_gas_total_unweighted"] = (
        out["generator_use_blast_furnace_gas"]
        + out["generator_use_coke_gas"]
        + out["generator_use_converter_gas"]
    )
    out["feat_holder_delta_1"] = out["blast_furnace_gas_holder_2"].diff(1)
    out["feat_holder_delta_4"] = out["blast_furnace_gas_holder_2"].diff(4)
    out["feat_load_hierarchy_gap"] = out["feat_p120_current"]
    return out


def _base_engineered_features(causal: pd.DataFrame) -> pd.DataFrame:
    data = add_mechanism_features(causal)
    dynamic_columns = [
        "feat_p50_current", "feat_p120_current", "feat_generator_all_filled",
        "generator_use_blast_furnace_gas", "generator_use_coke_gas", "generator_use_converter_gas",
        "feat_blast_furnace_observed_sum", "feat_air_heater_observed_sum",
        "feat_blast_furnace_user_observed_sum", "feat_converter_user_observed_sum",
        "blast_furnace_gas_holder_2", "coke_oven_1", "converter_1",
        "into_gas_mixed_blast_furnace", "into_gas_mixed_coke", "into_gas_mixed_converter",
        "feat_bfg_balance_proxy", "feat_converter_balance_proxy", "feat_generator_gas_total_unweighted",
    ]
    rolling_columns = [
        "feat_p50_current", "feat_p120_current", "feat_generator_all_filled",
        "generator_use_blast_furnace_gas", "generator_use_converter_gas",
        "feat_blast_furnace_observed_sum", "feat_blast_furnace_user_observed_sum",
        "blast_furnace_gas_holder_2",
    ]
    engineered: dict[str, pd.Series] = {}
    for horizon in HORIZONS:
        for component in ("feat_p50_current", "feat_p120_current"):
            stem = component.removeprefix("feat_")
            engineered[f"feat_seasonal_day_{stem}_h{horizon}"] = data[component].shift(96 - horizon)
            engineered[f"feat_seasonal_week_{stem}_h{horizon}"] = data[component].shift(672 - horizon)
    for column in dynamic_columns:
        stem = column.removeprefix("feat_")
        for lag in LAGS:
            engineered[f"feat_lag{lag}_{stem}"] = data[column].shift(lag)
        for lag in (1, 4, 8):
            engineered[f"feat_diff{lag}_{stem}"] = data[column] - data[column].shift(lag)
    for column in rolling_columns:
        stem = column.removeprefix("feat_")
        for window in ROLLING_WINDOWS:
            rolling = data[column].rolling(window=window, min_periods=window)
            engineered[f"feat_roll{window}_mean_{stem}"] = rolling.mean()
            engineered[f"feat_roll{window}_std_{stem}"] = rolling.std(ddof=0)
            engineered[f"feat_roll{window}_min_{stem}"] = rolling.min()
            engineered[f"feat_roll{window}_max_{stem}"] = rolling.max()
    return pd.concat([data, pd.DataFrame(engineered, index=data.index)], axis=1)


def _state_run_length(mask: pd.Series) -> pd.Series:
    groups = mask.ne(mask.shift()).cumsum()
    return mask.groupby(groups).cumcount().add(1).astype(float)


def _enhanced_features(causal: pd.DataFrame) -> pd.DataFrame:
    added: dict[str, pd.Series] = {}
    outlier_flags = [column for column in causal if column.startswith("feat_outlier_")]
    for flag_column in outlier_flags:
        stem = flag_column.removeprefix("feat_outlier_")
        source = f"feat_{stem}_filled" if stem in {"generator_1", "generator_all"} else stem
        if source not in causal:
            continue
        values = pd.to_numeric(causal[source], errors="coerce")
        past = values.shift(1)
        median = past.rolling(96, min_periods=24).median()
        mad = past.rolling(96, min_periods=24).apply(
            lambda items: float(np.median(np.abs(items - np.median(items)))), raw=True
        )
        scale = 1.4826 * mad
        valid = scale > np.maximum(1e-9, median.abs() * 1e-6)
        cleaned = values.where(
            ~(causal[flag_column].eq(1) & valid), values.clip(median - 8.0 * scale, median + 8.0 * scale)
        )
        added[f"feat_cleanview_{stem}"] = cleaned.fillna(values)

    zero_state_columns = [
        "converter_user1", "into_gas_mixed_blast_furnace", "blast_furnace_user3", "air_heater_5",
        "blast_furnace_user2", "converter_user2", "generator_use_converter_gas", "generator_use_coke_gas",
        "blast_furnace_user4", "air_heater_4", "blast_furnace_5", "into_gas_mixed_converter", "air_heater_2",
    ]
    for column in zero_state_columns:
        zero = causal[column].eq(0)
        added[f"feat_state_{column}_is_zero"] = zero.astype(float)
        added[f"feat_state_{column}_state_duration"] = _state_run_length(zero).clip(upper=672)
        added[f"feat_state_{column}_turned_on"] = ((~zero) & zero.shift(1, fill_value=False)).astype(float)
        added[f"feat_state_{column}_turned_off"] = (zero & (~zero.shift(1, fill_value=True))).astype(float)

    p50 = causal["feat_generator_1_filled"]
    pall = causal["feat_generator_all_filled"]
    p120 = pall - p50
    trend_columns = {
        "p50": p50, "p120": p120, "pall": pall,
        "holder": causal["blast_furnace_gas_holder_2"],
        "gen_bfg": causal["generator_use_blast_furnace_gas"],
        "gen_cog": causal["generator_use_coke_gas"],
        "gen_cvg": causal["generator_use_converter_gas"],
        "bfg_supply": causal["feat_blast_furnace_observed_sum"],
        "bfg_users": causal["feat_blast_furnace_user_observed_sum"],
    }
    for stem, values in trend_columns.items():
        values = pd.to_numeric(values, errors="coerce")
        for span in (4, 8, 16, 32, 96):
            ewm = values.ewm(span=span, adjust=False, min_periods=1).mean()
            added[f"feat_smooth_{stem}_span{span}_ewm"] = ewm
            added[f"feat_smooth_{stem}_span{span}_gap"] = values - ewm
        step = values.diff()
        for window in (4, 8, 16, 32):
            rolling = step.rolling(window, min_periods=1)
            added[f"feat_transition_{stem}_w{window}_step_mean"] = rolling.mean().fillna(0.0)
            added[f"feat_transition_{stem}_w{window}_step_std"] = rolling.std(ddof=0).fillna(0.0)
            added[f"feat_transition_{stem}_w{window}_step_absmax"] = rolling.apply(
                lambda items: float(np.max(np.abs(items))), raw=True
            ).fillna(0.0)
        added[f"feat_transition_{stem}_acceleration"] = values.diff().diff().fillna(0.0)

    safe_p50 = p50.clip(lower=1e-6)
    safe_pall = pall.clip(lower=1e-6)
    safe_p120 = p120.clip(lower=1e-6)
    gen_bfg = causal["generator_use_blast_furnace_gas"].clip(lower=1e-6)
    total_generator_gas = (
        causal["generator_use_blast_furnace_gas"]
        + causal["generator_use_coke_gas"]
        + causal["generator_use_converter_gas"]
    ).clip(lower=1e-6)
    ratios = {
        "feat_ratio_p50_share": safe_p50 / safe_pall,
        "feat_ratio_p120_share": safe_p120 / safe_pall,
        "feat_ratio_power_per_total_generator_gas": safe_pall / total_generator_gas,
        "feat_ratio_p50_per_bfg": safe_p50 / gen_bfg,
        "feat_ratio_coke_share_generator_gas": causal["generator_use_coke_gas"] / total_generator_gas,
        "feat_ratio_converter_share_generator_gas": causal["generator_use_converter_gas"] / total_generator_gas,
        "feat_ratio_holder_to_bfg_use": causal["blast_furnace_gas_holder_2"] / gen_bfg,
        "feat_ratio_bfg_generator_use_to_supply": causal["generator_use_blast_furnace_gas"]
        / causal["feat_blast_furnace_observed_sum"].clip(lower=1e-6),
        "feat_ratio_bfg_users_to_supply": causal["feat_blast_furnace_user_observed_sum"]
        / causal["feat_blast_furnace_observed_sum"].clip(lower=1e-6),
    }
    for name, values in ratios.items():
        # No bfill: invalid initial values get a fixed neutral fallback, never a future observation.
        added[name] = values.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    missing_flags = [column for column in causal if column.startswith("feat_missing_")]
    source_missing = [column for column in causal if column.startswith("feat_source_missing_")]
    added["feat_quality_missing_count"] = causal[missing_flags].sum(axis=1).astype(float)
    added["feat_quality_source_missing_count"] = causal[source_missing].sum(axis=1).astype(float)
    added["feat_quality_outlier_count"] = causal[outlier_flags].sum(axis=1).astype(float)
    added["feat_quality_any_outlier"] = causal[outlier_flags].any(axis=1).astype(float)
    for stem, values in (("p50", p50), ("pall", pall)):
        relative_step = values.diff().abs() / values.shift(1).abs().clip(lower=1e-6)
        for threshold in (0.03, 0.06, 0.10):
            event = relative_step.ge(threshold).fillna(False)
            since = event.groupby(event.cumsum()).cumcount().astype(float)
            percentage = int(threshold * 100)
            added[f"feat_change_{stem}_ge{percentage:02d}pct"] = event.astype(float)
            added[f"feat_change_{stem}_since_ge{percentage:02d}pct"] = since.clip(upper=672)
    return pd.DataFrame(added, index=causal.index)


def build_inference_feature_frame(
    causal: pd.DataFrame,
) -> pd.DataFrame:
    """Return a feature superset; no label or negative-shift column is created."""
    if "datetime" not in causal:
        raise ValueError("Causal frame requires datetime")
    ordered = causal.copy()
    ordered["datetime"] = pd.to_datetime(ordered["datetime"], errors="raise")
    ordered = ordered.sort_values("datetime").reset_index(drop=True)
    if ordered["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps in causal feature input")
    if not ordered["datetime"].diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Expected a complete 15-minute grid")
    base = _base_engineered_features(ordered)
    enhanced = _enhanced_features(ordered)
    anomaly = add_causal_multivariate_anomaly_features(ordered)
    output = pd.concat([base, enhanced, anomaly], axis=1)
    output = add_physical_balance_features(output)
    output = add_domain_interaction_features(output)
    if any(column.startswith("label_") for column in output):
        raise AssertionError("Inference builder must not create label columns")
    return output


def select_model_features(feature_frame: pd.DataFrame, feature_schema: Sequence[str]) -> pd.DataFrame:
    missing = [feature for feature in feature_schema if feature not in feature_frame]
    if missing:
        raise ValueError(f"Missing {len(missing)} frozen model features; first={missing[:5]}")
    selected = feature_frame.loc[:, list(feature_schema)].copy()
    return selected
