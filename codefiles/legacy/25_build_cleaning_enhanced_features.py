"""Build causal cleaning views and operating-state features.

Raw observed values are retained. Added features provide robustly clipped views,
state durations, smooth trends, and mechanism ratios without using future data.
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
CAUSAL_PATH = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train_causal.csv"
OUTPUT_MATRIX = ROOT / "results" / "features" / "train_supervised_features_cleaning_enhanced.pkl"
OUTPUT_CATALOG = ROOT / "results" / "features" / "feature_catalog_cleaning_enhanced.csv"
AUDIT_PATH = ROOT / "results" / "features" / "cleaning_enhanced_feature_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_run_length(mask: pd.Series) -> pd.Series:
    groups = mask.ne(mask.shift()).cumsum()
    return mask.groupby(groups).cumcount().add(1).astype(float)


def main() -> None:
    base = pd.read_pickle(BASE_MATRIX)
    base["datetime"] = pd.to_datetime(base["datetime"], errors="raise")
    causal = pd.read_csv(CAUSAL_PATH, encoding="utf-8-sig", low_memory=False)
    causal["datetime"] = pd.to_datetime(causal["datetime"], errors="raise")
    causal = causal.sort_values("datetime").reset_index(drop=True)
    if set(causal["split"].unique()) != {"train"}:
        raise ValueError("Input is not training-only")
    if not causal["datetime"].diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Expected complete 15-minute training grid")

    added: dict[str, pd.Series] = {}
    groups: dict[str, str] = {}

    # Causal robust clipping is an additional view; original values remain available.
    outlier_flags = [c for c in causal.columns if c.startswith("feat_outlier_")]
    for flag_column in outlier_flags:
        stem = flag_column.removeprefix("feat_outlier_")
        source = f"feat_{stem}_filled" if stem in {"generator_1", "generator_all"} else stem
        if source not in causal:
            continue
        values = pd.to_numeric(causal[source], errors="coerce")
        past = values.shift(1)
        median = past.rolling(96, min_periods=24).median()
        mad = past.rolling(96, min_periods=24).apply(
            lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
        )
        scale = 1.4826 * mad
        lower, upper = median - 8.0 * scale, median + 8.0 * scale
        valid = scale > np.maximum(1e-9, median.abs() * 1e-6)
        cleaned = values.where(~(causal[flag_column].eq(1) & valid), values.clip(lower, upper))
        cleaned = cleaned.fillna(values)
        name = f"feat_cleanview_{stem}"
        added[name] = cleaned
        groups[name] = "clean_view"

    zero_state_columns = [
        "converter_user1", "into_gas_mixed_blast_furnace", "blast_furnace_user3",
        "air_heater_5", "blast_furnace_user2", "converter_user2",
        "generator_use_converter_gas", "generator_use_coke_gas", "blast_furnace_user4",
        "air_heater_4", "blast_furnace_5", "into_gas_mixed_converter", "air_heater_2",
    ]
    for column in zero_state_columns:
        zero = causal[column].eq(0)
        for suffix, values in (
            ("is_zero", zero.astype(float)),
            ("state_duration", state_run_length(zero).clip(upper=672)),
            ("turned_on", ((~zero) & zero.shift(1, fill_value=False)).astype(float)),
            ("turned_off", (zero & (~zero.shift(1, fill_value=True))).astype(float)),
        ):
            name = f"feat_state_{column}_{suffix}"
            added[name] = values
            groups[name] = "state"

    causal["_p50"] = causal["feat_generator_1_filled"]
    causal["_pall"] = causal["feat_generator_all_filled"]
    causal["_p120"] = causal["_pall"] - causal["_p50"]
    trend_columns = {
        "p50": "_p50", "p120": "_p120", "pall": "_pall",
        "holder": "blast_furnace_gas_holder_2",
        "gen_bfg": "generator_use_blast_furnace_gas",
        "gen_cog": "generator_use_coke_gas",
        "gen_cvg": "generator_use_converter_gas",
        "bfg_supply": "feat_blast_furnace_observed_sum",
        "bfg_users": "feat_blast_furnace_user_observed_sum",
    }
    for stem, column in trend_columns.items():
        values = pd.to_numeric(causal[column], errors="coerce")
        for span in (4, 8, 16, 32, 96):
            ewm = values.ewm(span=span, adjust=False, min_periods=1).mean()
            for suffix, feature in (("ewm", ewm), ("gap", values - ewm)):
                name = f"feat_smooth_{stem}_span{span}_{suffix}"
                added[name] = feature
                groups[name] = "smooth"
        step = values.diff()
        for window in (4, 8, 16, 32):
            rolling = step.rolling(window, min_periods=1)
            for suffix, feature in (
                ("step_mean", rolling.mean()),
                ("step_std", rolling.std(ddof=0)),
                ("step_absmax", rolling.apply(lambda x: float(np.max(np.abs(x))), raw=True)),
            ):
                name = f"feat_transition_{stem}_w{window}_{suffix}"
                added[name] = feature.fillna(0.0)
                groups[name] = "transition"
        acceleration = values.diff().diff().fillna(0.0)
        name = f"feat_transition_{stem}_acceleration"
        added[name] = acceleration
        groups[name] = "transition"

    p50 = causal["_p50"].clip(lower=1e-6)
    pall = causal["_pall"].clip(lower=1e-6)
    p120 = causal["_p120"].clip(lower=1e-6)
    gen_bfg = causal["generator_use_blast_furnace_gas"].clip(lower=1e-6)
    total_generator_gas = (
        causal["generator_use_blast_furnace_gas"]
        + causal["generator_use_coke_gas"]
        + causal["generator_use_converter_gas"]
    ).clip(lower=1e-6)
    ratio_features = {
        "feat_ratio_p50_share": p50 / pall,
        "feat_ratio_p120_share": p120 / pall,
        "feat_ratio_power_per_total_generator_gas": pall / total_generator_gas,
        "feat_ratio_p50_per_bfg": p50 / gen_bfg,
        "feat_ratio_coke_share_generator_gas": causal["generator_use_coke_gas"] / total_generator_gas,
        "feat_ratio_converter_share_generator_gas": causal["generator_use_converter_gas"] / total_generator_gas,
        "feat_ratio_holder_to_bfg_use": causal["blast_furnace_gas_holder_2"] / gen_bfg,
        "feat_ratio_bfg_generator_use_to_supply": causal["generator_use_blast_furnace_gas"] / causal["feat_blast_furnace_observed_sum"].clip(lower=1e-6),
        "feat_ratio_bfg_users_to_supply": causal["feat_blast_furnace_user_observed_sum"] / causal["feat_blast_furnace_observed_sum"].clip(lower=1e-6),
    }
    for name, values in ratio_features.items():
        added[name] = values.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        groups[name] = "mechanism_ratio"

    missing_flags = [c for c in causal.columns if c.startswith("feat_missing_")]
    source_missing = [c for c in causal.columns if c.startswith("feat_source_missing_")]
    added["feat_quality_missing_count"] = causal[missing_flags].sum(axis=1).astype(float)
    added["feat_quality_source_missing_count"] = causal[source_missing].sum(axis=1).astype(float)
    added["feat_quality_outlier_count"] = causal[outlier_flags].sum(axis=1).astype(float)
    added["feat_quality_any_outlier"] = causal[outlier_flags].any(axis=1).astype(float)
    for name in (
        "feat_quality_missing_count", "feat_quality_source_missing_count",
        "feat_quality_outlier_count", "feat_quality_any_outlier",
    ):
        groups[name] = "quality"

    # Transition-state persistence for the two target levels.
    for stem, values in (("p50", causal["_p50"]), ("pall", causal["_pall"])):
        relative_step = values.diff().abs() / values.shift(1).abs().clip(lower=1e-6)
        for threshold in (0.03, 0.06, 0.10):
            event = relative_step.ge(threshold).fillna(False)
            event_groups = event.cumsum()
            since = event.groupby(event_groups).cumcount().astype(float)
            flag_name = f"feat_change_{stem}_ge{int(threshold * 100):02d}pct"
            since_name = f"feat_change_{stem}_since_ge{int(threshold * 100):02d}pct"
            added[flag_name] = event.astype(float)
            added[since_name] = since.clip(upper=672)
            groups[flag_name] = "change_state"
            groups[since_name] = "change_state"

    added_frame = pd.concat(
        [causal[["datetime"]], pd.DataFrame(added, index=causal.index)], axis=1
    )
    augmented = base.merge(added_frame, on="datetime", how="left", validate="one_to_one")
    added_columns = [c for c in added_frame.columns if c != "datetime"]
    if augmented[added_columns].isna().any().any():
        missing = augmented[added_columns].isna().sum()
        raise ValueError(f"Enhanced features contain missing values: {missing[missing > 0].to_dict()}")
    if not np.isfinite(augmented[added_columns].to_numpy(dtype=float)).all():
        raise ValueError("Enhanced features contain non-finite values")

    base_catalog = pd.read_csv(BASE_CATALOG, encoding="utf-8-sig")
    added_catalog = pd.DataFrame(
        {"feature": added_columns, "group": [groups[c] for c in added_columns]}
    )
    catalog = pd.concat([base_catalog, added_catalog], ignore_index=True)
    augmented.to_pickle(OUTPUT_MATRIX)
    catalog.to_csv(OUTPUT_CATALOG, index=False, encoding="utf-8-sig")
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "causal preprocessed training data only",
        "external_scoring_data_accessed": False,
        "raw_observations_replaced": False,
        "design": "retain raw features and add causal clean views plus operating-state features",
        "rows": len(augmented),
        "base_feature_count": len(base_catalog),
        "added_feature_count": len(added_catalog),
        "total_feature_count": len(catalog),
        "added_group_counts": added_catalog["group"].value_counts().to_dict(),
        "output": str(OUTPUT_MATRIX.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT_MATRIX),
    }
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
