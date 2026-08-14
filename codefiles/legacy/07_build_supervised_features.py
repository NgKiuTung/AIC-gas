"""Build a leakage-audited, training-only, direct multi-horizon feature matrix."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train_causal.csv"
OUTPUT_DIR = ROOT / "results" / "features"
LOG_DIR = ROOT / "results" / "preprocessing" / "logs"
HORIZONS = tuple(range(1, 9))
LAGS = (1, 2, 3, 4, 8, 12, 16, 32, 96, 192, 672)
ROLLING_WINDOWS = (4, 8, 16, 32, 96)


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_supervised_features")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "07_build_supervised_features.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    data = add_mechanism_features(df)
    label_columns: list[str] = []
    p50 = data["generator_1"]
    pall = data["generator_all"]
    p120 = pall - p50
    for horizon in HORIZONS:
        data[f"label_p50_h{horizon}"] = p50.shift(-horizon)
        data[f"label_p120_h{horizon}"] = p120.shift(-horizon)
        data[f"label_generator_1_h{horizon}"] = p50.shift(-horizon)
        data[f"label_generator_all_h{horizon}"] = pall.shift(-horizon)
        label_columns.extend(
            [
                f"label_p50_h{horizon}",
                f"label_p120_h{horizon}",
                f"label_generator_1_h{horizon}",
                f"label_generator_all_h{horizon}",
            ]
        )

    excluded_base = {"datetime", "split", "generator_1", "generator_all", *label_columns}
    base_features = [c for c in data.columns if c not in excluded_base]
    dynamic_columns = [
        "feat_p50_current",
        "feat_p120_current",
        "feat_generator_all_filled",
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
        "feat_blast_furnace_observed_sum",
        "feat_air_heater_observed_sum",
        "feat_blast_furnace_user_observed_sum",
        "feat_converter_user_observed_sum",
        "blast_furnace_gas_holder_2",
        "coke_oven_1",
        "converter_1",
        "into_gas_mixed_blast_furnace",
        "into_gas_mixed_coke",
        "into_gas_mixed_converter",
        "feat_bfg_balance_proxy",
        "feat_converter_balance_proxy",
        "feat_generator_gas_total_unweighted",
    ]
    rolling_columns = [
        "feat_p50_current",
        "feat_p120_current",
        "feat_generator_all_filled",
        "generator_use_blast_furnace_gas",
        "generator_use_converter_gas",
        "feat_blast_furnace_observed_sum",
        "feat_blast_furnace_user_observed_sum",
        "blast_furnace_gas_holder_2",
    ]
    engineered: dict[str, pd.Series] = {}
    for horizon in HORIZONS:
        for component in ("feat_p50_current", "feat_p120_current"):
            stem = component.removeprefix("feat_")
            engineered[f"feat_seasonal_day_{stem}_h{horizon}"] = data[component].shift(96 - horizon)
            engineered[f"feat_seasonal_week_{stem}_h{horizon}"] = data[component].shift(672 - horizon)
    for col in dynamic_columns:
        for lag in LAGS:
            engineered[f"feat_lag{lag}_{col.removeprefix('feat_')}"] = data[col].shift(lag)
        for lag in (1, 4, 8):
            engineered[f"feat_diff{lag}_{col.removeprefix('feat_')}"] = data[col] - data[col].shift(lag)
    for col in rolling_columns:
        for window in ROLLING_WINDOWS:
            rolling = data[col].rolling(window=window, min_periods=window)
            stem = col.removeprefix("feat_")
            engineered[f"feat_roll{window}_mean_{stem}"] = rolling.mean()
            engineered[f"feat_roll{window}_std_{stem}"] = rolling.std(ddof=0)
            engineered[f"feat_roll{window}_min_{stem}"] = rolling.min()
            engineered[f"feat_roll{window}_max_{stem}"] = rolling.max()
    data = pd.concat([data, pd.DataFrame(engineered, index=data.index)], axis=1)
    feature_columns = base_features + list(engineered)
    constant_columns = [c for c in feature_columns if data[c].nunique(dropna=True) <= 1]
    feature_columns = [c for c in feature_columns if c not in constant_columns]
    return data, feature_columns, label_columns


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", low_memory=False)
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="raise")
    if set(raw["split"].unique()) != {"train"}:
        raise ValueError("Input is not a training-only dataset")
    data, feature_columns, label_columns = build_features(raw)
    required = feature_columns + label_columns
    valid = data[required].notna().all(axis=1)
    supervised = data.loc[valid, ["datetime", *feature_columns, *label_columns]].reset_index(drop=True)
    if not np.isfinite(supervised[feature_columns + label_columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values remain in supervised matrix")
    if any("lead" in c.lower() or "future" in c.lower() or c.startswith("label_") for c in feature_columns):
        raise ValueError("Potential label leakage detected in feature names")
    matrix_path = OUTPUT_DIR / "train_supervised_features.pkl"
    supervised.to_pickle(matrix_path)
    pd.DataFrame(
        {
            "feature": feature_columns,
            "group": [
                "lag" if "feat_lag" in c else "rolling" if "feat_roll" in c else "difference" if "feat_diff" in c else "base"
                for c in feature_columns
            ],
        }
    ).to_csv(OUTPUT_DIR / "feature_catalog.csv", index=False, encoding="utf-8-sig")
    audit: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(INPUT_PATH.relative_to(ROOT)),
        "source_sha256": sha256(INPUT_PATH),
        "official_test_accessed": False,
        "rows_input": len(raw),
        "rows_supervised": len(supervised),
        "datetime_min": supervised["datetime"].min().isoformat(),
        "datetime_max": supervised["datetime"].max().isoformat(),
        "feature_count": len(feature_columns),
        "label_count": len(label_columns),
        "horizons_steps": list(HORIZONS),
        "horizon_minutes": [15 * h for h in HORIZONS],
        "max_feature_lag_steps": max(LAGS),
        "max_feature_lookback_days": max(LAGS) * 15 / 60 / 24,
        "feature_rule": "Only current and positive lags/left-aligned history; rolling windows end at reference timestamp.",
        "label_rule": "Negative shifts are used only for label_* columns.",
        "dropped_rows": int((~valid).sum()),
        "matrix": str(matrix_path.relative_to(ROOT)),
        "matrix_sha256": sha256(matrix_path),
    }
    (OUTPUT_DIR / "feature_leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Supervised matrix=%s rows=%d features=%d labels=%d", matrix_path, len(supervised), len(feature_columns), len(label_columns))


if __name__ == "__main__":
    main()
