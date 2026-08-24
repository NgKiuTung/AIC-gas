"""Advanced feature engineering: future prices and interaction features.

This module adds two critical feature types:
1. Future known prices: Leverage the fact that future electricity prices are known
2. Interaction features: Capture non-linear relationships between features
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

HORIZONS = tuple(range(1, 9))


def add_future_price_features(
    frame: pd.DataFrame,
    price_lookup: Mapping[tuple[int, int], float]
) -> pd.DataFrame:
    """Add future known electricity price features for each forecast horizon.

    This is CRITICAL: we're predicting power generation 15-120 minutes ahead,
    and electricity prices are KNOWN in advance. Not using future prices is
    leaving money on the table.

    Args:
        frame: DataFrame with 'datetime' and 'feat_known_price' columns
        price_lookup: {(month, half_hour_slot): price} mapping

    Returns:
        DataFrame with added future price features
    """
    out = frame.copy()
    dt = out["datetime"]

    # Basic future prices for each horizon
    future_prices = []
    for horizon in HORIZONS:
        future_dt = dt + pd.Timedelta(minutes=15 * horizon)
        future_month = future_dt.dt.month
        future_half_hour = future_dt.dt.hour * 2 + (future_dt.dt.minute >= 30).astype(int)

        prices = np.array([
            price_lookup[(int(m), int(slot))]
            for m, slot in zip(future_month, future_half_hour, strict=True)
        ], dtype=np.float32)

        out[f"feat_future_price_h{horizon}"] = prices
        future_prices.append(prices)

    future_prices_matrix = np.column_stack(future_prices)
    current_price = out["feat_known_price"].to_numpy()

    # Derived features for each horizon
    for idx, horizon in enumerate(HORIZONS):
        price_h = future_prices_matrix[:, idx]

        # Price change (absolute and relative)
        out[f"feat_future_price_delta_h{horizon}"] = price_h - current_price
        out[f"feat_future_price_pct_change_h{horizon}"] = (
            (price_h - current_price) / np.maximum(current_price, 1e-6)
        )

        # Direction indicators
        out[f"feat_future_price_up_h{horizon}"] = (price_h > current_price).astype(np.float32)
        out[f"feat_future_price_down_h{horizon}"] = (price_h < current_price).astype(np.float32)

        # Path statistics (from now to horizon)
        path = future_prices_matrix[:, :idx + 1]
        out[f"feat_future_price_path_mean_h{horizon}"] = path.mean(axis=1).astype(np.float32)
        out[f"feat_future_price_path_max_h{horizon}"] = path.max(axis=1).astype(np.float32)
        out[f"feat_future_price_path_min_h{horizon}"] = path.min(axis=1).astype(np.float32)
        out[f"feat_future_price_path_range_h{horizon}"] = (
            path.max(axis=1) - path.min(axis=1)
        ).astype(np.float32)

    # Global future price features (across all horizons)
    out["feat_future_price_max_all"] = future_prices_matrix.max(axis=1).astype(np.float32)
    out["feat_future_price_min_all"] = future_prices_matrix.min(axis=1).astype(np.float32)
    out["feat_future_price_range_all"] = (
        future_prices_matrix.max(axis=1) - future_prices_matrix.min(axis=1)
    ).astype(np.float32)
    out["feat_future_price_mean_all"] = future_prices_matrix.mean(axis=1).astype(np.float32)

    # Price trend features
    out["feat_future_price_trend_h1_h4"] = (
        future_prices_matrix[:, 3] - future_prices_matrix[:, 0]
    ).astype(np.float32)
    out["feat_future_price_trend_h4_h8"] = (
        future_prices_matrix[:, 7] - future_prices_matrix[:, 3]
    ).astype(np.float32)

    # Count price changes in the future path
    price_changes = np.abs(
        np.diff(np.column_stack([current_price[:, None], future_prices_matrix]), axis=1)
    ) > 1e-6
    out["feat_future_price_change_count"] = price_changes.sum(axis=1).astype(np.float32)

    # Steps to next price change
    first_change_idx = np.where(
        price_changes.any(axis=1),
        price_changes.argmax(axis=1) + 1,
        9
    ).astype(np.float32)
    out["feat_future_price_steps_to_change"] = first_change_idx

    return out


def add_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add critical interaction features that capture non-linear relationships.

    Why interactions matter:
    - "High price" effect depends on "gas availability"
    - "Weekend" pattern is completely different from "weekday" for same hour
    - "Gas balance" impact varies with "current power level"

    Args:
        frame: DataFrame with base features

    Returns:
        DataFrame with added interaction features
    """
    out = frame.copy()

    # Helper: safe normalization
    def normalize(series: pd.Series, typical_max: float) -> np.ndarray:
        """Normalize to [0, 1] range with clipping."""
        return np.clip(series.to_numpy() / typical_max, 0, 1).astype(np.float32)

    # Normalize key features (with safe column access)
    holder_col = "blast_furnace_gas_holder_2" if "blast_furnace_gas_holder_2" in out else "feat_cleanview_blast_furnace_gas_holder_2"
    holder_norm = normalize(out[holder_col], 100000)

    p50_col = "feat_p50_current" if "feat_p50_current" in out else "feat_generator_1_filled"
    p50_norm = normalize(out[p50_col], 120)

    pall_col = "feat_pall_current" if "feat_pall_current" in out else "feat_generator_all_filled"
    pall_norm = normalize(out[pall_col], 200)

    gen_bfg_col = "generator_use_blast_furnace_gas" if "generator_use_blast_furnace_gas" in out else None
    gen_bfg_norm = normalize(out[gen_bfg_col], 50000) if gen_bfg_col else np.zeros(len(out), dtype=np.float32)

    bf_supply_col = "feat_blast_furnace_observed_sum" if "feat_blast_furnace_observed_sum" in out else None
    bf_supply_norm = normalize(out[bf_supply_col], 150000) if bf_supply_col else np.zeros(len(out), dtype=np.float32)

    # ===== Category 1: Price Interactions =====
    # These are CRITICAL: price impact depends on system state

    # Price × Holder: High price matters only if we have gas
    out["feat_interact_price_holder"] = out["feat_known_price"] * holder_norm

    # Price × Current power: Price sensitivity varies with load level
    out["feat_interact_price_p50"] = out["feat_known_price"] * p50_norm
    out["feat_interact_price_pall"] = out["feat_known_price"] * pall_norm

    # Price × Gas balance: Can we respond to price signals?
    if "feat_bfg_balance_proxy" in out:
        bfg_balance_norm = np.clip(
            out["feat_bfg_balance_proxy"].to_numpy() / 10000, -1, 1
        ).astype(np.float32)
        out["feat_interact_price_bfg_balance"] = out["feat_known_price"] * bfg_balance_norm

    # Future price interactions (if future prices exist)
    if "feat_future_price_delta_h4" in out:
        # Future price change × Holder: Can we shift load to high-price periods?
        out["feat_interact_price_change_h4_holder"] = (
            out["feat_future_price_delta_h4"] * holder_norm
        )

        # Future price trend × Current power
        out["feat_interact_price_trend_h1_h4_pall"] = (
            out["feat_future_price_trend_h1_h4"] * pall_norm
        )

    # Price × Price level (quadratic price effect)
    out["feat_interact_price_price_level"] = (
        out["feat_known_price"] * out["feat_price_level"]
    )

    # ===== Category 2: Time Interactions =====
    # Weekend patterns are fundamentally different from weekday

    # Check for time-related columns
    weekend_col = None
    for col_name in ["feat_is_weekend", "feat_weekend", "is_weekend"]:
        if col_name in out:
            weekend_col = col_name
            break

    if weekend_col:
        # Weekend × Hour
        out["feat_interact_weekend_hour"] = (
            out[weekend_col] * out["feat_hour"]
        ).astype(np.float32)

        # Weekend × Peak hours (9-17)
        is_peak = ((out["feat_hour"] >= 9) & (out["feat_hour"] < 17)).astype(np.float32)
        out["feat_interact_weekend_peak"] = out[weekend_col] * is_peak

        # Weekend × Price level
        out["feat_interact_weekend_price_level"] = (
            out[weekend_col] * out["feat_price_level"]
        ).astype(np.float32)

    # Month × Hour (seasonal patterns vary by time of day)
    out["feat_interact_month_hour"] = (
        out["feat_month"] * out["feat_hour"]
    ).astype(np.float32)

    # ===== Category 3: Gas System Interactions =====
    # Physical constraints create non-linear relationships

    # Gas balance × Power level
    if "feat_bfg_balance_proxy" in out:
        out["feat_interact_bfg_balance_p50"] = bfg_balance_norm * p50_norm
        out["feat_interact_bfg_balance_pall"] = bfg_balance_norm * pall_norm

    # Holder × Generator BFG use: Storage buffering effect
    out["feat_interact_holder_gen_bfg"] = holder_norm * gen_bfg_norm

    # BF supply × Air heater use: Production coordination
    if "feat_air_heater_observed_sum" in out:
        ah_use_norm = normalize(out["feat_air_heater_observed_sum"], 50000)
        out["feat_interact_bf_supply_ah_use"] = bf_supply_norm * ah_use_norm

    # Holder level × BF supply: Storage strategy
    out["feat_interact_holder_bf_supply"] = holder_norm * bf_supply_norm

    # ===== Category 4: State/Quality Interactions =====

    # Count simultaneous zero states (system fragility)
    zero_state_cols = [c for c in out.columns if c.startswith("feat_state_") and c.endswith("_is_zero")]
    if zero_state_cols:
        total_zeros = out[zero_state_cols].sum(axis=1).astype(np.float32)
        out["feat_interact_total_zero_states"] = total_zeros
        out["feat_interact_critical_zero"] = (total_zeros >= 3).astype(np.float32)

        # Zero states × Power level (risk indicator)
        out["feat_interact_zero_states_pall"] = total_zeros * pall_norm

    # Outlier count × Power level (quality risk)
    if "feat_quality_outlier_count" in out:
        out["feat_interact_outlier_pall"] = (
            out["feat_quality_outlier_count"] * pall_norm
        ).astype(np.float32)

    # ===== Category 5: Three-way Interactions (Advanced) =====
    # Only the most critical ones to avoid explosion

    # Price × Holder × Weekend: Weekend storage strategy under price signals
    if weekend_col:
        out["feat_interact_price_holder_weekend"] = (
            out["feat_known_price"] * holder_norm * out[weekend_col]
        ).astype(np.float32)

    # Price × BFG balance × Peak hour
    if "feat_bfg_balance_proxy" in out and weekend_col:
        is_peak = ((out["feat_hour"] >= 9) & (out["feat_hour"] < 17)).astype(np.float32)
        out["feat_interact_price_balance_peak"] = (
            out["feat_known_price"] * bfg_balance_norm * is_peak
        ).astype(np.float32)

    return out


def build_enhanced_features(
    causal_frame: pd.DataFrame,
    price_lookup: Mapping[tuple[int, int], float]
) -> pd.DataFrame:
    """Complete pipeline: add future prices (interactions removed due to zero impact).

    Args:
        causal_frame: Output from causal_preprocessing.preprocess_causal_raw_tables
        price_lookup: {(month, half_hour_slot): price} mapping

    Returns:
        DataFrame with future price features only
    """
    # Add future price features only (interactions had zero impact in testing)
    with_prices = add_future_price_features(causal_frame, price_lookup)

    return with_prices
