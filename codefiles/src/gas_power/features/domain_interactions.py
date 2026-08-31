"""Causal domain interaction features for gas storage and generation fuel mix."""

from __future__ import annotations

import numpy as np
import pandas as pd


HOLDER_CAPACITY_M3 = 200_000.0
EPSILON = 1e-6


def add_domain_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add two finite, current-and-past-only domain summaries.

    The holder interaction uses the normalized BFG holder level and the current
    high-furnace-gas balance ratio. Fuel structure is represented by the HHI of
    the three generator fuel shares, so no arbitrary fuel calorific weights are
    introduced.
    """
    required = {
        "feat_blast_furnace_observed_sum",
        "feat_air_heater_observed_sum",
        "feat_blast_furnace_user_observed_sum",
        "into_gas_mixed_blast_furnace",
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
        "blast_furnace_gas_holder_2",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing domain-interaction columns: {sorted(missing)}")

    out = frame.copy()
    supply = pd.to_numeric(out["feat_blast_furnace_observed_sum"], errors="coerce")
    bfg_balance = (
        supply
        - pd.to_numeric(out["feat_air_heater_observed_sum"], errors="coerce")
        - pd.to_numeric(out["feat_blast_furnace_user_observed_sum"], errors="coerce")
        - pd.to_numeric(out["into_gas_mixed_blast_furnace"], errors="coerce")
        - pd.to_numeric(out["generator_use_blast_furnace_gas"], errors="coerce")
    )
    holder_fraction = (
        pd.to_numeric(out["blast_furnace_gas_holder_2"], errors="coerce")
        / HOLDER_CAPACITY_M3
    )
    flow_scale = pd.concat(
        [
            supply.abs(),
            pd.to_numeric(out["feat_air_heater_observed_sum"], errors="coerce").abs(),
            pd.to_numeric(out["feat_blast_furnace_user_observed_sum"], errors="coerce").abs(),
            pd.to_numeric(out["into_gas_mixed_blast_furnace"], errors="coerce").abs(),
            pd.to_numeric(out["generator_use_blast_furnace_gas"], errors="coerce").abs(),
        ],
        axis=1,
    ).max(axis=1).clip(lower=EPSILON)
    balance_ratio = bfg_balance / flow_scale
    out["feat_interact_holder_bfg_balance"] = holder_fraction * balance_ratio

    fuels = pd.concat(
        [
            pd.to_numeric(out["generator_use_blast_furnace_gas"], errors="coerce"),
            pd.to_numeric(out["generator_use_coke_gas"], errors="coerce"),
            pd.to_numeric(out["generator_use_converter_gas"], errors="coerce"),
        ],
        axis=1,
    ).clip(lower=0.0)
    total_fuel = fuels.sum(axis=1)
    shares = fuels.div(total_fuel.where(total_fuel > EPSILON, 1.0), axis=0)
    hhi = shares.pow(2).sum(axis=1).where(total_fuel > EPSILON, 0.0)
    out["feat_generation_fuel_structure_hhi"] = hhi

    values = out[
        ["feat_interact_holder_bfg_balance", "feat_generation_fuel_structure_hhi"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Domain-interaction features contain NaN or Inf")
    return out
