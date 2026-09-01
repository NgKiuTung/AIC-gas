"""Causal gas-balance residual features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_physical_balance_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add supply-demand-storage residual signals without changing raw values."""
    required = {
        "feat_blast_furnace_observed_sum",
        "feat_air_heater_observed_sum",
        "feat_blast_furnace_user_observed_sum",
        "generator_use_blast_furnace_gas",
        "into_gas_mixed_blast_furnace",
        "blast_furnace_gas_holder_2",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing physical-balance columns: {sorted(missing)}")
    out = frame.copy()
    supply = out["feat_blast_furnace_observed_sum"].to_numpy(dtype=np.float64)
    demand = (
        out["feat_air_heater_observed_sum"]
        + out["feat_blast_furnace_user_observed_sum"]
        + out["generator_use_blast_furnace_gas"]
        + out["into_gas_mixed_blast_furnace"]
    ).to_numpy(dtype=np.float64)
    holder = out["blast_furnace_gas_holder_2"].to_numpy(dtype=np.float64)
    residual = supply - demand - np.r_[0.0, np.diff(holder)]
    out["feat_energy_balance_residual"] = residual
    out["feat_energy_balance_abs"] = np.abs(residual)
    out["feat_energy_balance_ratio"] = residual / np.maximum(np.abs(supply), 1e-6)
    values = out[[
        "feat_energy_balance_residual",
        "feat_energy_balance_abs",
        "feat_energy_balance_ratio",
    ]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Physical-balance features contain NaN or Inf")
    return out
