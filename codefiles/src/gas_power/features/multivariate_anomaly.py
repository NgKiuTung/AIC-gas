"""Strictly causal multivariate anomaly features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2


DEFAULT_ANOMALY_GROUPS: dict[str, tuple[str, ...]] = {
    "gas_supply": (
        "feat_blast_furnace_observed_sum",
        "blast_furnace_gas_holder_2",
        "feat_blast_furnace_user_observed_sum",
        "generator_use_blast_furnace_gas",
    ),
    "gas_composition": (
        "into_gas_mixed_blast_furnace",
        "into_gas_mixed_coke",
        "into_gas_mixed_converter",
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
    ),
    "power_gas": (
        "feat_generator_1_filled",
        "feat_generator_all_filled",
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
    ),
    "user_balance": (
        "feat_blast_furnace_user_observed_sum",
        "feat_converter_user_observed_sum",
        "feat_blast_furnace_observed_sum",
        "feat_air_heater_observed_sum",
    ),
}

ANOMALY_WINDOW = 96
ANOMALY_MIN_HISTORY = 24
ANOMALY_SHRINKAGE = 0.10
ANOMALY_QUANTILE = 0.999


def _resolve_value(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    filled_column = f"feat_{column}_filled"
    if filled_column in frame:
        return pd.to_numeric(frame[filled_column], errors="coerce")
    return None


def _mahalanobis_history_scores(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(len(values), dtype=float)
    flags = np.zeros(len(values), dtype=np.int8)
    dimensions = values.shape[1]
    threshold = float(np.sqrt(chi2.ppf(ANOMALY_QUANTILE, df=dimensions)))
    for index in range(len(values)):
        start = max(0, index - ANOMALY_WINDOW)
        history = values[start:index]
        if len(history) < ANOMALY_MIN_HISTORY:
            continue

        center = np.median(history, axis=0)
        mad = 1.4826 * np.median(np.abs(history - center), axis=0)
        standard_deviation = np.std(history, axis=0)
        scale = np.where(mad > 1e-9, mad, standard_deviation)
        scale = np.where(scale > 1e-9, scale, 1.0)
        history_scaled = np.clip((history - center) / scale, -8.0, 8.0)
        current_scaled = np.clip((values[index] - center) / scale, -8.0, 8.0)
        covariance = np.atleast_2d(np.cov(history_scaled, rowvar=False))
        diagonal = np.diag(np.diag(covariance))
        covariance = (1.0 - ANOMALY_SHRINKAGE) * covariance + ANOMALY_SHRINKAGE * diagonal
        covariance += np.eye(dimensions) * 1e-3
        inverse = np.linalg.pinv(covariance)
        distance = float(np.sqrt(max(current_scaled @ inverse @ current_scaled, 0.0)))
        scores[index] = distance
        flags[index] = int(distance > threshold)
    return scores, flags


def add_causal_multivariate_anomaly_features(
    frame: pd.DataFrame,
    groups: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Add past-window Mahalanobis scores without changing source measurements."""
    selected_groups = DEFAULT_ANOMALY_GROUPS if groups is None else groups
    added: dict[str, pd.Series] = {}
    for group_name, columns in selected_groups.items():
        resolved = [_resolve_value(frame, column) for column in columns]
        available = [series for series in resolved if series is not None]
        if len(available) < 2:
            continue
        values = pd.concat(available, axis=1).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in anomaly group {group_name}")
        scores, flags = _mahalanobis_history_scores(values)
        added[f"feat_multivar_anomaly_score_{group_name}"] = pd.Series(scores, index=frame.index)
        added[f"feat_multivar_anomaly_{group_name}"] = pd.Series(flags, index=frame.index)
    return pd.DataFrame(added, index=frame.index)
