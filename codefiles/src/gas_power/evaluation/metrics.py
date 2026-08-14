"""Metrics shared by tests and stable pipelines."""

from __future__ import annotations

import numpy as np


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denominator = np.maximum(np.abs(truth), epsilon)
    return float(np.mean(np.abs(truth - prediction) / denominator) * 100.0)

