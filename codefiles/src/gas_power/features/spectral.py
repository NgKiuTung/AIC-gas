"""Causal Fourier and Haar-wavelet descriptors for industrial time series."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _spectral_array(values: np.ndarray, window: int) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("spectral input must be one-dimensional")
    if window < 16 or window % 16:
        raise ValueError("window must be at least 16 and divisible by 16")
    if not np.isfinite(values).all():
        raise ValueError("spectral input must be finite")
    names = [
        *(f"fft_bin{index}_power" for index in range(1, 5)),
        "fft_low_band_power",
        "fft_mid_band_power",
        "fft_high_band_power",
        "fft_spectral_centroid",
        *(f"haar_detail_l{level}_power" for level in range(1, 5)),
    ]
    output = {name: np.zeros(len(values), dtype=np.float32) for name in names}
    if len(values) < window:
        return output

    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    centered = windows - windows.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    non_dc = power[:, 1:]
    total = np.maximum(non_dc.sum(axis=1), 1e-12)
    for index in range(1, 5):
        output[f"fft_bin{index}_power"][window - 1 :] = power[:, index] / total
    output["fft_low_band_power"][window - 1 :] = power[:, 1:4].sum(axis=1) / total
    output["fft_mid_band_power"][window - 1 :] = power[:, 4:13].sum(axis=1) / total
    output["fft_high_band_power"][window - 1 :] = power[:, 13:].sum(axis=1) / total
    normalized_frequency = np.arange(1, non_dc.shape[1] + 1, dtype=float) / non_dc.shape[1]
    output["fft_spectral_centroid"][window - 1 :] = (
        (non_dc * normalized_frequency).sum(axis=1) / total
    )

    approximation = centered
    time_energy = np.maximum(np.square(centered).sum(axis=1), 1e-12)
    for level in range(1, 5):
        even = approximation[:, 0::2]
        odd = approximation[:, 1::2]
        detail = (even - odd) / np.sqrt(2.0)
        approximation = (even + odd) / np.sqrt(2.0)
        output[f"haar_detail_l{level}_power"][window - 1 :] = np.square(detail).sum(axis=1) / time_energy
    return output


def add_causal_spectral_features(
    frame: pd.DataFrame,
    signals: Mapping[str, str],
    *,
    window: int = 96,
) -> pd.DataFrame:
    """Return spectral features using windows ending at the current timestamp.

    Rows before a complete window is available receive zero plus an explicit
    readiness flag.  No backfill or centered window is used.
    """
    if "datetime" not in frame:
        raise ValueError("spectral feature input requires datetime")
    out = pd.DataFrame({"datetime": pd.to_datetime(frame["datetime"], errors="raise")})
    out[f"feat_spectral_window{window}_ready"] = (np.arange(len(frame)) >= window - 1).astype("int8")
    for stem, column in signals.items():
        if column not in frame:
            raise ValueError(f"missing spectral source column: {column}")
        arrays = _spectral_array(pd.to_numeric(frame[column], errors="raise").to_numpy(), window)
        for descriptor, values in arrays.items():
            out[f"feat_spectral_{stem}_w{window}_{descriptor}"] = values
    return out
