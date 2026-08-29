"""Multiscale temporal features for time series forecasting.

This module implements advanced time series features:
1. Multi-scale rolling statistics (15min, 1h, 6h, 12h, 24h, 7d)
2. Trend decomposition (STL)
3. Fourier features for capturing periodicity
4. Exponential weighted moving averages (EWMA)
5. Autocorrelation features
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# Optional imports
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def add_multiscale_rolling_stats(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    scales: Optional[List[int]] = None
) -> pd.DataFrame:
    """Add rolling statistics at multiple time scales.

    Args:
        df: Input DataFrame
        target_cols: Columns to compute rolling stats for (default: generator columns)
        scales: Time scales in steps (default: [4, 8, 24, 48, 96, 672])
                4=1h, 8=2h, 24=6h, 48=12h, 96=24h, 672=7d

    Returns:
        DataFrame with added rolling statistics
    """
    out = df.copy()

    if target_cols is None:
        target_cols = [
            'feat_generator_1_filled',
            'feat_generator_all_filled',
            'feat_p50_current',
            'feat_p120_current',
        ]

    if scales is None:
        # Default scales: 1h, 2h, 6h, 12h, 24h, 7d
        scales = [4, 8, 24, 48, 96, 672]

    for col in target_cols:
        if col not in df.columns:
            continue

        base_name = col.replace('feat_', '').replace('_filled', '').replace('_current', '')

        for scale in scales:
            # Scale name
            if scale < 96:
                scale_name = f"{scale*15}min"
            elif scale < 672:
                scale_name = f"{scale//4}h"
            else:
                scale_name = f"{scale//96}d"

            # Rolling mean
            out[f'feat_temporal_{base_name}_mean_{scale_name}'] = (
                df[col].rolling(scale, min_periods=max(1, scale//4)).mean()
            )

            # Rolling std
            out[f'feat_temporal_{base_name}_std_{scale_name}'] = (
                df[col].rolling(scale, min_periods=max(1, scale//4)).std()
            )

            # Rolling min/max
            out[f'feat_temporal_{base_name}_min_{scale_name}'] = (
                df[col].rolling(scale, min_periods=max(1, scale//4)).min()
            )
            out[f'feat_temporal_{base_name}_max_{scale_name}'] = (
                df[col].rolling(scale, min_periods=max(1, scale//4)).max()
            )

            # Rolling range (max - min)
            out[f'feat_temporal_{base_name}_range_{scale_name}'] = (
                out[f'feat_temporal_{base_name}_max_{scale_name}'] -
                out[f'feat_temporal_{base_name}_min_{scale_name}']
            )

            # Coefficient of variation (std / mean)
            mean_col = out[f'feat_temporal_{base_name}_mean_{scale_name}']
            std_col = out[f'feat_temporal_{base_name}_std_{scale_name}']
            out[f'feat_temporal_{base_name}_cv_{scale_name}'] = (
                std_col / np.maximum(np.abs(mean_col), 1e-6)
            )

    return out


def add_ewma_features(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    spans: Optional[List[int]] = None
) -> pd.DataFrame:
    """Add exponential weighted moving average features.

    EWMA gives more weight to recent observations, better for trend tracking.

    Args:
        df: Input DataFrame
        target_cols: Columns to compute EWMA for
        spans: Span parameters (default: [4, 12, 48, 96])

    Returns:
        DataFrame with added EWMA features
    """
    out = df.copy()

    if target_cols is None:
        target_cols = [
            'feat_generator_1_filled',
            'feat_generator_all_filled',
            'feat_p50_current',
            'feat_p120_current',
        ]

    if spans is None:
        # Default spans: 1h, 3h, 12h, 24h
        spans = [4, 12, 48, 96]

    for col in target_cols:
        if col not in df.columns:
            continue

        base_name = col.replace('feat_', '').replace('_filled', '').replace('_current', '')

        for span in spans:
            # Scale name
            if span < 96:
                scale_name = f"{span*15}min"
            else:
                scale_name = f"{span//4}h"

            # EWMA
            out[f'feat_temporal_{base_name}_ewma_{scale_name}'] = (
                df[col].ewm(span=span, adjust=False).mean()
            )

            # EWMA std
            out[f'feat_temporal_{base_name}_ewmstd_{scale_name}'] = (
                df[col].ewm(span=span, adjust=False).std()
            )

            # Distance from EWMA (current - EWMA)
            out[f'feat_temporal_{base_name}_dist_ewma_{scale_name}'] = (
                df[col] - out[f'feat_temporal_{base_name}_ewma_{scale_name}']
            )

    return out


def add_trend_decomposition_features(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    period: int = 96  # 24 hours
) -> pd.DataFrame:
    """Add STL trend decomposition features.

    STL decomposes time series into:
    - Trend: Long-term pattern
    - Seasonal: Repeating pattern
    - Residual: Random noise

    Args:
        df: Input DataFrame
        target_cols: Columns to decompose
        period: Seasonal period (default: 96 = 24 hours)

    Returns:
        DataFrame with added decomposition features
    """
    out = df.copy()

    if not HAS_STATSMODELS:
        print("  Warning: statsmodels not installed, skipping STL decomposition")
        return out

    if target_cols is None:
        target_cols = [
            'feat_generator_1_filled',
            'feat_generator_all_filled',
        ]

    for col in target_cols:
        if col not in df.columns or len(df) < period * 2:
            continue

        base_name = col.replace('feat_', '').replace('_filled', '')

        try:
            # STL decomposition
            stl = STL(df[col].fillna(method='ffill').fillna(method='bfill'),
                     period=period,
                     seasonal=13,
                     trend=None)
            result = stl.fit()

            # Trend component
            out[f'feat_temporal_{base_name}_trend'] = result.trend

            # Seasonal component
            out[f'feat_temporal_{base_name}_seasonal'] = result.seasonal

            # Residual component
            out[f'feat_temporal_{base_name}_residual'] = result.resid

            # Trend strength (1 - Var(residual) / Var(trend + residual))
            detrended = result.seasonal + result.resid
            out[f'feat_temporal_{base_name}_trend_strength'] = (
                1 - np.var(result.resid) / np.var(detrended + 1e-10)
            )

            # Seasonal strength
            deseasonal = result.trend + result.resid
            out[f'feat_temporal_{base_name}_seasonal_strength'] = (
                1 - np.var(result.resid) / np.var(deseasonal + 1e-10)
            )

        except Exception as e:
            print(f"Warning: STL decomposition failed for {col}: {e}")
            continue

    return out


def add_fourier_features(
    df: pd.DataFrame,
    period: int = 96,  # 24 hours
    n_terms: int = 3
) -> pd.DataFrame:
    """Add Fourier features to capture periodicity.

    Fourier features are sine/cosine functions that capture periodic patterns.

    Args:
        df: Input DataFrame with 'datetime' column
        period: Main period in steps (default: 96 = 24h)
        n_terms: Number of Fourier terms (default: 3)

    Returns:
        DataFrame with added Fourier features
    """
    out = df.copy()

    if 'datetime' not in df.columns:
        return out

    # Create time index (steps since start)
    time_idx = np.arange(len(df))

    for k in range(1, n_terms + 1):
        # Frequency
        freq = 2 * np.pi * k / period

        # Sine and cosine
        out[f'feat_temporal_fourier_sin_{k}'] = np.sin(freq * time_idx)
        out[f'feat_temporal_fourier_cos_{k}'] = np.cos(freq * time_idx)

    # Additional daily and weekly cycles
    if 'datetime' in df.columns:
        # Hour of day cycle (0-23)
        hour = df['datetime'].dt.hour + df['datetime'].dt.minute / 60
        out['feat_temporal_hour_sin'] = np.sin(2 * np.pi * hour / 24)
        out['feat_temporal_hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Day of week cycle (0-6)
        dow = df['datetime'].dt.dayofweek
        out['feat_temporal_dow_sin'] = np.sin(2 * np.pi * dow / 7)
        out['feat_temporal_dow_cos'] = np.cos(2 * np.pi * dow / 7)

    return out


def add_autocorrelation_features(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    lags: Optional[List[int]] = None
) -> pd.DataFrame:
    """Add autocorrelation features.

    Captures how correlated the series is with its past values.

    Args:
        df: Input DataFrame
        target_cols: Columns to compute autocorrelation for
        lags: Lags to compute (default: [4, 8, 24, 96])

    Returns:
        DataFrame with added autocorrelation features
    """
    out = df.copy()

    if target_cols is None:
        target_cols = [
            'feat_generator_1_filled',
            'feat_generator_all_filled',
        ]

    if lags is None:
        # 1h, 2h, 6h, 24h
        lags = [4, 8, 24, 96]

    for col in target_cols:
        if col not in df.columns:
            continue

        base_name = col.replace('feat_', '').replace('_filled', '')

        for lag in lags:
            # Rolling autocorrelation
            window = min(96, max(48, lag * 4))

            rolled = df[col].rolling(window, min_periods=window//2)

            def compute_autocorr(x):
                if len(x) < lag + 1:
                    return np.nan
                return np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else np.nan

            out[f'feat_temporal_{base_name}_autocorr_lag{lag}'] = (
                rolled.apply(compute_autocorr, raw=True)
            )

    return out


def add_change_point_features(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    window: int = 24
) -> pd.DataFrame:
    """Add change point detection features.

    Detects sudden changes in mean or variance.

    Args:
        df: Input DataFrame
        target_cols: Columns to detect change points in
        window: Window size for change detection

    Returns:
        DataFrame with added change point features
    """
    out = df.copy()

    if target_cols is None:
        target_cols = [
            'feat_generator_1_filled',
            'feat_generator_all_filled',
        ]

    for col in target_cols:
        if col not in df.columns:
            continue

        base_name = col.replace('feat_', '').replace('_filled', '')

        # Mean change detection
        mean_before = df[col].rolling(window, min_periods=1).mean()
        mean_after = df[col].shift(-window).rolling(window, min_periods=1).mean()

        out[f'feat_temporal_{base_name}_mean_change'] = (
            (mean_after - mean_before) / np.maximum(np.abs(mean_before), 1e-6)
        )

        # Variance change detection
        var_before = df[col].rolling(window, min_periods=1).var()
        var_after = df[col].shift(-window).rolling(window, min_periods=1).var()

        out[f'feat_temporal_{base_name}_var_change'] = (
            (var_after - var_before) / np.maximum(var_before, 1e-6)
        )

    return out


def add_all_multiscale_temporal_features(
    df: pd.DataFrame,
    add_rolling: bool = True,
    add_ewma: bool = True,
    add_trend: bool = False,  # Expensive, set to False by default
    add_fourier: bool = True,
    add_autocorr: bool = False,  # Expensive, set to False by default
    add_changepoint: bool = True
) -> pd.DataFrame:
    """Add all multiscale temporal features.

    Args:
        df: Input DataFrame
        add_rolling: Add rolling statistics
        add_ewma: Add EWMA features
        add_trend: Add trend decomposition (expensive)
        add_fourier: Add Fourier features
        add_autocorr: Add autocorrelation features (expensive)
        add_changepoint: Add change point features

    Returns:
        DataFrame with all temporal features
    """
    out = df.copy()

    print("Adding multiscale temporal features...")

    if add_rolling:
        print("  - Rolling statistics...")
        out = add_multiscale_rolling_stats(out)

    if add_ewma:
        print("  - EWMA features...")
        out = add_ewma_features(out)

    if add_trend:
        print("  - Trend decomposition (this may take a while)...")
        out = add_trend_decomposition_features(out)

    if add_fourier:
        print("  - Fourier features...")
        out = add_fourier_features(out)

    if add_autocorr:
        print("  - Autocorrelation features (this may take a while)...")
        out = add_autocorrelation_features(out)

    if add_changepoint:
        print("  - Change point features...")
        out = add_change_point_features(out)

    # Count new features
    new_features = [c for c in out.columns if c.startswith('feat_temporal_')]
    print(f"  Total new temporal features added: {len(new_features)}")

    return out


def get_multiscale_feature_summary() -> dict:
    """Get summary of multiscale temporal features.

    Returns:
        Dictionary with feature counts
    """
    # Based on default settings
    return {
        'rolling_stats': 6 * 4 * 6,  # 6 stats × 4 cols × 6 scales = 144
        'ewma': 3 * 4 * 4,  # 3 stats × 4 cols × 4 spans = 48
        'fourier': 2 * 3 + 4,  # 2 (sin/cos) × 3 terms + 4 cycle = 10
        'changepoint': 2 * 2,  # 2 stats × 2 cols = 4
        'total_fast': 206,  # Without trend and autocorr
        'trend_decomp': 5 * 2,  # 5 features × 2 cols = 10 (expensive)
        'autocorr': 4 * 2 * 4,  # 4 lags × 2 cols = 8 (expensive)
        'total_all': 224  # With all features
    }
