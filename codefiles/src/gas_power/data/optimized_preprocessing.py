"""Optimized data preprocessing with advanced imputation and outlier handling.

This module enhances the original causal_preprocessing.py with:
1. Device-specific missing value imputation
2. Outlier detection and correction
3. Data smoothing options
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def advanced_fill_missing(
    series: pd.Series,
    timestamps: pd.Series,
    device_type: str = 'generic'
) -> tuple[pd.Series, pd.Series]:
    """Advanced missing value imputation with device-specific strategies.

    Args:
        series: Series with missing values
        timestamps: Corresponding datetime series
        device_type: Type of device ('blast_furnace', 'converter', 'air_heater', 'generic')

    Returns:
        Tuple of (filled_series, method_series)
    """
    filled = series.copy()
    methods = pd.Series(['original'] * len(series), index=series.index)

    # Step 1: Short-term linear interpolation (<=4 consecutive missing)
    missing_mask = filled.isna()
    groups = (missing_mask != missing_mask.shift()).cumsum()
    gap_lengths = missing_mask.groupby(groups).transform('sum')

    short_gaps = missing_mask & (gap_lengths <= 4)
    if short_gaps.any():
        filled_temp = filled.interpolate(method='linear', limit=4)
        filled.loc[short_gaps] = filled_temp.loc[short_gaps]
        methods.loc[short_gaps] = 'linear_interpolation'

    # Step 2: Device-specific filling
    if device_type == 'blast_furnace':
        # Blast furnace: continuous operation, use spline interpolation
        still_missing = filled.isna()
        if still_missing.any():
            filled_temp = filled.interpolate(method='spline', order=2, limit=12)
            filled.loc[still_missing] = filled_temp.loc[still_missing]
            methods.loc[still_missing & filled.notna()] = 'blast_furnace_spline'

    elif device_type == 'converter':
        # Converter: intermittent operation, forward fill
        still_missing = filled.isna()
        if still_missing.any():
            filled_temp = filled.fillna(method='ffill', limit=8)
            filled.loc[still_missing] = filled_temp.loc[still_missing]
            methods.loc[still_missing & filled.notna()] = 'converter_forward_fill'

    elif device_type == 'air_heater':
        # Air heater: periodic operation, use same-hour median
        still_missing = filled.isna()
        if still_missing.any() and timestamps is not None:
            hour = pd.to_datetime(timestamps).dt.hour
            hour_median = filled.groupby(hour).transform('median')
            filled.loc[still_missing] = hour_median.loc[still_missing]
            methods.loc[still_missing & filled.notna()] = 'air_heater_hourly_median'

    # Step 3: Seasonal median (same hour same weekday)
    still_missing = filled.isna()
    if still_missing.any() and timestamps is not None:
        dt = pd.to_datetime(timestamps)
        hour = dt.dt.hour
        weekday = dt.dt.weekday

        # Group by hour and weekday
        hour_weekday_key = hour.astype(str) + '_' + weekday.astype(str)
        seasonal_median = filled.groupby(hour_weekday_key).transform('median')
        filled.loc[still_missing] = seasonal_median.loc[still_missing]
        methods.loc[still_missing & filled.notna()] = 'seasonal_hourly_weekday_median'

    # Step 4: Global median (last resort)
    still_missing = filled.isna()
    if still_missing.any():
        global_median = filled.median()
        filled.loc[still_missing] = global_median
        methods.loc[still_missing] = 'global_median'

    return filled, methods


def detect_outliers_multi_method(
    series: pd.Series,
    timestamps: pd.Series,
    methods: list[str] = None
) -> pd.Series:
    """Detect outliers using multiple methods and combine results.

    Args:
        series: Series to check for outliers
        timestamps: Corresponding datetime series
        methods: List of methods to use ('iqr', 'zscore', 'rate', 'isolation')

    Returns:
        Boolean series indicating outliers (True = outlier)
    """
    if methods is None:
        methods = ['iqr', 'rate']

    outlier_flags = []

    # Method 1: IQR with rolling window
    if 'iqr' in methods:
        rolling = series.rolling(96, min_periods=24)
        Q1 = rolling.quantile(0.25)
        Q3 = rolling.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR

        iqr_outliers = (series < lower) | (series > upper)
        outlier_flags.append(iqr_outliers)

    # Method 2: Change rate
    if 'rate' in methods:
        change_rate = series.pct_change().abs()
        rate_outliers = change_rate > 0.5  # 50% change in 15 minutes
        outlier_flags.append(rate_outliers)

    # Method 3: Z-score with rolling window
    if 'zscore' in methods:
        rolling_mean = series.rolling(96, min_periods=24).mean()
        rolling_std = series.rolling(96, min_periods=24).std()

        z_scores = ((series - rolling_mean) / rolling_std).abs()
        zscore_outliers = z_scores > 4
        outlier_flags.append(zscore_outliers)

    # Combine: outlier if ANY method flags it
    if outlier_flags:
        combined = pd.concat(outlier_flags, axis=1).any(axis=1)
        return combined
    else:
        return pd.Series([False] * len(series), index=series.index)


def apply_smoothing(
    series: pd.Series,
    method: str = 'savgol',
    **kwargs
) -> pd.Series:
    """Apply smoothing to reduce noise.

    Args:
        series: Series to smooth
        method: Smoothing method ('savgol', 'ema', 'rolling')
        **kwargs: Additional parameters for the smoothing method

    Returns:
        Smoothed series
    """
    if method == 'savgol':
        window = kwargs.get('window', 11)
        polyorder = kwargs.get('polyorder', 3)

        # Handle NaN values
        mask = series.notna()
        if mask.sum() < window:
            return series

        smoothed = series.copy()
        smoothed[mask] = savgol_filter(
            series[mask].values,
            window_length=window,
            polyorder=polyorder
        )
        return smoothed

    elif method == 'ema':
        span = kwargs.get('span', 4)
        return series.ewm(span=span, adjust=False).mean()

    elif method == 'rolling':
        window = kwargs.get('window', 8)
        return series.rolling(window, min_periods=1).mean()

    else:
        return series


def preprocess_with_optimization(
    tables: Mapping[str, pd.DataFrame],
    price_lookup: Mapping[tuple[int, int], float],
    split: str = 'train',
    fix_outliers: bool = True,
    advanced_imputation: bool = True,
    apply_smoothing_flag: bool = False
) -> pd.DataFrame:
    """Enhanced preprocessing pipeline with all optimizations.

    Args:
        tables: Dictionary of raw tables
        price_lookup: Price mapping
        split: 'train' or 'test'
        fix_outliers: Whether to detect and fix outliers
        advanced_imputation: Whether to use advanced imputation
        apply_smoothing_flag: Whether to apply smoothing

    Returns:
        Preprocessed DataFrame ready for feature engineering
    """
    # Import the original preprocessing
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / 'codefiles' / 'src'))

    from gas_power.data.causal_preprocessing import (
        merge_raw_tables,
        add_known_features
    )

    # Step 1: Merge tables
    merged = merge_raw_tables(tables)

    # Step 2: Outlier detection and correction (if enabled)
    if fix_outliers:
        # Physical constraints
        if 'generator_1' in merged.columns:
            merged['generator_1'] = merged['generator_1'].clip(0, 120)

        if 'generator_all' in merged.columns:
            merged['generator_all'] = merged['generator_all'].clip(0, 200)

        if 'blast_furnace_gas_holder_2' in merged.columns:
            merged['blast_furnace_gas_holder_2'] = merged['blast_furnace_gas_holder_2'].clip(0, 100000)

        # Detect and fix change rate outliers
        power_cols = [c for c in ['generator_1', 'generator_all'] if c in merged.columns]
        for col in power_cols:
            outliers = detect_outliers_multi_method(
                merged[col],
                merged['datetime'],
                methods=['rate']
            )

            # Replace outliers with NaN and interpolate
            merged.loc[outliers, col] = np.nan
            merged[col] = merged[col].interpolate(method='linear', limit=4)

    # Step 3: Advanced missing value imputation (if enabled)
    if advanced_imputation:
        # Define device types for key columns
        device_mapping = {
            'blast_furnace_1': 'blast_furnace',
            'blast_furnace_2': 'blast_furnace',
            'blast_furnace_4': 'blast_furnace',
            'blast_furnace_5': 'blast_furnace',
            'converter_1': 'converter',
            'converter_2': 'converter',
            'air_heater_1': 'air_heater',
            'air_heater_2': 'air_heater',
            'air_heater_4': 'air_heater',
            'air_heater_5': 'air_heater',
        }

        for col, device_type in device_mapping.items():
            if col in merged.columns and merged[col].isna().any():
                filled, methods = advanced_fill_missing(
                    merged[col],
                    merged['datetime'],
                    device_type
                )
                merged[col] = filled
                merged[f'{col}_imputation_method'] = methods

    # Step 4: Add known features (time, price, etc.)
    causal = add_known_features(merged, price_lookup)

    # Step 5: Apply smoothing to key variables (optional)
    if apply_smoothing_flag:
        smooth_cols = [
            'generator_1', 'generator_all',
            'blast_furnace_gas_holder_2'
        ]

        for col in smooth_cols:
            if col in causal.columns:
                causal[f'{col}_smooth'] = apply_smoothing(
                    causal[col],
                    method='savgol',
                    window=11,
                    polyorder=3
                )

    return causal


def get_preprocessing_stats(causal: pd.DataFrame) -> dict:
    """Get statistics about preprocessing results.

    Args:
        causal: Preprocessed DataFrame

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_rows': len(causal),
        'missing_counts': causal.isna().sum().to_dict(),
        'outlier_flags': {}
    }

    # Count outlier flags if present
    outlier_cols = [c for c in causal.columns if 'outlier' in c.lower()]
    for col in outlier_cols:
        if causal[col].dtype in [np.float32, np.float64, int]:
            stats['outlier_flags'][col] = int(causal[col].sum())

    return stats
