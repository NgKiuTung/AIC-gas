"""Enhanced preprocessing with advanced data quality improvements."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from scipy import stats


def detect_anomalies_multivariate(
    frame: pd.DataFrame,
    feature_groups: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    多变量异常检测 - 考虑变量间的相关性

    改善点：
    - 不仅看单变量的异常，还要看多变量组合的异常
    - 例如：储罐水平高 + 产气量低 = 异常
    """
    flags = frame.copy()

    for group_name, columns in feature_groups.items():
        available = [col for col in columns if col in frame.columns]
        if len(available) < 2:
            continue

        # 使用Mahalanobis距离检测多变量异常
        group_data = frame[available].fillna(method='ffill').fillna(0)

        # 计算协方差矩阵
        try:
            cov = np.cov(group_data.T)
            cov_inv = np.linalg.inv(cov)
            mean = group_data.mean()

            # 计算每个样本的Mahalanobis距离
            diff = group_data - mean
            mahal_dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

            # 使用卡方分布确定阈值
            threshold = np.sqrt(stats.chi2.ppf(0.999, df=len(available)))
            flags[f'feat_multivar_anomaly_{group_name}'] = (mahal_dist > threshold).astype('int8')
        except np.linalg.LinAlgError:
            # 奇异矩阵，跳过
            continue

    return flags


def add_data_quality_features(frame: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame:
    """
    添加数据质量特征 - 让模型知道数据的可信度

    改善点：
    - 标记哪些数据是插值的
    - 标记数据的稳定性（高方差 = 不稳定）
    - 标记数据的新鲜度（距离上次观测的时间）
    """
    out = frame.copy()

    for col in value_columns:
        if col not in frame.columns:
            continue

        # 1. 数据新鲜度：距离上次真实观测的时间步数
        is_observed = ~frame[f'feat_missing_{col}'].astype(bool) if f'feat_missing_{col}' in frame.columns else pd.Series([True] * len(frame))
        time_since_observed = (~is_observed).groupby(is_observed.cumsum()).cumsum()
        out[f'feat_staleness_{col}'] = time_since_observed.astype('int16')

        # 2. 局部稳定性：最近4个时间步的变异系数
        rolling_std = frame[col].rolling(4, min_periods=2).std()
        rolling_mean = frame[col].rolling(4, min_periods=2).mean()
        cv = rolling_std / (rolling_mean.abs() + 1e-6)
        out[f'feat_stability_{col}'] = cv.fillna(0).clip(0, 10)

        # 3. 突变检测：与前值的相对变化
        pct_change = frame[col].pct_change().fillna(0)
        out[f'feat_sudden_change_{col}'] = (pct_change.abs() > 0.3).astype('int8')

    return out


def smart_interpolation(
    series: pd.Series,
    timestamps: pd.Series,
    related_series: Dict[str, pd.Series] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    智能插值 - 考虑相关变量的信息

    改善点：
    - 如果储罐水平缺失，可以用产气量和用气量推算
    - 使用相关变量的趋势辅助插值
    """
    values = series.copy()
    methods = pd.Series(['observed'] * len(series), index=series.index)

    missing_mask = series.isna()

    if related_series and any(missing_mask):
        # 尝试用相关变量进行线性回归插值
        for related_name, related_data in related_series.items():
            if related_data is None:
                continue

            # 在有数据的地方训练简单线性模型
            valid_mask = ~missing_mask & ~related_data.isna()
            if valid_mask.sum() < 10:
                continue

            X_train = related_data[valid_mask].values.reshape(-1, 1)
            y_train = series[valid_mask].values

            # 简单线性回归
            if len(X_train) > 0 and np.std(X_train) > 1e-6:
                slope = np.cov(X_train.flatten(), y_train)[0, 1] / np.var(X_train)
                intercept = np.mean(y_train) - slope * np.mean(X_train)

                # 预测缺失值
                X_pred = related_data[missing_mask].values.reshape(-1, 1)
                if len(X_pred) > 0 and not np.any(np.isnan(X_pred)):
                    pred = slope * X_pred.flatten() + intercept
                    values.loc[missing_mask] = pred
                    methods.loc[missing_mask] = f'correlated_{related_name}'
                    break

    return values, methods


def add_physical_constraint_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    添加物理约束特征 - 基于燃气系统的物理规律

    改善点：
    - 燃气平衡：产气量 ≈ 用气量 + 储罐变化
    - 能量守恒：发电量与用气量的比例关系
    - 设备约束：发电机不能超过额定功率
    """
    out = frame.copy()

    # 1. 燃气平衡检查（产气 - 用气 - 储罐变化应该接近0）
    gas_columns = [col for col in frame.columns if 'blast_furnace' in col and 'holder' not in col]
    user_columns = [col for col in frame.columns if 'user' in col]

    if gas_columns and user_columns:
        total_production = frame[gas_columns].sum(axis=1)
        total_consumption = frame[user_columns].sum(axis=1)

        if 'blast_furnace_gas_holder_2' in frame.columns:
            holder_change = frame['blast_furnace_gas_holder_2'].diff()
            balance = total_production - total_consumption - holder_change
            out['feat_gas_balance_residual'] = balance.fillna(0)
            out['feat_gas_balance_violation'] = (balance.abs() > balance.abs().quantile(0.95)).astype('int8')

    # 2. 发电效率特征
    if 'generator_use_blast_furnace_gas' in frame.columns and 'feat_generator_all_filled' in frame.columns:
        efficiency = frame['feat_generator_all_filled'] / (frame['generator_use_blast_furnace_gas'] + 1e-6)
        out['feat_generation_efficiency'] = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 3. 储罐利用率
    if 'blast_furnace_gas_holder_2' in frame.columns:
        # 假设储罐容量上限（需要根据实际数据调整）
        holder_max = frame['blast_furnace_gas_holder_2'].quantile(0.99)
        holder_min = frame['blast_furnace_gas_holder_2'].quantile(0.01)

        utilization = (frame['blast_furnace_gas_holder_2'] - holder_min) / (holder_max - holder_min + 1e-6)
        out['feat_holder_utilization'] = utilization.clip(0, 1)
        out['feat_holder_near_full'] = (utilization > 0.9).astype('int8')
        out['feat_holder_near_empty'] = (utilization < 0.1).astype('int8')

    return out


def add_temporal_context_features(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    添加时间上下文特征 - 捕捉更丰富的时间模式

    改善点：
    - 不仅看当前值，还要看趋势和加速度
    - 周期性特征（日、周、月）
    - 特殊时段标记（节假日、维护期）
    """
    out = frame.copy()

    if target_col in frame.columns:
        # 1. 趋势特征（一阶和二阶导数）
        out[f'feat_trend_1st_{target_col}'] = frame[target_col].diff()
        out[f'feat_trend_2nd_{target_col}'] = frame[target_col].diff().diff()

        # 2. 动量特征（最近的平均变化率）
        out[f'feat_momentum_4_{target_col}'] = frame[target_col].diff(4)
        out[f'feat_momentum_96_{target_col}'] = frame[target_col].diff(96)  # 一天前

        # 3. 波动率（最近的标准差）
        out[f'feat_volatility_96_{target_col}'] = frame[target_col].rolling(96).std()

    # 4. 时间周期特征增强
    if 'datetime' in frame.columns:
        dt = pd.to_datetime(frame['datetime'])

        # 月内进度（0-1）
        out['feat_month_progress'] = (dt.dt.day - 1) / (dt.dt.days_in_month - 1)

        # 工作日vs周末的小时交互
        is_weekend = (dt.dt.dayofweek >= 5).astype(int)
        hour = dt.dt.hour
        out['feat_weekend_hour_interaction'] = is_weekend * hour

        # 是否月初/月末（通常有不同的生产计划）
        out['feat_is_month_start'] = (dt.dt.day <= 3).astype('int8')
        out['feat_is_month_end'] = (dt.dt.day >= dt.dt.days_in_month - 2).astype('int8')

    return out
