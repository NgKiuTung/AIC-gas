"""私有赛事研究：冻结历史模板与仅向历史计算的快慢状态修正。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gasstage.baselines import reconcile
from gasstage.targets import future_index
from gasrefine.common import message

TEMPLATES = {"week4": (7, 4), "week2": (7, 2), "day7": (1, 7)}


def template(history: pd.DataFrame, index: pd.DatetimeIndex, contract: dict,
             name: str) -> np.ndarray:
    """只查询已完整结束的历史块；缺失类比使用训练历史中位数回退。"""
    period_days, count = TEMPLATES[name]
    stamps = future_index(index, contract)
    period = np.timedelta64(period_days * 1440, "m")
    last = history.index[-1].to_datetime64()
    jumps = np.maximum(1, np.ceil((stamps - last) / period).astype("int64"))
    lookup = stamps[..., None] - (jumps[..., None] + np.arange(count)) * period
    values = history.reindex(pd.DatetimeIndex(lookup.ravel())).to_numpy()
    values = values.reshape(len(index), 96, count, 2)
    counts = np.isfinite(values).sum(axis=2)
    means = np.nansum(values, axis=2) / np.maximum(counts, 1)
    return np.where(counts > 0, means, history.median().to_numpy())


def mixed_template(history, index, contract, day_weight: float) -> np.ndarray:
    """日周模板的固定凸组合；权重来源为开发选择而非测试标签。"""
    if not 0 <= day_weight <= 1:
        raise ValueError(message("template.weight"))
    if day_weight == 0:
        return template(history, index, contract, "week4")
    day = template(history, index, contract, "day7")
    if day_weight == 1:
        return day
    return day_weight * day + (1-day_weight) * template(history, index, contract, "week4")


def smooth_delta(delta: np.ndarray, index: pd.DatetimeIndex,
                 window_hours: float) -> np.ndarray:
    """按时间窗(t-W,t]平滑代理偏差，不采用居中窗口或完整缺口长度。"""
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(message("index.invalid"))
    if window_hours <= 0:
        return np.zeros_like(delta)
    return pd.DataFrame(delta, index=index).rolling(
        pd.Timedelta(hours=window_hours), closed="right", min_periods=1).mean().to_numpy()


def corrected_curve(week: np.ndarray, delta: np.ndarray, level: np.ndarray,
                    recipe: dict, offset_minutes: int = 0) -> np.ndarray:
    """快项描述瞬时偏差，慢项描述持续水平；仅确定的非负/包含关系约束。"""
    h = (np.arange(96)*15 + 7.5 + offset_minutes) / 60
    fast, slow = recipe["fast_hours"], recipe["slow_hours"]
    if fast <= 0 or slow <= 0:
        raise ValueError(message("decay.invalid"))
    gain = recipe.get("level_gain", 1.0)
    slow_level = gain * level
    out = week + (delta-slow_level)[:, None, :] * np.exp(-h/fast)[None, :, None]
    out += slow_level[:, None, :] * np.exp(-h/slow)[None, :, None]
    return reconcile(out)


def forecast(model, features: pd.DataFrame, index: pd.DatetimeIndex,
             recipe: dict, return_context: bool = False):
    """model为截止前冻结快照模型；features仅过程视图，支持一致的任意分批推理。"""
    cutoff = pd.Timestamp(model.metadata["cutoff"])
    if index.min() < cutoff or index.max() > features.index.max():
        raise ValueError(message("origin.invalid"))
    # NOTE: 以模型截止而非批次首行定义暖启动，保证分批/全批的数值序列一致。
    window = recipe["smooth_hours"]
    start = cutoff - pd.Timedelta(hours=max(window, 1))
    support = features.loc[start:index.max()]
    current = model.current(support)
    week = mixed_template(model.history, support.index, model.metadata["contract"], recipe["day_weight"])
    delta = current - week[:, 0, :]
    level = smooth_delta(delta, support.index, window)
    positions = support.index.get_indexer(index)
    if (positions < 0).any():
        raise ValueError(message("origin.missing"))
    predicted = corrected_curve(week[positions], delta[positions], level[positions], recipe,
                                model.metadata["contract"]["protocol"]["block_start_offset_minutes"])
    if not return_context:
        return predicted
    context = np.column_stack([current[positions], week[positions, 0], delta[positions], level[positions]])
    columns = [f"{part}_{target}" for part in ("proxy", "template_now", "delta", "slow_level")
               for target in ("g1", "gall")]
    return predicted, pd.DataFrame(context, index=index, columns=columns)
