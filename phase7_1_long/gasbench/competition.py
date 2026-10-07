"""私有AIC赛事研究：把本地MAPE映射为已观测平台线性评分代理。"""
from __future__ import annotations

import numpy as np
from gasbench.common import msg

# NOTE: 三次官方回执均满足：短acc_score=3*accuracy-210，长=6*accuracy-450，
# 总分=(短acc_score+长acc_score)/2。这里只用于方案级比较，不反推测试标签。
SHORT_SCORE_WEIGHT = 150.0
LONG_SCORE_WEIGHT = 300.0
SCORE_INTERCEPT = 120.0


def add_platform_proxy(metrics: dict) -> dict:
    """复制指标并加入长周期双权重的竞争口径代理。"""
    value = dict(metrics)
    short = value.get("short_mape")
    long = value.get("long_mape")
    if short is None or long is None:
        value.update(competition_mape=None, platform_score_proxy=None)
        return value
    value["competition_mape"] = float((short + 2.0 * long) / 3.0)
    value["platform_score_proxy"] = float(
        SCORE_INTERCEPT - SHORT_SCORE_WEIGHT * short - LONG_SCORE_WEIGHT * long
    )
    return value


def split_score(truth: np.ndarray, short_pred: np.ndarray, long_pred: np.ndarray) -> dict:
    """短表与长表允许不同模型；分别按官方两个任务口径计算。"""
    if truth.shape != short_pred.shape or truth.shape != long_pred.shape:
        raise ValueError(msg('metric.shape'))
    if truth.ndim != 3 or truth.shape[1:] != (96, 2):
        raise ValueError(msg('metric.shape'))
    if not np.isfinite(short_pred).all() or not np.isfinite(long_pred).all():
        raise ValueError(msg('metric.prediction'))
    mask = np.isfinite(truth)
    if ((truth <= 0) & mask).any():
        raise ValueError(msg('metric.truth'))

    values: dict[str, float | int] = {}
    for target, name in enumerate(("g1", "gall")):
        short_mask = mask[:, :8, target]
        long_mask = mask[:, :, target]
        s = np.abs(short_pred[:, :8, target] - truth[:, :8, target]) / truth[:, :8, target]
        l = np.abs(long_pred[:, :, target] - truth[:, :, target]) / truth[:, :, target]
        values[f"short_{name}"] = float(s[short_mask].mean())
        values[f"long_{name}"] = float(l[long_mask].mean())
    values["short_mape"] = float((values["short_g1"] + values["short_gall"]) / 2)
    values["long_mape"] = float((values["long_g1"] + values["long_gall"]) / 2)
    values["combined_mape"] = float((values["short_mape"] + values["long_mape"]) / 2)
    values["valid_pairs"] = int(mask.sum())
    values["expected_pairs"] = int(mask.size)
    values["coverage"] = float(mask.mean())
    return add_platform_proxy(values)
