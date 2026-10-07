"""私有赛事研究：本地短长周期等权MAPE，与未公开官方acc_score分开。"""
from __future__ import annotations
from gasstage.common import message
import numpy as np


def score(truth: np.ndarray, pred: np.ndarray) -> dict:
    """先按目标计算均值再等权；真实缺失单独计覆盖率，预测不可缺失。"""
    if truth.shape != pred.shape or truth.ndim != 3 or truth.shape[1:] != (96,2):
        raise ValueError(message('metrics.error.01'))
    if not np.isfinite(pred).all():
        raise ValueError(message('metrics.error.02'))
    mask = np.isfinite(truth)
    if ((truth <= 0) & mask).any():
        raise ValueError(message('metrics.error.03'))
    error = np.abs(pred-truth)/truth
    short, long = [], []
    for target in range(2):
        for end, dest in [(8,short),(96,long)]:
            m = mask[:,:end,target]
            dest.append(float(error[:,:end,target][m].mean()) if m.any() else None)
    if any(v is None for v in short+long):
        return {'short_mape':None,'long_mape':None,'combined_mape':None,'coverage':float(mask.mean())}
    s, l = float(np.mean(short)), float(np.mean(long))
    return {'short_mape':s, 'long_mape':l, 'combined_mape':0.5*(s+l),
            'short_g1':short[0], 'short_gall':short[1], 'long_g1':long[0], 'long_gall':long[1],
            'valid_pairs':int(mask.sum()), 'expected_pairs':int(mask.size), 'coverage':float(mask.mean())}


def per_horizon(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """按偏移和目标记录MAPE，用于定位短期与远期差异。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        errors = np.abs(pred-truth)/truth
    count = np.isfinite(errors).sum(axis=0)
    return np.nansum(errors,axis=0)/np.maximum(count,1)
