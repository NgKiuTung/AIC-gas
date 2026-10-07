"""私有赛事研究：仅用更早且标签完整兑现的折学习融合，禁止同折拟合后报泛化。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gas2.common import text
from gasstage.baselines import reconcile

FALLBACK = "stage1_process_week_joint"

def default_weights(names: list[str], bands: list) -> np.ndarray:
    """没有更早验证证据时固定沿用Stage1，不用当前折选择权重。"""
    w = np.zeros((len(bands),2,len(names)),dtype="float64")
    w[:,:,names.index(FALLBACK)] = 1.0
    return w

def eligible_past(records: list, cutoff: pd.Timestamp, offset: int = 0) -> list:
    """只有整折最长标签已兑现才能用于权重学习。"""
    return [r for r in records if pd.Timestamp(r["end"]) + pd.Timedelta(minutes=1440+offset) <= cutoff]

def fit_weights(records: list, names: list[str], settings: dict, cutoff: pd.Timestamp,
                offset: int = 0) -> tuple[np.ndarray, dict]:
    """按目标/跨度拟合simplex权重；已用日期和最晚标签时间写入证据。"""
    bands = settings["bands"]
    prior = default_weights(names,bands)
    for record in records:
        if pd.Timestamp(record["end"]) + pd.Timedelta(minutes=1440+offset) > cutoff:
            raise ValueError(text("fold.overlap"))
    if len(records) < settings["minimum_prior_folds"]:
        return prior, {"mode":"fixed_stage1", "source_folds":[], "knowledge_cutoff":None}
    weights = prior.copy()
    solutions = []
    stride = settings["sample_origin_stride"]
    for b,(lo,hi) in enumerate(bands):
        for g in range(2):
            truth = np.concatenate([r["truth"][::stride,lo:hi,g].ravel() for r in records])
            pred = np.stack([np.concatenate([r["pred"][n][::stride,lo:hi,g].ravel() for r in records])
                             for n in names], axis=1)
            mask = np.isfinite(truth) & (truth > 0)
            target = truth[mask]
            relative = pred[mask] / target[:,None]
            init = prior[b,g]
            def fun(w):
                residual = relative @ w - 1
                return float(np.mean(np.abs(residual)) + settings["ridge"]*np.sum((w-init)**2))
            def jac(w):
                return relative.T @ np.sign(relative@w-1)/len(target) + 2*settings["ridge"]*(w-init)
            opt = minimize(fun, init, jac=jac, method="SLSQP", bounds=[(0,1)]*len(names),
                           constraints=[{"type":"eq","fun":lambda w: w.sum()-1,
                                         "jac":lambda w: np.ones_like(w)}],
                           options={"maxiter":120, "ftol":1e-9})
            if opt.success and np.isfinite(opt.x).all() and fun(opt.x) <= fun(init):
                current = np.maximum(opt.x,0)
                weights[b,g] = current/current.sum()
            solutions.append({"band":b,"target":g,"optimizer_success":bool(opt.success),
                              "optimizer_message":str(opt.message), "calibration_rows":int(mask.sum())})
    latest = max(pd.Timestamp(r["end"])+pd.Timedelta(minutes=1440+offset) for r in records)
    return weights, {"mode":"past_folds_only","source_folds":[r["name"] for r in records],
                     "knowledge_cutoff":str(latest), "solutions":solutions}

def combine(predictions: dict[str,np.ndarray], names: list, weights: np.ndarray, bands: list) -> np.ndarray:
    """仅做已冻结线性融合，最后沿用Stage1非负/包含校验，不修改任何真值。"""
    result = np.empty_like(predictions[names[0]])
    for b,(lo,hi) in enumerate(bands):
        for g in range(2):
            stacked = np.stack([predictions[n][:,lo:hi,g] for n in names],axis=-1)
            result[:,lo:hi,g] = stacked @ weights[b,g]
    return reconcile(result)
