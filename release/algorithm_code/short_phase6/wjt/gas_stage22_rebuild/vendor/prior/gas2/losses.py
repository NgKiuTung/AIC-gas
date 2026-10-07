"""私有赛事研究：训练损失对照；奖励为负误差，不建立强化学习环境。"""
from __future__ import annotations

import numpy as np

def horizon_weights(blocks: int = 96, short: int = 8, short_share: float = 0.5) -> np.ndarray:
    """返回sum=1的本地短长等权损失；前8块同时进入短表与长表。"""
    if not 0 < short <= blocks or not 0 <= short_share <= 1:
        raise ValueError("invalid horizon geometry")
    weights = np.full(blocks, (1.0 - short_share) / blocks, dtype="float64")
    weights[:short] += short_share / short
    return weights

def relative_tail_loss(residual: np.ndarray, epsilon: float, strength: float, threshold: float) -> np.ndarray:
    """平滑绝对相对误差＋超阈值的平方惩罚；正负误差完全对称。"""
    if epsilon <= 0 or strength < 0 or threshold < 0:
        raise ValueError("invalid penalty parameters")
    return np.sqrt(residual**2 + epsilon**2) - epsilon + strength * np.maximum(np.abs(residual) - threshold, 0)**2

def relative_derivatives(pred: np.ndarray, truth: np.ndarray, center: float,
                         epsilon: float, strength: float, threshold: float) -> tuple:
    """返回对残差模型输出的解析梯度/Hessian，包含真实标签分母。"""
    if np.any(~np.isfinite(truth)) or np.any(truth <= 0):
        raise ValueError("training truth must be finite and positive")
    r = (pred + center - truth) / truth
    norm = np.sqrt(r*r + epsilon*epsilon)
    excess = np.maximum(np.abs(r) - threshold, 0.0)
    grad = (r / norm + 2.0 * strength * excess * np.sign(r)) / truth
    hess = (epsilon*epsilon / norm**3 + 2.0 * strength * (np.abs(r) > threshold)) / truth**2
    return grad, np.maximum(hess, 1e-12)

def make_objective(center: float, settings: dict):
    """创建LightGBM目标；权重仅取训练样本，缩放不改变最优解。"""
    def objective(pred, train):
        truth = train.get_label().astype("float64") + center
        grad, hess = relative_derivatives(pred, truth, center, settings["epsilon"],
                                         settings["tail_lambda"], settings["tail_threshold"])
        weight = train.get_weight()
        weight = np.ones(len(truth)) if weight is None else weight
        # NOTE: L1样本权重包含1/y；自定义目标内已除y，需抵消避免双重MAPE加权。
        scale = weight * truth
        return grad * scale, hess * scale
    return objective
