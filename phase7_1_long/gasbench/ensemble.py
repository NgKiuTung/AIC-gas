"""私有AIC赛事研究：只用更早且标签已兑现的折学习非负融合权重。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from gasstage.baselines import reconcile
BANDS = ((0, 8), (8, 24), (24, 48), (48, 96))

def matured_folds(folds: list[dict], before: pd.Timestamp) -> list[dict]:
    """末个起点的24小时标签必须已经完整结束，避免融合层跨界。"""
    return [f for f in folds if pd.Timestamp(f['end']) + pd.Timedelta(days=1) <= before]

def weights_for(predictions: np.ndarray, truth: np.ndarray, reference_index: int, ridge: float) -> tuple:
    """在校准数据求simplex权重；失败时显式回退参照并保存求解状态。"""
    experts = predictions.shape[0]
    weights = np.zeros((len(BANDS), 2, experts))
    diagnostics = []
    initial = np.eye(experts)[reference_index]
    for b, (lo, hi) in enumerate(BANDS):
        for g in range(2):
            y = truth[:, lo:hi, g].ravel()
            p = predictions[:, :, lo:hi, g].reshape(experts, -1).T
            keep = np.isfinite(y) & (y > 0) & np.isfinite(p).all(axis=1)
            a = p[keep] / y[keep, None]
            if len(a) == 0:
                weights[b, g] = initial
                diagnostics.append({'band': b, 'target': g, 'success': False, 'reason': 'no labels'})
                continue

            def objective(w):
                """相对绝对误差与向固定参照的正则，提供解析次梯度。"""
                residual = np.einsum('ij,j->i', a, w) - 1
                value = np.abs(residual).mean() + ridge * ((w - initial) ** 2).sum()
                grad = np.einsum('ij,i->j', a, np.sign(residual)) / len(a) + 2 * ridge * (w - initial)
                return (value, grad)
            result = minimize(objective, initial, jac=True, method='SLSQP', bounds=[(0, 1)] * experts, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones_like(w)}], options={'maxiter': 100, 'ftol': 1e-08})
            valid = result.success and np.isfinite(result.x).all() and (abs(result.x.sum() - 1) < 1e-05)
            w = np.maximum(result.x, 0) if valid else initial.copy()
            weights[b, g] = w / w.sum()
            diagnostics.append({'band': b, 'target': g, 'success': bool(valid), 'message': str(result.message), 'calibration_rows': int(keep.sum()), 'calibration_objective': float(objective(weights[b, g])[0])})
    return (weights, diagnostics)

def apply_weights(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """短长表共用同一96步曲线；两个目标分别融合后做非负包含关系投影。"""
    result = np.empty(predictions.shape[1:], dtype='float64')
    for b, (lo, hi) in enumerate(BANDS):
        for g in range(2):
            result[:, lo:hi, g] = np.einsum('enb,e->nb', predictions[:, :, lo:hi, g], weights[b, g])
    return reconcile(result)
