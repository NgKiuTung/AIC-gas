"""私有赛事研究：固定正则的线性基线，仅验证信息增益，不进行超参数搜索。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from gasstage.common import message, save_npz


@dataclass
class RidgeBaseline:
    """列变换只在训练子集拟合；缺失指示并列保存，填补不回写原始数据。"""
    active: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeBaseline:
        """标准化后求解L2正则最小二乘，不是MAPE优化器；多输出共享设计矩阵。"""
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 3 or not np.isfinite(y).all():
            raise ValueError(message('ridge.error.01'))
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(message('ridge.error.02'))
        active = np.isfinite(x).sum(axis=0) >= 3
        xx = np.asarray(x[:,active],dtype='float64')
        if not active.any():
            raise ValueError(message('ridge.error.03'))
        center = np.nanmedian(xx,axis=0)
        q = np.nanpercentile(xx,[25,75],axis=0)
        scale = q[1]-q[0]
        scale[scale < 1e-9] = 1.0
        z = cls._design(xx,center,scale)
        y0 = y.mean(axis=0)
        zm = z.mean(axis=0)
        z = z-zm
        # NOTE: 限制病态值影响只发生于基线内部；截幅常数固定，不从验证中调节。
        gram = z.T@z
        gram.flat[::len(gram)+1] += alpha
        coef = solve(gram,z.T@(y-y0),assume_a='pos')
        return cls(active,center,scale,coef,y0-zm@coef)

    @staticmethod
    def _design(x, center, scale):
        finite = np.isfinite(x)
        z = np.where(finite,np.clip((x-center)/scale,-20,20),0.)
        return np.concatenate([z,(~finite).astype(float)],axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """只使用冻结的列选择和尺度，输出不依赖待预测批次的均值或分位数。"""
        z = self._design(np.asarray(x[:,self.active],dtype='float64'),self.center,self.scale)
        return z@self.coef+self.intercept

    def save(self, path: Path) -> None:
        """以非pickle数值文件保存模型，支持离线核验重放。"""
        save_npz(path,active=self.active,center=self.center,scale=self.scale,coef=self.coef,intercept=self.intercept)

    @classmethod
    def load(cls, path: Path) -> RidgeBaseline:
        """载入本模型的固定参数，不运行归档内代码。"""
        with np.load(path,allow_pickle=False) as x:
            return cls(**{name:x[name].copy() for name in ['active','center','scale','coef','intercept']})
