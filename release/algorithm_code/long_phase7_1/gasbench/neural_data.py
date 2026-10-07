"""私有AIC赛事研究：神经网络batch只含历史过程、冻结背景和已知未来时间。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from gasbench.sequences import Scaler, sequence_values, window_rows
from gasbench.samples import known_future, training_template
from gasbench.common import msg
from gasstage.targets import future_truth

@dataclass
class NeuralArrays:
    """训练真值只在损失阶段转入GPU，输入序列与真值物理分离。"""
    sequence: np.ndarray
    static: np.ndarray
    future: np.ndarray
    anchor: np.ndarray
    rows: np.ndarray
    length: int
    truth: np.ndarray | None = None

    def batch(self, selection: np.ndarray, device: str):
        """有限batch窗口装配；不把N×T×C完整张量常驻内存。"""
        window = window_rows(self.rows[selection], self.length, len(self.sequence))
        arrays = (self.sequence[window], self.static[self.rows[selection]], self.future[selection], self.anchor[selection])
        x = tuple((torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays))
        if self.truth is None:
            return (x, None)
        return (x, torch.from_numpy(np.ascontiguousarray(self.truth[selection])).to(device))

def fit_normalization(bundle, rows: np.ndarray, length: int) -> dict:
    """只用训练起点及其历史窗；外层验证和测试分布不参与缩放。"""
    grid, _ = sequence_values(bundle.features, list(bundle.native.process.columns))
    used = np.unique(window_rows(rows, length, len(grid)))
    seq = Scaler.fit(grid[used])
    static = Scaler.fit(bundle.features.iloc[rows].to_numpy(dtype='float32'))
    truth = future_truth(bundle.truth, bundle.features.index.take(rows), bundle.contract)
    scale = np.nanmedian(truth, axis=(0, 1)).astype('float32')
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError(msg('data.labels'))
    return {'sequence': seq.as_dict(), 'static': static.as_dict(), 'target_scale': scale.tolist()}

def arrays_for(bundle, rows: np.ndarray, length: int, normalization: dict, anchor: np.ndarray | None=None, training: bool=True) -> NeuralArrays:
    """训练模板按10日模拟段冻结；预测模板由模型截止历史生成并显式传入。"""
    grid, _ = sequence_values(bundle.features, list(bundle.native.process.columns))
    seq = Scaler.from_dict(normalization['sequence']).transform(grid)
    static = Scaler.from_dict(normalization['static']).transform(bundle.features.to_numpy(dtype='float32'))
    index = bundle.features.index.take(rows)
    if anchor is None:
        if not training:
            raise ValueError(msg('data.future'))
        anchor = training_template(bundle, rows)
    scale = np.asarray(normalization['target_scale'], dtype='float32')
    truth = None
    if training:
        truth = future_truth(bundle.truth, index, bundle.contract)
        finite = np.isfinite(truth)
        if ((truth <= 0) & finite).any():
            raise ValueError(msg('data.labels'))
        truth = (truth / scale).astype('float32')
    return NeuralArrays(seq, static, known_future(index, bundle.contract), (anchor / scale).astype('float32'), rows, length, truth)
