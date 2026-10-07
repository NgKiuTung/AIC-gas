"""私有AIC赛事研究：仅历史过程量的81通道序列，按需装配不复制大滑窗。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from gasbench.common import msg, write_json
OPS = ('mean_15m', 'coverage_60m', 'age_minutes')

def sequence_values(features: pd.DataFrame, process_names: list[str]) -> tuple[np.ndarray, list]:
    """15分钟网格的历史摘要；一分钟原始观测已进入均值，不读取目标。"""
    arrays, names = ([], [])
    for operation in OPS:
        columns = [f'feat_{name}__{operation}' for name in process_names]
        a = features[columns].to_numpy(dtype='float32')
        if operation == 'age_minutes':
            a = np.log1p(np.clip(a, 0, 7 * 1440))
        arrays.append(a)
        names.extend(columns)
    return (np.column_stack(arrays), names)

def save_sequence_grid(features: pd.DataFrame, process_names: list[str], path: Path) -> None:
    """单一连续矩阵落盘；长窗口在batch内索引，不展开为全量三维副本。"""
    value, names = sequence_values(features, process_names)
    np.save(path / 'sequence_grid.npy', value, allow_pickle=False)
    write_json(path / 'sequence_dictionary.json', {'names': names, 'shape': list(value.shape), 'frequency_minutes': 15, 'available': '<=origin', 'age_transform': 'log1p(min(age_minutes,10080))', 'sequence_contains_target': False, 'operations': list(OPS)})

@dataclass
class Scaler:
    """用训练段有效值拟合均值/标准差；全空列置零，缺失掩码仍在输入中。"""
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> 'Scaler':
        """不接受无样本矩阵；不以验证数据补训练全空字段。"""
        if len(x) == 0 or x.ndim != 2:
            raise ValueError(msg('data.empty'))
        finite = np.isfinite(x)
        count = finite.sum(axis=0)
        mean = np.where(count, np.where(finite, x, 0).sum(axis=0) / np.maximum(count, 1), 0)
        delta = np.where(finite, x - mean, 0)
        scale = np.sqrt((delta * delta).sum(axis=0) / np.maximum(count, 1))
        return cls(mean.astype('float32'), np.where(scale > 1e-06, scale, 1).astype('float32'))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """训练统计变换；极端值截断范围是固定的数值保护，不拟合测试分布。"""
        a = (x - self.mean) / self.scale
        return np.clip(np.where(np.isfinite(a), a, 0), -12, 12).astype('float32')

    def as_dict(self) -> dict:
        """无pickle序列化。"""
        return {'mean': self.mean.tolist(), 'scale': self.scale.tolist()}

    @classmethod
    def from_dict(cls, value: dict) -> 'Scaler':
        """只恢复已保存的训练统计。"""
        return cls(np.asarray(value['mean'], dtype='float32'), np.asarray(value['scale'], dtype='float32'))

def window_rows(rows: np.ndarray, length: int, total: int) -> np.ndarray:
    """拒绝负索引和越界，防止首行用-1悄悄读取未来尾行。"""
    rows = np.asarray(rows, dtype='int64')
    if length < 1 or len(rows) == 0 or rows.min() < length - 1 or (rows.max() >= total):
        raise ValueError(msg('data.shape'))
    return rows[:, None] - np.arange(length - 1, -1, -1)
