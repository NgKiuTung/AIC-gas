"""私有AIC赛事研究：任务特定标签/尺度，过程序列按批装配。"""
import numpy as np
from gasauto.samples import future_for
from gasauto.protocol import BLOCKS, truth_for
from gasbench.sequences import Scaler, sequence_values, window_rows
from gasbench.neural_data import NeuralArrays

def normalization(bundle, rows, length, task):
    """缩放只看训练行和所需历史窗口；不看外层验证或测试分布。"""
    grid, _ = sequence_values(bundle.features, list(bundle.native.process.columns))
    used = np.unique(window_rows(rows, length, len(grid)))
    truth = truth_for(bundle, bundle.features.index.take(rows), task)
    scale = np.nanmedian(truth, axis=(0, 1)).astype('float32')
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('invalid target scale')
    return {'sequence': Scaler.fit(grid[used]).as_dict(), 'static': Scaler.fit(bundle.features.iloc[rows].to_numpy(dtype='float32')).as_dict(), 'target_scale': scale.tolist()}

def make_arrays(bundle, rows, length, norm, task, anchor, with_truth=True):
    """anchor必须由调用方按真实截止产生；禁止默认使用未来标签构造。"""
    grid, _ = sequence_values(bundle.features, list(bundle.native.process.columns))
    sequence = Scaler.from_dict(norm['sequence']).transform(grid)
    static = Scaler.from_dict(norm['static']).transform(bundle.features.to_numpy(dtype='float32'))
    index = bundle.features.index.take(rows)
    scale = np.asarray(norm['target_scale'], dtype='float32')
    y = (truth_for(bundle, index, task) / scale).astype('float32') if with_truth else None
    arrays = NeuralArrays(sequence, static, future_for(index, bundle.contract, BLOCKS[task]), (anchor / scale).astype('float32'), rows, length, y)
    arrays.target_scale = scale
    return arrays
