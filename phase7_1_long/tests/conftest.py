"""私有AIC赛事研究：合成原生分钟数据，独立于未分发的官方原始ZIP。"""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pytest
from gasbench.data import Bundle, source_contract
from gasstage.features import ProcessView, build_features
from gasstage.ingest import NativeDataset
from gasstage.targets import block_truth

@pytest.fixture(scope='session')
def bundle():
    """测试数据有完整分钟标签和真实时间轴；不含任何官方测试标签。"""
    cfg = source_contract()
    index = pd.date_range('2025-05-03', '2025-06-10 23:59', freq='min', name='datetime')
    t = np.arange(len(index), dtype=float)
    x = np.column_stack([50 + j * 2 + 4 * np.sin(t / 500 + j * 0.2) + np.cos(t / 57 + j) for j in range(27)])
    y = np.column_stack([100 + 0.3 * x[:, 0] + 2 * np.sin(t / 800), 250 + 0.5 * x[:, 1] + 4 * np.cos(t / 400)])
    names = [f'p{i:02d}' for i in range(27)] + cfg['targets']
    values = pd.DataFrame(np.c_[x, y], index=index, columns=names)
    values.loc[index[100:103], names[2]] = np.nan
    native = NativeDataset(values, pd.DataFrame(0, index=index, columns=names, dtype='uint8'), pd.DataFrame(4, index=index, columns=['gas'], dtype='uint8'), tuple(cfg['targets']))
    idx = pd.date_range(index.min(), index.max().floor('15min'), freq='15min', name='datetime')
    features, _ = build_features(ProcessView.from_native(native, cfg), idx, cfg)
    truth, _ = block_truth(native)
    return Bundle(native, features, truth, cfg, Path('.'))

@pytest.fixture
def tiny_tree():
    return {'id': 'sample', 'family': 'lightgbm', 'kind': 'direct', 'device': 'cpu', 'rounds': 4, 'seed': 7, 'origin_stride': 4, 'sample_horizons': 8, 'max_train_origins': 160, 'params': {}}

@pytest.fixture
def tiny_neural():
    return {'id': 'sample_nn', 'family': 'tcn', 'kind': 'sequence', 'device': 'cpu', 'history_hours': 24, 'hidden': 8, 'projection': 4, 'levels': 2, 'dropout': 0.0, 'epochs': 1, 'batch_size': 64, 'predict_batch_size': 64, 'learning_rate': 0.001, 'patience': 2, 'inner_days': 5, 'seed': 7, 'max_train_origins': 320, 'amp': False}
