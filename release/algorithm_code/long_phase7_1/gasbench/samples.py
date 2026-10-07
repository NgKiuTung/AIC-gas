"""私有AIC赛事研究：共享采样、24小时标签清除、训练内周期黑窗模拟。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from gasbench.common import array_hash, msg
from gas2.samples import proxy_training, future_context
from gasstage.baselines import frozen_week
from gasstage.targets import eligible_training, future_truth, history_only

def proxy_samples(bundle, cutoff: pd.Timestamp, pre_weight: float=0.25) -> tuple:
    """与58分参照逐行相同：120天复赛＋初赛低权重，15分钟采样名义标签。"""
    x, y, weight, info = proxy_training(bundle, cutoff, {'pre_weight': pre_weight}, {'lookback_days': 120})
    info['sample_hashes'] = {'x': array_hash(x), 'y': array_hash(y), 'source_weights': array_hash(weight)}
    return (x, y, weight, info)

def sequence_origins(bundle, cutoff: pd.Timestamp, candidate: dict) -> np.ndarray:
    """拟合区间均值仅用复赛；标签尾端<=cutoff，保留训练窗口和稀疏预算。"""
    index = bundle.features.index
    first = pd.Timestamp(bundle.contract['sources'][2]['start']) + pd.Timedelta(days=7)
    first = max(first, cutoff - pd.Timedelta(days=candidate.get('lookback_days', 120)))
    keep = (index >= first) & eligible_training(index, cutoff, bundle.contract)
    rows = np.flatnonzero(keep)[::candidate.get('origin_stride', 2)]
    limit = candidate.get('max_train_origins', 0)
    if limit and len(rows) > limit:
        rows = rows[np.linspace(0, len(rows) - 1, limit, dtype=int)]
    if len(rows) < 32:
        raise ValueError(msg('data.empty'))
    return rows

def training_template(bundle, rows: np.ndarray) -> np.ndarray:
    """每10天模拟一次整段负荷隐藏；不以外层最终历史反向填早期输入。"""
    idx = bundle.features.index.take(rows)
    start = pd.Timestamp(bundle.contract['sources'][2]['start']) + pd.Timedelta(days=7)
    days = ((idx - start).total_seconds() // (86400 * 10)).astype(int) * 10
    cutoffs = pd.DatetimeIndex(start + pd.to_timedelta(days, unit='D'))
    result = np.empty((len(rows), 96, 2), dtype='float64')
    for cutoff in cutoffs.unique():
        take = cutoffs == cutoff
        if idx[take].min() < cutoff:
            raise ValueError(msg('data.future'))
        hist = history_only(bundle.truth, cutoff)
        result[take] = frozen_week(hist, idx[take], bundle.contract)
    return result

def known_future(index: pd.DatetimeIndex, contract: dict) -> np.ndarray:
    """返回N×96×6，只包含预测距离、日历和公开参考电价。"""
    repeated = pd.DatetimeIndex(np.repeat(index.to_numpy(dtype='datetime64[ns]'), 96))
    return future_context(repeated, np.tile(np.arange(96), len(index)), contract).reshape(-1, 96, 6)

def direct_samples(bundle, cutoff: pd.Timestamp, candidate: dict) -> tuple:
    """所有树家族共用起点与分层跨度，样本权重修正抽样而非偷换评分口径。

NOTE: 家族随机种子不改变共同样本。"""
    rows = sequence_origins(bundle, cutoff, candidate)
    index = bundle.features.index.take(rows)
    y = future_truth(bundle.truth, index, bundle.contract)
    template = training_template(bundle, rows)
    future = known_future(index, bundle.contract)
    count = int(candidate.get('sample_horizons', 32))
    ns = min(8, count // 2)
    nl = count - ns
    rng = np.random.default_rng(20261003)
    blocks = np.stack([np.r_[rng.choice(8, ns, replace=False), rng.choice(np.arange(8, 96), nl, replace=False)] for _ in rows])
    rr = np.repeat(np.arange(len(rows)), count)
    hh = blocks.ravel()
    # NOTE: 平台长周期acc_score斜率约为短周期2倍。首8块同时进入短/长评分，
    # 因此组权重为7/18；其余88块只进入长评分，组权重为11/18。
    source = np.tile(np.r_[np.full(ns, 7 / 18 / ns), np.full(nl, 11 / 18 / nl)], len(rows))
    base = bundle.features.to_numpy(dtype='float32')[rows[rr]]
    x = np.column_stack([base, future[rr, hh], template[rr, hh]]).astype('float32')
    labels, anchors = (y[rr, hh], template[rr, hh])
    ok = np.isfinite(labels).all(axis=1) & (labels > 0).all(axis=1)
    info = {'origins': len(rows), 'pairs': int(ok.sum()), 'sample_horizons': count, 'last_origin': str(index[-1]), 'label_end_exclusive': str(index[-1] + pd.Timedelta(days=1)), 'coarse_mean_labels_used': False, 'training_template': '10-day historical target blackout', 'sample_hashes': {'x': array_hash(x[ok]), 'labels': array_hash(labels[ok]), 'blocks': array_hash(hh[ok]), 'weights': array_hash(source[ok])}}
    return (x[ok], labels[ok], anchors[ok], hh[ok], source[ok], info)

def direct_matrix(features: pd.DataFrame, template: np.ndarray, contract: dict) -> np.ndarray:
    """小批次全部96区间，禁止插值少量节点冒充完整多步回归。"""
    base = np.repeat(features.to_numpy(dtype='float32'), 96, axis=0)
    return np.column_stack([base, known_future(features.index, contract).reshape(-1, 6), template.reshape(-1, 2)]).astype('float32')
