"""私有AIC赛事研究：独立指标公式、原始过程前缀与隐藏目标污染检查。"""
from __future__ import annotations
import copy
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import pandas as pd
from gasbench.common import msg
from gasbench.factory import fit_model, predict_model
from gasbench.samples import proxy_samples, direct_samples, sequence_origins, training_template
from gasstage.ingest import NativeDataset
from gasstage.features import ProcessView, build_features
from gasstage.targets import block_truth

def independent_score(y: np.ndarray, p: np.ndarray) -> dict:
    """不调用原评分器，重新逐目标/短长累计有效样本的相对误差。"""
    values = {}
    for end, band in ((8, 'short'), (96, 'long')):
        for g, name in enumerate(('g1', 'gall')):
            real = y[:, :end, g].reshape(-1)
            pred = p[:, :end, g].reshape(-1)
            mask = np.isfinite(real)
            if not mask.any() or (real[mask] <= 0).any() or (not np.isfinite(pred).all()):
                raise ValueError(msg('data.labels'))
            values[f'{band}_{name}'] = float(np.sum(np.abs(pred[mask] - real[mask]) / real[mask], dtype='float64') / mask.sum())
        values[f'{band}_mape'] = (values[f'{band}_g1'] + values[f'{band}_gall']) / 2
    values['combined_mape'] = (values['short_mape'] + values['long_mape']) / 2
    from gasbench.competition import add_platform_proxy
    return add_platform_proxy(values)

def poisoned_bundle(bundle, cutoff: pd.Timestamp):
    """复制目标视图，仅在内存替换cutoff之后标签；原文件和过程值不写入。"""
    values = bundle.native.values.copy()
    values.loc[values.index >= cutoff, list(bundle.native.target_names)] = 777777.0
    native = NativeDataset(values, bundle.native.flags, bundle.native.sources, bundle.native.target_names)
    truth, _ = block_truth(native)
    return replace(bundle, native=native, truth=truth)

def prefix_features(bundle, stamp: pd.Timestamp) -> pd.DataFrame:
    """从原始NativeDataset的ProcessView截断后重建，不只切全量特征数组。"""
    view = ProcessView.from_native(bundle.native, bundle.contract).prefix(stamp)
    idx = bundle.features.index[bundle.features.index <= stamp]
    features, _ = build_features(view, idx, bundle.contract)
    np.testing.assert_allclose(features.to_numpy(), bundle.features.loc[idx].to_numpy(), rtol=0, atol=0, equal_nan=True)
    return features

def poison_refit(bundle, cutoff: pd.Timestamp, candidate: dict, output: Path) -> dict:
    """用固定小规模配置对同一模型族两次真拟合；CPU复核不伪称GPU重训。"""
    cfg = copy.deepcopy(candidate)
    if cfg['kind'] != 'reference':
        cfg['device'] = 'cpu'
        if cfg['kind'] == 'sequence':
            cfg.update(epochs=1, batch_size=64, predict_batch_size=64, hidden=16, levels=2, projection=4, max_train_origins=320, amp=False)
        else:
            cfg['rounds'] = 8
            if cfg['kind'] == 'direct':
                cfg.update(max_train_origins=320, sample_horizons=8)
    poison = poisoned_bundle(bundle, cutoff)
    if cfg['kind'] == 'proxy' or cfg['kind'] == 'reference':
        a = proxy_samples(bundle, cutoff)[3]['sample_hashes']
        b = proxy_samples(poison, cutoff)[3]['sample_hashes']
    elif cfg['kind'] == 'direct':
        a = direct_samples(bundle, cutoff, cfg)[-1]['sample_hashes']
        b = direct_samples(poison, cutoff, cfg)[-1]['sample_hashes']
    else:
        rr = sequence_origins(bundle, cutoff, cfg)
        np.testing.assert_array_equal(training_template(bundle, rr), training_template(poison, rr))
        a = b = {'training_template_equal': True}
    if a != b:
        raise ValueError(msg('data.future'))
    idx = pd.date_range(cutoff, periods=3, freq='15min')
    first = fit_model(bundle, cutoff, cfg, 2, output / 'original', time.time() + 600)
    p = predict_model(first, cfg['kind'], bundle.features, idx)
    del first
    second = fit_model(poison, cutoff, cfg, 2, output / 'poisoned', time.time() + 600)
    q = predict_model(second, cfg['kind'], poison.features, idx)
    np.testing.assert_allclose(p, q, rtol=0, atol=0)
    return {'candidate': candidate['id'], 'max_error': float(np.max(np.abs(p - q))), 'scope': 'reduced-round CPU actual refits', 'sample_hashes_equal': True}
