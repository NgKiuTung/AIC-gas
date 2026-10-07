"""私有AIC赛事研究：共享代理/直接多步接口，隔离验证目标并保存全部来源。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from gasbench.common import load_json, write_json, msg, seal, check_seal
from gasbench.backends import EXTENSIONS, active_columns, fit_backend, predict_backend, save_backend, load_backend
from gasbench.samples import proxy_samples, direct_samples, direct_matrix
from gasstage.baselines import frozen_week, reconcile
from gasstage.targets import history_only

@dataclass
class TreePredictor:
    """模型持有冻结历史而非NativeDataset，推理只接只含过程的特征表。"""
    metadata: dict
    models: dict
    history: pd.DataFrame

    @classmethod
    def fit(cls, bundle, cutoff: pd.Timestamp, candidate: dict, threads: int, path: Path):
        """共享训练样本，不在eval_set中放外层测试目标。"""
        path.mkdir(parents=True, exist_ok=True)
        scope, family = (candidate['kind'], candidate['family'])
        if scope == 'proxy':
            x, y, weight, provenance = proxy_samples(bundle, cutoff, candidate.get('pre_weight', 0.25))
            anchors, blocks = (np.zeros_like(y), np.zeros(len(y), dtype=int))
            bands = [(0, 1)]
        else:
            x, y, anchors, blocks, weight, provenance = direct_samples(bundle, cutoff, candidate)
            bands = [(0, 8), (8, 96)]
        metadata = {'candidate': candidate, 'cutoff': str(cutoff), 'threads': threads, 'features': list(bundle.features.columns), 'contract': bundle.contract, 'process_names': list(bundle.native.process.columns), 'training': provenance, 'models': {}, 'interval_means': scope != 'proxy'}
        models = {}
        for target in range(2):
            for lo, hi in bands:
                take = (blocks >= lo) & (blocks < hi)
                key = f't{target}_b{lo}'
                model, info = cls._fit_part(x[take], y[take, target], anchors[take, target], weight[take], candidate, threads)
                save_backend(family, model, path / (key + EXTENSIONS[family]))
                info.update(target=target, lo=lo, hi=hi)
                metadata['models'][key], models[key] = (info, model)
        history = history_only(bundle.truth, cutoff)
        history.to_csv(path / 'history.csv', float_format='%.17g')
        write_json(path / 'model.json', metadata)
        seal(path)
        return cls(metadata, models, history)

    @staticmethod
    def _fit_part(x, truth, anchor, source_weight, candidate, threads):
        active = active_columns(x)
        center = float(np.median(truth - anchor))
        weight = source_weight / truth
        weight /= weight.mean()
        model = fit_backend(candidate['family'], np.ascontiguousarray(x[:, active]), truth - anchor - center, weight, candidate, threads)
        return (model, {'active': active.tolist(), 'center': center})

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """每起点输出96个区间；同一投影用于短/长结果，防止重叠矛盾。"""
        if list(features.columns) != self.metadata['features'] or len(features) == 0:
            raise ValueError(msg('data.shape'))
        if features.index.min() < pd.Timestamp(self.metadata['cutoff']):
            raise ValueError(msg('data.future'))
        template = frozen_week(self.history, features.index, self.metadata['contract'])
        if self.metadata['candidate']['kind'] == 'proxy':
            columns = [f'feat_{n}__last' for n in self.metadata['process_names']]
            x = features[columns].to_numpy(dtype='float32')
            current = np.zeros((len(features), 2))
            for key, info in self.metadata['models'].items():
                current[:, info['target']] = self._predict_part(key, x)
            decay = np.exp(-(np.arange(96) * 15 + 7.5) / 360)
            return reconcile(template + (current - template[:, 0])[:, None] * decay[None, :, None])
        value = np.empty_like(template)
        for start in range(0, len(features), 64):
            batch = features.iloc[start:start + 64]
            x = direct_matrix(batch, template[start:start + len(batch)], self.metadata['contract'])
            blocks = np.tile(np.arange(96), len(batch))
            for key, info in self.metadata['models'].items():
                take = (blocks >= info['lo']) & (blocks < info['hi'])
                residual = self._predict_part(key, x[take]).reshape(len(batch), -1)
                value[start:start + len(batch), info['lo']:info['hi'], info['target']] = residual
        return reconcile(template + value)

    def _predict_part(self, key, x):
        info = self.metadata['models'][key]
        return predict_backend(self.metadata['candidate']['family'], self.models[key], x[:, info['active']], self.metadata['threads']) + info['center']

    @classmethod
    def load(cls, path: Path):
        """独立从磁盘校验和加载，重放不借用内存中的训练实例。"""
        check_seal(path)
        meta = load_json(path / 'model.json')
        family = meta['candidate']['family']
        models = {key: load_backend(family, path / (key + EXTENSIONS[family])) for key in meta['models']}
        hist = pd.read_csv(path / 'history.csv', index_col=0, parse_dates=True, float_precision='round_trip')
        return cls(meta, models, hist)
