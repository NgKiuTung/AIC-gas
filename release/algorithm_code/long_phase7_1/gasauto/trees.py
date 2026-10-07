"""私有AIC赛事研究：复用原生树后端，短长任务独立预测与保存。"""
import shutil
import numpy as np
import pandas as pd
from gasauto.protocol import BLOCKS
from gasauto.samples import direct_samples, direct_matrix, wide_proxy_samples
from gasbench.samples import proxy_samples
from gasbench.backends import active_columns, fit_backend, predict_backend, save_backend, load_backend, EXTENSIONS
from gasbench.common import load_json, write_json, seal, check_seal
from gasbench.reference import load_reference
from gasstage.baselines import frozen_week, reconcile
from gasstage.targets import history_only, future_index


def frozen_day(history: pd.DataFrame, index: pd.DatetimeIndex, contract: dict, blocks: int) -> np.ndarray:
    """Frozen seven-day same-clock template; hidden validation/test truth is never reused."""
    stamps = future_index(index, contract)[:, :blocks]
    last = history.index[-1].to_datetime64()
    day = np.timedelta64(24 * 60, "m")
    jump = np.maximum(1, np.ceil((stamps - last) / day).astype("int64"))
    offsets = jump[..., None] + np.arange(7)
    lookup = stamps[..., None] - offsets * day
    vals = history.reindex(pd.DatetimeIndex(lookup.ravel())).to_numpy().reshape(len(index), blocks, 7, 2)
    count = np.isfinite(vals).sum(axis=2)
    totals = np.nansum(vals, axis=2)
    fallback = history.median().to_numpy()
    return np.where(count > 0, totals / np.maximum(count, 1), fallback)

class TaskTree:
    """模型推理只接过程特征；直接残差的基线模型随模型目录保存。"""

    def __init__(self, meta, models, history, reference=None):
        self.metadata, self.models, self.history, self.reference = (meta, models, history, reference)

    @classmethod
    def fit(cls, bundle, cutoff, cfg, task, threads, path, bank, clock):
        """当前代理维持同记录训练；直接模型按task跨度独立采样。"""
        path.mkdir(parents=True, exist_ok=True)
        if cfg['kind'] == 'proxy':
            x, y, w, info = proxy_samples(bundle, cutoff, cfg.get('pre_weight', 0.25))
            base = np.zeros_like(y)
        elif cfg['kind'] == 'proxy_wide':
            x, y, w, info = wide_proxy_samples(bundle, cutoff, cfg)
            base = np.zeros_like(y)
        else:
            x, y, base, info = direct_samples(bundle, cutoff, cfg, task, bank)
            w = np.ones(len(y))
        meta = {'candidate': cfg, 'task': task, 'blocks': BLOCKS[task], 'cutoff': str(cutoff), 'threads': threads, 'features': list(bundle.features.columns), 'contract': bundle.contract, 'process_names': list(bundle.native.process.columns), 'training': info, 'models': {}}
        active = active_columns(x)
        models = {}
        for g in range(2):
            center = float(np.median(y[:, g] - base[:, g]))
            weight = w / y[:, g]
            weight /= weight.mean()
            with clock.compute():
                model = fit_backend(cfg['family'], np.ascontiguousarray(x[:, active]), y[:, g] - base[:, g] - center, weight, cfg, threads)
            key = f't{g}'
            save_backend(cfg['family'], model, path / (key + EXTENSIONS[cfg['family']]))
            models[key] = model
            meta['models'][key] = {'target': g, 'active': active.tolist(), 'center': center}
        history = history_only(bundle.truth, cutoff)
        history.to_csv(path / 'history.csv', float_format='%.17g')
        reference = None
        if cfg['kind'] == 'direct':
            reference, source = bank.get(cutoff)
            shutil.copytree(source, path / 'reference', dirs_exist_ok=True)
        write_json(path / 'model.json', meta)
        seal(path)
        return cls(meta, models, history, reference)

    def predict(self, features, index):
        """按任务返回N×8×2或N×96×2，绝不以填充伪装成另一任务结果。"""
        m = self.metadata
        cfg = m['candidate']
        blocks = m['blocks']
        if list(features.columns) != m['features'] or index.min() < pd.Timestamp(m['cutoff']):
            raise ValueError('prediction availability mismatch')
        chosen = features.loc[index]
        if cfg['kind'] in ('proxy', 'proxy_wide'):
            if cfg['kind'] == 'proxy':
                x = chosen[[f'feat_{n}__last' for n in m['process_names']]].to_numpy(dtype='float32')
            else:
                x = chosen.to_numpy(dtype='float32')
            current = self._predict(x)
            week = frozen_week(self.history, index, m['contract'])[:, :blocks]
            day_weight = float(cfg.get('day_weight', 0.0))
            if not 0.0 <= day_weight <= 1.0:
                raise ValueError('invalid day_weight')
            base = week
            if day_weight:
                day = frozen_day(self.history, index, m['contract'], blocks)
                base = (1.0 - day_weight) * week + day_weight * day
            decay_hours = float(cfg.get('decay_hours', 6.0))
            if decay_hours <= 0:
                raise ValueError('invalid decay_hours')
            decay = np.exp(-(np.arange(blocks) * 15 + 7.5) / (decay_hours * 60.0))
            return reconcile(base + (current - base[:, 0])[:, None] * decay[None, :, None])
        pred = []
        for start in range(0, len(index), 64):
            batch = chosen.iloc[start:start + 64]
            base = self.reference.predict(batch)[:, :blocks]
            x = direct_matrix(batch, base, m['contract'])
            pred.append(reconcile(base + self._predict(x).reshape(len(batch), blocks, 2)))
        return np.concatenate(pred)

    def _predict(self, x):
        output = np.empty((len(x), 2))
        for key, info in self.metadata['models'].items():
            output[:, info['target']] = predict_backend(self.metadata['candidate']['family'], self.models[key], x[:, info['active']], self.metadata['threads']) + info['center']
        return output

    @classmethod
    def load(cls, path, device='cpu'):
        """只载入自身原生权重，不反序列化未知对象。"""
        check_seal(path)
        meta = load_json(path / 'model.json')
        family = meta['candidate']['family']
        models = {k: load_backend(family, path / (k + EXTENSIONS[family])) for k in meta['models']}
        history = pd.read_csv(path / 'history.csv', index_col=0, parse_dates=True, float_precision='round_trip')
        ref = load_reference(path / 'reference') if meta['candidate']['kind'] == 'direct' else None
        return cls(meta, models, history, ref)
