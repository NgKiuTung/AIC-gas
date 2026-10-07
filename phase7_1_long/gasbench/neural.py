"""私有AIC赛事研究：多步序列模型，外层验证不参与早停或标准化。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from gasbench.common import write_json, load_json, msg, seal, check_seal, array_hash
from gasbench.samples import sequence_origins, known_future
from gasbench.neural_data import fit_normalization, arrays_for, NeuralArrays
from gasbench.neural_fit import fit_epochs, new_network, atomic_torch
from gasbench.networks import build_network
from gasbench.sequences import sequence_values, Scaler
from gasstage.baselines import frozen_week, reconcile
from gasstage.targets import history_only

@dataclass
class NeuralPredictor:
    """仅持有冻结周期和训练统计；输入视图只含过程特征，没有真实目标。"""
    metadata: dict
    net: torch.nn.Module
    history: pd.DataFrame

    @classmethod
    def fit(cls, bundle, cutoff: pd.Timestamp, cfg: dict, threads: int, path: Path, deadline: float):
        """先用截止前的内层留出选择epoch，再从零用全训练段重训。"""
        path.mkdir(parents=True, exist_ok=True)
        inner = cutoff - pd.Timedelta(days=cfg.get('inner_days', 5))
        length = cfg['history_hours'] * 4
        fit_rows = sequence_origins(bundle, inner, cfg)
        all_rows = sequence_origins(bundle, cutoff, cfg)
        valid_rows = all_rows[bundle.features.index.take(all_rows) >= inner]
        if len(valid_rows) < 8:
            raise ValueError(msg('data.empty'))
        norm = fit_normalization(bundle, fit_rows, length)
        train = arrays_for(bundle, fit_rows, length, norm)
        hist = history_only(bundle.truth, inner)
        anchor = frozen_week(hist, bundle.features.index.take(valid_rows), bundle.contract)
        valid = arrays_for(bundle, valid_rows, length, norm, anchor=anchor)
        tuned_net, tuned = fit_epochs(train, valid, cfg, threads, path, 'tune', deadline)
        del train, valid, tuned_net
        if cfg.get('device') == 'cuda':
            torch.cuda.empty_cache()
        norm = fit_normalization(bundle, all_rows, length)
        full = arrays_for(bundle, all_rows, length, norm)
        net, fitted = fit_epochs(full, None, cfg, threads, path, 'refit', deadline, epochs=max(1, tuned['best_epoch']))
        meta = {'candidate': cfg, 'cutoff': str(cutoff), 'threads': threads, 'normalization': norm, 'features': list(bundle.features.columns), 'process_names': list(bundle.native.process.columns), 'contract': bundle.contract, 'training': {'inner_cutoff': str(inner), 'inner_fit_rows': len(fit_rows), 'inner_validation_rows': len(valid_rows), 'final_rows': len(all_rows), 'best_epoch': tuned['best_epoch'], 'final_epoch': fitted['epoch'], 'last_fit_origin': str(bundle.features.index[all_rows[-1]]), 'label_end_exclusive': str(bundle.features.index[all_rows[-1]] + pd.Timedelta(days=1)), 'training_rows_sha256': array_hash(all_rows), 'pre_interval_labels_used': False, 'true_target_history_input': False}}
        history = history_only(bundle.truth, cutoff)
        history.to_csv(path / 'history.csv', float_format='%.17g')
        write_json(path / 'model.json', meta)
        atomic_torch(path / 'weights.pt', {'state_dict': net.cpu().state_dict()})
        net.to(cfg.get('device', 'cpu'))
        seal(path)
        return cls(meta, net, history)

    def predict(self, features: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        """features可含全批未来行，但索引窗口永远<=当前起点；前缀测试覆盖此约束。"""
        if list(features.columns) != self.metadata['features'] or len(origins) == 0:
            raise ValueError(msg('data.shape'))
        if origins.min() < pd.Timestamp(self.metadata['cutoff']):
            raise ValueError(msg('data.future'))
        rows = features.index.get_indexer(origins)
        if (rows < 0).any():
            raise ValueError(msg('data.shape'))
        norm, cfg = (self.metadata['normalization'], self.metadata['candidate'])
        grid, _ = sequence_values(features, self.metadata['process_names'])
        sequence = Scaler.from_dict(norm['sequence']).transform(grid)
        static = Scaler.from_dict(norm['static']).transform(features.to_numpy(dtype='float32'))
        template = frozen_week(self.history, origins, self.metadata['contract'])
        scale = np.asarray(norm['target_scale'], dtype='float32')
        arrays = NeuralArrays(sequence, static, known_future(origins, self.metadata['contract']), (template / scale).astype('float32'), rows, cfg['history_hours'] * 4)
        device = next(self.net.parameters()).device.type
        batch = cfg.get('predict_batch_size', cfg['batch_size'])
        predictions = []
        self.net.eval()
        with torch.no_grad():
            for start in range(0, len(rows), batch):
                x, _ = arrays.batch(np.arange(start, min(start + batch, len(rows))), device)
                predictions.append(self.net(*x).float().cpu().numpy() * scale)
        out = np.concatenate(predictions).astype('float64')
        if not np.isfinite(out).all():
            raise ValueError(msg('model.finite'))
        return reconcile(out)

    @classmethod
    def load(cls, path: Path, device: str='cpu'):
        """只从自有tensor状态恢复；预测设备可以不同，记录误差容差而非改权重。"""
        check_seal(path)
        meta = load_json(path / 'model.json')
        cfg = meta['candidate']
        if device == 'cuda' and (not torch.cuda.is_available()):
            raise RuntimeError(msg('device.unavailable'))
        torch.set_num_threads(meta['threads'])
        net = build_network(cfg['family'], 81, len(meta['features']), cfg['history_hours'] * 4, cfg)
        net.load_state_dict(torch.load(path / 'weights.pt', map_location='cpu', weights_only=True)['state_dict'])
        net.to(device).eval()
        history = pd.read_csv(path / 'history.csv', index_col=0, parse_dates=True, float_precision='round_trip')
        return cls(meta, net, history)
