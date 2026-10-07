"""私有AIC赛事研究：参考锚点上的独立Short/Long多步残差。"""
import shutil
import numpy as np
import pandas as pd
import torch
from gasauto.protocol import BLOCKS, shift
from gasauto.samples import rows_for, future_for
from gasauto.neural_data import normalization, make_arrays
from gasauto.trainer import fit_phase
from gasauto.networks import build
from gasbench.common import write_json, load_json, seal, check_seal, array_hash
from gasbench.neural_fit import atomic_torch
from gasbench.neural_data import NeuralArrays
from gasbench.reference import load_reference
from gasbench.sequences import sequence_values, Scaler
from gasstage.baselines import reconcile

class TaskNeural:
    """外层验证从不用于训练轮数选择；预测只接过程视图。"""

    def __init__(self, meta, net, reference):
        self.metadata, self.net, self.reference = (meta, net, reference)

    @classmethod
    def fit(cls, bundle, cutoff, cfg, task, threads, path, bank, clock):
        """内层10天校准，训练标签另行purge；更好epoch0允许原样部署。"""
        cfg = {**cfg, 'blocks': BLOCKS[task]}
        length = cfg['history_hours'] * 4
        path.mkdir(parents=True, exist_ok=True)
        inner = shift(cutoff, -11 * 1440)
        fit = rows_for(bundle, inner, cfg, task)
        allrows = rows_for(bundle, cutoff, cfg, task)
        idx = bundle.features.index
        valid = np.flatnonzero((idx >= inner) & (idx < shift(inner, 10 * 1440)))
        cap = cfg.get('max_calibration_origins', 0)
        if cap and len(valid) > cap:
            valid = valid[np.linspace(0, len(valid) - 1, cap, dtype=int)]
        norm = normalization(bundle, fit, length, task)
        train = make_arrays(bundle, fit, length, norm, task, bank.training(fit, BLOCKS[task]))
        val = make_arrays(bundle, valid, length, norm, task, bank.predict(idx.take(valid), inner, BLOCKS[task]))
        _, tuned = fit_phase(train, val, cfg, threads, path, 'tune', clock)
        del train, val
        norm = normalization(bundle, allrows, length, task)
        full = make_arrays(bundle, allrows, length, norm, task, bank.training(allrows, BLOCKS[task]))
        net, fitted = fit_phase(full, None, cfg, threads, path, 'refit', clock, epochs=tuned['best_epoch'])
        ref, source = bank.get(cutoff)
        shutil.copytree(source, path / 'reference', dirs_exist_ok=True)
        meta = {'candidate': cfg, 'task': task, 'blocks': BLOCKS[task], 'cutoff': str(cutoff), 'threads': threads, 'features': list(bundle.features.columns), 'process_names': list(bundle.native.process.columns), 'contract': bundle.contract, 'normalization': norm, 'epoch0_fallback': tuned['best_epoch'] == 0, 'training': {'inner_cutoff': str(inner), 'inner_rows': len(fit), 'calibration_rows': len(valid), 'final_rows': len(allrows), 'training_rows_hash': array_hash(allrows), 'label_end_exclusive': str(shift(idx[allrows[-1]], BLOCKS[task] * 15)), 'tune': tuned, 'refit': fitted, 'true_target_history_input': False}}
        atomic_torch(path / 'weights.pt', {'state_dict': net.cpu().state_dict()})
        net.to(cfg.get('device', 'cpu')).eval()
        write_json(path / 'model.json', meta)
        seal(path)
        return cls(meta, net, ref)

    def predict(self, features, index):
        """零轮精确返回双精度原参照，避免归一化往返产生伪改动。"""
        m = self.metadata
        cfg = m['candidate']
        h = m['blocks']
        if list(features.columns) != m['features'] or index.min() < pd.Timestamp(m['cutoff']):
            raise ValueError('future model')
        anchor = self.reference.predict(features.loc[index])[:, :h]
        if m['epoch0_fallback']:
            return anchor
        rows = features.index.get_indexer(index)
        if (rows < 0).any():
            raise ValueError('origin missing')
        norm = m['normalization']
        scale = np.asarray(norm['target_scale'], dtype='float32')
        grid, _ = sequence_values(features, m['process_names'])
        arrays = NeuralArrays(Scaler.from_dict(norm['sequence']).transform(grid), Scaler.from_dict(norm['static']).transform(features.to_numpy(dtype='float32')), future_for(index, m['contract'], h), (anchor / scale).astype('float32'), rows, cfg['history_hours'] * 4)
        device = next(self.net.parameters()).device.type
        parts = []
        self.net.eval()
        with torch.no_grad():
            for lo in range(0, len(rows), cfg['batch_size']):
                x, _ = arrays.batch(np.arange(lo, min(lo + cfg['batch_size'], len(rows))), device)
                parts.append(self.net(*x).float().cpu().numpy() * scale)
        return reconcile(np.concatenate(parts).astype('float64'))

    @classmethod
    def load(cls, path, device='cpu'):
        """本机重放与跨设备重放分开记录，不改变容差换通过。"""
        check_seal(path)
        m = load_json(path / 'model.json')
        cfg = m['candidate']
        torch.set_num_threads(m['threads'])
        net = build(cfg, 81, len(m['features']), cfg['history_hours'] * 4)
        net.load_state_dict(torch.load(path / 'weights.pt', map_location='cpu', weights_only=True)['state_dict'])
        return cls(m, net.to(device).eval(), load_reference(path / 'reference'))
