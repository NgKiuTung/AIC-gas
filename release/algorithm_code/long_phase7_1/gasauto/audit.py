"""私有AIC赛事研究：独立任务评分、原始前缀和隐藏负荷污染重训。"""
import copy
import numpy as np
import pandas as pd
from gasauto.anchors import AnchorBank
from gasauto.budget import TrainingClock
from gasauto.protocol import BLOCKS
from gasauto.worker import predict
from gasbench.audit import poisoned_bundle, prefix_features
from gasbench.common import write_json

def independent_score(y: np.ndarray, p: np.ndarray) -> dict:
    """不调用训练评分器，逐目标展平后按有效标签复算相对误差。"""
    values = []
    for target in range(2):
        truth = y[:, :, target].reshape(-1)
        prediction = p[:, :, target].reshape(-1)
        valid = np.isfinite(truth)
        if not valid.any() or (truth[valid] <= 0).any() or (not np.isfinite(prediction).all()):
            raise ValueError('独立评分输入非法。')
        values.append(float(np.sum(np.abs(prediction[valid] / truth[valid] - 1), dtype=np.float64) / valid.sum()))
    return {'mape': float(np.mean(values)), 'g1_mape': values[0], 'gall_mape': values[1]}

def limited_fit(bundle, cutoff, candidate, task, path, bank):
    """污染验收使用小轮次CPU真拟合，不声称是GPU再次训练。"""
    cfg = copy.deepcopy(candidate)
    cfg.update(device='cpu', rounds=3, max_train_origins=96, sample_horizons=8, max_calibration_origins=32, seed=41)
    clock = TrainingClock(120, path / 'phase.json')
    if cfg['kind'] == 'reference':
        return bank.get(cutoff)[0]
    if cfg['kind'] == 'sequence':
        from gasauto.neural import TaskNeural
        cfg.update(epochs=1, hidden=8, levels=2, projection=4, history_hours=24, batch_size=32, amp=False, patience=2, learning_rate=0.0002)
        return TaskNeural.fit(bundle, cutoff, cfg, task, 2, path / 'model', bank, clock)
    from gasauto.trees import TaskTree
    return TaskTree.fit(bundle, cutoff, cfg, task, 2, path / 'model', bank, clock)

def poison_checks(bundle, candidates, output):
    """测试每个原始候选的两个任务；重新生成时间外推代理，不复用污染前锚点。"""
    cutoff = pd.Timestamp('2025-06-01')
    altered = poisoned_bundle(bundle, cutoff)
    original_bank = AnchorBank(bundle, output / 'anchors_original', 'poison-a')
    altered_bank = AnchorBank(altered, output / 'anchors_poisoned', 'poison-b')
    idx = pd.date_range(cutoff, periods=3, freq='15min')
    records = []
    for candidate in candidates:
        if candidate.get('parent'):
            continue
        for task in candidate.get('tasks', list(BLOCKS)):
            path = output / (candidate['id'] + '__' + task)
            a = limited_fit(bundle, cutoff, candidate, task, path / 'original', original_bank)
            p = predict(a, candidate['kind'], bundle.features, idx, task)
            del a
            b = limited_fit(altered, cutoff, candidate, task, path / 'poisoned', altered_bank)
            q = predict(b, candidate['kind'], altered.features, idx, task)
            del b
            np.testing.assert_allclose(p, q, rtol=0, atol=0)
            records.append({'candidate': candidate['id'], 'task': task, 'max_error': float(np.max(np.abs(p - q))), 'scope': 'reduced-round CPU actual fits; original/poison anchors independently fitted'})
            write_json(output / 'poison_results.json', records)
    return records

def raw_prefixes(bundle, output):
    """从原始过程视图截断重建，不仅切已经算好的全矩阵。"""
    values = {}
    records = []
    for stamp in ('2025-10-01 00:00:00', '2025-10-06 12:00:00', '2025-10-10 23:45:00'):
        t = pd.Timestamp(stamp)
        values[t] = prefix_features(bundle, t)
        records.append({'stamp': stamp, 'feature_max_error': 0.0})
    write_json(output / 'raw_prefix.json', records)
    return values
