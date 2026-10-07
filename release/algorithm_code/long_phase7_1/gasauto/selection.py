"""私有AIC赛事研究：只使用登记的筛选/确认结果，拒绝同折拟合权重回报。"""
import numpy as np
from gasauto.protocol import score

def gate(values, reference, rule):
    """单位为MAPE小数；阈值启动前冻结，输出全部拒绝原因。"""
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if values.shape != reference.shape or not np.isfinite(values).all() or (not np.isfinite(reference).all()) or (values.size == 0):
        return {'accepted': False, 'reason': ['missing_or_invalid_metrics']}
    gain = reference - values
    reasons = []
    if gain.mean() < rule['min_mean_gain']:
        reasons.append('mean_gain')
    if gain.min() < -rule['max_fold_regression']:
        reasons.append('worst_fold')
    if gain[-1] < -rule['max_last_regression']:
        reasons.append('latest_fold')
    return {'accepted': not reasons, 'reason': reasons, 'mean_mape': float(values.mean()), 'reference_mape': float(reference.mean()), 'mean_gain': float(gain.mean()), 'worst_gain': float(gain.min()), 'last_gain': float(gain[-1]), 'wins': int((gain > 0).sum()), 'folds': int(gain.size)}

def choose_finalist(screen, candidates):
    """只在已完成且通过筛选的候选中选一个，确认标签不进入此函数。"""
    eligible = [(v['decision']['mean_mape'], k) for k, v in screen.items() if v['decision']['accepted']]
    if not eligible:
        return None
    best = min(eligible)[1]
    return next((c for c in candidates if c['id'] == best))

def confirm_summary(seed_predictions, truths, references, rule):
    """至少3个种子，按种子均值预测评分；不以最优单种子替代平均。"""
    if not seed_predictions or len(seed_predictions) != len(truths) or len(truths) != len(references) or any((len(p) < 3 for p in seed_predictions)):
        return {'accepted': False, 'reason': ['incomplete_seeds']}
    averages = [np.mean(p, axis=0) for p in seed_predictions]
    values = [score(y, p)['mape'] for y, p in zip(truths, averages)]
    refs = [score(y, p)['mape'] for y, p in zip(truths, references)]
    result = gate(values, refs, rule)
    seeds = [[score(y, p)['mape'] for p in ps] for y, ps in zip(truths, seed_predictions)]
    result['seed_mape_by_fold'] = seeds
    result['seed_std_mean'] = float(np.mean(np.std(seeds, axis=1)))
    if result['seed_std_mean'] > rule['max_seed_std']:
        result['accepted'] = False
        result['reason'].append('seed_variability')
    return result
