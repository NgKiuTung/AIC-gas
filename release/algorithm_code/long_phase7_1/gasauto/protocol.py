"""私有AIC赛事研究：冻结十日黑窗及短长纯任务指标。"""
from __future__ import annotations
import copy
import re
import math
from pathlib import Path
import numpy as np
import pandas as pd
from gasbench.common import load_json
from gasstage.targets import future_truth
BLOCKS = {'short': 8, 'long': 96}
LONG_BANDS = {
    'h01_08': (0, 8),
    'h09_24': (8, 24),
    'h25_48': (24, 48),
    'h49_96': (48, 96),
}

def shift(stamp, minutes: int) -> pd.Timestamp:
    """显式时间单位，避免generic timedelta在新NumPy中弃用。"""
    return pd.Timestamp(pd.Timestamp(stamp).to_datetime64().astype('datetime64[ns]') + np.timedelta64(int(minutes), 'm'))

def index_for(fold: dict) -> pd.DatetimeIndex:
    """窗口同时作为推理起点合同，不按有效标签截断。"""
    return pd.date_range(fold['start'], fold['end'], freq='15min')

def task_contract(contract: dict, task: str) -> dict:
    """保留数据合同，仅按独立任务改变标签右边界。"""
    value = copy.deepcopy(contract)
    value['protocol']['blocks'] = BLOCKS[task]
    return value

def truth_for(bundle, index, task: str) -> np.ndarray:
    """评分视图独立于特征；沿用原96块真值后切取所需跨度。"""
    return future_truth(bundle.truth, index, bundle.contract)[:, :BLOCKS[task]]

def score(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """两个目标等权；长任务对96块一次等权，不重复加权前8块。"""
    if prediction.shape != truth.shape or truth.ndim != 3 or truth.shape[-1] != 2:
        raise ValueError('task shape mismatch')
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('invalid prediction')
    valid = np.isfinite(truth)
    if np.any(valid & (truth <= 0)) or np.any(valid.sum(axis=(0, 1)) == 0):
        raise ValueError('nonpositive or empty truth')
    safe = np.where(valid, truth, 1)
    ape = np.where(valid, np.abs(prediction - safe) / safe, 0)
    by_target = ape.sum(axis=(0, 1)) / valid.sum(axis=(0, 1))
    return {'mape': float(by_target.mean()), 'g1_mape': float(by_target[0]), 'gall_mape': float(by_target[1]), 'valid_pairs': int(valid.sum()), 'expected_pairs': int(valid.size), 'coverage': float(valid.mean())}

def segment_scores(truth, pred, index) -> dict:
    """报告周末和负荷隐藏第6—10日，避免工作日均值掩盖变化。"""
    masks = {'first5': np.arange(len(index)) < 480, 'last5': np.arange(len(index)) >= 480, 'weekend': np.asarray(index.dayofweek >= 5)}
    return {k: score(truth[m], pred[m]) for k, m in masks.items() if m.any()}


def component_score(truth: np.ndarray, prediction: np.ndarray, target: int, start: int, end: int) -> float:
    """One target and one horizon band MAPE; selection never mixes targets implicitly."""
    y = truth[:, start:end, target]
    p = prediction[:, start:end, target]
    valid = np.isfinite(y) & np.isfinite(p) & (y > 0)
    if not valid.any():
        raise ValueError("empty component truth")
    return float(np.mean(np.abs(p[valid] - y[valid]) / y[valid]))

def long_component_scores(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """Return target×band MAPE used by Phase7 guarded component selection."""
    out = {}
    for target, name in enumerate(("g1", "gall")):
        for band, (start, end) in LONG_BANDS.items():
            out[f"{name}__{band}"] = component_score(truth, prediction, target, start, end)
    return out

def validate_plan(cfg: dict) -> dict:
    """数据、窗口和候选在开跑前语义检查；未来区间边界不可搜索。"""
    if 'protocol' not in cfg:
        raise ValueError('配置不是双任务研究配置，请使用a10_3h或dual_smoke系列。')
    protocol = cfg['protocol']
    if protocol.get('interval_assumption') != '[t,t+15min)':
        raise ValueError('区间定义必须在本版本保持冻结。')
    if not protocol['screen'] or not protocol['confirm']:
        raise ValueError('筛选和确认窗口不能为空。')
    allfolds = protocol['screen'] + protocol['confirm']
    names = set()
    for fold in allfolds:
        index = index_for(fold)
        if not re.fullmatch('[a-z0-9_]+', fold['name']) or fold['name'] in names or len(index) != 960 or (index[0] != index[0].normalize()) or (index.tz is not None) or (shift(index[-1], 1440) > pd.Timestamp('2025-10-01')) or (not (index.dayofweek >= 5).any()):
            raise ValueError('invalid complete ten-day fold')
        names.add(fold['name'])
    if max((shift(f['end'], 1440) for f in protocol['screen'])) > min((pd.Timestamp(f['start']) for f in protocol['confirm'])):
        raise ValueError('screen label overlaps confirmation')
    for group in ('screen', 'confirm'):
        starts = [pd.Timestamp(f['start']) for f in protocol[group]]
        if starts != sorted(starts):
            raise ValueError('窗口必须按时间顺序登记。')
    test = index_for(protocol['test'])
    if protocol['test']['name'] != 'test' or len(test) != 960 or test[0] != pd.Timestamp('2025-10-01') or (test[-1] != pd.Timestamp('2025-10-10 23:45:00')):
        raise ValueError('正式测试起点不符。')
    if len(set(cfg['confirm_seeds'])) < 3 or not all((isinstance(s, int) and s >= 0 for s in cfg['confirm_seeds'])):
        raise ValueError('three preregistered confirmation seeds required')
    for key in ('train_seconds', 'job_seconds', 'reserve_seconds', 'min_job_seconds', 'export_reserve_seconds'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError('非法预算数值。')
    if cfg['train_seconds'] <= 0 or cfg['job_seconds'] <= cfg['train_seconds']:
        raise ValueError('invalid training/job budget')
    if not 1 <= cfg['threads'] <= 64:
        raise ValueError('invalid threads')
    if not 0 < cfg['min_coverage'] <= 1:
        raise ValueError('标签覆盖率门槛非法。')
    for group in ('screen_gate', 'confirm_gate'):
        if any((not math.isfinite(x) or x < 0 for x in cfg[group].values())):
            raise ValueError('门槛须为非负有限值。')
    seen = set()
    for c in cfg['candidates']:
        if not re.fullmatch('[a-z0-9_]+', c['id']) or c['id'] in seen:
            raise ValueError('invalid candidate identity')
        seen.add(c['id'])
        if c['kind'] not in ('proxy', 'proxy_wide', 'direct', 'sequence') or c['family'] not in ('lightgbm', 'catboost', 'xgboost', 'tcn', 'tide_style', 'mlp'):
            raise ValueError('unsupported model')
        if c.get('device', 'cpu') not in ('cpu', 'cuda'):
            raise ValueError('unsupported device')
        if c.get('parent') and (c['parent'] not in seen or c['parent'] == c['id']):
            raise ValueError('parent must precede single-factor child')
        if not c['tasks'] or len(set(c['tasks'])) != len(c['tasks']) or (not set(c['tasks']) <= set(BLOCKS)):
            raise ValueError('invalid task')
        if (c['kind'] == 'sequence') != (c['family'] in ('tcn', 'tide_style', 'mlp')):
            raise ValueError('模型类型与模型族不一致。')
        for key in ('rounds', 'origin_stride', 'batch_size', 'epochs', 'history_hours', 'sample_horizons'):
            if key in c and (not isinstance(c[key], int) or c[key] < 1):
                raise ValueError('模型整数参数非法。')
        if c.get('parent'):
            parent = next((p for p in cfg['candidates'] if p['id'] == c['parent']))
            ignored = {'id', 'parent', 'hypothesis', 'mutation'}
            changed = [key for key in set(parent) | set(c) if key not in ignored and parent.get(key) != c.get(key)]
            if changed != [c['mutation']['field']] or c[changed[0]] != c['mutation']['value']:
                raise ValueError('条件子候选必须只改变一个登记字段。')
    return cfg

def read_plan(path: Path) -> dict:
    """调用方封存整份配置，不在恢复时重新选日期。"""
    return validate_plan(load_json(path))
