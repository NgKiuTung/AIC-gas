"""私有AIC赛事研究：明确模型族、设备与训练规模，不接受悄悄忽略的配置。"""
from __future__ import annotations
import re
from pathlib import Path
from gasbench.common import load_json, msg
ALLOWED = {'id', 'family', 'kind', 'device', 'rounds', 'seed', 'params', 'pre_weight', 'origin_stride', 'sample_horizons', 'lookback_days', 'max_train_origins', 'history_hours', 'hidden', 'projection', 'dropout', 'levels', 'epochs', 'batch_size', 'predict_batch_size', 'learning_rate', 'weight_decay', 'patience', 'inner_days', 'amp'}
PARAMS = {'lightgbm': {'learning_rate', 'num_leaves', 'min_data_in_leaf', 'max_bin', 'lambda_l2', 'lambda_l1', 'feature_fraction'}, 'xgboost': {'eta', 'max_depth', 'max_bin', 'lambda', 'alpha', 'min_child_weight', 'subsample', 'colsample_bytree'}, 'catboost': {'depth', 'learning_rate', 'l2_leaf_reg', 'border_count', 'random_strength'}}

def validate(cfg: dict) -> dict:
    """拒绝未知字段、不支持的设备和路径形模型名。"""
    allowed_top = {'version', 'threads', 'job_minutes', 'folds', 'candidates', 'fusion', 'selection'}
    if set(cfg) - allowed_top or not 1 <= cfg['threads'] <= 64 or cfg['job_minutes'] <= 0:
        raise ValueError(msg('config.invalid', detail='top-level/threads/budget'))
    if not cfg['folds'] or not cfg['candidates'] or set(cfg['fusion']) != {'ridge'} or (cfg['fusion']['ridge'] < 0):
        raise ValueError(msg('config.invalid', detail='top-level/threads/budget'))
    selection = cfg.get('selection', {})
    expected_selection = {'objective', 'freeze_short_to_reference', 'min_folds', 'min_mean_improvement_pp', 'min_win_fraction', 'max_worst_degradation_pp', 'recent_fold_count', 'min_recent_improvement_pp', 'blend_weight'}
    if set(selection) != expected_selection or selection['objective'] != 'platform_score_proxy':
        raise ValueError(msg('config.invalid', detail='selection'))
    if selection['freeze_short_to_reference'] is not True or selection['min_folds'] < 2:
        raise ValueError(msg('config.invalid', detail='selection guard'))
    if not 0 <= selection['min_win_fraction'] <= 1 or not 0 <= selection['blend_weight'] <= 1:
        raise ValueError(msg('config.invalid', detail='selection range'))
    if selection['recent_fold_count'] < 1 or selection['max_worst_degradation_pp'] < 0:
        raise ValueError(msg('config.invalid', detail='selection range'))
    names = set()
    for c in cfg['candidates']:
        if set(c) - ALLOWED or not re.fullmatch('[a-z0-9_]+', c['id']) or c['id'] in names:
            raise ValueError(msg('config.invalid', detail='candidate keys/id'))
        names.add(c['id'])
        _candidate(c)
    if 'reference_58' not in names:
        raise ValueError(msg('config.invalid', detail='reference_58 required'))
    import pandas as pd
    starts = []
    fold_names = set()
    for fold in cfg['folds']:
        if set(fold) != {'name', 'start', 'end'} or not re.fullmatch('[a-z0-9_]+', fold['name']):
            raise ValueError(msg('config.invalid', detail='fold'))
        if fold['name'] in fold_names or fold['name'] == 'test':
            raise ValueError(msg('config.invalid', detail='fold names'))
        fold_names.add(fold['name'])
        start, end = (pd.Timestamp(fold['start']), pd.Timestamp(fold['end']))
        if start.tz is not None or end.tz is not None or end + pd.Timedelta(days=1) > pd.Timestamp('2025-10-01'):
            raise ValueError(msg('config.invalid', detail='fold outside observable target history'))
        if start > end or start != start.floor('15min') or end != end.floor('15min'):
            raise ValueError(msg('config.invalid', detail='fold time'))
        starts.append(start)
    if starts != sorted(set(starts)):
        raise ValueError(msg('config.invalid', detail='fold order/duplicate'))
    return cfg

def _candidate(c):
    if c['id'] == 'reference_58':
        if c != {'id': 'reference_58', 'family': 'lightgbm', 'kind': 'reference', 'device': 'cpu'}:
            raise ValueError(msg('config.invalid', detail='reference_58 is immutable'))
        return
    if c.get('device', 'cpu') not in ('cpu', 'cuda'):
        raise ValueError(msg('config.invalid', detail='device'))
    if c['kind'] == 'sequence':
        if c['family'] not in ('tcn', 'tide_style') or c.get('history_hours') not in (24, 72):
            raise ValueError(msg('config.invalid', detail='sequence family/window'))
        for key in ('epochs', 'batch_size', 'inner_days', 'learning_rate'):
            if c.get(key, 0) <= 0:
                raise ValueError(msg('config.invalid', detail=key))
    elif c['kind'] in ('proxy', 'direct'):
        if c['family'] not in PARAMS or c.get('rounds', 0) < 1:
            raise ValueError(msg('config.invalid', detail='tree family/rounds'))
        if set(c.get('params', {})) - PARAMS[c['family']]:
            raise ValueError(msg('config.invalid', detail='backend params'))
        if c['family'] == 'lightgbm' and c.get('device', 'cpu') != 'cpu':
            raise ValueError(msg('config.invalid', detail='LightGBM CPU reference backend'))
    else:
        raise ValueError(msg('config.invalid', detail='kind'))
    if not 0 <= c.get('pre_weight', 0.25) <= 1:
        raise ValueError(msg('config.invalid', detail='pre_weight'))
    if not 1 <= c.get('origin_stride', 2) <= 96 or not 4 <= c.get('sample_horizons', 32) <= 96:
        raise ValueError(msg('config.invalid', detail='training sample geometry'))

def read_config(path: Path) -> dict:
    """读取UTF-8并执行语义校验。"""
    return validate(load_json(path))
