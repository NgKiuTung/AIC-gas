"""私有AIC赛事研究：三种提升树的真实训练API，共用MAPE样本权重。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from gasbench.common import msg
EXTENSIONS = {'lightgbm': '.txt', 'catboost': '.cbm', 'xgboost': '.ubj'}

def active_columns(x: np.ndarray) -> np.ndarray:
    """训练内剔除全空/常量列，不查看验证或测试分布。"""
    finite = np.isfinite(x)
    low = np.where(finite, x, np.inf).min(axis=0)
    high = np.where(finite, x, -np.inf).max(axis=0)
    keep = (finite.sum(axis=0) >= 16) & (high > low)
    if not keep.any():
        raise ValueError(msg('data.empty'))
    return np.flatnonzero(keep)

def fit_backend(family: str, x: np.ndarray, target: np.ndarray, weights: np.ndarray, cfg: dict, threads: int):
    """输入标签可以是残差，但weights已用真实正负荷作分母。"""
    p = cfg.get('params', {})
    rounds, seed = (cfg.get('rounds', 350), cfg.get('seed', 20261003))
    device = cfg.get('device', 'cpu')
    if family == 'lightgbm':
        import lightgbm as lgb
        params = {'objective': 'regression_l1', 'metric': 'None', 'verbosity': -1, 'num_threads': threads, 'seed': seed, 'deterministic': True, 'force_col_wise': True, 'learning_rate': 0.04, 'num_leaves': 31, 'min_data_in_leaf': 80, 'max_bin': 127, 'lambda_l2': 2.0, **p}
        return lgb.train(params, lgb.Dataset(x, label=target, weight=weights), num_boost_round=rounds)
    if family == 'catboost':
        from catboost import CatBoostRegressor
        params = {'iterations': rounds, 'loss_function': 'MAE', 'depth': 6, 'learning_rate': 0.04, 'l2_leaf_reg': 3, 'border_count': 127, 'random_seed': seed, 'thread_count': threads, 'allow_writing_files': False, 'verbose': False, 'nan_mode': 'Min', 'boosting_type': 'Plain', **p, 'task_type': 'GPU' if device == 'cuda' else 'CPU'}
        if device == 'cuda':
            params.update(devices='0', gpu_ram_part=0.7)
        model = CatBoostRegressor(**params)
        model.fit(x, target, sample_weight=weights)
        return model
    if family == 'xgboost':
        import xgboost as xgb
        params = {'objective': 'reg:absoluteerror', 'tree_method': 'hist', 'max_depth': 6, 'eta': 0.04, 'lambda': 3, 'max_bin': 127, 'seed': seed, 'base_score': 0.0, 'nthread': threads, 'validate_parameters': True, **p, 'device': device}
        model = xgb.train(params, xgb.DMatrix(x, label=target, weight=weights, nthread=threads), num_boost_round=rounds)
        actual = json.loads(model.save_config())['learner']['generic_param']['device']
        if device == 'cuda' and (not actual.startswith('cuda')):
            raise RuntimeError(msg('device.unavailable'))
        return model
    raise ValueError(msg('model.family', family=family))

def predict_backend(family: str, model, x: np.ndarray, threads: int) -> np.ndarray:
    """保存后用各家原生加载器预测，拒绝非有限数值。"""
    if family == 'xgboost':
        import xgboost as xgb
        value = model.predict(xgb.DMatrix(x, nthread=threads))
    elif family == 'catboost':
        value = model.predict(x, thread_count=threads)
    else:
        value = model.predict(x, num_threads=threads)
    value = np.asarray(value, dtype='float64')
    if not np.isfinite(value).all():
        raise ValueError(msg('model.finite'))
    return value

def save_backend(family: str, model, path: Path) -> None:
    """原生非pickle模型格式，路径由内部目标键生成。"""
    model.save_model(str(path))

def load_backend(family: str, path: Path):
    """调用方先校验产物hash，不从未知pickle恢复代码。

NOTE: 输出重放可在CPU完成，不因保存时GPU配置读取无GPU失败。"""
    if family == 'lightgbm':
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    if family == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor().load_model(str(path))
    if family == 'xgboost':
        import xgboost as xgb
        value = xgb.Booster()
        value.load_model(str(path))
        value.set_param({'device': 'cpu'})
        return value
    raise ValueError(msg('model.family', family=family))
