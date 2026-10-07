"""私有AIC赛事研究：唯一模型调度，测试和生产共用同一训练/推理实现。"""
from pathlib import Path
import pandas as pd
from gasbench.common import load_json, msg

def fit_model(bundle, cutoff: pd.Timestamp, candidate: dict, threads: int, path: Path, deadline: float):
    """模型实现全部真实训练，禁止用回退基线冒充目标家族。"""
    if candidate['kind'] == 'reference':
        from gasbench.reference import fit_reference
        return fit_reference(bundle, cutoff, path)
    if candidate['kind'] == 'sequence':
        from gasbench.neural import NeuralPredictor
        return NeuralPredictor.fit(bundle, cutoff, candidate, threads, path, deadline)
    from gasbench.tree import TreePredictor
    return TreePredictor.fit(bundle, cutoff, candidate, threads, path)

def load_model(path: Path, kind: str, device: str='cpu'):
    """仅载入签名核验过的自有产物；设备差异写入重放结果。"""
    if kind == 'reference':
        from gasbench.reference import load_reference
        return load_reference(path)
    if kind == 'sequence':
        from gasbench.neural import NeuralPredictor
        return NeuralPredictor.load(path, device)
    from gasbench.tree import TreePredictor
    return TreePredictor.load(path)

def predict_model(model, kind: str, features: pd.DataFrame, index: pd.DatetimeIndex):
    """只有序列模型接完整过程特征视图，仍按origin索引取历史。"""
    if kind == 'sequence':
        return model.predict(features, index)
    return model.predict(features.loc[index])
