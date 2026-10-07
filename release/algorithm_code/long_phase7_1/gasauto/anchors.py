"""私有AIC赛事研究：参考58模型的按时间冻结锚点，禁止最终模型回填过去。"""
from pathlib import Path
import numpy as np
import pandas as pd
from gasbench.common import fingerprint, check_seal
from gasbench.reference import fit_reference, load_reference

class AnchorBank:
    """单worker串行使用；缓存签名包含任务运行固定身份和训练截止。"""

    def __init__(self, bundle, root: Path, identity: str):
        self.bundle, self.root, self.identity = (bundle, root, identity)
        self.used = {}
        self.training_segments = []

    def get(self, cutoff: pd.Timestamp):
        """只用cutoff之前标签拟合，旧缓存加载前完整验签。"""
        cutoff = pd.Timestamp(cutoff)
        key = fingerprint({'identity': self.identity, 'cutoff': str(cutoff)})[:24]
        path = self.root / key
        if (path / 'FILES.json').exists():
            check_seal(path)
            model = load_reference(path)
        else:
            path.mkdir(parents=True, exist_ok=True)
            model = fit_reference(self.bundle, cutoff, path)
        if pd.Timestamp(model.metadata['cutoff']) != cutoff:
            raise ValueError('anchor cutoff mismatch')
        self.used[str(cutoff)] = {'key': key, 'training': model.metadata['training']}
        return (model, path)

    def predict(self, index: pd.DatetimeIndex, cutoff: pd.Timestamp, blocks: int):
        """推理只传过程特征；每行必须不早于模型训练截止。"""
        if index.min() < cutoff:
            raise ValueError('anchor from future')
        model, _ = self.get(cutoff)
        return model.predict(self.bundle.features.loc[index])[:, :blocks]

    def training(self, rows: np.ndarray, blocks: int):
        """每10日冻结一次代理；训练期模拟目标整段隐藏，不借外层真值。"""
        index = self.bundle.features.index.take(rows)
        start = pd.Timestamp('2025-05-10')
        if index.min() < start:
            raise ValueError('training before anchor warmup')
        days = ((index - start).total_seconds() // (10 * 86400)).astype(int) * 10
        cutoffs = pd.DatetimeIndex(start.to_datetime64() + np.asarray(days).astype('timedelta64[D]'))
        out = np.empty((len(rows), blocks, 2), dtype='float64')
        for cut in cutoffs.unique():
            take = cutoffs == cut
            out[take] = self.predict(index[take], cut, blocks)
            self.training_segments.append({'anchor_cutoff': str(cut), 'first_origin': str(index[take][0]), 'last_origin': str(index[take][-1]), 'row_count': int(take.sum()), 'blocks': blocks})
        return out
