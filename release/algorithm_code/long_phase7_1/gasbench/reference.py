"""私有AIC赛事研究：保留58.1578分对应的原模型实现，不套烟雾缩减参数。"""
from pathlib import Path
import time
import pandas as pd
from gasbench.common import ROOT, load_json, seal, check_seal, write_json
from gas2.model import ProcessModel

def fit_reference(bundle, cutoff: pd.Timestamp, output: Path) -> ProcessModel:
    """继承原训练样本、损失、seed、220轮及6小时衰减；仅features名随276版本登记。"""
    settings = load_json(ROOT / 'reference/stage2_settings.json')
    candidate = next((c for c in settings['candidates'] if c['id'] == 'proxy_joint025_l1'))
    model = ProcessModel.fit(bundle, cutoff, candidate, settings, output, time.time() + 86400)
    seal(output)
    return model

def load_reference(output: Path) -> ProcessModel:
    """加载前先校验摘要，再调用原Stage2加载器。"""
    check_seal(output)
    return ProcessModel.load(output)
