"""私有AIC研究：双任务后端真训练、重放和时间外推来源测试。"""
import copy
import numpy as np
import pandas as pd
import pytest
from gasauto.anchors import AnchorBank
from gasauto.trees import TaskTree
from gasauto.neural import TaskNeural
from gasauto.budget import TrainingClock
from gasauto.samples import direct_samples,rows_for
from gasauto.protocol import BLOCKS,shift
from gasbench.audit import poisoned_bundle

@pytest.fixture(scope='session')
def bank(bundle,tmp_path_factory):
    return AnchorBank(bundle,tmp_path_factory.mktemp('auto_anchor'),'synthetic')

@pytest.mark.parametrize('family',['lightgbm','catboost','xgboost'])
@pytest.mark.parametrize('kind',['proxy','direct'])
@pytest.mark.parametrize('task',['short','long'])
def test_tree_fit_reload(bundle,bank,tmp_path,family,kind,task,tiny_tree):
    cfg={**tiny_tree,'family':family,'kind':kind,'rounds':3,'max_train_origins':64}
    cutoff=pd.Timestamp('2025-06-01')
    model=TaskTree.fit(bundle,cutoff,cfg,task,2,tmp_path/'model',bank,TrainingClock(30,tmp_path/'phase.json'))
    idx=pd.date_range(cutoff,periods=3,freq='15min')
    pred=model.predict(bundle.features,idx)
    saved=TaskTree.load(tmp_path/'model').predict(bundle.features,idx)
    assert pred.shape==(3,BLOCKS[task],2)
    np.testing.assert_array_equal(pred,saved)
    if kind=='direct':assert pd.Timestamp(model.metadata['training']['label_end_exclusive'])<=cutoff

@pytest.mark.parametrize('task',['short','long'])
def test_training_anchor_respects_cutoff(bundle,bank,task):
    cfg={'origin_stride':16,'max_train_origins':64}
    rows=rows_for(bundle,pd.Timestamp('2025-06-01'),cfg,task)
    isolated=AnchorBank(bundle,bank.root,bank.identity)
    pred=isolated.training(rows,BLOCKS[task])
    assert pred.shape==(len(rows),BLOCKS[task],2)
    assert max(pd.Timestamp(c) for c in isolated.used)<=bundle.features.index[rows[-1]]

@pytest.mark.parametrize('family',['tcn','tide_style','mlp'])
@pytest.mark.parametrize('task',['short','long'])
def test_neural_fit_reload(bundle,bank,tmp_path,tiny_neural,family,task):
    cfg={**tiny_neural,'family':family,'max_train_origins':64,'max_calibration_origins':32}
    cutoff=pd.Timestamp('2025-06-01')
    model=TaskNeural.fit(bundle,cutoff,cfg,task,2,tmp_path/'model',bank,TrainingClock(60,tmp_path/'phase.json'))
    idx=pd.date_range(cutoff,periods=3,freq='15min')
    p=model.predict(bundle.features,idx)
    q=TaskNeural.load(tmp_path/'model').predict(bundle.features,idx)
    np.testing.assert_array_equal(p,q)
    assert p.shape==(3,BLOCKS[task],2)
    assert model.metadata['training']['tune']['epoch0_loss'] is not None
    assert pd.Timestamp(model.metadata['training']['label_end_exclusive'])<=cutoff
