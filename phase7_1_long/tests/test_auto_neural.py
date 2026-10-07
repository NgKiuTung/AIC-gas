"""私有AIC研究：真实神经零轮回退、任务输出和最佳权重存取。"""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from gasauto.networks import build,task_loss
from gasauto.trainer import choose_best,fit_phase
from gasauto.budget import TrainingClock
from gasbench.neural_data import NeuralArrays

@pytest.fixture
def arrays():
    rng=np.random.default_rng(1)
    return NeuralArrays(rng.normal(size=(32,5)).astype('float32'),
        rng.normal(size=(32,7)).astype('float32'),np.ones((16,8,6),dtype='float32'),
        np.ones((16,8,2),dtype='float32'),np.arange(16,32),8,np.ones((16,8,2),dtype='float32'))

@pytest.fixture
def neural_cfg():
    return {'family':'tcn','blocks':8,'device':'cpu','seed':3,'hidden':8,'levels':2,
       'history_hours':2,'projection':4,'dropout':0,'epochs':2,'batch_size':8,
       'learning_rate':.03,'weight_decay':0,'patience':2,'amp':False}

@pytest.mark.parametrize('family',['tcn','tide_style','mlp'])
@pytest.mark.parametrize('horizon',[8,96])
def test_zero_head_dynamic_shapes(family,horizon):
    cfg={'family':family,'blocks':horizon,'hidden':8,'levels':2,'projection':4,'dropout':0}
    net=build(cfg,5,7,8)
    anchor=torch.ones((2,horizon,2))
    result=net(torch.ones((2,8,5)),torch.ones((2,7)),torch.ones((2,horizon,6)),anchor)
    torch.testing.assert_close(result,anchor,rtol=0,atol=0)

def test_task_loss_uniform_long():
    y=torch.ones(1,96,2);p=y.clone();p[:,:8]+=1
    assert float(task_loss(p,y))==pytest.approx(8/96)

def test_task_loss_masked():
    y=torch.ones(1,8,2);y[0,0,0]=float('nan')
    assert float(task_loss(torch.ones_like(y),y))==0

def test_best_epoch_zero_not_overwritten():
    state={'best_loss':.1,'best_epoch':0,'stale':0}
    actual,improved=choose_best(state,1,.2,1e-6)
    assert not improved and actual['best_epoch']==0

def test_better_epoch_kept():
    state={'best_loss':.1,'best_epoch':0,'stale':1}
    actual,improved=choose_best(state,2,.09,1e-6)
    assert improved and actual['best_epoch']==2 and actual['stale']==0

@pytest.mark.parametrize('family',['tcn','tide_style','mlp'])
def test_actual_training_keeps_epoch0(tmp_path,arrays,neural_cfg,family):
    neural_cfg['family']=family
    train=copy.deepcopy(arrays);train.truth=np.full_like(train.truth,1.5)
    clock=TrainingClock(20,tmp_path/'phase.json')
    net,state=fit_phase(train,arrays,neural_cfg,2,tmp_path,'tune',clock)
    x,_=arrays.batch(np.arange(16),'cpu')
    assert state['best_epoch']==0
    assert state['epoch0_loss']==0
    torch.testing.assert_close(net(*x),torch.ones((16,8,2)),rtol=0,atol=0)
    assert (tmp_path/'tune_best.pt').exists()

@pytest.mark.parametrize('family',['tcn','tide_style','mlp'])
def test_actual_refit_accepts_zero(tmp_path,arrays,neural_cfg,family):
    neural_cfg['family']=family
    net,state=fit_phase(arrays,None,neural_cfg,2,tmp_path,'refit',TrainingClock(20,tmp_path/'phase.json'),epochs=0)
    x,_=arrays.batch(np.arange(16),'cpu')
    assert state['final_epoch']==0 and state['steps']==0
    torch.testing.assert_close(net(*x),x[-1],rtol=0,atol=0)
