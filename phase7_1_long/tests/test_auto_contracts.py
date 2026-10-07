"""私有AIC研究：双任务数据、评估和保留规则回归。"""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from gasauto.protocol import (BLOCKS,read_plan,validate_plan,index_for,score,shift,truth_for,segment_scores)
from gasauto.samples import rows_for
from gasauto.selection import gate,confirm_summary,choose_finalist
from gasauto.identity import safe_path,ensure_same_source

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def plan():
    return read_plan(ROOT/'configs/dual_smoke_cpu.json')

@pytest.mark.parametrize('name', ['a10_3h.json','a10_gpu_trees_3h.json','dual_smoke_cpu.json','dual_smoke_a10.json','cpu_review_dual.json'])
def test_plans_are_complete(name):
    value=read_plan(ROOT/'configs'/name)
    assert len(index_for(value['protocol']['screen'][0]))==960
    assert len(value['confirm_seeds'])==3

@pytest.mark.parametrize('task,minutes',[('short',120),('long',1440)])
def test_task_purge(bundle,task,minutes):
    cutoff=pd.Timestamp('2025-06-01')
    rows=rows_for(bundle,cutoff,{'origin_stride':1},task)
    last=bundle.features.index[rows[-1]]
    assert shift(last,minutes)==cutoff
    assert truth_for(bundle,bundle.features.index.take(rows[:2]),task).shape==(2,BLOCKS[task],2)

def test_short_includes_more_latest_labels(bundle):
    cutoff=pd.Timestamp('2025-06-01')
    short=rows_for(bundle,cutoff,{'origin_stride':1},'short')
    long=rows_for(bundle,cutoff,{'origin_stride':1},'long')
    assert len(short)-len(long)==88

@pytest.mark.parametrize('error', ['weekday_only','late_tail','overlap','two_seeds','self_parent','two_changes','bad_test','bad_budget','bad_gate'])
def test_invalid_plan_stops(plan,error):
    if error=='weekday_only':plan['protocol']['screen'][0]['end']='2025-06-05 23:45:00'
    elif error=='late_tail':plan['protocol']['confirm'][1]={'name':'late','start':'2025-09-22','end':'2025-10-01 23:45'}
    elif error=='overlap':plan['protocol']['screen'][-1]={'name':'overlap','start':'2025-08-25','end':'2025-09-03 23:45'}
    elif error=='two_seeds':plan['confirm_seeds']=[1,2]
    elif error=='self_parent':plan['candidates'][0]['parent']=plan['candidates'][0]['id']
    elif error=='two_changes':
        c=copy.deepcopy(plan['candidates'][0]);c.update(id='child',parent=c['id'],rounds=7,device='cuda',mutation={'field':'rounds','value':7});plan['candidates'].append(c)
    elif error=='bad_test':plan['protocol']['test']['end']='2025-10-10 23:30'
    elif error=='bad_budget':plan['train_seconds']=float('nan')
    elif error=='bad_gate':plan['screen_gate']['min_mean_gain']=-1
    with pytest.raises(ValueError):validate_plan(plan)

def test_long_loss_does_not_double_short():
    y=np.full((2,96,2),100.);p=y.copy();p[:,:8]+=100
    result=score(y,p)
    assert result['mape']==pytest.approx(8/96)

def test_missing_label_not_filled():
    y=np.ones((2,8,2))*10;y[0,0,0]=np.nan
    result=score(y,np.full_like(y,11))
    assert result['mape']==pytest.approx(.1)
    assert result['valid_pairs']==31

@pytest.mark.parametrize('bad',['zero','empty','shape','nan_prediction'])
def test_score_rejects_invalid(bad):
    y=np.ones((2,8,2));p=y.copy()
    if bad=='zero':y[0,0,0]=0
    elif bad=='empty':y[:]=np.nan
    elif bad=='shape':p=p[:,:7]
    else:p[0,0,0]=np.nan
    with pytest.raises(ValueError):score(y,p)

def test_segment_coverage(plan):
    index=index_for(plan['protocol']['screen'][0]);y=np.ones((960,8,2))
    result=segment_scores(y,y,index)
    assert result['last5']['expected_pairs']==480*8*2
    assert result['weekend']['expected_pairs']>0

def test_gate_refuses_worst_and_recent():
    result=gate([.1,.12],[.2,.1],{'min_mean_gain':.0005,'max_fold_regression':.005,'max_last_regression':.001})
    assert not result['accepted']
    assert 'latest_fold' in result['reason']

def test_gate_nan_reference_fails():
    result=gate([.1],[np.nan],{'min_mean_gain':0,'max_fold_regression':0,'max_last_regression':0})
    assert not result['accepted']

def test_confirm_requires_three_seeds():
    y=np.ones((2,8,2))*10
    result=confirm_summary([[y,y]],[y],[y+1],{})
    assert not result['accepted']

def test_confirm_uses_average_not_best_seed():
    y=np.ones((2,8,2))*10
    result=confirm_summary([[y,y+4,y+4]],[y],[y+1],{'min_mean_gain':0,'max_fold_regression':1,'max_last_regression':1,'max_seed_std':1})
    assert not result['accepted']
    assert result['mean_mape']==pytest.approx(8/30)

def test_confirm_passes_real_gain():
    y=np.ones((2,8,2))*10
    result=confirm_summary([[y+.1,y+.2,y+.3]],[y],[y+1],{'min_mean_gain':.0005,'max_fold_regression':.005,'max_last_regression':.001,'max_seed_std':.02})
    assert result['accepted']

def test_finalist_no_invalid_scores():
    screen={'a':{'decision':{'accepted':False}},'b':{'decision':{'accepted':True,'mean_mape':.1}}}
    assert choose_finalist(screen,[{'id':'a'},{'id':'b'}])=={'id':'b'}

@pytest.mark.parametrize('path',['../escape','/etc/passwd'])
def test_path_escape(tmp_path,path):
    with pytest.raises(ValueError):safe_path(tmp_path,path)

def test_source_mutation_stops(monkeypatch):
    monkeypatch.setattr('gasauto.identity.source_identity',lambda:{'x':'new'})
    with pytest.raises(ValueError):ensure_same_source({'x':'old'})
