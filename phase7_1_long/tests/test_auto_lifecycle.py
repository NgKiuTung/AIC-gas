"""私有AIC研究：失败可见、预算、文件冻结与两份独立输出。"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import pytest
from gasauto.budget import TrainingClock
from gasauto.delivery import export_pair,pack
from gasauto.execution import append_tsv,Executor
from gasbench.common import write_json,seal,load_json
from gasauto.controller import _freeze


def test_budget_rejects_zero(tmp_path):
    clock=TrainingClock(0,tmp_path/'phase.json')
    with pytest.raises(TimeoutError):
        with clock.compute():pass


def test_budget_time_accounted(tmp_path):
    clock=TrainingClock(10,tmp_path/'phase.json')
    with clock.compute():time.sleep(.002)
    info=load_json(tmp_path/'phase.json')
    assert info['training_elapsed']>0
    assert info['active_since'] is None


def test_elapsed_budget_is_not_success(tmp_path):
    clock=TrainingClock(.001,tmp_path/'phase.json')
    with pytest.raises(TimeoutError):
        with clock.compute():time.sleep(.005)


def test_failure_metric_blank(tmp_path):
    p=tmp_path/'results.tsv';append_tsv(p,{'status':'crash','candidate':'x','mape':None})
    frame=pd.read_csv(p,sep='\t')
    assert np.isnan(frame.loc[0,'mape'])
    assert frame.loc[0,'status']=='crash'


def test_split_result_can_differ(tmp_path):
    idx=pd.date_range('2025-10-01',periods=960,freq='15min')
    short=np.tile([100.,200.],(960,8,1));long=np.tile([110.,210.],(960,96,1))
    report=export_pair(tmp_path,idx,short,long,{'source':'unit_test'})
    assert not report['overlap_equal']
    assert report['short_columns']==17 and report['long_columns']==193

@pytest.mark.parametrize('bad',['shape','nan','negative','relationship'])
def test_export_rejects_invalid(tmp_path,bad):
    idx=pd.date_range('2025-10-01',periods=960,freq='15min')
    s=np.tile([100.,200.],(960,8,1));l=np.tile([100.,200.],(960,96,1))
    if bad=='shape':s=s[:,:7]
    elif bad=='nan':s[0,0,0]=np.nan
    elif bad=='negative':s[0,0,0]=-1
    else:s[0,0,0]=201
    with pytest.raises(ValueError):export_pair(tmp_path,idx,s,l,{})


def test_pack_refuses_external_symlink(tmp_path):
    run=tmp_path/'run';run.mkdir();write_json(run/'summary.json',{})
    write_json(run/'verification/summary.json',{'status':'passed'})
    (tmp_path/'outside').write_text('private')
    (run/'bad').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError):pack(run,tmp_path/'packed.zip')


def test_pack_includes_hash_manifest(tmp_path):
    import zipfile
    run=tmp_path/'run';run.mkdir();write_json(run/'summary.json',{'status':'completed'})
    write_json(run/'verification/summary.json',{'status':'passed'})
    (run/'file.txt').write_text('payload')
    result=pack(run,tmp_path/'handoff.zip')
    assert result['status']=='passed'
    with zipfile.ZipFile(tmp_path/'handoff.zip') as archive:
        assert 'HANDOFF_MANIFEST.json' in archive.namelist()
        assert archive.testzip() is None
