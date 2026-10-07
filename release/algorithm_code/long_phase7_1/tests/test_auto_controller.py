"""私有AIC研究：使用确定性合成结果验证确认和部署控制，不冒充真实训练。"""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pytest
from gasauto.controller import Research,_freeze
from gasauto.protocol import read_plan,index_for,truth_for,BLOCKS
from gasauto.execution import Executor
from gasbench.common import write_json,seal,load_json,fingerprint
from gasbench.lifecycle import run_lock

ROOT=Path(__file__).resolve().parents[1]


def cache_stub(path):
    path.mkdir();write_json(path/'ready.json',{'dataset_sha256':'test'});seal(path)
    return path


def test_resume_identity_reject(tmp_path):
    cache=cache_stub(tmp_path/'cache');run=tmp_path/'run'
    with run_lock(run):
        _freeze(cache,ROOT/'configs/dual_smoke_cpu.json',run,1,False,0)
    cfg=read_plan(ROOT/'configs/dual_smoke_cpu.json');cfg['threads']=3
    write_json(tmp_path/'changed.json',cfg)
    with run_lock(run),pytest.raises(ValueError):
        _freeze(cache,tmp_path/'changed.json',run,1,True,0)


def test_resume_deadline_not_reset(tmp_path):
    cache=cache_stub(tmp_path/'cache');run=tmp_path/'run'
    with run_lock(run):
        _,_,before=_freeze(cache,ROOT/'configs/dual_smoke_cpu.json',run,1,False,0)
    with run_lock(run):
        _,_,after=_freeze(cache,ROOT/'configs/dual_smoke_cpu.json',run,3,True,0)
    assert before==after


def test_explicit_budget_extension_logged(tmp_path):
    cache=cache_stub(tmp_path/'cache');run=tmp_path/'run'
    with run_lock(run):
        _,_,before=_freeze(cache,ROOT/'configs/dual_smoke_cpu.json',run,1,False,0)
    with run_lock(run):
        _,_,after=_freeze(cache,ROOT/'configs/dual_smoke_cpu.json',run,1,True,.5)
    assert after==before+1800
    assert len(list(run.glob('extension_*.json')))==1


def test_second_supervisor_refused(tmp_path):
    with run_lock(tmp_path),pytest.raises(RuntimeError):
        with run_lock(tmp_path):pass


def test_budget_skip_visible_in_ledger(tmp_path):
    import time
    from gasauto.identity import source_identity
    write_json(tmp_path/'experiment.json',{'identity':{'source':source_identity()}})
    cfg=read_plan(ROOT/'configs/dual_smoke_cpu.json')
    executor=Executor(tmp_path/'cache',tmp_path,cfg,'test',time.time()-1)
    c=cfg['candidates'][0]
    result=executor.job('screen','short',c,cfg['protocol']['screen'][0],41)
    assert result['status']=='budget_skipped'
    assert 'skipped' in (tmp_path/'results.tsv').read_text()


class SyntheticExecutor:
    """只测试研究状态机；预测由合成标签决定，不是模型实验。"""
    def __init__(self,run,bundle,selected):
        self.run,self.bundle,self.selected=run,bundle,selected;self.calls=[]

    def job(self,phase,task,candidate,fold,seed,allow_reserved=False):
        self.calls.append((phase,task,candidate['id'],seed))
        relative=f'fixtures/{phase}_{task}_{candidate["id"]}_{fold["name"]}_{seed}'
        out=self.run/relative;out.mkdir(parents=True,exist_ok=True)
        idx=index_for(fold)
        if fold['name']=='test':
            y=np.tile([100.,200.],(len(idx),BLOCKS[task],1))
        else:y=truth_for(self.bundle,idx,task)
        factor=1.1 if candidate['id']=='reference_58' else 1.01
        pred=y*factor
        np.savez(out/'prediction.npz',origins=idx.values,prediction=pred)
        seal(out)
        from gasauto.protocol import score
        return {'status':'completed','output':relative,'candidate':{**candidate,'seed':seed},'metrics':score(y,pred)}


def test_complete_select_confirm_export_control(tmp_path,bundle,monkeypatch):
    cfg=read_plan(ROOT/'configs/dual_smoke_cpu.json')
    one=cfg['candidates'][0];cfg['candidates']=[one]
    # 合成数据范围内的窗口；这里单测状态机而非验证正式日期。
    fold={'name':'control_fold','start':'2025-05-11','end':'2025-05-20 23:45:00'}
    cfg['protocol']['screen']=[fold]
    cfg['protocol']['confirm']=[{**fold,'name':'confirm_control'}]
    from gasauto.identity import source_identity
    write_json(tmp_path/'experiment.json',{'identity':{'source':source_identity()}})
    engine=Research(tmp_path/'cache',tmp_path,cfg,'fixture',1e20)
    engine.executor=SyntheticExecutor(tmp_path,bundle,None)
    monkeypatch.setattr('gasauto.controller.load_bundle',lambda _:bundle)
    engine.baseline();engine.screening();picks=engine.freeze_finalists()
    confirmed=engine.confirmation(picks);selected=engine.finalize(picks,confirmed)
    assert confirmed['short']['accepted'] and confirmed['long']['accepted']
    assert len(selected['short']['records'])==3
    assert selected['long']['kind']=='confirmed_seed_average'
    assert (tmp_path/'candidate_results/selected/results_only.zip').exists()


def test_finalists_cannot_change_after_freeze(tmp_path,bundle):
    cfg=read_plan(ROOT/'configs/dual_smoke_cpu.json')
    from gasauto.identity import source_identity
    write_json(tmp_path/'experiment.json',{'identity':{'source':source_identity()}})
    engine=Research(tmp_path/'cache',tmp_path,cfg,'fixture',1e20)
    write_json(tmp_path/'screening.json',{})
    engine.freeze_finalists()
    c=cfg['candidates'][0]
    engine.screen['short'][c['id']]={'decision':{'accepted':True,'mean_mape':.01}}
    with pytest.raises(ValueError):engine.freeze_finalists()


def test_worker_timeout_only_own_process(tmp_path):
    import os
    import subprocess
    import sys
    import time
    from gasauto.identity import source_identity
    from gasbench.lifecycle import identity,same_process,terminate_child
    write_json(tmp_path/'experiment.json',{'identity':{'source':source_identity()}})
    cfg=read_plan(ROOT/'configs/dual_smoke_cpu.json')
    engine=Executor(tmp_path/'cache',tmp_path,cfg,'fixture',time.time()+5)
    self_id=identity(os.getpid())
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'],start_new_session=True)
    write_json(tmp_path/'owned_child.json',identity(process.pid))
    try:
        actual=engine._wait(process,tmp_path,time.time()+.02)
        assert actual=='timeout'
        assert process.poll() is not None
        assert same_process(self_id)
    finally:terminate_child(process)


def test_completed_job_is_reused_without_spawn(tmp_path,monkeypatch):
    import time
    from gasauto.identity import source_identity
    write_json(tmp_path/'experiment.json',{'identity':{'source':source_identity()}})
    cfg=read_plan(ROOT/'configs/dual_smoke_cpu.json');c={**cfg['candidates'][0],'seed':41}
    f=cfg['protocol']['screen'][0];task='short';phase='screen';sig='fixture'
    signature=fingerprint({'run':sig,'phase':phase,'task':task,'candidate':c,'fold':f})
    folder=tmp_path/'jobs'/f'{phase}__{task}__{c["id"]}__{f["name"]}__s41'
    output=folder/'attempt_001';output.mkdir(parents=True)
    (output/'sentinel.txt').write_text('completed model')
    write_json(output/'receipt.json',{'status':'completed','signature':signature})
    seal(output)
    write_json(folder/'result.json',{'status':'completed','signature':signature,
                                   'output':str(output.relative_to(tmp_path))})
    def forbidden(*args,**kwargs):raise AssertionError('不应创建第二个训练进程')
    monkeypatch.setattr('gasauto.execution.subprocess.Popen',forbidden)
    engine=Executor(tmp_path/'cache',tmp_path,cfg,sig,time.time()+1000)
    result=engine.job(phase,task,c,f,41)
    assert result['reused']


def test_corrupt_receipt_not_reused(tmp_path,monkeypatch):
    import time
    from gasauto.identity import source_identity
    write_json(tmp_path/'experiment.json',{'identity':{'source':source_identity()}})
    cfg=read_plan(ROOT/'configs/dual_smoke_cpu.json');c={**cfg['candidates'][0],'seed':41}
    f=cfg['protocol']['screen'][0];signature=fingerprint({'run':'fixture','phase':'screen','task':'short','candidate':c,'fold':f})
    folder=tmp_path/'jobs'/f'screen__short__{c["id"]}__{f["name"]}__s41'
    output=folder/'attempt_001';output.mkdir(parents=True)
    (output/'weights.txt').write_text('original');seal(output);(output/'weights.txt').write_text('changed')
    write_json(folder/'result.json',{'status':'completed','signature':signature,'output':str(output.relative_to(tmp_path))})
    with pytest.raises(ValueError):
        Executor(tmp_path/'cache',tmp_path,cfg,'fixture',time.time()+1000).job('screen','short',c,f,41)


def test_resume_wrong_new_folder_does_not_restart(tmp_path):
    cache=cache_stub(tmp_path/'cache');run=tmp_path/'mistyped_run'
    with run_lock(run),pytest.raises(ValueError):
        _freeze(cache,ROOT/'configs/dual_smoke_cpu.json',run,1,True,0)
    assert not (run/'experiment.json').exists()
