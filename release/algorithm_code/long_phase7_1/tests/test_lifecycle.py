"""私有AIC赛事研究：只操作测试自身创建进程的生命周期测试。"""
import os, subprocess, sys, time
from pathlib import Path
import pytest
from gasbench.lifecycle import identity, same_process, run_lock, wait_child, terminate_child
from gasbench.config import read_config
from gasbench.common import ROOT
from gasbench.preflight import preflight

def test_identity():
    assert same_process(identity(os.getpid()))

def test_pid_not_reused():
    assert not same_process({'pid': os.getpid(), 'boot_id': 'old', 'start_ticks': '-1'})

def test_lock_exclusion(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with run_lock(tmp_path):
                pass

def test_timeout_own_process():
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(10)'], start_new_session=True)
    assert wait_child(child, time.time() + 0.1)
    assert child.poll() is not None

def test_completed_child_not_killed():
    child = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    assert not wait_child(child, time.time() + 10)
    terminate_child(child)
    assert child.returncode == 0

def test_cpu_preflight(tmp_path):
    config = read_config(ROOT / 'configs/smoke_cpu.json')
    result = preflight(config, tmp_path / 'report.json')
    assert result['status'] == 'passed' and len(result['checks']) == 5

def test_gpu_missing_refuses(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    cfg = read_config(ROOT / 'configs/smoke_a10.json')
    with pytest.raises(RuntimeError):
        preflight(cfg, tmp_path / 'report.json')

def test_zombie_not_running():
    child = subprocess.Popen([sys.executable, '-c', 'pass'])
    recorded = identity(child.pid)
    _await_zombie(child)
    actual = same_process(recorded)
    child.wait()
    assert actual is False


def _await_zombie(child):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        done = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
        if done is not None:
            return
        time.sleep(0.02)
    child.kill()
    child.wait(timeout=5)
    raise TimeoutError('test child failed to exit')
