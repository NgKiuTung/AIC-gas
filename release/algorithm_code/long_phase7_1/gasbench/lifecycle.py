"""私有AIC赛事研究：单运行锁和本任务子进程生命周期，不操作其他实验。"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
from gasbench.common import msg

@contextmanager
def run_lock(path: Path):
    """同一个run只允许一个监督器；锁由内核在进程退出后释放。"""
    path.mkdir(parents=True, exist_ok=True)
    with (path / '.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(msg('run.locked', path=path)) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)

def identity(pid: int) -> dict:
    """登记boot_id及进程开始tick，防止恢复时PID复用导致误判。"""
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        ticks = text[text.rfind(')') + 2:].split()[19]
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return {'pid': pid, 'start_ticks': ticks, 'boot_id': boot}
    except (FileNotFoundError, ProcessLookupError):
        return {'pid': pid, 'start_ticks': None, 'boot_id': None}

def same_process(record: dict) -> bool:
    """只识别尚活着的同一个进程，不发送任何信号。"""
    if record.get('start_ticks') is None or identity(record['pid']) != record:
        return False
    try:
        stat = Path(f"/proc/{record['pid']}/stat").read_text()
        return stat[stat.rfind(')') + 2:].split()[0] != 'Z'
    except FileNotFoundError:
        return False

def terminate_child(proc: subprocess.Popen) -> None:
    """只处理本监督器start_new_session创建且仍未回收的子进程组。"""
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)

def wait_child(proc: subprocess.Popen, deadline: float) -> bool:
    """返回是否因预算结束；异常/人工中断也清理本worker，不泄漏子任务。"""
    try:
        while proc.poll() is None:
            if time.time() >= deadline:
                terminate_child(proc)
                return True
            time.sleep(0.2)
        return False
    except BaseException:
        terminate_child(proc)
        raise
