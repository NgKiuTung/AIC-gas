"""私有AIC赛事研究：串行跨模型实验、持久签名与明确的恢复会话预算。"""
from __future__ import annotations
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from gasbench.common import ROOT, check_seal, code_hashes, environment, fingerprint, load_json, logging_to, msg, write_json
from gasbench.config import read_config
from gasbench.lifecycle import identity, run_lock, same_process, wait_child
LOGGER = logging.getLogger(__name__)

def task_complete(path: Path, signature: str) -> bool:
    """回执和不可变文件全部一致才跳过任务，不能只看model存在。"""
    receipt = path / 'receipt.json'
    if not receipt.exists():
        return False
    saved = load_json(receipt)
    if saved['signature'] != signature:
        raise ValueError(msg('run.changed'))
    if saved['status'] != 'completed':
        return False
    if check_seal(path) != saved['output_hashes']:
        raise ValueError(msg('file.changed', path=path))
    return True

def _task(run: Path, cache: Path, cfg: dict, candidate: dict, fold: dict, contract: str, session_deadline: float, ledger: dict) -> dict:
    key = fold['name'] + '__' + candidate['id']
    path = run / 'tasks' / key
    signature = fingerprint({'run_contract': contract, 'candidate': candidate, 'fold': fold})
    if task_complete(path, signature):
        LOGGER.info(msg('job.resume', fold=fold['name'], candidate=candidate['id']))
        return load_json(path / 'receipt.json')
    old = ledger.get(key, {})
    if old.get('identity') and same_process(old['identity']):
        raise RuntimeError(msg('job.orphan', pid=old['identity']['pid']))
    path.mkdir(parents=True, exist_ok=True)
    deadline = min(session_deadline, time.time() + cfg['job_minutes'] * 60)
    spec = {'candidate': candidate, 'fold': fold, 'cache': str(cache), 'output': str(path), 'threads': cfg['threads'], 'deadline': deadline, 'parent_pid': os.getpid(), 'signature': signature}
    write_json(path / 'task.json', spec)
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': str(cfg['threads']), 'MKL_NUM_THREADS': str(cfg['threads']), 'OPENBLAS_NUM_THREADS': str(cfg['threads'])}
    started = time.time()
    with (path / 'console.log').open('a') as log:
        proc = subprocess.Popen([sys.executable, str(ROOT / 'run.py'), '_worker', str(path / 'task.json')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ledger[key] = {'status': 'running', 'identity': identity(proc.pid), 'started': started, 'deadline': deadline}
        write_json(run / 'jobs.json', ledger)
        LOGGER.info(msg('job.start', fold=fold['name'], candidate=candidate['id']))
        timed_out = wait_child(proc, deadline)
    status = 'budget_exhausted' if timed_out or proc.returncode == 124 else 'failed'
    if proc.returncode == 0 and task_complete(path, signature):
        status = 'completed'
    ledger[key].update(status=status, returncode=proc.returncode, seconds=time.time() - started)
    write_json(run / 'jobs.json', ledger)
    if status == 'completed':
        LOGGER.info(msg('job.end', fold=fold['name'], candidate=candidate['id'], seconds=ledger[key]['seconds']))
        return load_json(path / 'receipt.json')
    LOGGER.error(msg('job.failed', fold=fold['name'], candidate=candidate['id'], path=path / 'console.log'))
    return ledger[key]

def train(cache: Path, config: Path, output: Path, hours: float, resume: bool=False) -> dict:
    """新session明确给定有限时长；不修改旧session截止，不静默改变候选。

NOTE: 预检单独进程释放GPU上下文，正式运行不会同时留一个预检占用。"""
    if not 0 < hours <= 72:
        raise ValueError(msg('config.invalid', detail='hours in (0,72]'))
    cache, output = (cache.resolve(), output.resolve())
    cfg = read_config(config)
    cache_hashes = check_seal(cache)
    metadata = {'config': cfg, 'cache_hashes': cache_hashes, 'code_hashes': code_hashes(), 'environment': environment()}
    signature = fingerprint(metadata)
    with run_lock(output):
        cp = output / 'contract.json'
        if cp.exists():
            if not resume:
                raise ValueError(msg('run.exists'))
            if load_json(cp)['signature'] != signature:
                raise ValueError(msg('run.changed'))
        elif any((p.name != '.lock' for p in output.iterdir())):
            raise ValueError(msg('run.exists'))
        else:
            write_json(cp, {**metadata, 'signature': signature, 'cache_path': str(cache)})
        logging_to(output / 'events.jsonl')
        session = {'started': time.time(), 'hours': hours, 'pid': os.getpid(), 'resume': resume}
        session['deadline'] = session['started'] + hours * 3600
        sid = str(time.time_ns())
        write_json(output / 'sessions' / f'{sid}.json', session)
        write_json(output / 'run_status.json', {**session, 'status': 'running'})
        result = subprocess.run([sys.executable, str(ROOT / 'run.py'), 'preflight', '--config', str(config.resolve()), '--output', str(output / 'preflight.json')], cwd=ROOT, timeout=min(300, hours * 3600))
        if result.returncode:
            write_json(output / 'run_status.json', {**session, 'status': 'preflight_failed'})
            raise RuntimeError(msg('verify.failed', detail='preflight'))
        ledger = load_json(output / 'jobs.json') if (output / 'jobs.json').exists() else {}
        test = {'name': 'test', 'start': '2025-10-01 00:00:00', 'end': '2025-10-10 23:45:00'}
        results = []
        try:
            for fold in [*cfg['folds'], test]:
                for candidate in cfg['candidates']:
                    if time.time() >= session['deadline'] - 15:
                        break
                    results.append(_task(output, cache, cfg, candidate, fold, signature, session['deadline'], ledger))
                if time.time() >= session['deadline'] - 15:
                    LOGGER.warning(msg('budget.end'))
                    break
        except BaseException:
            write_json(output / 'run_status.json', {**session, 'status': 'interrupted', 'seconds': time.time() - session['started']})
            raise
        expected = (len(cfg['folds']) + 1) * len(cfg['candidates'])
        completed = sum((x['status'] == 'completed' for x in results))
        status = 'completed' if completed == expected else 'budget_exhausted' if time.time() >= session['deadline'] - 15 else 'failed'
        info = {**session, 'status': status, 'seconds': time.time() - session['started'], 'expected_tasks': expected, 'completed_tasks': completed, 'failed_tasks': sum((x['status'] == 'failed' for x in results)), 'timed_out_tasks': sum(x['status'] == 'budget_exhausted' for x in results), 'incomplete_tasks': expected - completed}
        write_json(output / 'run_status.json', info)
        from gasbench.reporting import summarize
        summarize(output, cache, cfg)
        info['seconds'] = time.time() - session['started']
        write_json(output / 'run_status.json', info)
        summary = load_json(output / 'summary.json')
        summary['status'] = info
        write_json(output / 'summary.json', summary)
        LOGGER.info(msg('run.done', status=status, path=output))
        return info
