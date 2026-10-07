"""私有AIC赛事研究：只终止自己登记的进程组，原子结果与完整失败账本。"""
import csv
import os
import subprocess
import sys
import time
import traceback
from gasbench.common import ROOT, load_json, write_json, fingerprint, check_seal
from gasbench.lifecycle import identity, terminate_child
from gasauto.identity import ensure_same_source, safe_path
import logging
LOGGER = logging.getLogger(__name__)
FIELDS = ['phase', 'task', 'candidate', 'parent', 'fold', 'seed', 'version', 'status', 'mape', 'g1_mape', 'gall_mape', 'wall_seconds', 'training_seconds', 'description', 'output']

def append_tsv(path, row):
    """失败指标为空，不能用0冒充最佳；每条结果立即flush/fsync。"""
    new = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter='\t', extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())

class Executor:
    """一个监督器串行执行；中断任务重跑，已完成且验签一致任务复用。"""

    def __init__(self, cache, run, cfg, run_signature, deadline):
        self.cache, self.run, self.cfg, self.signature, self.deadline = (cache, run, cfg, run_signature, deadline)
        self.active = None
        self.source = load_json(run / 'experiment.json')['identity']['source']

    def job(self, phase, task, candidate, fold, seed, allow_reserved=False):
        """保持任务ID及种子稳定；确认/测试调用由控制器顺序授权。"""
        ensure_same_source(self.source)
        reserve = self.cfg.get('export_reserve_seconds', 30.0) if allow_reserved else self.cfg['reserve_seconds']
        remaining = self.deadline - time.time() - reserve
        ident = f"{phase}__{task}__{candidate['id']}__{fold['name']}__s{seed}"
        jobdir = self.run / 'jobs' / ident
        jobdir.mkdir(parents=True, exist_ok=True)
        c = {**candidate, 'seed': seed}
        signature = fingerprint({'run': self.signature, 'phase': phase, 'task': task, 'candidate': c, 'fold': fold})
        resultfile = jobdir / 'result.json'
        if resultfile.exists():
            saved = load_json(resultfile)
            if saved['signature'] != signature:
                raise ValueError('job identity changed')
            if saved['status'] == 'completed':
                target = safe_path(self.run, saved['output'])
                check_seal(target)
                rec = load_json(target / 'receipt.json')
                if rec['signature'] != signature:
                    raise ValueError('receipt mismatch')
                return {**rec, 'output': saved['output'], 'reused': True}
        if remaining < self.cfg['min_job_seconds']:
            rec = {'status': 'budget_skipped', 'candidate': c, 'task': task, 'fold': fold, 'signature': signature, 'seconds': 0.0}
            write_json(resultfile, rec)
            append_tsv(self.run / 'results.tsv', {'phase': phase, 'task': task, 'candidate': c['id'], 'fold': fold['name'], 'seed': seed, 'version': signature, 'status': 'skipped', 'description': 'budget_reserve', 'wall_seconds': 0.0})
            return rec
        attempts = list(jobdir.glob('attempt_*'))
        out = jobdir / f'attempt_{len(attempts) + 1:03d}'
        out.mkdir()
        spec = {'candidate': c, 'task': task, 'fold': fold, 'threads': self.cfg['threads'], 'train_seconds': self.cfg['train_seconds'], 'signature': signature, 'run_signature': self.signature, 'cache': str(self.cache.resolve()), 'bank': str((self.run / 'anchor_bank').resolve()), 'directory': str(out.resolve()), 'parent_pid': os.getpid(), 'source_files': self.source}
        write_json(out / 'spec.json', spec)
        start = time.time()
        status = 'crash'
        LOGGER.info('开始：%s / %s / %s / %s / seed=%s', phase, task, c['id'], fold['name'], seed)
        with (out / 'console.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(ROOT / 'run.py'), '_worker', str((out / 'spec.json').resolve())], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.active = proc
            write_json(self.run / 'active_worker.json', {'identity': identity(proc.pid), 'job': ident, 'output': str(out.relative_to(self.run))})
            try:
                status = self._wait(proc, out, min(self.deadline - self.cfg.get('export_reserve_seconds', 30.0), start + self.cfg['job_seconds']))
            finally:
                terminate_child(proc)
                self.active = None
                write_json(self.run / 'active_worker.json', {'identity': None, 'job': None})
        rec = {'status': status, 'candidate': c, 'task': task, 'fold': fold, 'signature': signature, 'seconds': time.time() - start}
        if status == 'completed':
            try:
                check_seal(out)
                rec = load_json(out / 'receipt.json')
            except Exception:
                status = 'crash'
                rec['status'] = status
                write_json(out / 'receipt_error.json', {'traceback': traceback.format_exc()})
        rec['output'] = str(out.relative_to(self.run))
        write_json(resultfile, rec)
        metric = rec.get('metrics') or {}
        append_tsv(self.run / 'results.tsv', {'phase': phase, 'task': task, 'candidate': c['id'], 'parent': c.get('parent', 'reference_58'), 'fold': fold['name'], 'seed': seed, 'version': signature, 'status': rec['status'], 'mape': metric.get('mape'), 'g1_mape': metric.get('g1_mape'), 'gall_mape': metric.get('gall_mape'), 'wall_seconds': rec['seconds'], 'training_seconds': rec.get('training_seconds'), 'description': c.get('hypothesis', 'fixed reference'), 'output': rec['output']})
        LOGGER.info('结束：%s / %s / %s；%s，%.2f秒', task, c['id'], fold['name'], rec['status'], rec['seconds'])
        return rec

    def _wait(self, proc, out, deadline):
        while proc.poll() is None:
            if time.time() >= deadline:
                terminate_child(proc)
                return 'timeout'
            phasepath = out / 'phase.json'
            if phasepath.exists():
                p = load_json(phasepath)
                active = p.get('active_since')
                if active is not None and p['training_elapsed'] + time.time() - active > p['training_limit']:
                    terminate_child(proc)
                    return 'timeout'
            time.sleep(0.15)
        return 'completed' if proc.returncode == 0 and (out / 'receipt.json').exists() else 'crash'
