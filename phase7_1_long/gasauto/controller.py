"""私有AIC赛事研究：固定评估的短长独立有限队列与一次性确认。"""
import math
import os
import time
import numpy as np
from gasauto.protocol import read_plan, BLOCKS, index_for, truth_for
from gasauto.execution import Executor, append_tsv
from gasauto.selection import gate, choose_finalist, confirm_summary
from gasauto.delivery import export_run
from gasauto.identity import source_identity, safe_path, ensure_same_source
from gasbench.preflight import preflight
from gasbench.common import ROOT, load_json, write_json, fingerprint, environment, check_seal, logging_to, sha256
from gasbench.data import load_bundle
from gasbench.lifecycle import run_lock, identity, same_process
REFERENCE = {'id': 'reference_58', 'family': 'lightgbm', 'kind': 'reference', 'device': 'cpu'}

def load_prediction(run, record):
    """只加载完成的已验签模型任务预测，不从失败回执取数据。"""
    if record.get('status') != 'completed':
        raise ValueError('incomplete prediction')
    path = safe_path(run, record['output'])
    check_seal(path)
    with np.load(path / 'prediction.npz', allow_pickle=False) as z:
        return z['prediction']

def _decision_row(run, task, candidate, decision):
    append_tsv(run / 'results.tsv', {'phase': 'screen_decision', 'task': task, 'candidate': candidate['id'], 'parent': candidate.get('parent', 'reference_58'), 'version': fingerprint(candidate), 'status': 'keep' if decision['accepted'] else 'discard', 'mape': decision.get('mean_mape'), 'description': ','.join(decision.get('reason', [])) or 'eligible_for_confirmation'})

class Research:
    """确认结果不反馈候选队列；失败候选不掩盖为成功参照。"""

    def __init__(self, cache, run, cfg, signature, deadline):
        self.cache, self.run, self.cfg = (cache, run, cfg)
        self.executor = Executor(cache, run, cfg, signature, deadline)
        self.protocol = cfg['protocol']
        self.refs = {}
        self.screen = {t: {} for t in BLOCKS}

    def baseline(self):
        """先生成测试参照作为预算耗尽时可追溯的保底产物。"""
        test = self.protocol['test']
        for task in BLOCKS:
            record = self.executor.job('baseline', task, REFERENCE, test, 0, True)
            self.refs[task, 'test'] = record
            if record['status'] != 'completed':
                raise RuntimeError('reference baseline failed')
        for f in self.protocol['screen']:
            for task in BLOCKS:
                self.refs[task, f['name']] = self.executor.job('baseline', task, REFERENCE, f, 0, True)

    def screening(self):
        """按预登记假设轮流研究短长，父项未通过的单因素变体明确跳过。"""
        for c in self.cfg['candidates']:
            for task in c['tasks']:
                parent = c.get('parent')
                if parent and (not self.screen[task].get(parent, {}).get('decision', {}).get('accepted', False)):
                    decision = {'accepted': False, 'reason': ['parent_not_kept']}
                    records = []
                else:
                    records = [self.executor.job('screen', task, c, f, self.cfg['screen_seed']) for f in self.protocol['screen']]
                    refs = [self.refs[task, f['name']] for f in self.protocol['screen']]
                    if any((r['status'] != 'completed' for r in records + refs)):
                        decision = {'accepted': False, 'reason': ['incomplete_or_failed']}
                    elif any((r['metrics']['coverage'] < self.cfg['min_coverage'] for r in records)):
                        decision = {'accepted': False, 'reason': ['coverage']}
                    else:
                        decision = gate([r['metrics']['mape'] for r in records], [r['metrics']['mape'] for r in refs], self.cfg['screen_gate'])
                self.screen[task][c['id']] = {'records': records, 'decision': decision}
                _decision_row(self.run, task, c, decision)
                write_json(self.run / 'screening.json', self.screen)

    def freeze_finalists(self):
        """确认前锁定每任务一个候选；恢复时必须保持同一选择。"""
        picks = {t: choose_finalist(self.screen[t], self.cfg['candidates']) for t in BLOCKS}
        path = self.run / 'selection_frozen.json'
        if path.exists() and load_json(path)['finalists'] != picks:
            raise ValueError('finalists changed after confirmation freeze')
        if not path.exists():
            write_json(path, {'finalists': picks, 'screening_sha256': sha256(self.run / 'screening.json'), 'time': time.time()})
        return picks

    def confirmation(self, picks):
        """三种子在两个更晚完整黑窗复验；仅决定接受或回退，不再调参。"""
        result = {}
        bundle = load_bundle(self.cache)
        for task, c in picks.items():
            if c is None:
                result[task] = {'accepted': False, 'reason': ['no_screen_winner']}
                continue
            seed_preds = []
            ys = []
            refs = []
            records = []
            valid = True
            for f in self.protocol['confirm']:
                r = self.executor.job('confirm_reference', task, REFERENCE, f, 0)
                trials = [self.executor.job('confirm', task, c, f, s) for s in self.cfg['confirm_seeds']]
                records.extend(trials)
                if any((t['status'] != 'completed' for t in [r] + trials)):
                    valid = False
                    continue
                if any((t['metrics']['coverage'] < self.cfg['min_coverage'] for t in [r] + trials)):
                    valid = False
                    continue
                refs.append(load_prediction(self.run, r))
                seed_preds.append([load_prediction(self.run, t) for t in trials])
                ys.append(truth_for(bundle, index_for(f), task))
            result[task] = confirm_summary(seed_preds, ys, refs, self.cfg['confirm_gate']) if valid else {'accepted': False, 'reason': ['confirmation_incomplete']}
            result[task]['records'] = records
        write_json(self.run / 'confirmation.json', result)
        return result

    def finalize(self, picks, confirmed):
        """确认通过才训练最终种子模型，预算不足显式保留参照。"""
        selected = {}
        for task in BLOCKS:
            c = picks[task]
            records = []
            if c and confirmed[task]['accepted']:
                records = [self.executor.job('final', task, c, self.protocol['test'], s, True) for s in self.cfg['confirm_seeds']]
            if records and all((r['status'] == 'completed' for r in records)):
                selected[task] = {'kind': 'confirmed_seed_average', 'candidate': c['id'], 'records': records, 'reason': []}
            else:
                selected[task] = {'kind': 'reference_retained', 'candidate': 'reference_58', 'records': [self.refs[task, 'test']], 'reason': confirmed[task].get('reason', []) or ['final_fit_incomplete']}
        ensure_same_source(load_json(self.run / 'experiment.json')['identity']['source'])
        write_json(self.run / 'selection.json', selected)
        export_run(self.run, selected, {t: self.refs[t, 'test'] for t in BLOCKS}, self.protocol['test'])
        return selected

def _freeze(cache, config, run, hours, resume, extend):
    check_seal(cache)
    cfg = read_plan(config)
    source = source_identity()
    ident = {'config': cfg, 'source': source, 'cache_files': load_json(cache / 'FILES.json'), 'environment': environment()}
    sig = fingerprint(ident)
    contract = run / 'experiment.json'
    if contract.exists():
        old = load_json(contract)
        if not resume or old['signature'] != sig:
            raise ValueError('resume identity mismatch or --resume missing')
        active = load_json(run / 'active_worker.json') if (run / 'active_worker.json').exists() else {}
        if active.get('identity') and same_process(active['identity']):
            raise RuntimeError('worker still active')
        deadline = old['deadline']
        if extend:
            deadline = max(deadline, time.time()) + extend * 3600
            write_json(run / f'extension_{time.time_ns()}.json', {'hours': extend, 'old_deadline': old['deadline'], 'new_deadline': deadline})
            old['deadline'] = deadline
            write_json(contract, old)
    else:
        if resume:
            raise ValueError('恢复目录没有原实验身份，禁止把路径错误当成新实验。')
        if any((p.name != '.lock' for p in run.iterdir())):
            raise ValueError('new run directory must be empty')
        deadline = time.time() + hours * 3600
        write_json(contract, {'signature': sig, 'identity': ident, 'started': time.time(), 'deadline': deadline})
        frozen = run / 'frozen'
        for rel in source:
            target = frozen / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / rel).read_bytes())
    return (cfg, sig, deadline)

def run_research(cache, config, run, hours=3.0, resume=False, extend_hours=0.0):
    """绝对总截止包含加载与评价；额外时长必须显式授权并留账。"""
    if not math.isfinite(hours) or hours <= 0 or (not math.isfinite(extend_hours)) or (extend_hours < 0):
        raise ValueError('invalid hour budget')
    if extend_hours and (not resume):
        raise ValueError('延长预算必须显式使用--resume。')
    cache, run = (cache.resolve(), run.resolve())
    with run_lock(ROOT / '.research_lock'), run_lock(run):
        cfg, sig, deadline = _freeze(cache, config, run, hours, resume, extend_hours)
        if (run / 'summary.json').exists() and load_json(run / 'summary.json')['status'] == 'completed':
            return {**load_json(run / 'summary.json'), 'already_completed': True}
        logging_to(run / 'console.log')
        start = time.time()
        state = {'status': 'running', 'identity': identity(os.getpid()), 'started': start, 'deadline': deadline}
        write_json(run / 'run_status.json', state)
        engine = Research(cache, run, cfg, sig, deadline)
        try:
            preflight({'threads': cfg['threads'], 'candidates': cfg['candidates']}, run / 'preflight.json')
            engine.baseline()
            if (run / 'selection_frozen.json').exists():
                engine.screen = load_json(run / 'screening.json')
                if sha256(run / 'screening.json') != load_json(run / 'selection_frozen.json')['screening_sha256']:
                    raise ValueError('确认开始后筛选记录被修改。')
                picks = load_json(run / 'selection_frozen.json')['finalists']
            else:
                engine.screening()
                picks = engine.freeze_finalists()
            confirmed = engine.confirmation(picks)
            selected = engine.finalize(picks, confirmed)
            results = [load_json(p) for p in (run / 'jobs').glob('*/result.json')]
            failures = sum((r['status'] == 'crash' for r in results))
            timed = sum((r['status'] == 'timeout' for r in results))
            skipped = sum((r['status'] == 'budget_skipped' for r in results))
            state.update(status='budget_limited' if skipped else 'completed' if not failures and (not timed) else 'completed_with_failures', seconds=time.time() - start, job_records=len(results), failed_jobs=failures, timed_out_jobs=timed, skipped_jobs=skipped, selected={k: v['candidate'] for k, v in selected.items()}, budget_exhausted=time.time() >= deadline or skipped > 0, remaining_seconds=max(0.0, deadline - time.time()), elapsed_since_first_start=time.time() - load_json(run / 'experiment.json')['started'])
            write_json(run / 'summary.json', state)
            return state
        except BaseException:
            state.update(status='interrupted_or_failed', seconds=time.time() - start)
            raise
        finally:
            write_json(run / 'run_status.json', state)
