"""Phase7: freeze official-best Phase6 short output and autoresearch only Long components."""
from __future__ import annotations
from pathlib import Path
import io
import json
import math
import os
import time
import zipfile

import numpy as np
import pandas as pd

from gasauto.execution import Executor, append_tsv
from gasauto.identity import source_identity, ensure_same_source, safe_path
from gasauto.protocol import index_for, truth_for, long_component_scores, LONG_BANDS
from gasauto.selection import gate
from gasauto.delivery import task_frame, merged_prediction, pack, validate_long_file
from gasbench.common import ROOT, load_json, write_json, fingerprint, environment, check_seal, sha256, logging_to
from gasbench.data import load_bundle
from gasbench.lifecycle import run_lock, identity, same_process
from gasbench.preflight import preflight
from gasbench.split_submission import validate_split_pair

REFERENCE = {'id': 'reference_58', 'family': 'lightgbm', 'kind': 'reference', 'device': 'cpu'}
TARGET_NAMES = ('g1', 'gall')


def _read_short_source(source: Path) -> pd.DataFrame:
    """Accept a Phase6 s_result.csv or a results_only.zip and validate exact official shape."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as z:
            choices = [n for n in z.namelist() if Path(n).name == 's_result.csv']
            if len(choices) != 1:
                raise ValueError('short source ZIP must contain exactly one s_result.csv')
            frame = pd.read_csv(io.BytesIO(z.read(choices[0])), float_precision='round_trip')
    else:
        frame = pd.read_csv(source, float_precision='round_trip')
    expected_index = pd.date_range('2025-10-01 00:00:00', '2025-10-10 23:45:00', freq='15min')
    if len(frame) != 960 or frame.shape[1] != 17 or frame.columns[0] != 'datetime':
        raise ValueError('Phase6 short source must be 960x17 including datetime')
    parsed = pd.to_datetime(frame['datetime'], format='%Y-%m-%d %H:%M:%S')
    if not pd.DatetimeIndex(parsed).equals(expected_index):
        raise ValueError('Phase6 short source origins mismatch')
    values = frame.iloc[:, 1:].to_numpy(dtype='float64')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Phase6 short source has invalid values')
    g1, gall = values[:, :8], values[:, 8:]
    if (gall < g1).any():
        raise ValueError('Phase6 short source violates generator hierarchy')
    return frame


def _freeze_short(run: Path, source: Path) -> dict:
    """Copy only validated short predictions into run identity; later steps never depend on external path."""
    frame = _read_short_source(source)
    folder = run / 'frozen_short'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 's_result.csv'
    frame.to_csv(target, index=False, encoding='utf-8', float_format='%.6f')
    meta = {
        'source_path_at_start': str(source.resolve()),
        'source_sha256': sha256(source.resolve()),
        'frozen_csv_sha256': sha256(target),
        'official_feedback': {
            'short_accuracy_percent': 90.48,
            'g1_accuracy_percent': 87.17,
            'gall_accuracy_percent': 93.78,
            'short_score_50': 30.9525,
            'total_of_original_phase6_submission': 59.0888,
        },
        'role': 'frozen_short_incumbent_only; long is researched independently',
    }
    write_json(folder / 'source.json', meta)
    return meta


def _freeze_missing_short(run: Path) -> dict:
    """Record explicit Long-only mode without fabricating a Phase6 Short file."""
    folder = run / 'frozen_short'
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        'status': 'unavailable',
        'source_path_at_start': None,
        'source_sha256': None,
        'frozen_csv_sha256': None,
        'official_feedback': {
            'short_accuracy_percent': 90.48,
            'g1_accuracy_percent': 87.17,
            'gall_accuracy_percent': 93.78,
            'short_score_50': 30.9525,
            'total_of_original_phase6_submission': 59.0888,
        },
        'role': 'long_only_research; external Phase6 Short is not attached',
    }
    write_json(folder / 'source.json', meta)
    return meta


def _component_arrays(record, run, bundle, task='long'):
    if record.get('status') != 'completed':
        raise ValueError('incomplete component record')
    path = safe_path(run, record['output'])
    check_seal(path)
    with np.load(path / 'prediction.npz', allow_pickle=False) as z:
        pred = z['prediction']
    truth = truth_for(bundle, index_for(record['fold']), task)
    return pred, truth


def _component_metric(pred, truth, component):
    target_name, band = component.split('__', 1)
    target = 0 if target_name == 'g1' else 1
    start, end = LONG_BANDS[band]
    y = truth[:, start:end, target]
    p = pred[:, start:end, target]
    valid = np.isfinite(y) & np.isfinite(p) & (y > 0)
    if not valid.any():
        raise ValueError('empty component metric')
    return float(np.mean(np.abs(p[valid] - y[valid]) / y[valid]))


def _all_components():
    return [f'{target}__{band}' for target in TARGET_NAMES for band in LONG_BANDS]


def _component_gate(values, refs, rule):
    """Reuse task gate thresholds on target×horizon-band MAPE, plus minimum fold wins."""
    result = gate(values, refs, rule)
    gains = np.asarray(refs) - np.asarray(values)
    wins_required = int(rule.get('min_wins', 2))
    result['wins'] = int((gains > 0).sum())
    if result['wins'] < wins_required:
        result['accepted'] = False
        result['reason'].append('wins')
    return result


def _freeze_identity(cache, config, run, short_source, hours, resume, extend, long_only=False):
    """Freeze source/cache identity; Long-only mode is explicit and never invents Short predictions."""
    check_seal(cache)
    cfg = load_json(config)
    from gasauto.protocol import validate_plan
    cfg = validate_plan(cfg)
    if cfg.get('mode') != 'phase7_long_component_focus':
        raise ValueError('use Phase7 long-focus config')
    source = source_identity()
    contract = run / 'experiment.json'
    if contract.exists():
        old = load_json(contract)
        if not resume:
            raise ValueError('existing run requires --resume')
        frozen = load_json(run / 'frozen_short/source.json')
        if frozen.get('status') == 'unavailable':
            if short_source is not None or not long_only:
                raise ValueError('Long-only run identity cannot attach or replace Short during resume')
        else:
            if short_source is None:
                raise ValueError('resume requires the same Phase6 short source')
            frame = _read_short_source(short_source)
            tmp_hash = fingerprint(frame.iloc[:, 1:].round(6).to_numpy().tolist())
            frozen_frame = pd.read_csv(run / 'frozen_short/s_result.csv')
            frozen_hash = fingerprint(frozen_frame.iloc[:, 1:].round(6).to_numpy().tolist())
            if tmp_hash != frozen_hash:
                raise ValueError('short incumbent changed on resume')
        ident = {'config': cfg, 'source': source, 'cache_files': load_json(cache / 'FILES.json'),
                 'environment': environment(), 'short_incumbent': frozen}
        sig = fingerprint(ident)
        if old['signature'] != sig:
            raise ValueError('resume identity mismatch')
        active = load_json(run / 'active_worker.json') if (run / 'active_worker.json').exists() else {}
        if active.get('identity') and same_process(active['identity']):
            raise RuntimeError('worker still active')
        deadline = old['deadline']
        if extend:
            deadline = max(deadline, time.time()) + extend * 3600
            write_json(run / f'extension_{time.time_ns()}.json',
                       {'hours': extend, 'old_deadline': old['deadline'], 'new_deadline': deadline})
            old['deadline'] = deadline
            write_json(contract, old)
        return cfg, sig, deadline
    if resume:
        raise ValueError('resume directory has no experiment identity')
    if run.exists() and any(x.name != '.lock' for x in run.iterdir()):
        raise ValueError('new Phase7 run directory must be empty except runtime lock')
    run.mkdir(parents=True, exist_ok=True)
    if short_source is None:
        if not long_only:
            raise ValueError('Phase7 requires --short-source unless --long-only is explicit')
        short_meta = _freeze_missing_short(run)
    else:
        short_meta = _freeze_short(run, short_source)
    ident = {'config': cfg, 'source': source, 'cache_files': load_json(cache / 'FILES.json'),
             'environment': environment(), 'short_incumbent': short_meta}
    sig = fingerprint(ident)
    deadline = time.time() + hours * 3600
    write_json(contract, {'signature': sig, 'identity': ident, 'started': time.time(), 'deadline': deadline})
    frozen = run / 'frozen'
    for rel in source:
        target = frozen / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / rel).read_bytes())
    return cfg, sig, deadline


class Phase7Research:
    """Long-only model search with guarded target×horizon-band confirmation."""

    def __init__(self, cache, run, cfg, signature, deadline):
        self.cache, self.run, self.cfg = cache, run, cfg
        self.executor = Executor(cache, run, cfg, signature, deadline)
        self.protocol = cfg['protocol']
        self.refs = {}
        self.screen_records = {}

    def baseline(self):
        for fold in [*self.protocol['screen'], *self.protocol['confirm'], self.protocol['test']]:
            rec = self.executor.job('baseline', 'long', REFERENCE, fold, 0, fold['name'] == 'test')
            if rec['status'] != 'completed':
                raise RuntimeError('reference long baseline failed')
            self.refs[fold['name']] = rec

    def screening(self):
        bundle = load_bundle(self.cache)
        decisions = {}
        components = _all_components()
        for candidate in self.cfg['candidates']:
            if candidate.get('tasks') != ['long']:
                raise ValueError('Phase7 candidates must be long-only')
            records = [self.executor.job('screen', 'long', candidate, fold, self.cfg['screen_seed'])
                       for fold in self.protocol['screen']]
            self.screen_records[candidate['id']] = records
            if any(r['status'] != 'completed' for r in records):
                decisions[candidate['id']] = {'status': 'discard', 'reason': ['incomplete_or_failed']}
                continue
            candidate_decision = {'status': 'complete', 'components': {}}
            for comp in components:
                vals, refs = [], []
                for fold, rec in zip(self.protocol['screen'], records):
                    p, y = _component_arrays(rec, self.run, bundle)
                    rp, ry = _component_arrays(self.refs[fold['name']], self.run, bundle)
                    vals.append(_component_metric(p, y, comp))
                    refs.append(_component_metric(rp, ry, comp))
                candidate_decision['components'][comp] = _component_gate(vals, refs, self.cfg['component_screen_gate'])
            decisions[candidate['id']] = candidate_decision
            append_tsv(self.run / 'results.tsv', {
                'phase': 'screen_decision', 'task': 'long', 'candidate': candidate['id'],
                'version': fingerprint(candidate), 'status': 'keep' if any(
                    x['accepted'] for x in candidate_decision['components'].values()) else 'discard',
                'description': 'phase7 component screening',
            })
        write_json(self.run / 'screening.json', decisions)
        return decisions

    def freeze_components(self, decisions):
        picks = {}
        for comp in _all_components():
            eligible = []
            for c in self.cfg['candidates']:
                d = decisions.get(c['id'], {}).get('components', {}).get(comp)
                if d and d.get('accepted'):
                    eligible.append((d['mean_mape'], c['id']))
            picks[comp] = min(eligible)[1] if eligible else 'reference_58'
        path = self.run / 'selection_frozen.json'
        value = {'components': picks, 'screening_sha256': sha256(self.run / 'screening.json'), 'time': time.time()}
        if path.exists() and load_json(path) != value:
            # time differs, compare immutable fields only
            old = load_json(path)
            if old['components'] != picks or old['screening_sha256'] != value['screening_sha256']:
                raise ValueError('Phase7 component selection changed after freeze')
            return old['components']
        if not path.exists():
            write_json(path, value)
        return picks

    def confirmation(self, picks):
        bundle = load_bundle(self.cache)
        by_id = {c['id']: c for c in self.cfg['candidates']}
        needed = sorted(set(picks.values()) - {'reference_58'})
        all_records = {}
        for cid in needed:
            c = by_id[cid]
            all_records[cid] = {}
            for fold in self.protocol['confirm']:
                all_records[cid][fold['name']] = [
                    self.executor.job('confirm', 'long', c, fold, seed)
                    for seed in self.cfg['confirm_seeds']
                ]
        out = {}
        for comp, cid in picks.items():
            if cid == 'reference_58':
                out[comp] = {'accepted': False, 'candidate': cid, 'reason': ['reference_from_screen']}
                continue
            vals, refs, seed_values = [], [], []
            valid = True
            for fold in self.protocol['confirm']:
                trials = all_records[cid][fold['name']]
                ref = self.refs[fold['name']]
                if any(r['status'] != 'completed' for r in trials):
                    valid = False
                    break
                preds = []
                per_seed = []
                truth = None
                for trial in trials:
                    p, y = _component_arrays(trial, self.run, bundle)
                    preds.append(p)
                    truth = y
                    per_seed.append(_component_metric(p, y, comp))
                rp, ry = _component_arrays(ref, self.run, bundle)
                avg = np.mean(preds, axis=0)
                vals.append(_component_metric(avg, truth, comp))
                refs.append(_component_metric(rp, ry, comp))
                seed_values.append(per_seed)
            if not valid:
                out[comp] = {'accepted': False, 'candidate': cid, 'reason': ['confirmation_incomplete']}
                continue
            decision = _component_gate(vals, refs, self.cfg['component_confirm_gate'])
            seed_std = float(np.mean(np.std(np.asarray(seed_values), axis=1)))
            decision.update(candidate=cid, seed_mape_by_fold=seed_values, seed_std_mean=seed_std)
            if seed_std > self.cfg['component_confirm_gate']['max_seed_std']:
                decision['accepted'] = False
                decision['reason'].append('seed_variability')
            out[comp] = decision
        write_json(self.run / 'confirmation.json', {'components': out, 'records': all_records})
        return out

    def finalize(self, picks, confirmed):
        by_id = {c['id']: c for c in self.cfg['candidates']}
        accepted = {comp: cid for comp, cid in picks.items()
                    if cid != 'reference_58' and confirmed.get(comp, {}).get('accepted')}
        needed = sorted(set(accepted.values()))
        final_records = {}
        for cid in needed:
            final_records[cid] = [
                self.executor.job('final', 'long', by_id[cid], self.protocol['test'], seed, True)
                for seed in self.cfg['confirm_seeds']
            ]
            if any(r['status'] != 'completed' for r in final_records[cid]):
                for comp in list(accepted):
                    if accepted[comp] == cid:
                        del accepted[comp]
        reference = merged_prediction(self.run, [self.refs['test']])
        long_pred = reference.copy()
        for comp, cid in accepted.items():
            pred = merged_prediction(self.run, final_records[cid])
            target_name, band = comp.split('__', 1)
            target = 0 if target_name == 'g1' else 1
            start, end = LONG_BANDS[band]
            long_pred[:, start:end, target] = pred[:, start:end, target]
        # Preserve physical hierarchy after component stitching; g1 remains unchanged.
        long_pred[:, :, 0] = np.clip(long_pred[:, :, 0], 0, None)
        long_pred[:, :, 1] = np.maximum(np.clip(long_pred[:, :, 1], 0, None), long_pred[:, :, 0])
        short_meta = load_json(self.run / 'frozen_short/source.json')
        short_kind = 'external_phase6' if short_meta.get('status') != 'unavailable' else 'unavailable_external_phase6'
        selected = {
            'short': {'kind': short_kind, 'candidate': 'phase6_xgboost_process_proxy',
                      'source': short_meta},
            'long': {'kind': 'component_stitch', 'candidate': 'phase7_long_components',
                     'accepted_components': accepted, 'screen_picks': picks,
                     'confirmation': confirmed, 'final_records': final_records,
                     'reference_record': self.refs['test']},
        }
        write_json(self.run / 'selection.json', selected)
        self._export(long_pred, reference, selected)
        return selected

    def _export(self, long_pred, reference_long, provenance):
        """Export a submission pair only when the external Phase6 Short file is actually present."""
        test_index = index_for(self.protocol['test'])
        short_path = self.run / 'frozen_short/s_result.csv'
        selected_dir = self.run / 'candidate_results/selected'
        selected_dir.mkdir(parents=True, exist_ok=True)
        task_frame(test_index, long_pred, 96).to_csv(
            selected_dir / 'l_result.csv', index=False, encoding='utf-8', float_format='%.6f')

        if short_path.exists():
            short = pd.read_csv(short_path, float_precision='round_trip')
            short.to_csv(selected_dir / 's_result.csv', index=False, encoding='utf-8', float_format='%.6f')
            report = validate_split_pair(selected_dir, test_index)
            report.update(provenance=provenance, submission_pair_ready=True)
            write_json(selected_dir / 'validation.json', report)
            with zipfile.ZipFile(selected_dir / 'results_only.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(selected_dir / 's_result.csv', 'submissions/s_result.csv')
                z.write(selected_dir / 'l_result.csv', 'submissions/l_result.csv')

            # Known component baseline is valid only with the exact external Phase6 Short file.
            base_dir = self.run / 'candidate_results/phase6short_reference58long'
            base_dir.mkdir(parents=True, exist_ok=True)
            short.to_csv(base_dir / 's_result.csv', index=False, encoding='utf-8', float_format='%.6f')
            task_frame(test_index, reference_long, 96).to_csv(
                base_dir / 'l_result.csv', index=False, encoding='utf-8', float_format='%.6f')
            base_report = validate_split_pair(base_dir, test_index)
            base_report['official_component_scores'] = {
                'short_50': 30.9525, 'reference58_long_50': 28.7356,
                'sum_if_scored_independently': 59.6881}
            write_json(base_dir / 'validation.json', base_report)
            with zipfile.ZipFile(base_dir / 'results_only.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(base_dir / 's_result.csv', 'submissions/s_result.csv')
                z.write(base_dir / 'l_result.csv', 'submissions/l_result.csv')
        else:
            report = validate_long_file(selected_dir / 'l_result.csv', test_index)
            report.update(
                provenance=provenance,
                reason='external Phase6 Short is unavailable; this artifact is Long research only',
            )
            write_json(selected_dir / 'validation.json', report)
            with zipfile.ZipFile(selected_dir / 'long_only_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(selected_dir / 'l_result.csv', 'submissions/l_result.csv')

        # Preserve the exact original 58.1578 reference pair for identity verification.
        ref_dir = self.run / 'candidate_results/reference_58'
        ref_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ROOT / 'reference/reference_58_results.zip') as archive:
            for name in ('s_result.csv', 'l_result.csv'):
                choices = [n for n in archive.namelist() if Path(n).name == name]
                if len(choices) != 1:
                    raise ValueError('reference_58 archive schema changed')
                (ref_dir / name).write_bytes(archive.read(choices[0]))
        validate_split_pair(ref_dir, test_index)
        with zipfile.ZipFile(ref_dir / 'results_only.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(ref_dir / 's_result.csv', 'submissions/s_result.csv')
            z.write(ref_dir / 'l_result.csv', 'submissions/l_result.csv')



def run_phase7(cache: Path, config: Path, run: Path, short_source: Path | None,
               hours=3.0, resume=False, extend_hours=0.0, long_only=False):
    if not math.isfinite(hours) or hours <= 0 or not math.isfinite(extend_hours) or extend_hours < 0:
        raise ValueError('invalid Phase7 budget')
    if extend_hours and not resume:
        raise ValueError('budget extension requires --resume')
    cache, run = cache.resolve(), run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with run_lock(ROOT / '.research_lock'), run_lock(run):
        cfg, sig, deadline = _freeze_identity(cache, config, run, short_source, hours, resume, extend_hours, long_only)
        if (run / 'summary.json').exists() and load_json(run / 'summary.json').get('status') == 'completed':
            return {**load_json(run / 'summary.json'), 'already_completed': True}
        logging_to(run / 'console.log')
        start = time.time()
        state = {'status': 'running', 'identity': identity(os.getpid()), 'started': start, 'deadline': deadline}
        write_json(run / 'run_status.json', state)
        engine = Phase7Research(cache, run, cfg, sig, deadline)
        try:
            preflight({'threads': cfg['threads'], 'candidates': cfg['candidates']}, run / 'preflight.json')
            engine.baseline()
            if (run / 'selection_frozen.json').exists():
                decisions = load_json(run / 'screening.json')
                frozen = load_json(run / 'selection_frozen.json')
                if sha256(run / 'screening.json') != frozen['screening_sha256']:
                    raise ValueError('screening changed after freeze')
                picks = frozen['components']
            else:
                decisions = engine.screening()
                picks = engine.freeze_components(decisions)
            confirmed = engine.confirmation(picks)
            selected = engine.finalize(picks, confirmed)
            results = [load_json(p) for p in (run / 'jobs').glob('*/result.json')]
            failures = sum(r['status'] == 'crash' for r in results)
            timed = sum(r['status'] == 'timeout' for r in results)
            skipped = sum(r['status'] == 'budget_skipped' for r in results)
            state.update(
                status='budget_limited' if skipped else 'completed' if not failures and not timed else 'completed_with_failures',
                seconds=time.time() - start,
                job_records=len(results),
                failed_jobs=failures,
                timed_out_jobs=timed,
                skipped_jobs=skipped,
                selected_components=selected['long']['accepted_components'],
                short_incumbent='phase6_xgboost_process_proxy' if (run / 'frozen_short/s_result.csv').exists() else 'unavailable_external_phase6',
                combined_submission_ready=(run / 'frozen_short/s_result.csv').exists(),
                budget_exhausted=time.time() >= deadline or skipped > 0,
                remaining_seconds=max(0.0, deadline - time.time()),
            )
            write_json(run / 'summary.json', state)
            return state
        except BaseException:
            state.update(status='interrupted_or_failed', seconds=time.time() - start)
            raise
        finally:
            write_json(run / 'run_status.json', state)
