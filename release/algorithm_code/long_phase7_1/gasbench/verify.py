"""私有AIC赛事研究：独立回放、评分、原始前缀和可选各族污染重训。"""
from __future__ import annotations
import logging
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
from gasbench.common import ROOT, sha256, check_seal, load_json, write_json, msg
from gasbench.data import load_bundle
from gasbench.reporting import read_prediction
from gasbench.factory import load_model, predict_model
from gasbench.audit import independent_score, prefix_features, poison_refit
from gasbench.fusion_audit import verify_fusion, compare_csv_prediction
from gasbench.guarded_audit import verify_guarded
from gasbench.split_submission import validate_split_pair
from gasstage.targets import future_truth
from gasstage.submission import validate_pair

def compare_reference(pred: np.ndarray, index: pd.DatetimeIndex) -> dict:
    """参照真实58分输出；列名、时间和浮点数值完整比较。"""
    from gasstage.submission import frame_for
    result = {}
    with zipfile.ZipFile(ROOT / 'reference/reference_58_results.zip') as z:
        for name, blocks in (('s_result.csv', 8), ('l_result.csv', 96)):
            member = next((n for n in z.namelist() if n.endswith('/' + name) or n == name))
            with z.open(member) as stream:
                old = pd.read_csv(stream, float_precision='round_trip')
            current = frame_for(index, pred, blocks)
            if list(current.columns) != list(old.columns) or current.datetime.tolist() != old.datetime.tolist():
                raise ValueError(msg('data.shape'))
            diff = np.max(np.abs(np.round(current.iloc[:, 1:].to_numpy(), 6) - old.iloc[:, 1:].to_numpy()))
            if diff > 1e-06:
                raise ValueError(msg('verify.failed', detail=f'reference58 {name}: {diff}'))
            result[name] = float(diff)
    return result

def verify(cache: Path, run: Path, dataset: Path, deep: bool=False, device: str='cpu') -> dict:
    """失败抛错并保留日志；未执行的GPU或污染检查必须在结果中标明。"""
    started = time.time()
    check_seal(cache)
    if load_json(run / 'run_status.json')['status'] != 'completed':
        raise ValueError(msg('verify.failed', detail='run is not completed'))
    bundle = load_bundle(cache)
    digest = sha256(dataset)
    if digest != bundle.contract['expected_dataset_sha256']:
        raise ValueError(msg('cache.invalid'))
    contract = load_json(run / 'contract.json')
    cfg = contract['config']
    out = run / 'verification'
    out.mkdir(exist_ok=True)
    prefix_times = [pd.Timestamp('2025-08-01 01:45'), pd.Timestamp('2025-10-01 01:45'), pd.Timestamp('2025-10-05 10:00')]
    prefix = []
    for stamp in prefix_times:
        f = prefix_features(bundle, stamp)
        prefix.append({'cutoff': str(stamp), 'columns': f.shape[1], 'max_error': 0.0})
        if stamp == prefix_times[-1]:
            last_prefix = f
    replays = []
    scores = []
    batch = []
    submissions = []
    ref = {}
    for path in sorted((run / 'tasks').glob('*')):
        if not (path / 'receipt.json').exists():
            continue
        receipt = load_json(path / 'receipt.json')
        if receipt['status'] != 'completed':
            continue
        check_seal(path)
        meta = load_json(path / 'metadata.json')
        c = meta['candidate']
        index, saved = read_prediction(path)
        model = load_model(path / 'model', c['kind'], device)
        pred = predict_model(model, c['kind'], bundle.features, index)
        np.testing.assert_allclose(pred, saved, rtol=2e-05, atol=0.0001)
        replays.append({'task': path.name, 'max_error': float(np.max(np.abs(pred - saved))), 'replay_device': device})
        if meta['fold']['name'] != 'test':
            truth = future_truth(bundle.truth, index, bundle.contract)
            metric = independent_score(truth, pred)
            old = load_json(path / 'metrics.json')
            error = max((abs(metric[key] - old[key]) for key in metric))
            if error > 1e-05:
                raise ValueError(msg('verify.failed', detail=f'metric {path.name}: {error}'))
            scores.append({'task': path.name, 'max_error': error})
        else:
            validate_pair(path / 'results', index)
            compare_csv_prediction(path / 'results', index, saved)
            submissions.append(c['id'])
            points = index[[0, 1, 2, 100, 400]]
            a = predict_model(model, c['kind'], bundle.features, points)
            b = np.concatenate([predict_model(model, c['kind'], bundle.features, points[:2]), predict_model(model, c['kind'], bundle.features, points[2:])])
            np.testing.assert_allclose(a, b, rtol=2e-05, atol=0.0001)
            upto = points[points <= prefix_times[-1]]
            partial = predict_model(model, c['kind'], last_prefix, upto)
            whole = predict_model(model, c['kind'], bundle.features, upto)
            np.testing.assert_allclose(partial, whole, rtol=2e-05, atol=0.0001)
            batch.append({'candidate': c['id'], 'batch_max_error': float(np.max(np.abs(a - b))), 'prefix_prediction_max_error': float(np.max(np.abs(partial - whole)))})
            if c['id'] == 'reference_58':
                ref = compare_reference(pred, index)
        del model
    poisoned = []
    if deep:
        for c in cfg['candidates']:
            dest = out / 'target_poison' / c['id']
            if dest.exists():
                dest = out / 'target_poison' / f"{c['id']}_{time.time_ns()}"
            poisoned.append(poison_refit(bundle, pd.Timestamp('2025-09-01'), c, dest))
    test_index = pd.date_range('2025-10-01', '2025-10-10 23:45', freq='15min')
    for name in ('forward_ensemble',):
        path = run / 'candidate_results' / name
        if path.exists():
            validate_pair(path, test_index)
            submissions.append(name)
    for path in sorted((run / 'candidate_results').glob('refshort__*__long')):
        validate_split_pair(path, test_index)
        submissions.append(path.name)
    guarded = verify_guarded(bundle, run)
    if (run / 'candidate_results/guarded_long_hybrid').exists():
        submissions.append('guarded_long_hybrid')
    write_json(out / 'guarded.json', guarded)
    fusion = verify_fusion(bundle, run, cfg)
    write_json(out / 'fusion.json', fusion)
    if sha256(dataset) != digest:
        raise ValueError(msg('file.changed', path=dataset))
    result = {'status': 'passed', 'seconds': time.time() - started, 'original_source_unchanged': True, 'run_status': load_json(run / 'run_status.json')['status'], 'model_replays': len(replays), 'replay_max_error': max([x['max_error'] for x in replays], default=0), 'score_recomputations': len(scores), 'score_max_error': max([x['max_error'] for x in scores], default=0), 'raw_prefix_checks': len(prefix), 'test_model_batch_checks': len(batch), 'reference58_csv_max_errors': ref, 'poison_refits': poisoned if deep else 'NOT_RUN; use --deep', 'submission_pairs': len(submissions), 'fusion_replays': len(fusion), 'fusion_max_error': max([x['max_error'] for x in fusion], default=0), 'guarded_replays': len(guarded), 'guarded_max_error': max([x['max_error'] for x in guarded], default=0), 'device': device, 'test_targets_used': False}
    write_json(out / 'replays.json', replays)
    write_json(out / 'scores.json', scores)
    write_json(out / 'prefix.json', prefix)
    write_json(out / 'batches.json', batch)
    write_json(out / 'summary.json', result)
    logging.getLogger(__name__).info(msg('verify.done', replays=len(replays), scores=len(scores), submissions=len(submissions)))
    return result
