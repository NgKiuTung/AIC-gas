"""私有AIC赛事研究：已执行任务、确认来源和短长结果的独立验收。"""
from pathlib import Path
import io
import time
import zipfile
import numpy as np
import pandas as pd
from gasauto.protocol import index_for, truth_for, LONG_BANDS
from gasauto.worker import load_predictor, predict
from gasauto.identity import safe_path, ensure_same_source
from gasauto.audit import independent_score, poison_checks, raw_prefixes
from gasauto.delivery import merged_prediction, task_frame, validate_long_file
from gasbench.common import ROOT, sha256, load_json, write_json, check_seal, logging_to
from gasbench.data import load_bundle
from gasbench.split_submission import validate_split_pair

def _reference_csv(run):
    """固定参照与原58分归档逐数值比较，不将新预测冒称原参照。"""
    report = {}
    with zipfile.ZipFile(ROOT / 'reference/reference_58_results.zip') as archive:
        for name in ('s_result.csv', 'l_result.csv'):
            choices = [s for s in archive.namelist() if Path(s).name == name]
            if len(choices) != 1:
                raise ValueError('原参照归档不唯一。')
            a = pd.read_csv(io.BytesIO(archive.read(choices[0])), float_precision='round_trip')
            b = pd.read_csv(run / 'candidate_results/reference_58' / name, float_precision='round_trip')
            if list(a.columns) != list(b.columns) or a.datetime.tolist() != b.datetime.tolist():
                raise ValueError('参照格式变化。')
            np.testing.assert_array_equal(a.iloc[:, 1:].to_numpy(), b.iloc[:, 1:].to_numpy())
            report[name] = float(np.max(np.abs(a.iloc[:, 1:].to_numpy() - b.iloc[:, 1:].to_numpy())))
    return report

def _selected_csv(run, protocol):
    """Rebuild selected outputs. Phase7 may freeze an external Short incumbent and stitch Long components."""
    selected = load_json(run / 'selection.json')
    idx = index_for(protocol['test'])
    # Short may be intentionally absent while Long research proceeds.
    short_available = selected['short']['kind'] != 'unavailable_external_phase6'
    if selected['short']['kind'] == 'external_phase6':
        expected_short = pd.read_csv(run / 'frozen_short/s_result.csv', float_precision='round_trip')
        actual_short = pd.read_csv(run / 'candidate_results/selected/s_result.csv', float_precision='round_trip')
        if list(expected_short.columns) != list(actual_short.columns) or expected_short.datetime.tolist() != actual_short.datetime.tolist():
            raise ValueError('external short format changed')
        np.testing.assert_allclose(expected_short.iloc[:,1:].to_numpy(), actual_short.iloc[:,1:].to_numpy(), rtol=0, atol=5.00001e-7)
    elif short_available:
        item = selected['short']
        p = merged_prediction(run, item['records'])
        expected = task_frame(idx, p, 8)
        actual_short = pd.read_csv(run / 'candidate_results/selected/s_result.csv', float_precision='round_trip')
        np.testing.assert_allclose(expected.iloc[:,1:].to_numpy(), actual_short.iloc[:,1:].to_numpy(), rtol=0, atol=5.00001e-7)

    # Long
    item = selected['long']
    if item['kind'] == 'component_stitch':
        reference = merged_prediction(run, [item['reference_record']])
        rebuilt = reference.copy()
        for component, cid in item.get('accepted_components', {}).items():
            records = item['final_records'][cid]
            pred = merged_prediction(run, records)
            target_name, band = component.split('__',1)
            target = 0 if target_name == 'g1' else 1
            a,b = LONG_BANDS[band]
            rebuilt[:,a:b,target] = pred[:,a:b,target]
        rebuilt[:,:,0] = np.clip(rebuilt[:,:,0],0,None)
        rebuilt[:,:,1] = np.maximum(np.clip(rebuilt[:,:,1],0,None),rebuilt[:,:,0])
        expected_long = task_frame(idx, rebuilt, 96)
    else:
        records = item['records']
        if item['kind'] == 'confirmed_seed_average':
            seeds=[r['candidate']['seed'] for r in records]
            if len(seeds)<3 or len(set(seeds))!=len(seeds):
                raise ValueError('final seed provenance invalid')
        expected_long=task_frame(idx, merged_prediction(run,records),96)
    actual_long = pd.read_csv(run / 'candidate_results/selected/l_result.csv', float_precision='round_trip')
    np.testing.assert_allclose(expected_long.iloc[:,1:].to_numpy(), actual_long.iloc[:,1:].to_numpy(), rtol=0, atol=5.00001e-7)
    if short_available:
        return validate_split_pair(run / 'candidate_results/selected', idx)
    return validate_long_file(run / 'candidate_results/selected/l_result.csv', idx)

def _replay_job(bundle, run, record, device, prefixes):
    path = safe_path(run, record['output'])
    check_seal(path)
    spec = load_json(path / 'spec.json')
    c = spec['candidate']
    task = spec['task']
    fold = spec['fold']
    if record['signature'] != spec['signature']:
        raise ValueError('任务身份不一致。')
    for segment in record.get('oot_training_segments', []):
        if pd.Timestamp(segment['anchor_cutoff']) > pd.Timestamp(segment['first_origin']):
            raise ValueError('训练锚点来自输入起点之后。')
    idx = index_for(fold)
    with np.load(path / 'prediction.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['origins'], idx.to_numpy(dtype='datetime64[ns]'))
        stored = data['prediction']
    model = load_predictor(path / 'model', c['kind'], device if c['kind'] == 'sequence' else 'cpu')
    replay = predict(model, c['kind'], bundle.features, idx, task)
    np.testing.assert_allclose(replay, stored, rtol=2e-05, atol=0.0001)
    scored = None
    if fold['name'] != 'test':
        scored = independent_score(truth_for(bundle, idx, task), stored)
        for key in scored:
            np.testing.assert_allclose(scored[key], record['metrics'][key], rtol=0, atol=1e-12)
    prefix_error = 0.0
    batch_error = 0.0
    if fold['name'] == 'test':
        for stamp, features in prefixes.items():
            index = pd.DatetimeIndex([stamp])
            a = predict(model, c['kind'], features, index, task)
            b = predict(model, c['kind'], bundle.features, index, task)
            np.testing.assert_allclose(a, b, rtol=2e-05, atol=0.0001)
            prefix_error = max(prefix_error, float(np.max(np.abs(a - b))))
        pieces = [predict(model, c['kind'], bundle.features, idx[:7], task), predict(model, c['kind'], bundle.features, idx[7:], task)]
        merged = np.concatenate(pieces)
        np.testing.assert_allclose(merged, replay, rtol=2e-05, atol=0.0001)
        batch_error = float(np.max(np.abs(merged - replay)))
    return {'task': task, 'candidate': c['id'], 'fold': fold['name'], 'replay_max_error': float(np.max(np.abs(replay - stored))), 'score': scored, 'prefix_max_error': prefix_error, 'batch_max_error': batch_error, 'output': record['output']}

def verify(cache: Path, run: Path, dataset: Path, device='cpu', deep=False):
    """容差固定；跨CPU/GPU超差时停止，不扩大阈值换取成功。"""
    start = time.time()
    contract = load_json(run / 'experiment.json')
    ensure_same_source(contract['identity']['source'])
    check_seal(cache)
    if load_json(cache / 'FILES.json') != contract['identity']['cache_files']:
        raise ValueError('输入缓存变化。')
    before = sha256(dataset)
    if before != load_json(cache / 'ready.json')['dataset_sha256']:
        raise ValueError('原始数据身份变化。')
    cfg = contract['identity']['config']
    bundle = load_bundle(cache)
    folder = run / 'verification'
    folder.mkdir(exist_ok=True)
    logging_to(folder / 'console.log')
    prefixes = raw_prefixes(bundle, folder)
    records = []
    for file in sorted((run / 'jobs').glob('*/result.json')):
        rec = load_json(file)
        if rec['status'] == 'completed':
            records.append(_replay_job(bundle, run, rec, device, prefixes))
    write_json(folder / 'replays.json', records)
    selected = _selected_csv(run, cfg['protocol'])
    reference = _reference_csv(run)
    poison = []
    if deep:
        ref_tasks = ['long'] if cfg.get('mode') == 'phase7_long_component_focus' else ['short','long']
        initial = [{'id': 'reference_58', 'kind': 'reference', 'family': 'lightgbm', 'tasks': ref_tasks}, *cfg['candidates']]
        poison = poison_checks(bundle, initial, folder / 'poison')
    if sha256(dataset) != before:
        raise ValueError('验收改动了原始数据。')
    report = {'status': 'passed', 'seconds': time.time() - start, 'source_unchanged': True, 'model_replays': len(records), 'scores_recomputed': sum((r['score'] is not None for r in records)), 'replay_max_error': max((r['replay_max_error'] for r in records), default=0), 'prefix_checks': len(prefixes), 'batch_checks': sum((r['fold'] == 'test' for r in records)), 'reference58_csv_max_errors': reference, 'selected_pair': selected, 'poison_refits': poison, 'replay_device': device, 'test_targets_used': False}
    write_json(folder / 'summary.json', report)
    return report
