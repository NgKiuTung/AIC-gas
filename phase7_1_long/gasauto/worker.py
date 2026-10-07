"""私有AIC赛事研究：单候选/任务/窗口/种子隔离执行；未完成不得写成功回执。"""
from pathlib import Path
import resource
import shutil
import time
import numpy as np
import pandas as pd
from gasauto.anchors import AnchorBank
from gasauto.identity import ensure_same_source
from gasauto.budget import TrainingClock
from gasauto.protocol import BLOCKS, index_for, truth_for, score, segment_scores, long_component_scores
from gasbench.common import load_json, write_json, save_npz, seal, logging_to
from gasbench.data import load_bundle
from gasbench.reference import load_reference
from gasbench.worker import guard_parent

def load_predictor(path, kind, device='cpu'):
    """加载家族与元数据一致的模型；参照保持原实现。"""
    if kind == 'reference':
        return load_reference(path)
    if kind == 'sequence':
        from gasauto.neural import TaskNeural
        return TaskNeural.load(path, device)
    from gasauto.trees import TaskTree
    return TaskTree.load(path, device)

def predict(model, kind, features, index, task):
    """参照只投影成当前任务，不生成伪造长预测。"""
    if kind == 'reference':
        return model.predict(features.loc[index])[:, :BLOCKS[task]]
    return model.predict(features, index)

def execute(path: Path):
    """训练后才创建外层真值视图；评分与输入在不同阶段消费。"""
    spec = load_json(path)
    out = Path(spec['directory'])
    logging_to(out / 'worker.log')
    if spec.get('parent_pid'):
        guard_parent(spec['parent_pid'])
    if spec.get('source_files'):
        ensure_same_source(spec['source_files'])
    start = time.time()
    cpu0 = time.process_time()
    bundle = load_bundle(Path(spec['cache']))
    candidate = spec['candidate']
    task = spec['task']
    fold = spec['fold']
    cutoff = pd.Timestamp(fold['start'])
    index = index_for(fold)
    bank = AnchorBank(bundle, Path(spec['bank']), spec['run_signature'])
    clock = TrainingClock(spec['train_seconds'], out / 'phase.json')
    gpu = candidate.get('device') == 'cuda'
    if gpu:
        import torch
        torch.cuda.reset_peak_memory_stats()
    if candidate['kind'] == 'reference':
        with clock.compute():
            model, source = bank.get(cutoff)
        shutil.copytree(source, out / 'model', dirs_exist_ok=True)
    elif candidate['kind'] == 'sequence':
        from gasauto.neural import TaskNeural
        model = TaskNeural.fit(bundle, cutoff, candidate, task, spec['threads'], out / 'model', bank, clock)
    else:
        from gasauto.trees import TaskTree
        model = TaskTree.fit(bundle, cutoff, candidate, task, spec['threads'], out / 'model', bank, clock)
    clock.mark('evaluating')
    predstart = time.time()
    pred = predict(model, candidate['kind'], bundle.features, index, task)
    reloaded = load_predictor(out / 'model', candidate['kind'], candidate.get('device', 'cpu'))
    replay = predict(reloaded, candidate['kind'], bundle.features, index, task)
    np.testing.assert_allclose(replay, pred, rtol=2e-05, atol=0.0001)
    save_npz(out / 'prediction.npz', origins=index.to_numpy(dtype='datetime64[ns]'), prediction=pred)
    metrics = None
    segments = {}
    if fold['name'] != 'test':
        truth = truth_for(bundle, index, task)
        metrics = score(truth, pred)
        if task == 'long':
            metrics['components'] = long_component_scores(truth, pred)
        segments = segment_scores(truth, pred, index)
        write_json(out / 'metrics.json', metrics)
        write_json(out / 'segments.json', segments)
        rows = []
        for day in index.normalize().unique():
            take = index.normalize() == day
            rows.append({'date': str(day.date()), **score(truth[take], pred[take])})
        pd.DataFrame(rows).to_csv(out / 'by_day.csv', index=False)
    info = {'status': 'completed', 'signature': spec['signature'], 'task': task, 'candidate': candidate, 'fold': fold, 'seconds': time.time() - start, 'training_seconds': clock.elapsed, 'evaluation_seconds': time.time() - predstart, 'cpu_seconds': time.process_time() - cpu0, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'peak_vram_mib': None, 'replay_max_error': float(np.max(np.abs(replay - pred))), 'metrics': metrics, 'training': model.metadata.get('training', {}), 'test_targets_used': False, 'anchor_provenance': bank.used, 'oot_training_segments': bank.training_segments}
    if gpu:
        info['peak_vram_mib'] = float(torch.cuda.max_memory_allocated() / 1024 ** 2)
    if spec.get('source_files'):
        ensure_same_source(spec['source_files'])
    clock.mark('completed')
    write_json(out / 'summary.json', info)
    seal(out)
    write_json(out / 'receipt.json', info)
    return info
