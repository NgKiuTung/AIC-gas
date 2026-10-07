"""私有AIC赛事研究：单候选/单时间折进程，训练、保存、评分和模型重放。"""
from __future__ import annotations
import ctypes
import logging
import os
from pathlib import Path
import signal
import time
import numpy as np
import pandas as pd
from gasbench.common import load_json, write_json, save_npz, seal, msg, logging_to
from gasbench.data import load_bundle
from gasbench.factory import fit_model, load_model, predict_model
from gasstage.baselines import frozen_week
from gasstage.targets import history_only, future_truth
from gasstage.metrics import per_horizon
from gasbench.competition import add_platform_proxy
from gasstage.submission import export_pair
LOGGER = logging.getLogger(__name__)

def guard_parent(pid: int) -> None:
    """Linux父进程死亡时只终止自身，避免关终端后遗留GPU worker。"""
    if os.name == 'posix' and Path('/proc').exists():
        libc = ctypes.CDLL(None)
        if libc.prctl(1, signal.SIGTERM) != 0:
            raise OSError('PR_SET_PDEATHSIG')
        if os.getppid() != pid:
            raise RuntimeError(msg('job.orphan', pid=pid))

def run_task(spec_path: Path) -> dict:
    """外层无提前注入验证标签，训练结束后评分独立读取future_truth。"""
    spec = load_json(spec_path)
    path = Path(spec['output'])
    logging_to(path / 'worker.log')
    guard_parent(spec['parent_pid'])
    started = time.time()
    cfg, fold = (spec['candidate'], spec['fold'])
    bundle = load_bundle(Path(spec['cache']))
    cutoff = pd.Timestamp(fold['start'])
    index = pd.date_range(fold['start'], fold['end'], freq='15min')
    LOGGER.info(msg('job.start', fold=fold['name'], candidate=cfg['id']))
    fit_started = time.perf_counter()
    model = fit_model(bundle, cutoff, cfg, spec['threads'], path / 'model', spec['deadline'])
    fit_seconds = time.perf_counter() - fit_started
    prediction_started = time.perf_counter()
    pred = predict_model(model, cfg['kind'], bundle.features, index)
    prediction_seconds = time.perf_counter() - prediction_started
    save_npz(path / 'predictions.npz', origins=index.to_numpy(dtype='datetime64[ns]'), prediction=pred)
    reloaded = load_model(path / 'model', cfg['kind'], cfg.get('device', 'cpu'))
    replay = predict_model(reloaded, cfg['kind'], bundle.features, index)
    np.testing.assert_allclose(replay, pred, rtol=2e-05, atol=0.0001)
    metadata = {'candidate': cfg, 'fold': fold, 'seconds': time.time() - started, 'replay_max_error': float(np.max(np.abs(replay - pred))), 'fit_seconds': fit_seconds, 'predict_seconds': prediction_seconds, 'predict_seconds_per_origin': prediction_seconds / len(index), 'training': model.metadata.get('training', {}), 'hidden_evaluation_target_history_used': False}
    if fold['name'] != 'test':
        truth = future_truth(bundle.truth, index, bundle.contract)
        from gasstage.metrics import score as base_score
        report = add_platform_proxy(base_score(truth, pred))
        write_json(path / 'metrics.json', report)
        hh = per_horizon(truth, pred)
        pd.DataFrame({'offset_minutes': np.arange(1, 97) * 15, 'g1_mape': hh[:, 0], 'gall_mape': hh[:, 1]}).to_csv(path / 'by_horizon.csv', index=False)
        daily = [{'date': str(day.date()), **add_platform_proxy(base_score(truth[index.normalize() == day], pred[index.normalize() == day]))} for day in index.normalize().unique()]
        pd.DataFrame(daily).to_csv(path / 'by_day.csv', index=False)
    else:
        report = export_pair(path / 'results', index, pred, bundle.contract, cfg['id'])
    if cfg['id'] == 'reference_58':
        baseline = frozen_week(history_only(bundle.truth, cutoff), index, bundle.contract)
        save_npz(path / 'week_baseline.npz', origins=index.to_numpy(dtype='datetime64[ns]'), prediction=baseline)
    write_json(path / 'metadata.json', metadata)
    outputs = seal(path)
    receipt = {'status': 'completed', 'signature': spec['signature'], 'seconds': time.time() - started, 'metrics': report, 'output_hashes': outputs, 'candidate': cfg['id'], 'fold': fold['name']}
    write_json(path / 'receipt.json', receipt)
    LOGGER.info(msg('job.end', fold=fold['name'], candidate=cfg['id'], seconds=receipt['seconds']))
    return receipt
