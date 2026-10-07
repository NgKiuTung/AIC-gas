"""私有AIC赛事研究：内层早停与最终重训；检查点仅恢复自身安全tensor状态。"""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
import numpy as np
import torch
from gasbench.common import msg, write_json
from gasbench.networks import build_network, masked_competition_mape
LOGGER = logging.getLogger(__name__)

def atomic_torch(path: Path, value: dict) -> None:
    """同目录原子替换检查点，进程被终止时旧检查点仍可读。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)

def new_network(cfg: dict, arrays, threads: int):
    """固定可审计seed；不承诺GPU重复训练逐位相同。"""
    torch.set_num_threads(threads)
    torch.manual_seed(cfg.get('seed', 20261003))
    if cfg.get('device', 'cpu') == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(msg('device.unavailable'))
        torch.cuda.manual_seed_all(cfg.get('seed', 20261003))
    return build_network(cfg['family'], arrays.sequence.shape[1], arrays.static.shape[1], arrays.length, cfg).to(cfg.get('device', 'cpu'))

def evaluate_net(net, arrays, cfg: dict) -> float:
    """校准损失只在内层历史段计算，累积每目标/跨度的分子和有效计数。"""
    device = cfg.get('device', 'cpu')
    sums, counts = (np.zeros((2, 2)), np.zeros((2, 2)))
    net.eval()
    with torch.no_grad():
        for start in range(0, len(arrays.rows), cfg['batch_size']):
            sel = np.arange(start, min(start + cfg['batch_size'], len(arrays.rows)))
            inputs, truth = arrays.batch(sel, device)
            pred = net(*inputs).float().cpu().numpy()
            y = truth.cpu().numpy()
            err = np.abs(pred - y) / y
            for i, end in enumerate((8, 96)):
                for g in range(2):
                    finite = np.isfinite(y[:, :end, g])
                    sums[i, g] += np.where(finite, err[:, :end, g], 0).sum()
                    counts[i, g] += finite.sum()
    value = sums / np.maximum(counts, 1)
    # NOTE: 两个目标等权；长周期平台斜率约为短周期2倍。
    return float((value[0].sum() + 2.0 * value[1].sum()) / 6.0)

def _epoch(net, arrays, optimizer, scaler, cfg, epoch, deadline):
    """NOTE: 中断重跑从完整epoch边界恢复，相同epoch有固定采样和dropout种子。"""
    device = cfg.get('device', 'cpu')
    generator = np.random.default_rng(cfg.get('seed', 20261003) + epoch)
    order = generator.permutation(len(arrays.rows))
    torch.manual_seed(cfg.get('seed', 20261003) + epoch)
    if device == 'cuda':
        torch.cuda.manual_seed_all(cfg.get('seed', 20261003) + epoch)
    net.train()
    total, batches = (0.0, 0)
    for start in range(0, len(order), cfg['batch_size']):
        if time.time() >= deadline:
            raise TimeoutError(msg('job.timeout'))
        selected = order[start:start + cfg['batch_size']]
        x, y = arrays.batch(selected, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.get('amp', False) and device == 'cuda'):
            predicted = net(*x)
        loss = masked_competition_mape(predicted.float(), y.float())
        if not torch.isfinite(loss):
            raise ValueError(msg('model.finite'))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total, batches = (total + float(loss.detach()), batches + 1)
    return total / max(batches, 1)

def fit_epochs(arrays, valid, cfg: dict, threads: int, directory: Path, phase: str, deadline: float, epochs: int | None=None):
    """checkpoint含优化器和完整epoch；旧run签名检查由调度层先执行。"""
    device = cfg.get('device', 'cpu')
    net = new_network(cfg, arrays, threads)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg.get('weight_decay', 0.0001))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.get('amp', False) and device == 'cuda')
    state_path = directory / f'{phase}_checkpoint.pt'
    state = {'epoch': 0, 'best_epoch': 0, 'best_loss': float('inf'), 'stale': 0, 'trace': []}
    if state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=True)
        net.load_state_dict(saved.pop('model'))
        optimizer.load_state_dict(saved.pop('optimizer'))
        scaler.load_state_dict(saved.pop('scaler'))
        state.update(saved)
    maximum = epochs if epochs is not None else cfg['epochs']
    for epoch in range(state['epoch'] + 1, maximum + 1):
        if valid is not None and state['stale'] >= cfg.get('patience', 5):
            break
        training_loss = _epoch(net, arrays, optimizer, scaler, cfg, epoch, deadline)
        validation_loss = evaluate_net(net, valid, cfg) if valid is not None else training_loss
        improved = validation_loss < state['best_loss'] - 1e-07
        state.update(epoch=epoch, stale=0 if improved else state['stale'] + 1)
        if improved:
            state.update(best_epoch=epoch, best_loss=validation_loss)
        state['trace'].append({'epoch': epoch, 'train': training_loss, 'validation': validation_loss})
        atomic_torch(state_path, {**state, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()})
        write_json(directory / f'{phase}_trace.json', state['trace'])
        LOGGER.info(msg('model.train', family=cfg['family'], phase=phase, epoch=epoch, loss=training_loss, valid=validation_loss))
    return (net, state)
