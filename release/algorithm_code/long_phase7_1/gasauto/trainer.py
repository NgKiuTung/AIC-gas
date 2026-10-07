"""私有AIC赛事研究：epoch0先验保留，训练及校准计时分离，最佳状态真实保存。"""
import numpy as np
import torch
from gasauto.networks import build, task_loss
from gasbench.neural_fit import atomic_torch
from gasbench.common import write_json
from gasstage.baselines import reconcile
import logging
LOGGER = logging.getLogger(__name__)

def make_net(cfg, arrays, threads):
    """相同种子复建；GPU可用性由预检验证，不默默改CPU。"""
    torch.set_num_threads(threads)
    torch.manual_seed(cfg['seed'])
    device = cfg.get('device', 'cpu')
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.manual_seed_all(cfg['seed'])
    return build(cfg, arrays.sequence.shape[1], arrays.static.shape[1], arrays.length).to(device)

def evaluate(net, arrays, cfg):
    """跨批累积每目标误差与有效数，避免小尾批过度加权。"""
    total = np.zeros(2)
    count = np.zeros(2)
    net.eval()
    with torch.no_grad():
        for lo in range(0, len(arrays.rows), cfg['batch_size']):
            sel = np.arange(lo, min(lo + cfg['batch_size'], len(arrays.rows)))
            x, y = arrays.batch(sel, cfg.get('device', 'cpu'))
            p = net(*x).float().cpu().numpy()
            y = y.cpu().numpy()
            scale = getattr(arrays, 'target_scale', np.ones(2))
            p = reconcile(p * scale) / scale
            valid = np.isfinite(y)
            safe = np.where(valid, y, 1)
            err = np.where(valid, np.abs(p - safe) / safe, 0)
            total += err.sum(axis=(0, 1))
            count += valid.sum(axis=(0, 1))
    if not np.isfinite(total).all() or (count == 0).any():
        raise ValueError('invalid calibration')
    return float(np.mean(total / count))

def choose_best(state, epoch, loss, min_delta):
    """不允许较差epoch1覆盖较好的epoch0；独立函数便于回归测试。"""
    if loss < state['best_loss'] - min_delta:
        return ({**state, 'best_loss': loss, 'best_epoch': epoch, 'stale': 0}, True)
    return ({**state, 'stale': state['stale'] + 1}, False)

def train_epoch(net, arrays, opt, scaler, cfg, epoch, clock):
    """只计实际优化时间；整个epoch未结束不会产生成功检查点。"""
    rng = np.random.default_rng(cfg['seed'] + epoch)
    order = rng.permutation(len(arrays.rows))
    torch.manual_seed(cfg['seed'] + epoch)
    if cfg.get('device') == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'] + epoch)
    net.train()
    total = 0.0
    steps = 0
    with clock.compute():
        for lo in range(0, len(order), cfg['batch_size']):
            if clock.remaining() <= 0:
                raise TimeoutError('training allowance')
            x, y = arrays.batch(order[lo:lo + cfg['batch_size']], cfg.get('device', 'cpu'))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=cfg.get('device', 'cpu'), dtype=torch.float16, enabled=cfg.get('amp', False) and cfg.get('device') == 'cuda'):
                p = net(*x)
            loss = task_loss(p, y)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            steps += 1
            total += float(loss.detach())
        if cfg.get('device') == 'cuda':
            torch.cuda.synchronize()
    return (total / max(steps, 1), steps)

def fit_phase(arrays, valid, cfg, threads, path, phase, clock, epochs=None):
    """内层保存best.pt；最终refit允许epochs=0，不强制训练一轮。"""
    net = make_net(cfg, arrays, threads)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg.get('weight_decay', 0.0001))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.get('amp', False) and cfg.get('device') == 'cuda')
    initial = evaluate(net, valid, cfg) if valid is not None else None
    state = {'best_epoch': 0, 'best_loss': initial, 'stale': 0, 'final_epoch': 0, 'steps': 0, 'epoch0_loss': initial, 'stop_reason': 'max_epochs'}
    trace = []
    path.mkdir(parents=True, exist_ok=True)
    if valid is not None:
        atomic_torch(path / f'{phase}_best.pt', {'state_dict': net.state_dict(), 'epoch': 0})
        trace.append({'epoch': 0, 'train': None, 'validation': initial, 'best_epoch': 0})
    maximum = cfg['epochs'] if epochs is None else epochs
    for epoch in range(1, maximum + 1):
        if valid is not None and state['stale'] >= cfg.get('patience', 5):
            state['stop_reason'] = 'early_stop'
            break
        if valid is not None and clock.elapsed >= clock.limit * 0.45:
            state['stop_reason'] = 'tune_budget_reserve'
            break
        train, steps = train_epoch(net, arrays, opt, scaler, cfg, epoch, clock)
        loss = evaluate(net, valid, cfg) if valid is not None else train
        if valid is not None:
            state, improved = choose_best(state, epoch, loss, cfg.get('min_delta', 1e-06))
            if improved:
                atomic_torch(path / f'{phase}_best.pt', {'state_dict': net.state_dict(), 'epoch': epoch})
        state.update(final_epoch=epoch, steps=state['steps'] + steps)
        trace.append({'epoch': epoch, 'train': train, 'validation': loss, 'best_epoch': state['best_epoch']})
        atomic_torch(path / f'{phase}_checkpoint.pt', {'state_dict': net.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch})
        write_json(path / f'{phase}_trace.json', trace)
        LOGGER.info('训练：%s epoch=%s train=%.6f validation=%.6f best_epoch=%s', phase, epoch, train, loss, state['best_epoch'])
    if valid is not None:
        net.load_state_dict(torch.load(path / f'{phase}_best.pt', map_location=cfg.get('device', 'cpu'), weights_only=True)['state_dict'])
    write_json(path / f'{phase}_trace.json', trace)
    write_json(path / f'{phase}_state.json', state)
    return (net, state)
