"""私有AIC赛事研究：真实后端小训练与GPU算子检查，不根据版本号强装依赖。"""
from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
from gasbench.common import environment, msg, write_json
from gasbench.backends import fit_backend, predict_backend

def preflight(cfg: dict, output: Path) -> dict:
    """只检查请求的模型族；CUDA请求失败则停止，不自动静默CPU训练。"""
    report = environment()
    report['checks'] = []
    families = {(c['family'], c.get('device', 'cpu')) for c in cfg['candidates']}
    for library in {'lightgbm', 'numpy', 'pandas', 'scipy'}:
        importlib.import_module(library)
    for family, device in sorted(families):
        if family in ('tcn', 'tide_style', 'mlp'):
            import torch
            if device == 'cuda':
                _torch_cuda(report)
            else:
                x = torch.randn(4, 8, requires_grad=True)
                x.square().mean().backward()
            report['checks'].append({'family': family, 'device': device, 'status': 'passed'})
        else:
            try:
                importlib.import_module(family)
            except ImportError as error:
                raise RuntimeError(msg('model.missing', name=family)) from error
            rng = np.random.default_rng(7)
            x = rng.normal(size=(64, 4)).astype('float32')
            y = np.abs(x[:, 0]) * 20 + 50
            model = fit_backend(family, x, y - y.mean(), np.ones(64), {'rounds': 3, 'device': device, 'seed': 7}, cfg['threads'])
            pred = predict_backend(family, model, x, cfg['threads'])
            if not np.isfinite(pred).all():
                raise ValueError(msg('model.finite'))
            report['checks'].append({'family': family, 'device': device, 'status': 'passed'})
    report['status'] = 'passed'
    write_json(output, report)
    return report

def _torch_cuda(report):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(msg('device.unavailable'))
    torch.set_num_threads(2)
    x = torch.randn(32, 8, 96, device='cuda', requires_grad=True)
    layer = torch.nn.Conv1d(8, 16, 3, padding=1).cuda()
    with torch.autocast('cuda', dtype=torch.float16):
        y = layer(x)
        loss = y.float().square().mean()
    loss.backward()
    a = torch.randn(128, 128, device='cuda')
    product = a @ a
    torch.cuda.synchronize()
    if not torch.isfinite(product).all() or not torch.isfinite(x.grad).all():
        raise ValueError(msg('model.finite'))
    report['cuda'] = {'torch': torch.__version__, 'runtime': torch.version.cuda, 'name': torch.cuda.get_device_name(0), 'memory_bytes': torch.cuda.get_device_properties(0).total_memory, 'matmul_and_conv_backward': True}
    del x, layer, y, loss, a, product
    torch.cuda.empty_cache()
