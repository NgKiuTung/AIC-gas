"""私有AIC赛事研究：模型独立推理及带来源清单的私有交接归档。"""
from __future__ import annotations
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from gasbench.common import ROOT, check_seal, load_json, write_json, sha256, msg
from gasbench.data import load_bundle
from gasbench.factory import load_model, predict_model
from gasstage.submission import export_pair

def predict(cache: Path, model_path: Path, output: Path, device: str='cpu') -> dict:
    """加载已训练的单候选，为全部真实测试起点生成96步；不重新拟合。"""
    if output.exists() and any(output.iterdir()):
        raise ValueError(msg('cache.exists', path=output))
    check_seal(cache)
    bundle = load_bundle(cache)
    metadata = load_json(model_path / 'model.json')
    candidate = metadata['candidate']
    kind = 'reference' if candidate.get('id') == 'proxy_joint025_l1' else candidate['kind']
    model = load_model(model_path, kind, device)
    p = bundle.contract['protocol']
    index = pd.date_range(p['test_start'], p['test_end'], freq='15min')
    forecast = predict_model(model, kind, bundle.features, index)
    result = export_pair(output, index, forecast, bundle.contract, candidate['id'])
    return {'status': 'passed', 'output': str(output), 'rows': len(index), 'validation': result}

def pack(run: Path, output: Path) -> dict:
    """打包模型、结果与复现源码，不复制数据缓存、Torch或外部环境。"""
    if not (run / 'run_status.json').exists():
        raise ValueError(msg('file.changed', path=run / 'run_status.json'))
    if output.exists():
        raise ValueError(msg('cache.exists', path=output))
    files = {}
    for name in ('summary.json', 'run_status.json', 'contract.json', 'events.jsonl', 'jobs.json', 'preflight.json', 'task_times.json', 'fusion_provenance.json', 'guarded_provenance.json', 'REPORT.md'):
        if (run / name).is_file():
            files['run/' + name] = run / name
    for name in ('metrics', 'fusion', 'guarded', 'verification', 'candidate_results', 'sessions'):
        for p in (run / name).rglob('*'):
            if p.is_file():
                files['run/' + str(p.relative_to(run))] = p
    for task in (run / 'tasks').glob('*'):
        if not (task / 'receipt.json').exists():
            continue
        check_seal(task)
        for p in task.rglob('*'):
            if p.is_file() and 'checkpoint' not in p.name and (not p.name.endswith('.tmp')):
                files['run/' + str(p.relative_to(run))] = p
    for name in ('gasbench', 'vendor', 'resources', 'configs', 'tools', 'tests'):
        for p in (ROOT / name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and (p.suffix != '.pyc'):
                files['code/' + str(p.relative_to(ROOT))] = p
    for name in ('run.py', 'setup.sh', 'run_a10.sh', 'README.md', 'requirements.txt', 'AGENTS.md'):
        if (ROOT / name).is_file():
            files['code/' + name] = ROOT / name
    for p in (ROOT / 'reference').glob('*'):
        if p.is_file():
            files['code/reference/' + p.name] = p
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = {name: sha256(path) for name, path in files.items()}
    import json
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in sorted(files.items()):
            z.write(path, name)
        z.writestr('MANIFEST.json', json.dumps(digest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError(msg('file.changed', path=output))
    return {'status': 'passed', 'path': str(output), 'files': len(files), 'sha256': sha256(output), 'run_status': load_json(run / 'run_status.json')['status'], 'dataset_included': False}
