"""私有AIC赛事研究：只在项目venv补缺失库，复用宿主Torch和科学计算栈。"""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]

def install() -> dict:
    """不降级已经可用的宿主包；依赖安装日志保存，错误不被吞掉。"""
    cfg = json.loads((ROOT / 'resources/setup.json').read_text(encoding='utf-8'))
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        raise RuntimeError(cfg['outside'])
    before = {}
    for name in cfg['required_host']:
        try:
            module = importlib.import_module(name)
        except Exception as error:
            raise RuntimeError(cfg['no_host'].format(name=name)) from error
        before[name] = {'version': module.__version__, 'path': module.__file__}
    missing = []
    for name, spec in cfg['packages'].items():
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
            missing.append(spec)
        else:
            version = getattr(module, '__version__', '0')
            parts = tuple((int(x) for x in re.findall('\\d+', version)[:2]))
            if name in cfg['minimums'] and parts < tuple(cfg['minimums'][name]):
                raise RuntimeError(cfg['old_version'].format(name=name, version=version))
    target = ROOT / 'logs'
    target.mkdir(exist_ok=True)
    if missing:
        env = {**os.environ, 'PIP_CACHE_DIR': str(ROOT / '.runtime/pip-cache'), 'TMPDIR': str(ROOT / '.runtime/tmp')}
        Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *missing, '--index-url', os.environ.get('GAS_PIP_INDEX', cfg['index']), '--timeout', str(cfg['timeout']), '--retries', str(cfg['retries'])]
        with (target / 'setup.log').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'command': command}) + '\n')
            stream.flush()
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=3600)
    result = {'status': 'passed', 'installed_missing': missing, 'host_libraries_before': before, 'python': sys.executable, 'system_libraries_modified_by_installer': False}
    (target / 'setup.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result
if __name__ == '__main__':
    print(json.dumps(install(), ensure_ascii=False))
