"""私有AIC赛事研究：原子JSON、数组签名、日志和模型文件校验。"""
from __future__ import annotations
import hashlib
import json
import logging
import platform
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np
from gasbench.bootstrap import ROOT
from gasstage.common import load_json, save_npz, sha256, write_json

@lru_cache(maxsize=1)
def messages() -> dict:
    """加载中文错误与日志资源，库代码不配置全局日志处理器。"""
    return load_json(ROOT / 'resources' / 'messages.json')

def msg(key: str, **values) -> str:
    """格式化资源消息；不在日志中拼接原始生产记录。"""
    return messages()[key].format(**values)

def fingerprint(value) -> str:
    """稳定JSON签名；不允许NaN参与实验配置。"""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()

def array_hash(value: np.ndarray) -> str:
    """按dtype、形状和连续字节签名，用于核对跨模型的训练样本。"""
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def code_hashes() -> dict:
    """只签名执行代码与资源；不包含用户输出或日志。"""
    paths = [ROOT / 'run.py']
    for folder in ('gasbench', 'vendor', 'resources'):
        paths.extend((p for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.json')))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}

def environment() -> dict:
    """记录版本而非据版本名推断GPU兼容性，缺失库明确标记。"""
    packages = {}
    for name in ('numpy', 'pandas', 'scipy', 'lightgbm', 'catboost', 'xgboost', 'torch'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'NOT_INSTALLED'
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': packages}

def seal(directory: Path) -> dict:
    """训练成功后记录模型产物摘要；检查点不作为完成产物。"""
    hashes = {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*') if p.is_file() and p.name not in {'FILES.json', 'receipt.json', 'task.json', 'worker.log', 'console.log'} and ('checkpoint' not in p.name) and (not p.name.endswith('.tmp'))}
    write_json(directory / 'FILES.json', hashes)
    return hashes

def check_seal(directory: Path) -> dict:
    """加载模型前拒绝路径穿越、缺失和损坏文件。"""
    hashes = load_json(directory / 'FILES.json')
    for rel, expected in hashes.items():
        path = directory / rel
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(msg('path.invalid', path=rel))
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(msg('file.changed', path=path))
    return hashes

class JsonLog(logging.Formatter):
    """所有CLI使用同一日志结构，原始异常保留在消息内。"""

    def format(self, record: logging.LogRecord) -> str:
        """保持异常栈及结构化事件，不输出原始生产记录。"""
        value = {'time': self.formatTime(record), 'level': record.levelname, 'logger': record.name, 'message': record.getMessage()}
        if record.exc_info:
            value['exception'] = self.formatException(record.exc_info)
        return json.dumps(value, ensure_ascii=False)

def logging_to(path: Path | None=None) -> None:
    """仅主入口调用，控制台和文件同时记录关键事件。"""
    handlers = [logging.StreamHandler()]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding='utf-8'))
    for handler in handlers:
        handler.setFormatter(JsonLog())
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
