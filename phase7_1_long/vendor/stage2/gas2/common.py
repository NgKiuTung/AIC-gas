"""私有赛事研究：资源、状态和运行签名。所有输出仅写入明确的新实验目录。"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

from gas2.bootstrap import ROOT
from gasstage.common import load_json, save_npz, sha256, write_json, setup_logging

def text(key: str, **values) -> str:
    """从独立资源加载用户可见日志和错误；变量不包含生产原始记录。"""
    messages = load_json(ROOT / "resources/messages.json")
    return messages[key].format(**values)

def fingerprint(value: dict) -> str:
    """JSON稳定序列化签名，用于防止配置或输入变化后复用旧任务。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def code_hashes() -> dict:
    """仅计算执行代码、契约和参考表，排除日志、用户输出和字节码。"""
    paths = [ROOT / "run.py"]
    for folder in ["gas2", "vendor/stage1/gasstage", "vendor/stage1/resources"]:
        paths += sorted(p for p in (ROOT / folder).rglob("*") if p.suffix in {".py", ".json"})
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}

def environment() -> dict:
    """记录实际运行版本；不把CPU运行写成GPU或目标实例实测。"""
    import numpy, pandas, scipy, lightgbm
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "lightgbm": lightgbm.__version__,
            "gpu_used": False}

def load_settings(path: Path) -> dict:
    """只允许可复验候选；不接受静默忽略的键值或负数损失权重。"""
    cfg = load_json(path)
    if cfg["threads"] < 1 or cfg["max_hours"] <= 0:
        raise ValueError(text("config.invalid"))
    if not (0 < cfg["train_origin_stride"] <= 96):
        raise ValueError(text("config.invalid"))
    if cfg["loss"]["epsilon"] <= 0 or cfg["loss"]["tail_lambda"] < 0:
        raise ValueError(text("config.invalid"))
    for candidate in cfg["candidates"]:
        if candidate["kind"] not in {"proxy", "direct"}:
            raise ValueError(text("config.invalid"))
        if candidate["rounds"] < 1 or candidate["loss"] not in {"l1", "tail"}:
            raise ValueError(text("config.invalid"))
    return cfg
