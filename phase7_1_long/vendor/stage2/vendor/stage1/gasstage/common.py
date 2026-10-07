"""私有赛事研究：配置、可验证原子落盘和结构化日志公共入口。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """流式计算文件摘要，不把大数据文件整体载入内存。"""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    """读取显式UTF-8 JSON，解析错误直接向调用方传播。"""
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: Any) -> None:
    """原子写入严格JSON；禁止将NaN伪装为标准JSON数值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def save_npz(path: Path, **arrays: Any) -> None:
    """只保存非object数组，避免反序列化pickle及半写入文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = {k: np.asarray(v) for k, v in arrays.items()}
    if any(v.dtype == object for v in packed.values()):
        raise ValueError(message('common.error.01'))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **packed)
    os.replace(tmp, path)


def config(path: Path | None = None) -> dict:
    """加载与核验本阶段契约；不接受未定义的窗口/延时策略。"""
    cfg = load_json(path or ROOT / 'configs/stage1.json')
    p = cfg['protocol']
    if p['block_minutes'] != 15 or p['blocks'] != 96 or p['origin_minutes'] != 15:
        raise ValueError(message('common.error.02'))
    if p['block_start_offset_minutes'] not in (0, 15):
        raise ValueError(message('common.error.03'))
    if min(p['pre_observation_delay_minutes'], p['semi_observation_delay_minutes']) < 0:
        raise ValueError(message('common.error.04'))
    if p['minimum_label_coverage'] != 1.0:
        raise ValueError(message('common.error.05'))
    if cfg['quality']['fill_raw_missing']:
        raise ValueError(message('common.error.06'))
    if cfg['features']['last_max_age_minutes'] < 1:
        raise ValueError(message('common.error.07'))
    return cfg


class JsonFormatter(logging.Formatter):
    """在单进程批处理内统一日志格式，不记录原始生产数据行。"""
    def format(self, record: logging.LogRecord) -> str:
        """转换日志事件为可机器解析的一行JSON。"""
        return json.dumps({'time': self.formatTime(record), 'level': record.levelname,
                           'logger': record.name, 'message': record.getMessage()}, ensure_ascii=False)


def setup_logging(path: Path) -> None:
    """仅CLI入口配置日志；复用当前终端，不启动后台服务。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(path, encoding='utf-8'), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


@lru_cache(maxsize=1)
def _messages() -> dict:
    return load_json(ROOT / 'resources/messages.json')


def message(key: str) -> str:
    """用户日志模板唯一来源，技术schema字段名不在此重复定义。"""
    return _messages()[key]
