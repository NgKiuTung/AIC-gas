"""私有赛事研究：配置、资源和源码签名；不修改宿主环境。"""
from __future__ import annotations
from pathlib import Path
import math

from gas2.common import ROOT, load_json, write_json, sha256, fingerprint, environment


def message(key: str, **values) -> str:
    """运行日志和错误文案集中存储，便于审计修订。"""
    return load_json(ROOT / "resources/refinement_messages.json")[key].format(**values)


def execution_hashes() -> dict:
    """执行源码和资源签名，不包含输出日志及旧实验报告。"""
    paths = [ROOT / "refine.py"]
    for folder in ("gasrefine", "gas2", "vendor/stage1/gasstage", "vendor/stage1/resources", "resources"):
        paths += sorted(p for p in (ROOT/folder).rglob("*") if p.suffix in {".py", ".json"})
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def read_settings(path: Path) -> dict:
    """校验所有会改变训练或因果时间范围的配置；无效值不使用静默默认。"""
    cfg = load_json(path)
    if cfg["threads"] < 1 or not math.isfinite(cfg["max_hours"]) or cfg["max_hours"] <= 0:
        raise ValueError(message("config.invalid"))
    ids = [c["id"] for c in cfg["models"]]
    if len(ids) != len(set(ids)) or set(ids) != {"reference", "context60"}:
        raise ValueError(message("config.invalid"))
    for model in cfg["models"]:
        if (model["features"] not in {"raw", "context"} or model["source_stride_minutes"] not in {1, 5, 15}
                or model["pre_weight"] < 0 or model["rounds"] < 1 or model["lookback_days"] <= 0):
            raise ValueError(message("config.invalid"))
    for recipe in cfg["recipes"].values():
        numbers = [recipe[k] for k in ("day_weight", "smooth_hours", "fast_hours", "slow_hours", "level_gain")]
        if (not all(math.isfinite(v) for v in numbers) or not 0 <= recipe["day_weight"] <= 1
                or recipe["smooth_hours"] < 0 or min(recipe["fast_hours"], recipe["slow_hours"]) <= 0
                or not 0 <= recipe["level_gain"] <= 2):
            raise ValueError(message("config.invalid"))
    return cfg
