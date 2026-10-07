"""私有赛事研究：统一加载只读的上游实现。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "vendor/prior"
for directory in (PRIOR, PRIOR / "vendor/stage1"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
