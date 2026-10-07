"""私有赛事研究：定位随包上游实现，不修改用户已完成的Stage1工程。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "stage1"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
