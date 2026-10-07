"""私有AIC赛事研究：定位只读上游，不污染用户其他项目的导入路径。

noqa: E402,F401  NOTE: 上游声明自身Stage1资源根目录。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'stage2'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
import gas2.bootstrap
