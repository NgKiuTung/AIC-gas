"""私有AIC赛事研究：冻结执行代码并校验任务输入身份。"""
from pathlib import Path
from gasbench.common import ROOT, sha256

def source_identity() -> dict:
    """源码/资源变化必须新建运行；评估器不能在研究中被悄悄修改。"""
    paths = [ROOT / 'run.py']
    for directory in ('gasauto', 'gasbench', 'vendor', 'resources'):
        paths.extend((p for p in (ROOT / directory).rglob('*') if p.suffix in ('.py', '.json')))
    paths.extend((p for p in (ROOT / 'reference').rglob('*') if p.is_file()))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}

def ensure_same_source(expected: dict) -> None:
    """每次启动worker之前检查，拒绝混入未登记源码。"""
    if source_identity() != expected:
        raise ValueError('执行代码或评估资源发生变化，必须新建运行目录。')

def safe_path(root: Path, relative: str) -> Path:
    """回执路径不得越过run目录；不能用手改JSON加载外部文件。"""
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('回执或归档路径越界。')
    return target
