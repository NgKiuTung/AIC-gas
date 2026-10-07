"""私有AIC研究：发布独立代码包；只打包明确的源码、资源和验收摘要。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = 'gas_phase7_1_lgb_mlp'
TOP = ('AGENTS.md', 'README.md', 'PROGRAM.md', 'TODO.md', 'CHANGELOG.md',
       'UPSTREAM_ORIGINAL.json', 'pyproject.toml', 'requirements.txt',
       'run.py', 'run_research.sh', 'run_a10.sh', 'setup.sh', '.gitignore', '.coveragerc')
DIRS = ('gasauto', 'gasbench', 'vendor', 'reference', 'resources', 'configs', 'tests')
TOOLS = ('check_package.py', 'check_dual_source.py', 'install_missing.py', 'package_dual.py')
REPORTS = ('VALIDATION_PHASE7_1.md', 'STATIC_CHECK_PHASE7_1.json')

def digest(path: Path) -> str:
    """逐块读取而非将可选大证据全载入内存。"""
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def members() -> list[Path]:
    """仅收录白名单，不包含运行数据、绝对路径开发脚本或私有原始数据。"""
    paths = [ROOT / name for name in TOP]
    for folder in DIRS:
        paths.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ('.pyc', '.tmp'))
    paths += [ROOT / 'tools' / name for name in TOOLS]
    paths += [ROOT / 'reports' / name for name in REPORTS if (ROOT / 'reports' / name).is_file()]
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'发布成员缺失或为符号链接：{path}')
    return sorted(set(paths))

def package(output: Path) -> dict:
    """写入manifest后再次读取每个成员，核验CRC和逐文件摘要。"""
    paths = members()
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    (ROOT / 'CODE_MANIFEST.json').write_text(text, encoding='utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, PACKAGE_ROOT + '/' + str(path.relative_to(ROOT)))
        archive.writestr(PACKAGE_ROOT + '/CODE_MANIFEST.json', text)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(bad)
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(PACKAGE_ROOT + '/' + name)).hexdigest() != expected:
                raise ValueError(name)
    return {'status': 'passed', 'file': str(output), 'bytes': output.stat().st_size,
            'manifest_members': len(manifest), 'sha256': digest(output)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase7.1 Long-focused research code package release')
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(package(parser.parse_args().output), ensure_ascii=False))
