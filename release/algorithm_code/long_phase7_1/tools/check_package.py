"""私有AIC赛事研究：只读检查交付文件SHA-256，无第三方依赖。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def check(manifest: Path) -> dict:
    """只接受工程目录内文件，额外的用户运行产物不影响原件校验。"""
    data = json.loads(manifest.read_text(encoding='utf-8'))
    for rel, expected in data.items():
        path = ROOT / rel
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(rel)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'SHA-256 mismatch: {rel}')
    return {'status': 'passed', 'files': len(data), 'manifest': str(manifest)}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='交付包只读SHA-256检查')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'CODE_MANIFEST.json')
    print(json.dumps(check(parser.parse_args().manifest), ensure_ascii=False))
