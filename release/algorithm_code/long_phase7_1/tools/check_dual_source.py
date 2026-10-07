"""私有AIC研究：语法、危险导入、文档与上游保真检查，不冒充完整类型检查。"""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def check():
    """仅检查明确可机器判断的要求；允许未导出的私有帮助方法无docstring。"""
    files=sorted((ROOT/'gasauto').glob('*.py'))+[ROOT/'run.py']
    info=[]
    for path in files:
        source=path.read_text();tree=ast.parse(source)
        compile(tree,str(path),'exec')
        if not ast.get_docstring(tree):raise ValueError(f'模块缺少说明：{path}')
        if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('eval','exec') for n in ast.walk(tree)):
            raise ValueError(f'不允许动态执行字符串：{path}')
        info.append({'file':str(path.relative_to(ROOT)),'lines':len(source.splitlines())})
    upstream=json.loads((ROOT/'UPSTREAM_ORIGINAL.json').read_text())
    unchanged={}
    for name,digest in upstream.items():
        if name.startswith(('gasbench/','vendor/','reference/')):
            path=ROOT/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError(f'上游受保护文件变化：{name}')
            unchanged[name]=digest
    return {'status':'passed','source_files':len(info),'files':info,'unchanged_upstream_files':len(unchanged),
            'scope':'AST compilation and protected source check; not a full type/lint tool'}

if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
