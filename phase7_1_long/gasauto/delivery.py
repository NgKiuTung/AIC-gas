"""私有AIC赛事研究：独立短长冠军导出和交接归档，结果不冒称官方通过。"""
from pathlib import Path
import json
import zipfile
import numpy as np
from gasauto.protocol import index_for
from gasauto.identity import safe_path
from gasbench.common import load_json, write_json, check_seal, sha256
from gasbench.split_submission import validate_split_pair
from gasstage.submission import columns
import pandas as pd

def task_frame(index, prediction, blocks):
    """接收任务自身的8/96块，不将短预测补成伪96块。"""
    if prediction.shape != (len(index), blocks, 2) or not np.isfinite(prediction).all():
        raise ValueError('任务张量形状或数值不合法。')
    values = np.column_stack([prediction[:, :, 0], prediction[:, :, 1]])
    result = pd.DataFrame(values, columns=columns(blocks)[1:])
    result.insert(0, 'datetime', index.strftime('%Y-%m-%d %H:%M:%S'))
    return result

def merged_prediction(run, records):
    """部署为确认时预登记的种子等权，不挑测试表现最好seed。"""
    arrays = []
    for r in records:
        p = safe_path(run, r['output'])
        check_seal(p)
        with np.load(p / 'prediction.npz', allow_pickle=False) as z:
            arrays.append(z['prediction'])
    return np.mean(arrays, axis=0)


def validate_long_file(path: Path, index: pd.DatetimeIndex) -> dict:
    """Validate a Long-only artifact when the external Phase6 Short file is unavailable.

    NOTE: A passing Long-only report is not a valid competition submission pair.
    """
    frame = pd.read_csv(path, float_precision='round_trip')
    expected = columns(96)
    if list(frame.columns) != expected or len(frame) != len(index):
        raise ValueError('invalid long-only schema')
    if frame['datetime'].tolist() != index.strftime('%Y-%m-%d %H:%M:%S').tolist():
        raise ValueError('invalid long-only timeline')
    values = frame.iloc[:, 1:].to_numpy(dtype='float64')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('invalid long-only values')
    if (values[:, 96:] < values[:, :96]).any():
        raise ValueError('long-only generator hierarchy violation')
    return {
        'validation': 'passed',
        'long_only': True,
        'submission_pair_ready': False,
        'rows': len(frame),
        'columns': len(frame.columns),
        'start': str(index[0]),
        'end': str(index[-1]),
        'sha256': sha256(path),
    }

def export_pair(path, index, short, long, provenance):
    """Short真正8块，Long96块；允许重叠预测不同。"""
    if short.shape != (len(index), 8, 2) or long.shape != (len(index), 96, 2):
        raise ValueError('invalid output shape')
    path.mkdir(parents=True, exist_ok=True)
    for name, pred, h in [('s_result.csv', short, 8), ('l_result.csv', long, 96)]:
        if not np.isfinite(pred).all() or (pred < 0).any() or (pred[:, :, 1] < pred[:, :, 0]).any():
            raise ValueError('invalid output values')
        task_frame(index, pred, h).to_csv(path / name, index=False, encoding='utf-8', float_format='%.6f')
    report = validate_split_pair(path, index)
    report['provenance'] = provenance
    report['official_score'] = 'Unknown'
    report['interval_assumption'] = '[t,t+15min)'
    write_json(path / 'validation.json', report)
    with zipfile.ZipFile(path / 'results_only.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('s_result.csv', 'l_result.csv'):
            z.write(path / name, 'submissions/' + name)
    return report

def export_run(run, selected, refs, test):
    """参照和新选择分别保存，回退理由写入表旁清单。"""
    index = index_for(test)
    short = merged_prediction(run, selected['short']['records'])
    long = merged_prediction(run, selected['long']['records'])
    export_pair(run / 'candidate_results/selected', index, short, long, selected)
    s = merged_prediction(run, [refs['short']])
    l = merged_prediction(run, [refs['long']])
    export_pair(run / 'candidate_results/reference_58', index, s, l, {'candidate': 'reference_58'})

def pack(run: Path, output: Path):
    """交接包括冻结源码、模型和账本，原始私有数据不重复打包。"""
    if not (run / 'summary.json').exists():
        raise ValueError('run summary missing')
    if not (run / 'verification/summary.json').exists() or load_json(run / 'verification/summary.json').get('status') != 'passed':
        raise ValueError('请先通过verify再打包正式交接。')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(run.rglob('*')):
            if not p.is_file() or p.resolve() == output.resolve() or p.name == '.lock' or p.name.endswith('.tmp'):
                continue
            relative = str(p.relative_to(run))
            safe_path(run, relative)
            z.write(p, relative)
            manifest[relative] = sha256(p)
        z.writestr('HANDOFF_MANIFEST.json', json.dumps(manifest, indent=2))
    return {'status': 'passed', 'file': str(output), 'sha256': sha256(output), 'members': len(manifest), 'dataset_included': False}
