"""私有AIC赛事研究：配置、原子产物、源数据、边界与持久签名测试。"""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from gasbench.common import ROOT, array_hash, check_seal, fingerprint, seal, write_json, save_npz
from gasbench.config import read_config, validate
from gasbench.sequences import Scaler, window_rows
from gasbench.cli import parser
from gasbench.supervisor import task_complete

@pytest.mark.parametrize('name', ['a10', 'a10_gpu_trees', 'a10_72h', 'smoke_cpu', 'smoke_a10', 'cpu_review'])
def test_shipped_config(name):
    assert read_config(ROOT / f'configs/{name}.json')['candidates'][0]['id'] == 'reference_58'

@pytest.mark.parametrize('field,value', [('threads', 0), ('threads', 100), ('job_minutes', 0), ('unexpected', 1)])
def test_bad_top(field, value):
    cfg = read_config(ROOT / 'configs/smoke_cpu.json')
    cfg[field] = value
    with pytest.raises(ValueError):
        validate(cfg)

@pytest.mark.parametrize('mutate', [{'id': '../escape'}, {'family': 'rf'}, {'device': 'magic'}, {'rounds': 0}, {'pre_weight': 1.1}, {'sample_horizons': 200}, {'params': {'foo': 1}}, {'x': 1}])
def test_bad_candidate(mutate):
    cfg = read_config(ROOT / 'configs/smoke_cpu.json')
    cfg['candidates'][1].update(mutate)
    with pytest.raises(ValueError):
        validate(cfg)

def test_reference_immutable():
    cfg = read_config(ROOT / 'configs/smoke_cpu.json')
    cfg['candidates'][0]['rounds'] = 2
    with pytest.raises(ValueError):
        validate(cfg)

def test_reference_required():
    cfg = read_config(ROOT / 'configs/smoke_cpu.json')
    cfg['candidates'] = cfg['candidates'][1:]
    with pytest.raises(ValueError):
        validate(cfg)

@pytest.mark.parametrize('kind', ['duplicate', 'unordered', 'unaligned', 'backward'])
def test_fold_contract(kind):
    cfg = read_config(ROOT / 'configs/smoke_cpu.json')
    mapping = {'duplicate': [cfg['folds'][0]] * 2, 'unordered': cfg['folds'][::-1], 'unaligned': [dict(cfg['folds'][0], start='2025-08-01 00:01')], 'backward': [dict(cfg['folds'][0], end='2025-07-31 23:45')]}
    cfg['folds'] = mapping[kind]
    with pytest.raises(ValueError):
        validate(cfg)

def test_seal_roundtrip(tmp_path):
    (tmp_path / 'model.bin').write_bytes(b'abc')
    h = seal(tmp_path)
    assert check_seal(tmp_path) == h

def test_seal_changed(tmp_path):
    (tmp_path / 'model.bin').write_bytes(b'abc')
    seal(tmp_path)
    (tmp_path / 'model.bin').write_bytes(b'abd')
    with pytest.raises(ValueError):
        check_seal(tmp_path)

def test_seal_path_escape(tmp_path):
    write_json(tmp_path / 'FILES.json', {'../x': '0'})
    with pytest.raises(ValueError):
        check_seal(tmp_path)

def test_seal_excludes_running_log(tmp_path):
    (tmp_path / 'worker.log').write_text('start')
    (tmp_path / 'a').write_text('model')
    h = seal(tmp_path)
    (tmp_path / 'worker.log').write_text('completed')
    assert 'worker.log' not in h and check_seal(tmp_path) == h

def test_receipt_sig(tmp_path):
    assert not task_complete(tmp_path, 'a')
    write_json(tmp_path / 'receipt.json', {'signature': 'b', 'status': 'completed'})
    with pytest.raises(ValueError):
        task_complete(tmp_path, 'a')

def test_receipt_complete(tmp_path):
    (tmp_path / 'a').write_text('1')
    hashes = seal(tmp_path)
    write_json(tmp_path / 'receipt.json', {'signature': 'a', 'status': 'completed', 'output_hashes': hashes})
    assert task_complete(tmp_path, 'a')

def test_array_signature():
    assert array_hash(np.array([1], dtype='float32')) != array_hash(np.array([1], dtype='float64'))

def test_signature_order():
    assert fingerprint({'b': 2, 'a': 1}) == fingerprint({'a': 1, 'b': 2})

def test_npz_reject_object(tmp_path):
    with pytest.raises(ValueError):
        save_npz(tmp_path / 'bad.npz', x=np.array([{}], dtype=object))

@pytest.mark.parametrize('rows,length,total', [(np.array([0]), 2, 10), (np.array([-1]), 1, 10), (np.array([10]), 2, 10), (np.array([]), 2, 10), (np.array([2]), 0, 10)])
def test_window_rejects(rows, length, total):
    with pytest.raises(ValueError):
        window_rows(rows, length, total)

def test_window_offsets():
    np.testing.assert_array_equal(window_rows(np.array([3, 4]), 3, 10), [[1, 2, 3], [2, 3, 4]])

def test_scaler_nan():
    x = np.array([[1, np.nan, 2], [3, np.nan, 2]], dtype='float32')
    scaler = Scaler.fit(x)
    z = scaler.transform(x)
    np.testing.assert_allclose(z, [[-1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(z, Scaler.from_dict(scaler.as_dict()).transform(x))

def test_scaler_training_only():
    scaler = Scaler.fit(np.array([[1.0], [3.0]]))
    assert scaler.transform(np.array([[1000000.0]])).item() == 12
    assert scaler.mean.item() == 2

def test_scaler_no_rows():
    with pytest.raises(ValueError):
        Scaler.fit(np.empty((0, 3)))

def test_cli_train_budget_required():
    with pytest.raises(SystemExit):
        parser().parse_args(['train', '--cache', 'c', '--config', 'x', '--output', 'y'])

def test_duplicate_fold_name():
    cfg = read_config(ROOT / 'configs/smoke_cpu.json')
    cfg['folds'][1]['name'] = cfg['folds'][0]['name']
    with pytest.raises(ValueError):
        validate(cfg)

def test_fold_in_hidden_test():
    cfg = read_config(ROOT / 'configs/smoke_cpu.json')
    cfg['folds'][1]['end'] = '2025-10-01'
    with pytest.raises(ValueError):
        validate(cfg)

def test_fusion_config_reject():
    cfg = read_config(ROOT / 'configs/smoke_cpu.json')
    cfg['fusion']['ridge'] = -1
    with pytest.raises(ValueError):
        validate(cfg)

def test_declared_reference_hash():
    from gasbench.common import sha256
    assert len(sha256(ROOT / 'reference/reference_58_results.zip')) == 64


def test_selection_config_reject():
    cfg = read_config(ROOT / 'configs/smoke_cpu.json')
    cfg['selection']['blend_weight'] = 1.5
    with pytest.raises(ValueError):
        validate(cfg)
