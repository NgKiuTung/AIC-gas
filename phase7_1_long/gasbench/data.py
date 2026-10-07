"""私有AIC赛事研究：原16表和已验收Stage1双入口，标签与过程分开。"""
from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from gasbench.common import ROOT, load_json, write_json, sha256, msg, seal, check_seal
from gasstage.features import origins
from gasstage.ingest import NativeDataset, read_dataset
from gasstage.experiment import prepare as stage1_prepare
from gasstage.targets import block_truth
LOGGER = logging.getLogger(__name__)
STAGE1_ROOT = ROOT / 'vendor/stage2/vendor/stage1'
FILES = ('data/native_observations.npz', 'data/contract.json', 'features/history_process_features.npz', 'features/test_process_features.npz', 'features/dictionary.csv', 'labels/semi_truth_blocks_15m.npz')

@dataclass
class Bundle:
    """全量真实目标只供拟合/评分；模型推理不接收此结构。"""
    native: NativeDataset
    features: pd.DataFrame
    truth: pd.DataFrame
    contract: dict
    source: Path

def source_contract() -> dict:
    """复用Stage1唯一契约，不重新猜测单位和区间端点。"""
    return load_json(STAGE1_ROOT / 'configs/stage1.json')

def import_stage1(source: Path, output: Path, cfg: dict) -> None:
    """只复制最小数据产物，不需要旧模型/旧研究目录。"""
    if load_json(source / 'run_status.json')['status'] != 'completed':
        raise ValueError(msg('cache.invalid'))
    if load_json(source / 'audit/source.json').get('sha256') != cfg['expected_dataset_sha256']:
        raise ValueError(msg('cache.invalid'))
    if load_json(source / 'data/contract.json') != cfg:
        raise ValueError(msg('cache.invalid'))
    for rel in FILES:
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, target)

def prepare(dataset: Path, cache: Path, stage1: Path | None=None) -> dict:
    """新缓存有完整清单；已有缓存仅校验不覆盖。输入ZIP不解压执行。"""
    cfg = source_contract()
    source_hash = sha256(dataset)
    if source_hash != cfg['expected_dataset_sha256']:
        raise ValueError(msg('cache.invalid'))
    if (cache / 'ready.json').exists():
        check_seal(cache)
        ready = load_json(cache / 'ready.json')
        if ready['dataset_sha256'] != source_hash:
            raise ValueError(msg('cache.invalid'))
        return ready
    if cache.exists() and any(cache.iterdir()):
        raise ValueError(msg('cache.exists', path=cache))
    cache.mkdir(parents=True, exist_ok=True)
    if stage1 is None:
        native = read_dataset(dataset, cfg, cache / 'audit')
        stage1_prepare(native, cfg, cache)
    else:
        import_stage1(stage1, cache, cfg)
    bundle = load_bundle(cache)
    from gasbench.sequences import save_sequence_grid
    save_sequence_grid(bundle.features, bundle.native.process.columns.tolist(), cache / 'features')
    if sha256(dataset) != source_hash:
        raise ValueError(msg('file.changed', path=dataset))
    info = {'status': 'completed', 'dataset_sha256': source_hash, 'feature_count': bundle.features.shape[1], 'origins': len(bundle.features), 'test_origins': 960, 'test_targets_visible': False, 'source_stage1': str(stage1) if stage1 else None}
    write_json(cache / 'ready.json', info)
    seal(cache)
    LOGGER.info(msg('cache.done', rows=len(bundle.features), features=bundle.features.shape[1]))
    return info

def load_bundle(cache: Path) -> Bundle:
    """从数值缓存恢复，重新计算均值核验标签；禁止pickle对象数组。

NOTE: 名称不作为时间差异；索引值必须逐项相同。"""
    cfg = load_json(cache / 'data/contract.json')
    native = NativeDataset.load(cache / 'data/native_observations.npz')
    frames = []
    for kind in ('history', 'test'):
        with np.load(cache / f'features/{kind}_process_features.npz', allow_pickle=False) as z:
            frames.append(pd.DataFrame(z['features'], index=pd.DatetimeIndex(z['origins']), columns=z['names']))
    features = pd.concat(frames).sort_index()
    if not features.index.is_unique or features.shape[1] != 276:
        raise ValueError(msg('data.shape'))
    if any((c in cfg['targets'] for c in features.columns)):
        raise ValueError(msg('data.future'))
    means, _ = block_truth(native)
    with np.load(cache / 'labels/semi_truth_blocks_15m.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(means.to_numpy(), z['labels'])
    test = pd.Timestamp(cfg['protocol']['test_start'])
    if native.targets.loc[test:].notna().any().any():
        raise ValueError(msg('data.future'))
    expected = origins(test, cfg['protocol']['test_end'])
    if not features.loc[test:].index.equals(expected):
        np.testing.assert_array_equal(features.loc[test:].index.values, expected.values)
    return Bundle(native, features, means, cfg, cache)
