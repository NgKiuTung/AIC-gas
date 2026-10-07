"""私有赛事研究：读取Stage1产物、特征与标签分离、来源签名。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from gas2.common import load_json, save_npz, sha256, text, write_json
from gas2.features import extend_features
from gasstage.features import ProcessView, save_features
from gasstage.ingest import NativeDataset
from gasstage.targets import block_truth

LOGGER = logging.getLogger(__name__)
INPUTS = (
    "run_status.json", "data/contract.json", "data/native_observations.npz",
    "features/history_process_features.npz", "features/test_process_features.npz",
    "features/dictionary.csv", "labels/semi_truth_blocks_15m.npz",
)

@dataclass
class DataBundle:
    """native及truth只供拟合/评分；推理函数显式只接features与冻结历史。"""
    native: NativeDataset
    features: pd.DataFrame
    truth: pd.DataFrame
    contract: dict
    source: Path

def input_hashes(stage1: Path, dataset: Path | None = None) -> dict:
    """核验必须文件存在且Stage1成功；可选核对原zip，不静默重建或换数据。"""
    for rel in INPUTS:
        if not (stage1 / rel).is_file():
            raise FileNotFoundError(text("stage1.missing", path=stage1 / rel))
    cfg = load_json(stage1 / "data/contract.json")
    if load_json(stage1 / "run_status.json")["status"] != "completed":
        raise ValueError(text("stage1.invalid"))
    if dataset is not None and sha256(dataset) != cfg["expected_dataset_sha256"]:
        raise ValueError(text("stage1.invalid"))
    hashes = {rel: sha256(stage1 / rel) for rel in INPUTS}
    # NOTE: 固定基线模型也属于输入；不能只校验特征而遗漏其历史参数。
    folders = [stage1/"test/models"] + [stage1/"backtest"/f["name"]/"models" for f in cfg["folds"]]
    for folder in folders:
        if not (folder/"metadata.json").is_file():
            raise FileNotFoundError(text("stage1.missing", path=folder/"metadata.json"))
        for item in sorted(folder.glob("*")):
            if item.is_file():
                hashes[str(item.relative_to(stage1))] = sha256(item)
    return hashes

def read_feature_pair(stage1: Path) -> pd.DataFrame:
    """载入上游276列；不从验证期目标重建任何输入特征。"""
    parts = []
    for name in ["history_process_features", "test_process_features"]:
        with np.load(stage1 / "features" / f"{name}.npz", allow_pickle=False) as z:
            parts.append(pd.DataFrame(z["features"], index=pd.DatetimeIndex(z["origins"]), columns=z["names"]))
    result = pd.concat(parts).sort_index()
    if not result.index.is_unique or result.shape[1] != 276:
        raise ValueError(text("feature.invalid"))
    return result

def load_bundle(stage1: Path, output: Path, extended: bool = True) -> DataBundle:
    """按上游原值重建均值，和已保存标签逐项核验；只向output写入新增产物。"""
    cfg = load_json(stage1 / "data/contract.json")
    native = NativeDataset.load(stage1 / "data/native_observations.npz")
    features = read_feature_pair(stage1)
    dictionary = pd.read_csv(stage1 / "features/dictionary.csv").to_dict("records")
    if extended:
        features, extra = extend_features(ProcessView.from_native(native, cfg), features, cfg)
        dictionary.extend(extra)
    means, counts = block_truth(native)
    with np.load(stage1 / "labels/semi_truth_blocks_15m.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(means.to_numpy(), z["labels"])
        np.testing.assert_array_equal(means.index.to_numpy(dtype="datetime64[ns]"), z["time"])
    test_start = pd.Timestamp(cfg["protocol"]["test_start"])
    if native.targets.loc[test_start:].notna().any().any():
        raise ValueError(text("data.future"))
    output.mkdir(parents=True, exist_ok=True)
    save_features(output / "history_process_features.npz", features.loc[features.index < test_start])
    save_features(output / "test_process_features.npz", features.loc[features.index >= test_start])
    pd.DataFrame(dictionary).to_csv(output / "dictionary.csv", index=False)
    pd.DataFrame({"name": features.columns,
                  "history_missing": features.loc[features.index < test_start].isna().mean().to_numpy(),
                  "test_missing": features.loc[features.index >= test_start].isna().mean().to_numpy()
                  }).to_csv(output / "coverage.csv", index=False)
    write_json(output / "contract.json", {"base_features": 276, "features": len(features.columns),
               "no_test_targets": True, "native_frequency_preserved": True, "input_contract": cfg})
    LOGGER.info(text("prepare.done", rows=len(features), features=features.shape[1]))
    return DataBundle(native, features, means, cfg, stage1)

def load_cached_bundle(stage1: Path, feature_dir: Path) -> DataBundle:
    """从已签名的Stage2特征缓存恢复，不依赖另一个环境的pickle。"""
    cfg = load_json(stage1 / "data/contract.json")
    native = NativeDataset.load(stage1 / "data/native_observations.npz")
    frames = []
    for kind in ["history", "test"]:
        with np.load(feature_dir / f"{kind}_process_features.npz", allow_pickle=False) as z:
            frames.append(pd.DataFrame(z["features"], index=pd.DatetimeIndex(z["origins"]), columns=z["names"]))
    means, _ = block_truth(native)
    return DataBundle(native, pd.concat(frames), means, cfg, stage1)
