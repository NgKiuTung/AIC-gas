"""私有赛事研究：候选运行与恢复。仅复用同签名已完成模型，不修改旧run回执。"""
from __future__ import annotations
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd

from gas2.common import load_json, save_npz, sha256, text, write_json, fingerprint
from gas2.model import ProcessModel
from gasstage.baselines import predict_baselines, reload_pack

LOGGER = logging.getLogger(__name__)
BASELINES = {"stage1_frozen_week4":"frozen_week4",
             "stage1_proxy_hold_semi":"proxy_hold_semi",
             "stage1_process_week_joint":"process_week_joint"}

def stage1_predictions(stage1: Path, features: pd.DataFrame, contract: dict, fold_name: str) -> dict:
    """复用Stage1固定基线，沿用其训练截止，不重新根据本轮成绩调整。"""
    path = stage1/"test/models" if fold_name=="test" else stage1/"backtest"/fold_name/"models"
    pack = reload_pack(path)
    base = features.loc[:,pack.feature_names]
    pred = predict_baselines(pack,base,contract)
    return {new:pred[old] for new,old in BASELINES.items()}

def run_candidate(bundle, candidate: dict, settings: dict, cutoff: pd.Timestamp,
                  features: pd.DataFrame, path: Path, global_signature: str,
                  deadline: float, fold_name: str) -> np.ndarray:
    """逐候选原子落盘；回执校验包括预测文件及全部模型摘要。"""
    signature = fingerprint({"global":global_signature, "candidate":candidate,
                             "cutoff":str(cutoff), "start":str(features.index[0]),
                             "end":str(features.index[-1])})
    receipt_path = path/"receipt.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if (receipt["signature"] != signature
                or sha256(path/"predictions.npz") != receipt["prediction_sha256"]
                or sha256(path/"model/model.json") != receipt["model_metadata_sha256"]):
            raise ValueError(text("run.signature"))
        ProcessModel.load(path/"model")
        LOGGER.info(text("candidate.reuse",fold=fold_name,candidate=candidate["id"]))
        with np.load(path/"predictions.npz",allow_pickle=False) as z:
            np.testing.assert_array_equal(z["origins"],features.index.to_numpy(dtype="datetime64[ns]"))
            return z["prediction"].copy()
    if time.time() >= deadline:
        raise TimeoutError(text("candidate.timeout"))
    started = time.perf_counter()
    LOGGER.info(text("candidate.start",fold=fold_name,candidate=candidate["id"]))
    path.mkdir(parents=True,exist_ok=True)
    model = ProcessModel.fit(bundle,cutoff,candidate,settings,path/"model",deadline)
    pred = model.predict(features)
    save_npz(path/"predictions.npz",origins=features.index.to_numpy(dtype="datetime64[ns]"),prediction=pred)
    write_json(receipt_path,{"status":"completed","signature":signature,
               "seconds":time.perf_counter()-started,"prediction_sha256":sha256(path/"predictions.npz"),
               "model_metadata_sha256":sha256(path/"model/model.json")})
    return pred
