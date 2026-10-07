"""私有赛事研究：冻结的LightGBM过程模型，独立保存和可验证重载。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from gas2.common import load_json, text, write_json, sha256
from gas2.losses import make_objective
from gas2.samples import direct_training, proxy_training, pair_matrix, CONTEXT_NAMES
from gasstage.baselines import frozen_week, reconcile
from gasstage.targets import history_only

def _time_guard(deadline):
    def callback(env):
        if time.time() >= deadline:
            raise TimeoutError(text("candidate.timeout"))
    callback.order = 1
    callback.before_iteration = True
    return callback

def _active_columns(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    active = finite.sum(axis=0) >= 16
    low = np.min(np.where(finite, x, np.inf), axis=0)
    high = np.max(np.where(finite, x, -np.inf), axis=0)
    return active & (high > low)

def _fit_one(x, y, source_weight, candidate, settings, deadline):
    active = _active_columns(x)
    if not active.any():
        raise ValueError(text("data.empty"))
    center = float(np.median(y))
    weights = source_weight / y
    weights = weights / weights.mean()
    params = {**settings["tree"], "verbosity":-1, "num_threads":settings["threads"],
              "seed":settings["seed"], "deterministic":True, "force_col_wise":True,
              "objective":"regression_l1", "metric":"None"}
    if candidate["loss"] == "tail":
        params["objective"] = make_objective(center, settings["loss"])
    train = lgb.Dataset(np.ascontiguousarray(x[:,active]), label=y-center, weight=weights, free_raw_data=True)
    model = lgb.train(params, train, num_boost_round=candidate["rounds"],
                      callbacks=[_time_guard(deadline)])
    return model, active, center

@dataclass
class ProcessModel:
    """模型只持有训练截止前历史；预测调用不接收NativeDataset或真实标签。"""
    metadata: dict
    models: dict
    history: pd.DataFrame

    @classmethod
    def fit(cls, bundle, cutoff: pd.Timestamp, candidate: dict, settings: dict,
            directory: Path, deadline: float) -> ProcessModel:
        """训练点估计或共享跨度模型，任何失败都不写成功回执。"""
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"candidate":candidate, "cutoff":str(cutoff), "models":{},
                    "features":list(bundle.features.columns),
                    "process_names":list(bundle.native.process.columns),
                    "contract":bundle.contract, "settings":settings}
        models = {}
        if candidate["kind"] == "proxy":
            x,y,weights,info = proxy_training(bundle,cutoff,candidate,settings)
            blocks = np.zeros(len(y), dtype=int)
            bands = [(0,1)]
        else:
            x,y,blocks,info = direct_training(bundle,cutoff,settings)
            weights = np.ones(len(y))
            bands = [(0,8),(8,96)]
        for target in range(2):
            for lo,hi in bands:
                key = f"t{target}_b{lo}"
                take = (blocks>=lo) & (blocks<hi)
                model, active, center = _fit_one(x[take], y[take,target], weights[take],
                                                 candidate,settings,deadline)
                model.save_model(str(directory/f"{key}.txt"))
                models[key] = model
                metadata["models"][key] = {"active":np.flatnonzero(active).tolist(), "center":center,
                    "lo":lo, "hi":hi, "target":target, "sha256":sha256(directory/f"{key}.txt")}
        metadata["training"] = info
        history = history_only(bundle.truth,cutoff)
        history.to_csv(directory/"history.csv",float_format="%.17g")
        metadata["history_sha256"] = sha256(directory/"history.csv")
        write_json(directory/"model.json",metadata)
        return cls(metadata,models,history)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """返回(origin,96,2)，拒绝早于训练截止的起点；未来观测永远不作输入。"""
        if features.index.min() < pd.Timestamp(self.metadata["cutoff"]):
            raise ValueError(text("data.future"))
        if list(features.columns) != self.metadata["features"]:
            raise ValueError(text("feature.invalid"))
        if self.metadata["candidate"]["kind"] == "proxy":
            return self._predict_proxy(features)
        return self._predict_direct(features)

    def _predict_proxy(self, features):
        names = [f"feat_{col}__last" for col in self.metadata["process_names"]]
        x = features[names].to_numpy(dtype="float32")
        current = np.zeros((len(features),2))
        for key, info in self.metadata["models"].items():
            current[:,info["target"]] = self.models[key].predict(x[:,info["active"]]) + info["center"]
        cfg = self.metadata["contract"]
        week = frozen_week(self.history,features.index,cfg)
        offset = cfg["protocol"]["block_start_offset_minutes"]
        decay = np.exp(-(np.arange(96)*15+7.5+offset)/self.metadata["settings"]["proxy_decay_minutes"])
        return reconcile(week+(current-week[:,0,:])[:,None,:]*decay[None,:,None])

    def _predict_direct(self, features):
        result = np.empty((len(features),96,2), dtype="float64")
        cfg = self.metadata["contract"]
        for start in range(0,len(features),64):
            batch = features.iloc[start:start+64]
            for key,info in self.metadata["models"].items():
                row = np.repeat(np.arange(len(batch)),info["hi"]-info["lo"])
                block = np.tile(np.arange(info["lo"],info["hi"]),len(batch))
                x = pair_matrix(batch,row,block,cfg)
                p = self.models[key].predict(x[:,info["active"]]) + info["center"]
                result[start:start+len(batch),info["lo"]:info["hi"],info["target"]] = p.reshape(len(batch),-1)
        return reconcile(result)

    @classmethod
    def load(cls, directory: Path) -> ProcessModel:
        """校验模型及冻结历史文件，不接受摘要不符或路径跳转。"""
        meta = load_json(directory/"model.json")
        models = {}
        for key, info in meta["models"].items():
            if "/" in key or ".." in key or sha256(directory/f"{key}.txt") != info["sha256"]:
                raise ValueError(text("model.invalid",path=directory))
            models[key] = lgb.Booster(model_file=str(directory/f"{key}.txt"))
        if sha256(directory/"history.csv") != meta["history_sha256"]:
            raise ValueError(text("model.invalid",path=directory))
        history = pd.read_csv(directory/"history.csv", index_col=0, parse_dates=True, float_precision="round_trip")
        return cls(meta,models,history)
