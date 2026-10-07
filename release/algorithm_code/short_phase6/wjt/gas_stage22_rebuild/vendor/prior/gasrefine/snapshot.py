"""私有赛事研究：快照估计器，保存可重放模型，不读取验证负荷。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from gas2.common import load_json, write_json, sha256
from gas2.model import ProcessModel, _active_columns, _time_guard
from gasstage.targets import history_only
from gasrefine.common import message


def training_rows(bundle, cutoff: pd.Timestamp, candidate: dict):
    """按原始目标时刻和观测可用时刻共同过滤；初赛仅作为同记录名义目标。"""
    native = bundle.native
    time_index = native.values.index
    delay = np.where(native.coarse, bundle.contract["protocol"]["pre_observation_delay_minutes"], 0)
    available = pd.DatetimeIndex(time_index.to_numpy(dtype="datetime64[ns]") + delay.astype("timedelta64[m]"))
    pre = candidate["pre_weight"]
    keep = ((available < cutoff) & (time_index.minute % candidate["source_stride_minutes"] == 0)
            & ((time_index >= cutoff-pd.Timedelta(days=candidate["lookback_days"])) | (native.coarse & (pre > 0)))
            & (~native.coarse | (pre > 0)))
    y = native.targets.to_numpy()[keep]
    if candidate["features"] == "raw":
        names = [f"feat_{col}__last" for col in native.process.columns]
        x = native.process.to_numpy()[keep]
    else:
        names = list(bundle.features.columns[:276])
        x = bundle.features.reindex(available[keep])[names].to_numpy()
    stage_weight = np.where(native.coarse[keep], pre, 1.0)
    valid = np.isfinite(y).all(axis=1) & (y > 0).all(axis=1) & (np.isfinite(x).mean(axis=1) >= 0.5)
    if valid.sum() < 32:
        raise ValueError(message("training.empty"))
    age = (cutoff-time_index[keep][valid]).total_seconds().to_numpy()/86400
    half_life = candidate.get("half_life_days", 0)
    recency = np.exp2(-age/half_life) if half_life else np.ones(len(age))
    weights = stage_weight[valid] * recency
    return x[valid].astype("float32"), y[valid], weights, names, {
        "rows": int(valid.sum()), "pre_rows": int(native.coarse[keep][valid].sum()),
        "last_target_time": str(time_index[keep][valid][-1]), "max_available_time": str(available[keep][valid][-1]),
        "target_definition": "same-record nominal load; not pre interval means"}


@dataclass
class SnapshotModel:
    """两输出的冻结模型；ratio模式只作为实测对照，不假定为最佳。"""
    metadata: dict
    models: dict
    history: pd.DataFrame

    @classmethod
    def fit(cls, bundle, cutoff, candidate, settings, directory, deadline):
        """拟合明确候选并写模型摘要，训练和周期历史都截止于cutoff。"""
        directory.mkdir(parents=True, exist_ok=True)
        x, original, weights, names, info = training_rows(bundle, cutoff, candidate)
        y = original.copy()
        if candidate.get("target_mode", "direct") == "ratio":
            y[:, 0] = original[:, 0] / original[:, 1]
        meta = {"candidate": candidate, "cutoff": str(cutoff), "features": names,
                "models": {}, "contract": bundle.contract, "training": info, "settings": settings}
        models = {}
        active = _active_columns(x)
        for target in range(2):
            center = float(np.median(y[:, target]))
            weight = weights / y[:, target]
            params = {**settings["tree"], **candidate.get("tree", {}), "objective": "regression_l1",
                      "metric": "None", "verbosity": -1, "num_threads": settings["threads"],
                      "seed": settings["seed"], "deterministic": True, "force_col_wise": True}
            train = lgb.Dataset(np.ascontiguousarray(x[:, active]), label=y[:,target]-center,
                                weight=weight/weight.mean(), free_raw_data=True)
            model = lgb.train(params, train, num_boost_round=candidate["rounds"], callbacks=[_time_guard(deadline)])
            key = f"t{target}"
            model.save_model(str(directory/f"{key}.txt")); models[key] = model
            meta["models"][key] = {"target": target, "center": center, "active": np.flatnonzero(active).tolist(),
                                   "sha256": sha256(directory/f"{key}.txt")}
        history = history_only(bundle.truth, cutoff)
        history.to_csv(directory/"history.csv", float_format="%.17g")
        meta["history_sha256"] = sha256(directory/"history.csv")
        write_json(directory/"model.json", meta)
        return cls(meta, models, history)

    def current(self, features):
        """过程特征输入转为代理负荷；不宣称它是真实观测。"""
        x = features[self.metadata["features"]].to_numpy(dtype="float32")
        result = np.empty((len(features), 2))
        for key, info in self.metadata["models"].items():
            result[:,info["target"]] = self.models[key].predict(
                x[:,info["active"]], num_threads=self.metadata["settings"]["threads"]) + info["center"]
        if self.metadata["candidate"].get("target_mode", "direct") == "ratio":
            result[:,0] = np.clip(result[:,0],0,1)*np.maximum(result[:,1],0)
        return result

    @classmethod
    def load(cls, directory):
        """固定文件名与哈希校验，拒绝被修改的权重和冻结历史。"""
        meta = load_json(directory/"model.json")
        models = {}
        for key, info in meta["models"].items():
            if key not in {"t0", "t1"} or sha256(directory/f"{key}.txt") != info["sha256"]:
                raise ValueError(message("model.hash"))
            models[key] = lgb.Booster(model_file=str(directory/f"{key}.txt"))
        if sha256(directory/"history.csv") != meta["history_sha256"]:
            raise ValueError(message("history.hash"))
        history = pd.read_csv(directory/"history.csv", index_col=0, parse_dates=True, float_precision="round_trip")
        if (history.index.to_numpy(dtype="datetime64[ns]") + np.timedelta64(15,"m") > pd.Timestamp(meta["cutoff"]).to_datetime64()).any():
            raise ValueError(message("history.future"))
        return cls(meta, models, history)


class OriginalSnapshot:
    """只读适配原Stage2代理，保留58.1578分版本的所有输入和权重。"""
    def __init__(self, directory):
        self.model = ProcessModel.load(directory)
        self.metadata = self.model.metadata
        self.history = self.model.history

    def current(self, features):
        """提取原算法中已算出的当前负荷代理，不重新拟合。"""
        names = [f"feat_{n}__last" for n in self.metadata["process_names"]]
        x = features[names].to_numpy(dtype="float32")
        result = np.zeros((len(x),2))
        for key, info in self.metadata["models"].items():
            result[:,info["target"]] = self.model.models[key].predict(
                x[:,info["active"]], num_threads=4)+info["center"]
        return result
