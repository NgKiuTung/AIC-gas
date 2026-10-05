"""私有赛事研究：按输入起点与标签结束双重过滤的训练样本。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gas2.common import text
from gasstage.features import price_at
from gasstage.targets import eligible_training, future_truth

CONTEXT_NAMES = ["ctx_horizon_fraction", "ctx_hour_sin", "ctx_hour_cos",
                 "ctx_week_sin", "ctx_week_cos", "ctx_price"]

def future_context(index: pd.DatetimeIndex, block: np.ndarray, contract: dict) -> np.ndarray:
    """仅由目标区间时标和官方预知电价构成，不读取任何未来生产数据。"""
    p = contract["protocol"]
    offsets = p["block_start_offset_minutes"] + np.asarray(block) * p["block_minutes"]
    times = pd.DatetimeIndex(index.to_numpy(dtype="datetime64[ns]") + offsets.astype("timedelta64[m]"))
    minute = times.hour.to_numpy() * 60 + times.minute.to_numpy() + 7.5
    week = times.dayofweek.to_numpy() + minute / 1440
    return np.column_stack([np.asarray(block)/95, np.sin(2*np.pi*minute/1440),
                            np.cos(2*np.pi*minute/1440), np.sin(2*np.pi*week/7),
                            np.cos(2*np.pi*week/7), price_at(times)]).astype("float32")

def pair_matrix(features: pd.DataFrame, row: np.ndarray, block: np.ndarray, contract: dict) -> np.ndarray:
    """装配有限批次的起点特征＋跨度上下文；不会创建未来过程量。"""
    base = features.to_numpy(dtype="float32")[row]
    context = future_context(features.index.take(row), block, contract)
    return np.column_stack([base, context])

def direct_training(bundle, cutoff: pd.Timestamp, settings: dict) -> tuple:
    """仅使用复赛完整15分钟标签；保守清除24小时尾窗，随机数与标签无关。"""
    index = bundle.features.index
    semi_start = pd.Timestamp(bundle.contract["sources"][2]["start"])
    lookback = cutoff - pd.Timedelta(days=settings["lookback_days"])
    mask = eligible_training(index, cutoff, bundle.contract) & (index >= max(semi_start, lookback))
    features = bundle.features.loc[mask].iloc[::settings["train_origin_stride"]]
    if len(features) < 32:
        raise ValueError(text("data.empty"))
    truth = future_truth(bundle.truth, features.index, bundle.contract)
    rng = np.random.default_rng(settings["seed"])
    per = settings["horizons_per_origin"]
    # NOTE: 两个跨度带都被抽样；验证仍评估全部96块，不把节点插值冒充完整预测。
    short_count = max(1, per//2)
    blocks = np.concatenate([rng.integers(0, 8, (len(features),short_count)),
                             rng.integers(8, 96, (len(features),max(1,per-short_count)))], axis=1)
    row = np.repeat(np.arange(len(features)), blocks.shape[1])
    block = blocks.ravel()
    matrix = pair_matrix(features, row, block, bundle.contract)
    labels = truth[row, block]
    complete = np.isfinite(labels).all(axis=1) & (labels > 0).all(axis=1)
    if complete.sum() < 32:
        raise ValueError(text("data.empty"))
    meta = {"origins": len(features), "pairs": int(complete.sum()),
            "last_training_origin": str(features.index[-1]),
            "label_end_exclusive": str(pd.Timestamp(features.index[-1].to_datetime64()
                + np.timedelta64(1440 + bundle.contract["protocol"]["block_start_offset_minutes"], "m"))),
            "pre_interval_labels_used": False, "cutoff": str(cutoff)}
    return matrix[complete], labels[complete], block[complete], meta

def proxy_training(bundle, cutoff: pd.Timestamp, candidate: dict, settings: dict) -> tuple:
    """学习同一原始记录的过程-负荷关系，初赛标签不冒充区间均值。"""
    native = bundle.native
    time = native.values.index
    delay = np.where(native.coarse, bundle.contract["protocol"]["pre_observation_delay_minutes"], 0)
    available = time.to_numpy(dtype="datetime64[ns]") + delay.astype("timedelta64[m]")
    after = time >= cutoff - pd.Timedelta(days=settings["lookback_days"])
    after = after | (native.coarse & (candidate["pre_weight"] > 0))
    keep = (available < cutoff.to_datetime64()) & after & (time.minute % 15 == 0)
    keep &= (~native.coarse | (candidate["pre_weight"] > 0))
    x, y = native.process.to_numpy()[keep], native.targets.to_numpy()[keep]
    coarse = native.coarse[keep]
    good = np.isfinite(y).all(axis=1) & (y > 0).all(axis=1) & (np.isfinite(x).mean(axis=1) >= 0.5)
    weights = np.where(coarse[good], candidate["pre_weight"], 1.0)
    if good.sum() < 32:
        raise ValueError(text("data.empty"))
    return x[good].astype("float32"), y[good], weights, {
        "rows":int(good.sum()), "pre_rows":int(coarse[good].sum()),
        "pre_weight":candidate["pre_weight"], "cutoff":str(cutoff),
        "last_target_time":str(time[keep][good][-1]), "kind":"nominal_same_record_not_interval"}
