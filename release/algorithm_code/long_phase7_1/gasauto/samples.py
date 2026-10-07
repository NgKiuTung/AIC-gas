"""私有AIC赛事研究：分别清除2h/24h标签跨界，训练/预测协变量合同一致。"""
import numpy as np
import pandas as pd
from gasauto.protocol import BLOCKS, shift, truth_for
from gasbench.samples import known_future
from gasbench.common import array_hash

def rows_for(bundle, cutoff, cfg, task):
    """不把一个起点的标签拆散做随机验证；控制训练密度但不缩短外层验证。"""
    index = bundle.features.index
    first = max(pd.Timestamp('2025-05-10'), shift(cutoff, -1440 * cfg.get('lookback_days', 120)))
    right = index.to_numpy(dtype='datetime64[ns]') + np.timedelta64(BLOCKS[task] * 15, 'm')
    take = (index >= first) & (right <= pd.Timestamp(cutoff).to_datetime64())
    rows = np.flatnonzero(take)[::cfg.get('origin_stride', 2)]
    cap = cfg.get('max_train_origins', 0)
    if cap and len(rows) > cap:
        rows = rows[np.linspace(0, len(rows) - 1, cap, dtype=int)]
    if len(rows) < 32:
        raise ValueError('too few task training origins')
    return rows

def future_for(index, contract, blocks):
    """预测跨度/日历/已知电价；不含未来实际过程量。"""
    return known_future(index, contract)[:, :blocks]

def direct_samples(bundle, cutoff, cfg, task, bank):
    """全任务均匀跨度抽样；long不会额外偏向前8块。"""
    blocks = BLOCKS[task]
    rows = rows_for(bundle, cutoff, cfg, task)
    index = bundle.features.index.take(rows)
    y = truth_for(bundle, index, task)
    anchor = bank.training(rows, blocks)
    future = future_for(index, bundle.contract, blocks)
    count = min(blocks, cfg.get('sample_horizons', 48))
    rng = np.random.default_rng(20261004)
    hh = np.stack([np.sort(rng.choice(blocks, count, replace=False)) for _ in rows]).ravel()
    rr = np.repeat(np.arange(len(rows)), count)
    x = np.column_stack([bundle.features.to_numpy(dtype='float32')[rows[rr]], future[rr, hh], anchor[rr, hh]])
    truth, base = (y[rr, hh], anchor[rr, hh])
    valid = np.isfinite(truth).all(axis=1) & (truth > 0).all(axis=1)
    meta = {'task': task, 'blocks': blocks, 'origins': len(rows), 'pairs': int(valid.sum()), 'sample_horizons': count, 'last_origin': str(index[-1]), 'label_end_exclusive': str(shift(index[-1], blocks * 15)), 'cutoff': str(cutoff), 'truth_hash': array_hash(truth[valid]), 'rows_hash': array_hash(rows), 'x_hash': array_hash(x[valid]), 'anchor_policy': '10day past-trained reference_58', 'pre_interval_labels_used': False}
    return (x[valid].astype('float32'), truth[valid], base[valid], meta)

def direct_matrix(features, anchor, contract):
    """一次只装配有限起点批次。"""
    n, blocks, _ = anchor.shape
    future = future_for(features.index, contract, blocks)
    return np.column_stack([np.repeat(features.to_numpy(dtype='float32'), blocks, axis=0), future.reshape(-1, 6), anchor.reshape(-1, 2)]).astype('float32')



def wide_proxy_samples(bundle, cutoff, cfg):
    """Phase6-style wide causal proxy: 15-minute origins, preliminary becomes available after 15m.

    Labels are nominal same-record loads for process-proxy training only; they are not
    treated as official 15-minute interval-mean targets.
    """
    native = bundle.native
    time = native.values.index
    delay = np.where(native.coarse, bundle.contract["protocol"]["pre_observation_delay_minutes"], 0)
    available = pd.DatetimeIndex(time.to_numpy(dtype="datetime64[ns]") + delay.astype("timedelta64[m]"))
    lookback_days = int(cfg.get("lookback_days", 180))
    pre_weight = float(cfg.get("pre_weight", 0.25))
    after = time >= pd.Timestamp(cutoff) - pd.Timedelta(days=lookback_days)
    # Phase6 policy: preliminary history may remain in proxy training at reduced weight.
    after = after | (native.coarse & (pre_weight > 0))
    keep = (available < pd.Timestamp(cutoff)) & after & (time.minute % 15 == 0)
    keep &= (~native.coarse | (pre_weight > 0))
    raw_time = time[keep]
    feature_time = available[keep]
    y = native.targets.to_numpy()[keep]
    coarse = native.coarse[keep]
    # Stage1 15-minute feature grid represents information available at the origin.
    positions = bundle.features.index.get_indexer(feature_time)
    exists = positions >= 0
    if not exists.any():
        raise ValueError("wide proxy has no aligned feature origins")
    x = np.full((len(positions), bundle.features.shape[1]), np.nan, dtype="float32")
    x[exists] = bundle.features.to_numpy(dtype="float32")[positions[exists]]
    valid = (
        exists
        & np.isfinite(y).all(axis=1)
        & (y > 0).all(axis=1)
        & (np.isfinite(x).mean(axis=1) >= 0.5)
    )
    x = x[valid]
    y = y[valid]
    coarse = coarse[valid]
    raw_time = raw_time[valid]
    feature_time = feature_time[valid]
    weights = np.where(coarse, pre_weight, 1.0).astype("float64")
    if len(y) < 32:
        raise ValueError("too few wide proxy rows")
    meta = {
        "rows": int(len(y)),
        "pre_rows": int(coarse.sum()),
        "semi_rows": int((~coarse).sum()),
        "pre_weight": pre_weight,
        "lookback_days": lookback_days,
        "cutoff": str(pd.Timestamp(cutoff)),
        "last_target_time": str(raw_time[-1]),
        "last_feature_time": str(feature_time[-1]),
        "kind": "phase6_style_wide_nominal_proxy",
        "feature_count": int(x.shape[1]),
        "sample_hashes": {
            "x": array_hash(x),
            "y": array_hash(y),
            "source_weights": array_hash(weights),
        },
    }
    return x, y, weights, meta
