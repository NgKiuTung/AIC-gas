"""私有赛事研究：不搜索超参的六种回测基线，目标是验证真实输入条件。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from gasstage.common import message, write_json
from gasstage.ingest import NativeDataset
from gasstage.ridge import RidgeBaseline
from gasstage.targets import eligible_training, future_index, future_truth, history_only

BASELINES = ('frozen_last', 'frozen_week4', 'proxy_hold_semi', 'proxy_hold_joint',
             'process_week_joint', 'direct_blocks_semi')


def reconcile(pred: np.ndarray) -> np.ndarray:
    """共同输出约束：非负及总量不小于子组；不按未知开机数量裁剪下限。"""
    p = np.maximum(pred.copy(), 0.)
    p[...,1] = np.maximum(p[...,1],p[...,0])
    return p


def frozen_week(history: pd.DataFrame, index: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """每个目标区间只引用cutoff之前的完整历史块，测试期绝不更新负荷周模板。"""
    if history.empty or history.notna().sum().min() < 1:
        raise ValueError(message('baselines.error.01'))
    stamps = future_index(index,cfg)
    last = history.index[-1].to_datetime64()
    week = np.timedelta64(7*24*60,'m')
    jump = np.maximum(1, np.ceil((stamps-last)/week).astype('int64'))
    # NOTE: 验证超过一周后不能取验证前半周的隐藏负荷，必须继续向更早历史回退。
    offsets = jump[...,None] + np.arange(4)
    lookup = stamps[...,None]-offsets*week
    vals = history.reindex(pd.DatetimeIndex(lookup.ravel())).to_numpy().reshape(len(index),96,4,2)
    count = np.isfinite(vals).sum(axis=2)
    totals = np.nansum(vals,axis=2)
    fallback = history.median().to_numpy()
    return np.where(count > 0, totals/np.maximum(count,1), fallback)


def _snapshot_training(data: NativeDataset, cutoff: pd.Timestamp, cfg: dict, joint: bool) -> tuple:
    delay = np.where(data.coarse,cfg['protocol']['pre_observation_delay_minutes'],0)
    available = data.values.index + pd.to_timedelta(delay,unit='min')
    keep = (available < cutoff) & (data.values.index.minute % cfg['baselines']['source_training_stride_minutes'] == 0)
    if not joint:
        keep &= ~data.coarse
    x = data.process.to_numpy()[keep]
    y = data.targets.to_numpy()[keep]
    valid = np.isfinite(y).all(axis=1) & (y > 0).all(axis=1) & (np.isfinite(x).mean(axis=1) >= 0.5)
    x, y = x[valid], y[valid]
    coarse = data.coarse[keep][valid]
    # NOTE: 此处学习同一原始记录的过程-负荷关系；没有把初赛点值当区间平均标签。
    return x, y, {'rows':len(y),'pre_rows':int(coarse.sum()),'semi_rows':int((~coarse).sum()),
                  'pre_scope':'nominal same-record paired observations; interval semantics unresolved'}


@dataclass
class BaselinePack:
    """预测所需的历史真值已在构造时截断；持有冻结模型，不持有验证真值。"""
    history: pd.DataFrame
    last: np.ndarray
    proxy_semi: RidgeBaseline
    proxy_joint: RidgeBaseline
    direct: RidgeBaseline
    process_names: list[str]
    feature_names: list[str]
    metadata: dict


def fit_baselines(data: NativeDataset, features: pd.DataFrame, means: pd.DataFrame,
                  cutoff: pd.Timestamp, cfg: dict, model_dir: Path) -> BaselinePack:
    """冻结于预测段起点；每个模型保存训练来源计数和最远可见标签时间。"""
    hist = history_only(means,cutoff)
    last_rows = data.targets.loc[data.targets.index < cutoff].dropna()
    if last_rows.empty:
        raise ValueError(message('baselines.error.02'))
    models, meta = [], {'cutoff':str(cutoff),'fit_config':cfg['baselines'],'tuning':'none'}
    for joint, key in [(False,'proxy_semi'),(True,'proxy_joint')]:
        x,y,info = _snapshot_training(data,cutoff,cfg,joint)
        model = RidgeBaseline.fit(x,y,cfg['baselines']['ridge_alpha'])
        model.save(model_dir / f'{key}.npz')
        models.append(model); meta[key] = info
    select = eligible_training(features.index,cutoff,cfg) & (features.index >= pd.Timestamp(cfg['sources'][2]['start']))
    train_features = features.loc[select]
    labels = future_truth(means,train_features.index,cfg)
    complete = np.isfinite(labels).all(axis=(1,2)) & (labels>0).all(axis=(1,2))
    direct = RidgeBaseline.fit(train_features.to_numpy()[complete], labels[complete].reshape(-1,192),
                               cfg['baselines']['direct_ridge_alpha'])
    direct.save(model_dir/'direct.npz')
    meta['direct'] = {'rows':int(complete.sum()),'targets':'96 complete 15-minute means, two loads',
                      'last_training_origin':str(train_features.index[complete][-1]),
                      'pre_rows':0,'reason':'pre interval semantics not assumed'}
    meta['process_names'],meta['feature_names'] = list(data.process.columns),list(features.columns)
    meta['last_target_time'] = str(last_rows.index[-1])
    write_json(model_dir/'metadata.json',meta)
    hist.to_csv(model_dir/'history_blocks.csv',float_format='%.17g')
    np.save(model_dir/'last.npy',last_rows.iloc[-1].to_numpy(),allow_pickle=False)
    return BaselinePack(hist,last_rows.iloc[-1].to_numpy(),models[0],models[1],direct,
                        list(data.process.columns),list(features.columns),meta)


def predict_baselines(pack: BaselinePack, features: pd.DataFrame, cfg: dict) -> dict[str,np.ndarray]:
    """推理只接收过程特征和cutoff前历史；生成全部96区间，不访问未来过程或负荷。"""
    if list(features.columns) != pack.feature_names:
        raise ValueError(message('baselines.error.03'))
    last_cols = [f'feat_{name}__last' for name in pack.process_names]
    x = features[last_cols].to_numpy()
    semi = pack.proxy_semi.predict(x)
    joint = pack.proxy_joint.predict(x)
    week = frozen_week(pack.history,features.index,cfg)
    decay = np.exp(-(np.arange(96)*15+7.5+cfg['protocol']['block_start_offset_minutes'])/
                   cfg['baselines']['proxy_decay_minutes'])
    out = {'frozen_last':np.broadcast_to(pack.last,(len(features),96,2)),
           'frozen_week4':week,
           'proxy_hold_semi':np.repeat(semi[:,None,:],96,axis=1),
           'proxy_hold_joint':np.repeat(joint[:,None,:],96,axis=1),
           'process_week_joint':week+(joint-week[:,0,:])[:,None,:]*decay[None,:,None],
           'direct_blocks_semi':pack.direct.predict(features.to_numpy()).reshape(-1,96,2)}
    return {key:reconcile(value) for key,value in out.items()}


def reload_pack(model_dir: Path) -> BaselinePack:
    """从自有数值文件和元数据复原基线，用于磁盘重放验收。"""
    from gasstage.common import load_json
    m = load_json(model_dir/'metadata.json')
    hist = pd.read_csv(model_dir/'history_blocks.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    return BaselinePack(hist,np.load(model_dir/'last.npy',allow_pickle=False),
        RidgeBaseline.load(model_dir/'proxy_semi.npz'),RidgeBaseline.load(model_dir/'proxy_joint.npz'),
        RidgeBaseline.load(model_dir/'direct.npz'),m['process_names'],m['feature_names'],m)
