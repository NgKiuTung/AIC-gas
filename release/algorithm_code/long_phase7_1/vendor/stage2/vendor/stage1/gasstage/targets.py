"""私有赛事研究：真值专用模块。初赛15分钟观测不被伪装成已确认区间均值。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from gasstage.ingest import NativeDataset


def block_truth(data: NativeDataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """仅由复赛真实一分钟标签产生完整15分钟均值；缺少任何一分钟则不评分。"""
    y = data.targets.loc[~data.coarse]
    grid = pd.date_range(y.index.min().floor('15min'), pd.Timestamp(y.index.max().floor('15min').to_datetime64().astype('datetime64[ns]') + np.timedelta64(14, 'm')), freq='min')
    y = y.reindex(grid)
    counts = y.notna().resample('15min', label='left', closed='left').sum()
    means = y.resample('15min', label='left', closed='left').mean().where(counts == 15)
    means.index.name = counts.index.name = 'block_start'
    return means, counts


def future_index(index: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """生成未来96个区间左端点；列名t+15是区间编号标签，不直接作为点真值。"""
    p = cfg['protocol']
    offsets = p['block_start_offset_minutes'] + np.arange(p['blocks']) * p['block_minutes']
    return index.to_numpy(dtype='datetime64[ns]')[:,None] + offsets[None,:].astype('timedelta64[m]')


def future_truth(means: pd.DataFrame, index: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """返回(origin,96,2)真实均值；边界之外保留NaN，只允许评分端访问。"""
    stamps = future_index(index, cfg)
    return means.reindex(pd.DatetimeIndex(stamps.ravel())).to_numpy().reshape(len(index),96,2)


def history_only(means: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """只开放在cutoff之前已经完整结束的区间，隐藏当前评测整段及尾部真值。"""
    end_exclusive = pd.DatetimeIndex(means.index.to_numpy(dtype='datetime64[ns]') + np.timedelta64(15, 'm'))
    return means.loc[end_exclusive <= cutoff].copy()


def eligible_training(index: pd.DatetimeIndex, cutoff: pd.Timestamp, cfg: dict) -> np.ndarray:
    """以最远标签结束时间做purge，不能只按训练输入起点切分。"""
    p = cfg['protocol']
    end = pd.DatetimeIndex(index.to_numpy(dtype='datetime64[ns]') + np.timedelta64(int(p['block_start_offset_minutes'] + p['blocks']*p['block_minutes']), 'm'))
    return np.asarray(end <= cutoff)
