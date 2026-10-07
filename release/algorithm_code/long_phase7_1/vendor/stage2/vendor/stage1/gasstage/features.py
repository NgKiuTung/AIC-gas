"""私有赛事研究：仅过程变量的因果特征；保持原始分钟观测，按预测起点采样。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from gasstage.common import message, ROOT, load_json, save_npz
from gasstage.ingest import NativeDataset


@dataclass
class ProcessView:
    """预测模块只能持有此视图；本结构根本不存generator真实目标。"""
    values: pd.DataFrame
    observed_time: pd.Series

    @classmethod
    def from_native(cls, data: NativeDataset, cfg: dict) -> ProcessView:
        """按可用时刻建索引；初赛15分钟延迟是保守假设而非确认的原始统计口径。"""
        delay = np.where(data.coarse, cfg['protocol']['pre_observation_delay_minutes'],
                         cfg['protocol']['semi_observation_delay_minutes'])
        available = data.values.index + pd.to_timedelta(delay, unit='min')
        values = data.process.copy()
        values.index = available
        times = pd.Series(data.values.index.to_numpy(), index=available)
        # NOTE: 边界同时到达时，更新的真实观测优先；不把初赛值复制为分钟观测。
        keep = ~available.duplicated(keep='last')
        return cls(values.loc[keep].sort_index(), times.loc[keep].sort_index())

    def prefix(self, time: pd.Timestamp) -> ProcessView:
        """只读历史切片，供真实前缀重放测试。"""
        return ProcessView(self.values.loc[:time], self.observed_time.loc[:time])


def origins(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """生成15分钟预测起点，拒绝非对齐边界。"""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or start != start.floor('15min') or end != end.floor('15min'):
        raise ValueError(message('features.error.01'))
    return pd.date_range(start, end, freq='15min', name='datetime')


def price_at(index: pd.DatetimeIndex) -> np.ndarray:
    """按官方已知月/半小时表查价，不插值价格边界。"""
    table = load_json(ROOT / 'resources/price.json')['values']
    values = np.asarray([row[1:] for row in table[1:]], dtype='float64')
    return values[(index.hour * 60 + index.minute) // 30, index.month - 1]


def build_features(view: ProcessView, index: pd.DatetimeIndex, cfg: dict) -> tuple[pd.DataFrame, list]:
    """所有时间窗为(t-W,t]；只取截至各起点可见观测，最后值不回写原始缺失。"""
    if len(index) == 0 or not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError(message('features.error.02'))
    if any(x in view.values.columns for x in cfg['targets']):
        raise ValueError(message('features.error.03'))
    start = min(view.values.index.min(), index.min())
    grid = pd.date_range(start.floor('min'), index.max(), freq='min')
    dense = view.values.loc[:index.max()].reindex(grid)
    observed = view.observed_time.loc[:index.max()].reindex(grid)
    nominal = pd.Series(observed.values.astype('datetime64[ns]').astype('int64') / 60e9, index=grid)
    nominal = nominal.mask(observed.isna())
    positions = grid.get_indexer(index)
    if (positions < 0).any():
        raise ValueError(message('features.error.04'))
    # NOTE: 期望采样数来自公开阶段频率；没有记录仍算缺失，未计划分钟不算缺测。
    transition = pd.Timestamp(cfg['sources'][2]['start'])
    expected = pd.Series(((grid >= transition) | (grid.minute % 15 == 0)).astype(float), index=grid)
    expected_count = expected.rolling('60min', closed='right', min_periods=1).sum().to_numpy()[positions]
    result, dictionary = {}, []
    for col in dense.columns:
        s = dense[col]
        last = s.ffill()
        stamp = nominal.where(s.notna()).ffill()
        age = (pd.Series(grid.view('i8') / 60e9, index=grid) - stamp)
        last = last.mask(age > cfg['features']['last_max_age_minutes'])
        ops = {
            'observed': s, 'last': last, 'age_minutes': age,
            'missing_now': s.isna().astype(float),
            'mean_15m': s.rolling('15min', closed='right', min_periods=1).mean(),
            'mean_60m': s.rolling('60min', closed='right', min_periods=1).mean(),
            'mean_360m': s.rolling('360min', closed='right', min_periods=1).mean(),
            'std_60m': s.rolling('60min', closed='right', min_periods=2).std(ddof=0),
            'diff_15m': last - last.shift(15),
        }
        for op, values in ops.items():
            name = f'feat_{col}__{op}'
            result[name] = values.to_numpy(dtype='float64')[positions]
            dictionary.append({'name':name,'source':col,'operator':op,'availability':'<=origin',
                               'uses_target':False,'unit':'source_unit_unmodified' if op not in ['age_minutes','missing_now'] else op})
        name = f'feat_{col}__coverage_60m'
        result[name] = s.notna().astype(float).rolling('60min', closed='right', min_periods=1).sum().to_numpy()[positions] / np.maximum(expected_count, 1)
        dictionary.append({'name':name,'source':col,'operator':'valid_count / scheduled_count in past hour',
                           'availability':'<=origin','uses_target':False,'unit':'fraction'})
    minute = index.hour * 60 + index.minute
    extras = {'feat_day_sin':np.sin(2*np.pi*minute/1440),'feat_day_cos':np.cos(2*np.pi*minute/1440),
              'feat_week_sin':np.sin(2*np.pi*index.dayofweek/7),'feat_week_cos':np.cos(2*np.pi*index.dayofweek/7),
              'feat_price':price_at(index), 'feat_native_cadence_minutes':np.where(index < transition, 15., 1.)}
    for name, value in extras.items():
        result[name] = value
        dictionary.append({'name':name,'source':'deterministic_calendar_or_source_schedule',
                           'operator':name,'availability':'known_before_origin','uses_target':False,'unit':'specified'})
    return pd.DataFrame(result, index=index), dictionary


def save_features(path, features: pd.DataFrame) -> None:
    """特征文件只含预测起点和输入，绝不混入隐藏标签或未来观测。"""
    save_npz(path, origins=features.index.to_numpy(dtype='datetime64[ns]'),
             features=features.to_numpy(dtype='float64'), names=np.array(features.columns,dtype=str))
