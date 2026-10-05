"""私有赛事研究：原始值质量标记，修复规则不依赖未来分位数。"""
from __future__ import annotations
import numpy as np
import pandas as pd

# NOTE: 位标记可叠加；0是真实有效观测，非0不一律代表应删除。
MISSING, PARSE, NONFINITE, EXTREME, NEGATIVE, TINY_NEGATIVE = 1, 2, 4, 8, 16, 32


def numeric_frame(raw: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """解析十进制含科学计数法，保留微负数；返回清洗值、位标记、变更、词法统计。"""
    clean, flags, changes, profiles = {}, {}, [], []
    for col in raw.columns:
        text = raw[col].str.strip()
        empty = text.str.lower().isin(['', 'nan', 'na', 'null', 'none'])
        value = pd.to_numeric(text.mask(empty), errors='coerce').astype('float64')
        f = np.zeros(len(raw), dtype=np.uint8)
        f[empty] |= MISSING
        f[(~empty) & value.isna()] |= PARSE
        f[value.notna() & ~np.isfinite(value)] |= NONFINITE
        f[value.abs() > policy['hard_absolute_limit']] |= EXTREME
        f[value < policy['hard_negative_below']] |= NEGATIVE
        f[(value < 0) & (value >= policy['hard_negative_below'])] |= TINY_NEGATIVE
        invalid = (f & (PARSE | NONFINITE | EXTREME | NEGATIVE)) != 0
        for pos in np.flatnonzero(invalid):
            changes.append({'datetime': str(raw.index[pos]), 'field': col,
                            'raw_text': str(text.iloc[pos]), 'flag': int(f[pos]), 'action': 'isolate_nan'})
        profiles.append({'field': col, 'missing': int(empty.sum()), 'invalid': int(invalid.sum()),
                         'scientific_text': int(text.str.contains(r'[eE][+-]?\d+', regex=True).sum()),
                         'explicit_radix': int(text.str.contains(r'^[+-]?0[xXbBoO]', regex=True).sum()),
                         'tiny_negative': int(((f & TINY_NEGATIVE) != 0).sum())})
        clean[col], flags[col] = value.mask(invalid), f
    return pd.DataFrame(clean, index=raw.index), pd.DataFrame(flags, index=raw.index), changes, profiles


def collapse_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """完全一致的时间重复允许合并；任何列冲突包括空值冲突都直接报错。"""
    duplicate = frame.index[frame.index.duplicated(keep=False)].unique()
    records = []
    for t in duplicate:
        group = frame.loc[[t]]
        if (group.nunique(dropna=False) > 1).any():
            raise ValueError(f'conflicting duplicate timestamp: {t}')
        records.append({'datetime': str(t), 'copies': len(group), 'action': 'keep_first_identical'})
    return frame.loc[~frame.index.duplicated(keep='first')].sort_index(), records
