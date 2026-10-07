"""私有赛事研究：新增日/周过程状态；不包含真实负荷或未来过程量。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gas2.common import text
from gasstage.features import ProcessView

def extend_features(view: ProcessView, base: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """保留276列原特征；新增27×5项。所有窗口为(t-W,t]，以真实观测计算。"""
    if not base.index.is_unique or not base.index.is_monotonic_increasing:
        raise ValueError(text("feature.invalid"))
    if any(c in view.values for c in cfg["targets"]):
        raise ValueError(text("feature.invalid"))
    idx = base.index
    # NOTE: 首个预测起点可能早于首条可用观测（初赛有15分钟到达延迟）。
    # 网格必须涵盖该起点，禁止get_indexer返回-1后意外读到最后一行未来数据。
    start = idx.min() if view.values.empty else min(view.values.index.min(), idx.min())
    grid = pd.date_range(start.floor("min"), idx.max(), freq="min")
    dense = view.values.loc[:idx.max()].reindex(grid)
    positions = grid.get_indexer(idx)
    if (positions < 0).any():
        raise ValueError(text("feature.invalid"))
    transition = pd.Timestamp(cfg["sources"][2]["start"])
    schedule = pd.Series(((grid >= transition) | (grid.minute % 15 == 0)).astype(float), index=grid)
    expected = schedule.rolling("1440min", min_periods=1, closed="right").sum()
    values, dictionary = {}, []
    for col in dense:
        series = dense[col]
        # NOTE: 过去一天相同时间的值只取旧观测；不把缺失初赛分钟复制成观测。
        ops = {
            "mean_1440m": series.rolling("1440min", min_periods=1, closed="right").mean(),
            "std_1440m": series.rolling("1440min", min_periods=2, closed="right").std(ddof=0),
            "mean_10080m": series.rolling("10080min", min_periods=1, closed="right").mean(),
            "lag_1440m_observed": series.shift(1440),
            "coverage_1440m": series.notna().astype(float).rolling(
                "1440min", min_periods=1, closed="right").sum() / expected.clip(lower=1),
        }
        for op, item in ops.items():
            name = f"feat_{col}__{op}"
            values[name] = item.to_numpy(dtype="float64")[positions]
            dictionary.append({"name": name, "source": col, "operator": op,
                               "availability": "<=origin", "uses_target": False,
                               "unit": "fraction" if op.startswith("coverage") else "source_unit_unmodified"})
    extra = pd.DataFrame(values, index=idx)
    if set(extra).intersection(base):
        raise ValueError(text("feature.invalid"))
    return pd.concat([base, extra], axis=1), dictionary

def select_process_columns(names: list[str], process_names: list[str]) -> list[int]:
    """原生同记录模型只用对应过程量，不偷用事后派生的真实负荷代理。"""
    wanted = [f"feat_{col}__last" for col in process_names]
    return [names.index(col) for col in wanted]
