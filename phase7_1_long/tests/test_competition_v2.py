"""私有AIC赛事研究：平台代理指标、短长拆分导出和守门策略测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gasbench.competition import add_platform_proxy, split_score
from gasbench.guarded import apply_long_guard, choose_long_experts
from gasbench.split_submission import export_split_pair, validate_split_pair


def test_platform_proxy_matches_known_reference():
    """NOTE: 使用公开回执的显示精度，只要求与58.1578在舍入误差内一致。"""
    metrics = add_platform_proxy({"short_mape": 1 - 0.8961, "long_mape": 1 - 0.8458})
    assert metrics["platform_score_proxy"] == pytest.approx(58.155, abs=0.02)
    assert metrics["competition_mape"] == pytest.approx(((1 - 0.8961) + 2 * (1 - 0.8458)) / 3)


def test_split_score_keeps_reference_short():
    truth = np.full((2, 96, 2), 100.0)
    short = truth.copy()
    long = truth.copy()
    long[:, :, 0] = 110.0
    result = split_score(truth, short, long)
    assert result["short_mape"] == 0
    assert result["long_g1"] == pytest.approx(0.1)
    assert result["platform_score_proxy"] < 120


def test_split_submission_allows_different_overlap(tmp_path):
    index = pd.date_range("2025-10-01", periods=960, freq="15min")
    short = np.broadcast_to([100.0, 250.0], (960, 96, 2)).copy()
    long = short.copy()
    long[:, :8, 0] += 5
    report = export_split_pair(
        tmp_path,
        index,
        short,
        long,
        {"protocol": {"block_start_offset_minutes": 0}},
        "unit",
        {"short": "reference", "long": "other"},
    )
    assert report["overlap_equal"] is False
    assert validate_split_pair(tmp_path, index)["validation"] == "passed"


def test_guard_rejects_unstable_candidate():
    rows = []
    for i, fold in enumerate(("a", "b", "c", "d")):
        rows.append({"fold": fold, "candidate": "reference_58", "long_g1": 0.12, "long_gall": 0.10})
        rows.append({"fold": fold, "candidate": "other", "long_g1": 0.11 if i < 3 else 0.14, "long_gall": 0.09})
    policy = {
        "min_folds": 3,
        "min_mean_improvement_pp": 0.25,
        "min_win_fraction": 0.625,
        "max_worst_degradation_pp": 0.75,
        "recent_fold_count": 2,
        "min_recent_improvement_pp": 0.0,
        "blend_weight": 0.5,
    }
    decisions = choose_long_experts(pd.DataFrame(rows), ["reference_58", "other"], policy)
    assert decisions[0]["candidate"] == "reference_58"
    assert decisions[1]["candidate"] == "other"


def test_guard_blends_only_selected_target():
    reference = np.full((3, 96, 2), [100.0, 200.0])
    other = reference.copy()
    other[:, :, 0] = 120.0
    decisions = [
        {"target": 0, "candidate": "other", "blend_weight": 0.5},
        {"target": 1, "candidate": "reference_58", "blend_weight": 0.0},
    ]
    out = apply_long_guard(reference, {"reference_58": reference, "other": other}, decisions)
    assert np.all(out[:, :, 0] == 110.0)
    assert np.all(out[:, :, 1] == 200.0)
