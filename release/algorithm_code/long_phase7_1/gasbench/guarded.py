"""私有AIC赛事研究：冻结短周期参照，仅在有稳健证据时调整长周期。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_FIELDS = ((0, "long_g1"), (1, "long_gall"))


def choose_long_experts(metrics: pd.DataFrame, candidate_names: list[str], policy: dict) -> list[dict]:
    """按跨折改善、最坏退化和最近折表现筛选长周期专家。"""
    reference = metrics[metrics.candidate == "reference_58"].set_index("fold")
    decisions = []
    for target, field in TARGET_FIELDS:
        best = None
        for name in candidate_names:
            if name == "reference_58":
                continue
            current = metrics[metrics.candidate == name].set_index("fold")
            shared = reference.index.intersection(current.index)
            if len(shared) < policy["min_folds"]:
                continue
            delta = reference.loc[shared, field].to_numpy() - current.loc[shared, field].to_numpy()
            recent = delta[-min(policy["recent_fold_count"], len(delta)):]
            stats = {
                "candidate": name,
                "target": target,
                "mean_improvement_pp": float(delta.mean() * 100),
                "win_fraction": float((delta > 0).mean()),
                "worst_improvement_pp": float(delta.min() * 100),
                "recent_mean_improvement_pp": float(recent.mean() * 100),
                "folds": int(len(delta)),
            }
            eligible = (
                stats["mean_improvement_pp"] >= policy["min_mean_improvement_pp"]
                and stats["win_fraction"] >= policy["min_win_fraction"]
                and stats["worst_improvement_pp"] >= -policy["max_worst_degradation_pp"]
                and stats["recent_mean_improvement_pp"] >= policy["min_recent_improvement_pp"]
            )
            if eligible and (best is None or stats["mean_improvement_pp"] > best["mean_improvement_pp"]):
                best = stats
        if best is None:
            decisions.append({"target": target, "candidate": "reference_58", "blend_weight": 0.0, "reason": "guard_rejected_all"})
        else:
            best["blend_weight"] = policy["blend_weight"]
            best["reason"] = "guard_passed"
            decisions.append(best)
    return decisions


def apply_long_guard(reference: np.ndarray, predictions: dict[str, np.ndarray], decisions: list[dict]) -> np.ndarray:
    """仅修改长任务张量；每个目标独立且权重上限由固定策略控制。"""
    result = reference.copy()
    for decision in decisions:
        target = decision["target"]
        name = decision["candidate"]
        weight = float(decision["blend_weight"])
        if name == "reference_58" or weight <= 0:
            continue
        result[:, :, target] = (1.0 - weight) * reference[:, :, target] + weight * predictions[name][:, :, target]
    # NOTE: 混合后做最小物理一致性投影，不在测试标签上调值。
    result[:, :, 0] = np.maximum(result[:, :, 0], 0)
    result[:, :, 1] = np.maximum(result[:, :, 1], result[:, :, 0])
    return result
