"""私有AIC赛事研究：独立重放短周期冻结、长周期守门混合。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from gasbench.competition import split_score
from gasbench.common import msg
from gasbench.common import load_json
from gasbench.guarded import apply_long_guard
from gasbench.reporting import read_prediction
from gasbench.split_submission import validate_split_pair
from gasstage.targets import future_truth


def verify_guarded(bundle, run: Path) -> list[dict]:
    """从基模型和保存决策重算守门输出，不重新选择专家。"""
    metric_table = pd.read_csv(run / "metrics/by_fold.csv")
    records = []
    for path in sorted((run / "guarded").glob("*")):
        evidence = load_json(path / "selection.json")
        fold = evidence["fold"]
        names = {"reference_58"}
        names.update(item["candidate"] for item in evidence["long_decisions"])
        predictions = {}
        index = None
        for name in sorted(names):
            current, values = read_prediction(run / "tasks" / f"{path.name}__{name}")
            if index is None:
                index = current
            else:
                np.testing.assert_array_equal(current, index)
            predictions[name] = values
        reference = predictions["reference_58"]
        current, saved = read_prediction(path)
        np.testing.assert_array_equal(current, index)
        rebuilt = apply_long_guard(reference, predictions, evidence["long_decisions"])
        np.testing.assert_allclose(rebuilt, saved, rtol=0, atol=0)
        record = {"fold": path.name, "max_error": float(np.max(np.abs(rebuilt - saved)))}
        if path.name == "test":
            record["submission"] = validate_split_pair(run / "candidate_results/guarded_long_hybrid", index)
        else:
            truth = future_truth(bundle.truth, index, bundle.contract)
            recalculated = split_score(truth, reference, rebuilt)
            row = metric_table[(metric_table.fold == path.name) & (metric_table.candidate == "guarded_long_hybrid")]
            if len(row) != 1:
                raise ValueError(msg('guard.missing'))
            keys = ["short_mape", "long_mape", "combined_mape", "short_g1", "short_gall", "long_g1", "long_gall", "competition_mape", "platform_score_proxy"]
            record["score_max_error"] = max(abs(recalculated[key] - float(row.iloc[0][key])) for key in keys)
            if record["score_max_error"] > 1e-10:
                raise ValueError(msg('guard.metric'))
        records.append(record)
    return records
