"""私有AIC赛事研究：独立核验融合来源、时间成熟度、预测和CSV。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gasbench.audit import independent_score
from gasbench.common import load_json, msg
from gasbench.ensemble import BANDS, apply_weights, matured_folds
from gasbench.reporting import read_prediction
from gasstage.submission import frame_for
from gasstage.targets import future_truth


def compare_csv_prediction(path: Path, index: pd.DatetimeIndex, prediction: np.ndarray) -> dict:
    """CSV逐值核验，不以仅列名/行数通过代替预测一致性。"""
    errors = {}
    for name, blocks in (("s_result.csv", 8), ("l_result.csv", 96)):
        actual = pd.read_csv(path / name, float_precision="round_trip")
        expected = frame_for(index, prediction, blocks)
        if list(actual.columns) != list(expected.columns) or actual.datetime.tolist() != expected.datetime.tolist():
            raise ValueError(msg("data.shape"))
        difference = np.abs(actual.iloc[:, 1:].to_numpy() - np.round(expected.iloc[:, 1:].to_numpy(), 6))
        errors[name] = float(difference.max())
        if errors[name] > 1e-6:
            raise ValueError(msg("verify.failed", detail=f"CSV values changed: {path / name}"))
    return errors


def check_provenance(evidence: dict, config: dict) -> np.ndarray:
    """拒绝融合层使用当前折、未兑现标签或未登记来源。"""
    fold = evidence["fold"]
    allowed = [f for f in config["folds"] if f["name"] != fold["name"]]
    expected = matured_folds(allowed, pd.Timestamp(fold["start"]))
    if evidence["prior_calibration_folds"] != expected:
        raise ValueError(msg("data.future"))
    if evidence["same_fold_labels_used_to_fit_weights"] is not False:
        raise ValueError(msg("data.future"))
    if evidence["bands"] != [list(x) for x in BANDS]:
        raise ValueError(msg("data.shape"))
    names = evidence["experts"]
    weights = np.asarray(evidence["weights"], dtype="float64")
    if weights.shape != (len(BANDS), 2, len(names)) or not np.isfinite(weights).all():
        raise ValueError(msg("data.shape"))
    if (weights < -1e-10).any() or not np.allclose(weights.sum(axis=-1), 1, rtol=0, atol=1e-7):
        raise ValueError(msg("model.finite"))
    if not expected:
        reference = np.zeros_like(weights)
        reference[:, :, names.index("reference_58")] = 1
        np.testing.assert_array_equal(weights, reference)
    return weights


def verify_fusion(bundle, run: Path, config: dict) -> list[dict]:
    """从基模型磁盘预测重算全部融合，不重新选权重。"""
    records = []
    metric_table = pd.read_csv(run / "metrics/by_fold.csv")
    for path in sorted((run / "fusion").glob("*")):
        evidence = load_json(path / "ensemble.json")
        weights = check_provenance(evidence, config)
        index, saved = read_prediction(path)
        parts = []
        for name in evidence["experts"]:
            candidate_index, values = read_prediction(run / "tasks" / (path.name + "__" + name))
            np.testing.assert_array_equal(candidate_index, index)
            parts.append(values)
        forecast = apply_weights(np.stack(parts), weights)
        np.testing.assert_allclose(forecast, saved, rtol=0, atol=0)
        record = {"fold": path.name, "max_error": float(np.max(np.abs(forecast - saved))), "provenance_passed": True}
        if path.name == "test":
            record["csv_max_errors"] = compare_csv_prediction(run / "candidate_results/forward_ensemble", index, forecast)
        else:
            row = metric_table[(metric_table.fold == path.name) & (metric_table.candidate == "forward_ensemble")]
            if len(row) != 1:
                raise ValueError(msg("data.shape"))
            recalculated = independent_score(future_truth(bundle.truth, index, bundle.contract), forecast)
            record["score_max_error"] = max(abs(value - float(row.iloc[0][key])) for key, value in recalculated.items())
            if record["score_max_error"] > 1e-10:
                raise ValueError(msg("verify.failed", detail=f"fusion metric {path.name}"))
        records.append(record)
    return records
