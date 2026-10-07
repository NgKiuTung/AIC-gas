"""私有AIC赛事研究：按平台长周期权重汇总，并生成短周期冻结的派生候选。"""
from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from gasbench.common import check_seal, load_json, msg, save_npz, write_json
from gasbench.competition import add_platform_proxy, split_score
from gasbench.data import load_bundle
from gasbench.ensemble import BANDS, apply_weights, matured_folds, weights_for
from gasbench.guarded import apply_long_guard, choose_long_experts
from gasbench.split_submission import export_split_pair
from gasstage.metrics import score as base_score
from gasstage.submission import export_pair
from gasstage.targets import future_truth


def read_prediction(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """仅加载不含Python对象的预测数组。"""
    with np.load(path / "predictions.npz", allow_pickle=False) as data:
        return pd.DatetimeIndex(data["origins"]), data["prediction"]


def _available(run: Path, fold: dict, candidates: list[dict]) -> dict[str, Path]:
    """返回当前时间折真正完成且产物签名有效的模型任务。"""
    available = {}
    for candidate in candidates:
        path = run / "tasks" / f"{fold['name']}__{candidate['id']}"
        if not (path / "receipt.json").exists():
            continue
        if load_json(path / "receipt.json")["status"] != "completed":
            continue
        check_seal(path)
        available[candidate["id"]] = path
    return available


def _prediction_map(run: Path, fold: dict, names: list[str]) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """读取同一折所有模型并确认起点完全一致。"""
    index = None
    predictions = {}
    for name in names:
        current, values = read_prediction(run / "tasks" / f"{fold['name']}__{name}")
        if index is None:
            index = current
        else:
            np.testing.assert_array_equal(current, index)
        predictions[name] = values
    if index is None:
        raise ValueError(msg('guard.missing'))
    return index, predictions


def _build_fusion(run: Path, bundle, cfg: dict, fold: dict, names: list[str]) -> tuple[np.ndarray, dict]:
    """融合只用更早且96块标签已完整兑现的时间折。"""
    index, predictions = _prediction_map(run, fold, names)
    prior = matured_folds([item for item in cfg["folds"] if item["name"] != fold["name"]], pd.Timestamp(fold["start"]))
    if prior:
        past = np.concatenate(
            [np.stack([read_prediction(run / "tasks" / f"{p['name']}__{name}")[1] for name in names]) for p in prior],
            axis=1,
        )
        truth = np.concatenate(
            [future_truth(bundle.truth, read_prediction(run / "tasks" / f"{p['name']}__{names[0]}")[0], bundle.contract) for p in prior]
        )
        weights, diagnostics = weights_for(past, truth, names.index("reference_58"), cfg["fusion"]["ridge"])
    else:
        weights = np.zeros((len(BANDS), 2, len(names)))
        weights[:, :, names.index("reference_58")] = 1
        diagnostics = [{"reason": "no earlier matured calibration; fixed reference_58"}]
    forecast = apply_weights(np.stack([predictions[name] for name in names]), weights)
    evidence = {
        "fold": fold,
        "prior_calibration_folds": prior,
        "experts": names,
        "weights": weights.tolist(),
        "bands": [list(item) for item in BANDS],
        "optimizer": diagnostics,
        "same_fold_labels_used_to_fit_weights": False,
    }
    path = run / "fusion" / fold["name"]
    write_json(path / "ensemble.json", evidence)
    save_npz(path / "predictions.npz", origins=index.to_numpy(dtype="datetime64[ns]"), prediction=forecast)
    return forecast, evidence


def _derive_split_candidates(run: Path, bundle, cfg: dict, fold: dict, names: list[str], rows: list[dict]) -> list[dict]:
    """冻结reference短表；长表分别比较单模型、前向融合和守门混合。"""
    index, predictions = _prediction_map(run, fold, names)
    reference = predictions["reference_58"]
    truth = None if fold["name"] == "test" else future_truth(bundle.truth, index, bundle.contract)
    extra = {name: predictions[name] for name in names if name != "reference_58"}
    fusion, _ = _build_fusion(run, bundle, cfg, fold, names)
    extra["forward_ensemble"] = fusion
    if truth is not None:
        rows.append({"fold": fold["name"], "candidate": "forward_ensemble", **add_platform_proxy(base_score(truth, fusion))})
    else:
        export_pair(run / "candidate_results" / "forward_ensemble", index, fusion, bundle.contract, "forward_ensemble")

    for name, long_prediction in extra.items():
        derived = f"refshort__{name}__long"
        if truth is None:
            export_split_pair(
                run / "candidate_results" / derived,
                index,
                reference,
                long_prediction,
                bundle.contract,
                derived,
                {"short": "reference_58", "long": name},
            )
        else:
            rows.append({"fold": fold["name"], "candidate": derived, **split_score(truth, reference, long_prediction)})

    prior_names = [item["name"] for item in matured_folds(cfg["folds"], pd.Timestamp(fold["start"]))]
    metric_frame = pd.DataFrame(rows)
    calibration = metric_frame[metric_frame["fold"].isin(prior_names) & metric_frame["candidate"].isin(names)]
    decisions = choose_long_experts(calibration, names, cfg["selection"])
    guarded = apply_long_guard(reference, predictions, decisions)
    evidence = {"fold": fold, "short": "reference_58", "long_decisions": decisions, "policy": cfg["selection"], "same_fold_labels_used": False}
    path = run / "guarded" / fold["name"]
    write_json(path / "selection.json", evidence)
    save_npz(path / "predictions.npz", origins=index.to_numpy(dtype="datetime64[ns]"), prediction=guarded)
    if truth is None:
        export_split_pair(run / "candidate_results" / "guarded_long_hybrid", index, reference, guarded, bundle.contract, "guarded_long_hybrid", evidence)
    else:
        rows.append({"fold": fold["name"], "candidate": "guarded_long_hybrid", **split_score(truth, reference, guarded)})
    return [evidence]


def summarize(run: Path, cache: Path, cfg: dict) -> dict:
    """汇总基模型与派生候选；本地最佳按平台线性代理而非等权MAPE选择。"""
    bundle = load_bundle(cache)
    rows: list[dict] = []
    tasks = []
    folds = cfg["folds"]
    test = {"name": "test", "start": "2025-10-01 00:00:00", "end": "2025-10-10 23:45:00"}
    for fold in [*folds, test]:
        for name, path in _available(run, fold, cfg["candidates"]).items():
            tasks.append({"fold": fold["name"], "candidate": name, **load_json(path / "metadata.json")})
            if fold["name"] == "test":
                shutil.copytree(path / "results", run / "candidate_results" / name, dirs_exist_ok=True)
                continue
            index, prediction = read_prediction(path)
            truth = future_truth(bundle.truth, index, bundle.contract)
            rows.append({"fold": fold["name"], "candidate": name, **add_platform_proxy(base_score(truth, prediction))})
            if name == "reference_58":
                with np.load(path / "week_baseline.npz") as data:
                    rows.append({"fold": fold["name"], "candidate": "frozen_week", **add_platform_proxy(base_score(truth, data["prediction"]))})

    common = {candidate["id"] for candidate in cfg["candidates"]}
    for fold in [*folds, test]:
        common &= set(_available(run, fold, cfg["candidates"]))
    names = [candidate["id"] for candidate in cfg["candidates"] if candidate["id"] in common]
    guarded_log = []
    if "reference_58" in names:
        for fold in [*folds, test]:
            guarded_log.extend(_derive_split_candidates(run, bundle, cfg, fold, names, rows))

    metrics = pd.DataFrame(rows)
    target = run / "metrics"
    target.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(target / "by_fold.csv", index=False)
    aggregate = pd.DataFrame()
    local_best = None
    if len(metrics):
        columns = [name for name in metrics if name not in ("fold", "candidate")]
        aggregate = metrics.groupby("candidate")[columns].mean(numeric_only=True)
        aggregate["valid_pairs"] = metrics.groupby("candidate")["valid_pairs"].sum()
        aggregate["expected_pairs"] = metrics.groupby("candidate")["expected_pairs"].sum()
        aggregate["coverage"] = aggregate["valid_pairs"] / aggregate["expected_pairs"]
        aggregate["fold_count"] = metrics.groupby("candidate").size()
        aggregate["same_folds_complete"] = aggregate["fold_count"] == len(folds)
        aggregate.to_csv(target / "aggregate.csv")
        comparable = aggregate[aggregate["same_folds_complete"]]
        if len(comparable):
            local_best = comparable["platform_score_proxy"].idxmax()

    status = load_json(run / "run_status.json")
    summary = {
        "status": status,
        "local_best_platform_proxy": local_best,
        "selection_objective": "platform_score_proxy = 120 - 150*short_mape - 300*long_mape",
        "reference_official_score": 58.1578,
        "online_day3_official_score": 56.3906,
        "online_day3_is_not_a_training_candidate": True,
        "comparison_scope": "chronological development folds; not a new blind test",
        "models_with_complete_folds_and_test": names,
        "short_policy": "reference_58 frozen for split/guarded candidates",
        "candidate_results": sorted(path.parent.name for path in (run / "candidate_results").glob("*/results_only.zip")),
        "official_score_new_models": "Unknown",
        "test_targets_used": False,
    }
    write_json(run / "summary.json", summary)
    write_json(run / "task_times.json", [{"fold": item["fold"], "candidate": item["candidate"]["id"], "seconds": item["seconds"]} for item in tasks])
    write_json(run / "guarded_provenance.json", guarded_log)
    report = "# 跨模型正式运行报告\n\n"
    report += f"状态：{status['status']}；完成任务 {status['completed_tasks']}/{status['expected_tasks']}。\n\n"
    report += "本版根据已知平台回执，将长周期误差权重设为短周期的2倍；短表派生候选冻结58.1578分参照。\n\n"
    report += "历史折仍是开发数据，不等同官方隐藏测试。online_day3_mix50的56.3906只用于否决该规则族，不用于逐行修正。\n\n"
    if len(aggregate):
        report += aggregate.to_csv() + "\n"
    (run / "REPORT.md").write_text(report, encoding="utf-8")
    return summary
