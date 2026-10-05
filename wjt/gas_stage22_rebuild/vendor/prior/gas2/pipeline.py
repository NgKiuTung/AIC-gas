"""私有赛事研究：四折滚动输入/负荷隐藏模型实验与离线导出。"""
from __future__ import annotations
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd

from gas2.common import write_json, save_npz, text, load_json, sha256
from gas2.blend import fit_weights, eligible_past, combine
from gas2.data import load_bundle, load_cached_bundle
from gas2.evaluate import evaluate_prediction, aggregate_rows
from gas2.runner import stage1_predictions, run_candidate, BASELINES
from gasstage.features import origins
from gasstage.submission import export_pair
from gasstage.targets import future_truth

LOGGER = logging.getLogger(__name__)

def _persist_fold(root, fold, index, truth, pred):
    dest = root/"folds"/fold["name"]
    save_npz(dest/"evaluation.npz",origins=index.to_numpy(dtype="datetime64[ns]"),truth=truth,**pred)

def _evaluate_all(fold, index, truth, predictions):
    metrics, days, horizons = [], [], []
    for name,pred in predictions.items():
        m,d,h = evaluate_prediction(truth,pred,index,fold["name"],name)
        metrics.append(m); days.extend(d); horizons.extend(h)
    return metrics, days, horizons

def _write_results(root, metrics, days, horizons):
    folder = root/"metrics";folder.mkdir(parents=True,exist_ok=True)
    for name,records in [("by_fold",metrics),("by_day",days),("by_horizon",horizons)]:
        pd.DataFrame(records).to_csv(folder/f"{name}.csv",index=False,float_format="%.12g")
    aggregate = aggregate_rows(metrics)
    aggregate.to_csv(folder/"aggregate.csv",index=False,float_format="%.12g")
    return aggregate

def execute(stage1: Path, output: Path, settings: dict, signature: str, deadline: float) -> dict:
    """顺序执行候选；停止/断电后通过同签名回执恢复，不并行争抢用户资源。"""
    feature_dir = output/"features"
    if (feature_dir/"ready.json").exists():
        bundle = load_cached_bundle(stage1,feature_dir)
    else:
        bundle = load_bundle(stage1,feature_dir,settings["extended_features"])
        write_json(feature_dir/"ready.json",{"signature":signature,
            "hashes":{str(p.name):sha256(p) for p in feature_dir.glob("*.npz")}})
    ready = load_json(feature_dir/"ready.json")
    if ready["signature"] != signature or any(sha256(feature_dir/n)!=h for n,h in ready["hashes"].items()):
        raise ValueError(text("run.signature"))
    names = list(BASELINES)+[c["id"] for c in settings["candidates"]]
    records, metrics, days, horizons = [], [], [], []
    cfg = bundle.contract
    for fold in cfg["folds"]:
        cutoff = pd.Timestamp(fold["start"])
        index = origins(fold["start"],fold["end"])
        features = bundle.features.loc[index]
        truth = future_truth(bundle.truth,index,cfg)
        pred = stage1_predictions(stage1,features,cfg,fold["name"])
        for candidate in settings["candidates"]:
            started = time.perf_counter()
            pred[candidate["id"]] = run_candidate(bundle,candidate,settings,cutoff,features,
                 output/"folds"/fold["name"]/candidate["id"],signature,deadline,fold["name"])
            m = evaluate_prediction(truth,pred[candidate["id"]],index,fold["name"],candidate["id"])[0]
            LOGGER.info(text("candidate.done",fold=fold["name"],candidate=candidate["id"],
                             metric=m["combined_mape"],seconds=time.perf_counter()-started))
        past = eligible_past(records,cutoff,cfg["protocol"]["block_start_offset_minutes"])
        weights,evidence = fit_weights(past,names,settings["blend"],cutoff,
                                        cfg["protocol"]["block_start_offset_minutes"])
        pred["forward_ensemble"] = combine(pred,names,weights,settings["blend"]["bands"])
        write_json(output/"folds"/fold["name"]/"ensemble.json",
                   {"names":names,"weights":weights.tolist(),"bands":settings["blend"]["bands"],**evidence})
        _persist_fold(output,fold,index,truth,pred)
        m,d,h = _evaluate_all(fold,index,truth,pred)
        metrics.extend(m);days.extend(d);horizons.extend(h)
        records.append({**fold,"truth":truth,"pred":pred})
        _write_results(output,metrics,days,horizons)
    aggregate = _write_results(output,metrics,days,horizons)
    deployment = deploy(bundle,settings,output,records,names,signature,deadline)
    summary = {"status":"completed","folds":len(records),"models_per_fold":len(settings["candidates"]),
        "feature_count":len(bundle.features.columns),"mean_fold_metrics":aggregate.to_dict("records"),
        "validation_scope":"previously seen dates; forward ensemble uses only earlier matured folds",
        "local_score_definition":"equal targets; short/long MAPE each 0.5; not official acc_score",
        "official_test_truth_used":False,"gpu_used":False,"deployment":deployment}
    write_json(output/"summary.json",summary)
    return summary

def deploy(bundle,settings,output,records,names,signature,deadline):
    """全量公开历史重训；测试变量按起点更新；写结果而非人工修正隐藏预测。"""
    cfg=bundle.contract
    cutoff=pd.Timestamp(cfg["protocol"]["test_start"])
    index=origins(cfg["protocol"]["test_start"],cfg["protocol"]["test_end"])
    features=bundle.features.loc[index]
    pred=stage1_predictions(bundle.source,features,cfg,"test")
    for candidate in settings["candidates"]:
        pred[candidate["id"]]=run_candidate(bundle,candidate,settings,cutoff,features,
            output/"test"/candidate["id"],signature,deadline,"test")
    past=eligible_past(records,cutoff,cfg["protocol"]["block_start_offset_minutes"])
    weights,evidence=fit_weights(past,names,settings["blend"],cutoff,cfg["protocol"]["block_start_offset_minutes"])
    pred["forward_ensemble"]=combine(pred,names,weights,settings["blend"]["bands"])
    write_json(output/"test/ensemble.json",{"names":names,"weights":weights.tolist(),
                                         "bands":settings["blend"]["bands"],**evidence})
    save_npz(output/"test/predictions.npz",origins=index.to_numpy(dtype="datetime64[ns]"),**pred)
    for name,value in pred.items():
        report=export_pair(output/"candidate_results"/name,index,value,cfg,name)
        # NOTE: 明确此包只含结果；算法复现源码由工程包独立提供。
        report["status"]="stage2_research_candidate"
        write_json(output/"candidate_results"/name/"validation.json",report)
    return {"candidates":list(pred),"origins":len(index),"blocks":96,"targets":2,
            "cutoff":str(cutoff),"test_labels_visible":False}
