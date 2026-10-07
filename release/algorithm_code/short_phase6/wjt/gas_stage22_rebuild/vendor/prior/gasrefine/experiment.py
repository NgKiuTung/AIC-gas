"""私有赛事研究：固定本轮候选的四折重放、最终拟合及结果导出。"""
from __future__ import annotations
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from gas2.common import save_npz, write_json, load_json, sha256
from gas2.data import load_bundle, load_cached_bundle
from gas2.evaluate import evaluate_prediction, aggregate_rows
from gasstage.baselines import reconcile
from gasstage.features import origins
from gasstage.targets import future_truth
from gasstage.submission import export_pair
from gasrefine.common import message
from gasrefine.snapshot import SnapshotModel
from gasrefine.periodic import forecast

LOGGER = logging.getLogger(__name__)


def fitted_model(bundle, cutoff, candidate, settings, directory, signature, deadline):
    """同签名完整模型才复用；中断的半成品会重训并留原运行日志。"""
    receipt = directory/"receipt.json"
    if receipt.exists():
        prior = load_json(receipt)
        if prior["signature"] != signature or sha256(directory/"model.json") != prior["metadata_sha256"]:
            raise ValueError(message("signature.invalid"))
        model = SnapshotModel.load(directory)
        LOGGER.info(message("model.reuse", name=candidate["id"]))
        return model
    if time.time() >= deadline:
        raise TimeoutError(message("timeout"))
    start = time.perf_counter()
    model = SnapshotModel.fit(bundle, cutoff, candidate, settings, directory, deadline)
    write_json(receipt, {"signature": signature, "metadata_sha256": sha256(directory/"model.json"),
                         "fit_seconds": time.perf_counter()-start, "candidate": candidate["id"]})
    return model


def prediction_set(models, features, index, settings):
    """三份正式候选加固定消融；先按目标组合，最后只保证非负及包含关系。"""
    recipes = settings["recipes"]
    pred = {}
    pred["reference_58"] = forecast(models["reference"], features, index, recipes["old_week6"])
    pred["persistent_shared"], context_a = forecast(models["reference"], features, index,
                                                    recipes["mixed_slow"], return_context=True)
    other, context_b = forecast(models["context60"], features, index,
                               recipes["daily_slow6h"], return_context=True)
    pred["target_adaptive"] = reconcile(np.stack([pred["persistent_shared"][:,:,0],other[:,:,1]],axis=2))
    pred["ablation_decay24"] = forecast(models["reference"],features,index,recipes["decay24"])
    pred["ablation_daily_slow"] = forecast(models["reference"],features,index,recipes["daily_slow24h"])
    pred["ablation_week_slow"] = forecast(models["reference"],features,index,recipes["weekly_slow6h"])
    context = pd.concat([context_a.add_prefix("reference__"),context_b.add_prefix("context60__")],axis=1)
    return pred, context


def persist_metrics(output, rows, days, horizons):
    """报告保留全部固定对照；四折等权，不将重叠单元格称为独立样本。"""
    folder=output/"metrics";folder.mkdir(exist_ok=True)
    for name, records in [("by_fold",rows),("by_day",days),("by_horizon",horizons)]:
        pd.DataFrame(records).to_csv(folder/f"{name}.csv",index=False,float_format="%.12g")
    aggregate=aggregate_rows(rows)
    aggregate.to_csv(folder/"aggregate.csv",index=False,float_format="%.12g")
    return aggregate


def execute(stage1: Path, output: Path, settings: dict, signature: str, deadline: float) -> dict:
    """顺序拟合并保存阶段结果；不访问用户云实例，也不自动提交排行榜。"""
    feature_dir = output/"features"
    if (feature_dir/"ready.json").exists():
        ready=load_json(feature_dir/"ready.json")
        if ready["signature"]!=signature or any(sha256(feature_dir/n)!=h for n,h in ready["hashes"].items()):
            raise ValueError(message("signature.invalid"))
        bundle=load_cached_bundle(stage1,feature_dir)
    else:
        bundle=load_bundle(stage1,feature_dir,extended=False)
        write_json(feature_dir/"ready.json",{"signature":signature,"hashes":{
            p.name:sha256(p) for p in feature_dir.glob("*.npz")}})
    rows,days,horizons=[],[],[]
    cfg=bundle.contract
    phases=cfg["folds"]+[{"name":"test","start":cfg["protocol"]["test_start"],"end":cfg["protocol"]["test_end"]}]
    for phase in phases:
        phase_start=time.perf_counter();name=phase["name"]
        directory=output/"test" if name=="test" else output/"folds"/name
        cutoff=pd.Timestamp(phase["start"]); index=origins(phase["start"],phase["end"])
        LOGGER.info(message("phase.start", name=name))
        models={c["id"]:fitted_model(bundle,cutoff,c,settings,directory/c["id"],signature,deadline)
                for c in settings["models"]}
        prediction,context=prediction_set(models,bundle.features,index,settings)
        save_npz(directory/"online_context.npz",origins=index.to_numpy(dtype="datetime64[ns]"),
                 features=context.to_numpy(),names=np.asarray(context.columns,dtype="U"))
        context.to_csv(directory/"online_context.csv",float_format="%.17g")
        if name!="test":
            truth=future_truth(bundle.truth,index,cfg)
            for candidate,values in prediction.items():
                m,d,h=evaluate_prediction(truth,values,index,name,candidate)
                rows.append(m);days.extend(d);horizons.extend(h)
            save_npz(directory/"predictions.npz",origins=index.to_numpy(dtype="datetime64[ns]"),truth=truth,**prediction)
            persist_metrics(output,rows,days,horizons)
        else:
            save_npz(directory/"predictions.npz",origins=index.to_numpy(dtype="datetime64[ns]"),**prediction)
            for candidate in ("reference_58","persistent_shared","target_adaptive"):
                export_pair(output/"candidate_results"/candidate,index,prediction[candidate],cfg,candidate)
        write_json(directory/"summary.json",{"cutoff":str(cutoff),"origins":len(index),"status":"completed",
                   "online_context_features":len(context.columns),"seconds":time.perf_counter()-phase_start})
        LOGGER.info(message("phase.end",name=name,seconds=time.perf_counter()-phase_start))
    aggregate=persist_metrics(output,rows,days,horizons)
    summary={"status":"completed","base_features":276,"dynamic_context_features":16,
             "metrics":aggregate.to_dict("records"),"preferred_candidate":"target_adaptive",
             "selection_scope":"selected using previously seen development folds; not independent test",
             "official_score":None,"test_targets_used":False,"gpu_used":False,
             "counterfactual_reference_official_score":58.1578}
    write_json(output/"summary.json",summary)
    return summary
