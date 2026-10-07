"""私有赛事研究：全96区间评分、逐日/逐跨度诊断；不把公开开发日期称为盲测。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from gasstage.metrics import score, per_horizon

def evaluate_prediction(truth: np.ndarray, pred: np.ndarray, index: pd.DatetimeIndex,
                        fold: str, name: str) -> tuple[dict,list,list]:
    """同一个真实标签mask用于所有候选，除MAPE外记录偏差与尾部误差。"""
    metrics = {"fold":fold,"candidate":name,**score(truth,pred)}
    relative = (pred-truth)/truth
    metrics["signed_relative_bias"] = float(np.nanmean(relative))
    metrics["ape_q95"] = float(np.nanquantile(np.abs(relative),0.95))
    metrics["ape_over_20pct_fraction"] = float((np.abs(relative[np.isfinite(relative)]) > 0.2).mean())
    daily = []
    for date in index.normalize().unique():
        take = index.normalize()==date
        daily.append({"fold":fold,"candidate":name,"date":str(date.date()),**score(truth[take],pred[take])})
    horizon = per_horizon(truth,pred)
    rows = [{"fold":fold,"candidate":name,"offset_minutes":(h+1)*15,
             "g1_mape":float(horizon[h,0]),"gall_mape":float(horizon[h,1])} for h in range(96)]
    return metrics,daily,rows

def aggregate_rows(rows: list[dict]) -> pd.DataFrame:
    """四折等权，不把覆盖较长的窗口变成隐含更高权重。"""
    return pd.DataFrame(rows).groupby("candidate",sort=False).mean(numeric_only=True).reset_index()
