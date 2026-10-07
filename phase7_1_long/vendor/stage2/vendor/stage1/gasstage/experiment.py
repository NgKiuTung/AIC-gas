"""私有赛事研究：预登记十天负荷隐藏回测，以及同一契约下的基线测试推理。"""
from __future__ import annotations
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from gasstage.baselines import fit_baselines, predict_baselines
from gasstage.common import message, save_npz, write_json
from gasstage.features import origins, ProcessView, build_features, save_features, price_at
from gasstage.ingest import NativeDataset
from gasstage.metrics import score, per_horizon
from gasstage.submission import export_pair
from gasstage.targets import block_truth, future_truth, future_index

LOGGER = logging.getLogger(__name__)


def prepare(data: NativeDataset,cfg: dict,output: Path) -> tuple[pd.DataFrame,pd.DataFrame]:
    """构造可分享的输入特征与真值专用文件；两者物理分开，避免默认泄露。"""
    data.save(output/'data/native_observations.npz')
    view = ProcessView.from_native(data,cfg)
    idx = origins(data.values.index.min().floor('15min'),cfg['protocol']['test_end'])
    features,dictionary = build_features(view,idx,cfg)
    train = features.loc[features.index < pd.Timestamp(cfg['protocol']['test_start'])]
    test = features.loc[features.index >= pd.Timestamp(cfg['protocol']['test_start'])]
    save_features(output/'features/history_process_features.npz',train)
    save_features(output/'features/test_process_features.npz',test)
    pd.DataFrame(dictionary).to_csv(output/'features/dictionary.csv',index=False)
    quality = pd.DataFrame({'name':features.columns,'train_missing_fraction':train.isna().mean(),
                            'test_missing_fraction':test.isna().mean()})
    quality.to_csv(output/'features/coverage.csv',index=False)
    means, counts = block_truth(data)
    save_npz(output/'labels/semi_truth_blocks_15m.npz',time=means.index.to_numpy(dtype='datetime64[ns]'),
             labels=means.to_numpy(),counts=counts.to_numpy(),targets=np.array(cfg['targets']))
    # NOTE: 初赛原始标签保留供名义同记录基线；kind明确不是已确认的15分钟平均值。
    pre = data.targets.loc[data.coarse]
    save_npz(output/'labels/pre_nominal_records.npz',time=pre.index.to_numpy(dtype='datetime64[ns]'),
             labels=pre.to_numpy(),targets=np.array(cfg['targets']))
    stamps = future_index(test.index,cfg)
    known_price = price_at(pd.DatetimeIndex(stamps.ravel())).reshape(len(test),96)
    save_npz(output/'features/test_known_future.npz',origins=test.index.to_numpy(dtype='datetime64[ns]'),
             block_start=stamps,price=known_price)
    test.index.to_series().to_csv(output/'features/test_origins.csv',index=False,header=['datetime'])
    write_json(output/'data/contract.json',cfg)
    LOGGER.info(message('prepare.complete'),len(train),len(test),features.shape[1])
    return features,means


def _evaluate_fold(data,features,means,fold,cfg,output):
    start = time.perf_counter()
    cutoff = pd.Timestamp(fold['start'])
    idx = origins(fold['start'],fold['end'])
    root = output/'backtest'/fold['name']
    pack = fit_baselines(data,features,means,cutoff,cfg,root/'models')
    selected = features.loc[idx]
    predictions = predict_baselines(pack,selected,cfg)
    truth = future_truth(means,idx,cfg)
    shifted_cfg = {**cfg,'protocol':{**cfg['protocol'],'block_start_offset_minutes':15}}
    shifted_truth = future_truth(means,idx,shifted_cfg)
    save_npz(root/'evaluation.npz',origins=idx.to_numpy(dtype='datetime64[ns]'),truth=truth,**predictions)
    results,horizon,daily,sensitivity=[],[],[],[]
    for name,pred in predictions.items():
        results.append({'fold':fold['name'],'baseline':name,**score(truth,pred)})
        sensitivity.append({'fold':fold['name'],'baseline':name,**score(shifted_truth,pred)})
        hh=per_horizon(truth,pred)
        for h in range(96):
            horizon.append({'fold':fold['name'],'baseline':name,'offset_minutes':(h+1)*15,
                            'g1_mape':float(hh[h,0]),'gall_mape':float(hh[h,1])})
        for day in idx.normalize().unique():
            take = idx.normalize()==day
            daily.append({'fold':fold['name'],'date':str(day.date()),'baseline':name,**score(truth[take],pred[take])})
    elapsed=time.perf_counter()-start
    write_json(root/'summary.json',{'name':fold['name'],'status':'completed','elapsed_seconds':elapsed,
          'origins':len(idx),'target_hidden_start':str(cutoff),'target_hidden_end_exclusive':str(idx[-1]+pd.Timedelta(minutes=1440)),
          'historical_dates_previously_seen':True,'results':results})
    LOGGER.info(message('fold.complete'),fold['name'],elapsed)
    return results,horizon,daily,sensitivity


def backtest(data: NativeDataset,features: pd.DataFrame,means: pd.DataFrame,cfg: dict,output: Path) -> dict:
    """四折均按预登记参数执行，不基于验证成绩重新挑选窗口或融合权重。"""
    scores,horizons,days,sensitivity=[],[],[],[]
    for fold in cfg['folds']:
        r,h,d,s = _evaluate_fold(data,features,means,fold,cfg,output)
        scores.extend(r);horizons.extend(h);days.extend(d);sensitivity.extend(s)
    dest=output/'backtest';dest.mkdir(parents=True,exist_ok=True)
    for name,rows in [('metrics',scores),('by_horizon',horizons),('by_day',days),('offset15_sensitivity',sensitivity)]:
        pd.DataFrame(rows).to_csv(dest/f'{name}.csv',index=False,float_format='%.12g')
    metrics=pd.DataFrame(scores)
    means_out=metrics.groupby('baseline').mean(numeric_only=True).reset_index()
    means_out.to_csv(dest/'aggregate.csv',index=False,float_format='%.12g')
    summary={'status':'completed','folds':len(cfg['folds']),'baselines':len(means_out),
              'tuning':'none; descriptive comparison only','scores':means_out.to_dict('records'),
              'validation_scope':'all dates seen in prior research; no claim of independent hidden test'}
    write_json(dest/'summary.json',summary)
    return summary


def forecast_test(data: NativeDataset,features: pd.DataFrame,means: pd.DataFrame,cfg: dict,output: Path) -> None:
    """全历史冻结拟合，10月过程特征逐起点更新；测试真值永不进入拟合/评价。"""
    cutoff=pd.Timestamp(cfg['protocol']['test_start'])
    idx=origins(cfg['protocol']['test_start'],cfg['protocol']['test_end'])
    pack=fit_baselines(data,features,means,cutoff,cfg,output/'test/models')
    predictions=predict_baselines(pack,features.loc[idx],cfg)
    save_npz(output/'test/predictions.npz',origins=idx.to_numpy(dtype='datetime64[ns]'),**predictions)
    for name,pred in predictions.items():
        export_pair(output/'baseline_results'/name,idx,pred,cfg,name)
    write_json(output/'test/manifest.json',{'training_cutoff_exclusive':str(cutoff),
       'origins':len(idx),'first_origin':str(idx[0]),'last_origin':str(idx[-1]),
       'test_target_used':False,'mode':'rolling observed covariates; whole-period target blackout',
       'baseline_names':list(predictions),'public_score':'not evaluated'})
    LOGGER.info(message('forecast.complete'),len(idx),len(predictions))
