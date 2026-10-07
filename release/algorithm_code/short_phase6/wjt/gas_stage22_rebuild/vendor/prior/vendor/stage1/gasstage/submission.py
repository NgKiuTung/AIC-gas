"""私有赛事研究：以真实测试时间轴导出宽表，不复制其它参赛者预测值。"""
from __future__ import annotations
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from gasstage.common import message, sha256, write_json


def columns(blocks: int) -> list[str]:
    """列顺序遵循目标先后与递增偏移，datetime单独放第一列。"""
    return ['datetime'] + [f'{g}_t+{h*15}_pred' for g in ['generator_1','generator_all'] for h in range(1,blocks+1)]


def frame_for(index: pd.DatetimeIndex,pred: np.ndarray,blocks: int) -> pd.DataFrame:
    """由(origin,96,2)张量生成宽表；不做静默重排或补行。"""
    if pred.shape != (len(index),96,2) or not np.isfinite(pred).all():
        raise ValueError(message('submission.error.01'))
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(message('submission.error.02'))
    values = np.concatenate([pred[:,:blocks,0],pred[:,:blocks,1]],axis=1)
    df = pd.DataFrame(values,columns=columns(blocks)[1:])
    df.insert(0,'datetime',index.strftime('%Y-%m-%d %H:%M:%S'))
    return df


def export_pair(path: Path,index: pd.DatetimeIndex,pred: np.ndarray,cfg: dict,model_name: str) -> dict:
    """导出两个结果CSV和独立协议清单；不把基线结果称为优化后的正式最优模型。"""
    if (pred<0).any() or (pred[:,:,1] < pred[:,:,0]).any():
        raise ValueError(message('submission.error.03'))
    path.mkdir(parents=True,exist_ok=True)
    frames = {name:frame_for(index,pred,blocks) for name,blocks in [('s_result.csv',8),('l_result.csv',96)]}
    for name, frame in frames.items():
        frame.to_csv(path/name,index=False,encoding='utf-8',float_format='%.6f')
    report = validate_pair(path,index)
    report.update({'baseline':model_name,'status':'baseline_output_not_optimized_final',
                   'interval_start_offset_minutes':cfg['protocol']['block_start_offset_minutes'],
                   'label_semantics':'15-minute mean; first interval origin+offset through +offset+14',
                   'test_true_load_input':False,'test_process_input':'available_at <= origin'})
    write_json(path/'validation.json',report)
    with zipfile.ZipFile(path/'results_only.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in frames:
            z.write(path/name,'submissions/'+name)
    # NOTE: 此包仅装结果；题目要求的算法代码另在主工程交付，不能混称完整参赛包。
    return report


def validate_pair(path: Path,index: pd.DatetimeIndex) -> dict:
    """读回CSV并校验行列、起点、有限性、重叠预测及目标包含关系。"""
    dfs = {}
    for name,blocks in [('s_result.csv',8),('l_result.csv',96)]:
        df = pd.read_csv(path/name,float_precision='round_trip')
        if list(df.columns) != columns(blocks) or len(df)!=len(index):
            raise ValueError(message('submission.error.05'))
        if list(df.datetime) != list(index.strftime('%Y-%m-%d %H:%M:%S')):
            raise ValueError(message('submission.error.06'))
        vals=df.iloc[:,1:].to_numpy()
        if not np.isfinite(vals).all() or (vals<0).any() or (vals[:,blocks:]<vals[:,:blocks]).any():
            raise ValueError(message('submission.error.07'))
        dfs[name]=df
    short = dfs['s_result.csv']; long = dfs['l_result.csv']
    if not np.array_equal(short.iloc[:,1:].to_numpy(),long[short.columns[1:]].to_numpy()):
        raise ValueError(message('submission.error.04'))
    return {'rows':len(index),'short_columns':17,'long_columns':193,
            'start':str(index[0]),'end':str(index[-1]),
            'sha256':{name:sha256(path/name) for name in dfs},'validation':'passed'}
