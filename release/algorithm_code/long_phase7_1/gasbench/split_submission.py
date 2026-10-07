"""私有AIC赛事研究：短/长任务可使用不同模型时的独立结果导出。"""
from __future__ import annotations

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

from gasbench.common import msg, write_json
from gasstage.common import sha256
from gasstage.submission import columns, frame_for


def _validate_values(prediction: np.ndarray, blocks: int) -> None:
    """校验任务自身的有限性、非负性和总负荷包含关系。"""
    view = prediction[:, :blocks]
    if not np.isfinite(view).all() or (view < 0).any():
        raise ValueError(msg('submission.invalid', detail='non-finite or negative values'))
    if (view[:, :, 1] < view[:, :, 0]).any():
        raise ValueError(msg('submission.invalid', detail='generator_all < generator_1'))


def validate_split_pair(path: Path, index: pd.DatetimeIndex) -> dict:
    """校验两张结果表；不要求重叠8个区间数值相同。"""
    frames = {}
    for name, blocks in (("s_result.csv", 8), ("l_result.csv", 96)):
        frame = pd.read_csv(path / name, float_precision="round_trip")
        if list(frame.columns) != columns(blocks) or len(frame) != len(index):
            raise ValueError(msg('submission.invalid', detail='shape/columns'))
        if frame["datetime"].tolist() != index.strftime("%Y-%m-%d %H:%M:%S").tolist():
            raise ValueError(msg('submission.invalid', detail='timestamps'))
        raw = frame.iloc[:, 1:].to_numpy(dtype="float64")
        pred = np.stack((raw[:, :blocks], raw[:, blocks:]), axis=-1)
        _validate_values(pred, blocks)
        frames[name] = frame
    short = frames["s_result.csv"].iloc[:, 1:].to_numpy()
    long = frames["l_result.csv"][frames["s_result.csv"].columns[1:]].to_numpy()
    return {
        "rows": len(index),
        "short_columns": 17,
        "long_columns": 193,
        "start": str(index[0]),
        "end": str(index[-1]),
        "overlap_equal": bool(np.array_equal(short, long)),
        "sha256": {name: sha256(path / name) for name in frames},
        "validation": "passed",
    }


def export_split_pair(
    path: Path,
    index: pd.DatetimeIndex,
    short_prediction: np.ndarray,
    long_prediction: np.ndarray,
    contract: dict,
    model_name: str,
    provenance: dict,
) -> dict:
    """分别导出短/长任务，记录模型来源并保留完整算法复现边界。"""
    if short_prediction.shape != (len(index), 96, 2) or long_prediction.shape != (len(index), 96, 2):
        raise ValueError(msg('submission.invalid', detail='prediction shape'))
    _validate_values(short_prediction, 8)
    _validate_values(long_prediction, 96)
    path.mkdir(parents=True, exist_ok=True)
    short = frame_for(index, short_prediction, 8)
    long = frame_for(index, long_prediction, 96)
    short.to_csv(path / "s_result.csv", index=False, encoding="utf-8", float_format="%.6f")
    long.to_csv(path / "l_result.csv", index=False, encoding="utf-8", float_format="%.6f")
    report = validate_split_pair(path, index)
    report.update(
        {
            "model": model_name,
            "status": "candidate_output_not_officially_validated",
            "interval_start_offset_minutes": contract["protocol"]["block_start_offset_minutes"],
            "label_semantics": "15-minute mean; short and long tasks may use different models",
            "test_true_load_input": False,
            "test_process_input": "available_at <= origin",
            "provenance": provenance,
        }
    )
    write_json(path / "validation.json", report)
    with zipfile.ZipFile(path / "results_only.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(path / "s_result.csv", "submissions/s_result.csv")
        archive.write(path / "l_result.csv", "submissions/l_result.csv")
    return report
