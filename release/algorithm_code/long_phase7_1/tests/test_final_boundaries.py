"""私有AIC赛事研究：检查点恢复和融合层时间污染边界。"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest
import torch
from gasbench.ensemble import BANDS
from gasbench.fusion_audit import check_provenance, compare_csv_prediction
from gasbench.neural_data import arrays_for, fit_normalization
from gasbench.neural_fit import fit_epochs
from gasbench.samples import sequence_origins
from gasstage.submission import export_pair

def fusion_fixture():
    """第一折固定参照，其余专家不能从未兑现标签得到权重。"""
    fold = {'name': 'july', 'start': '2025-07-01', 'end': '2025-07-10 23:45'}
    weights = np.zeros((4, 2, 2))
    weights[:, :, 0] = 1
    return ({'fold': fold, 'prior_calibration_folds': [], 'same_fold_labels_used_to_fit_weights': False, 'experts': ['reference_58', 'other'], 'weights': weights.tolist(), 'bands': [list(x) for x in BANDS]}, {'folds': [fold]})

def test_fusion_provenance_accepts_reference():
    evidence, config = fusion_fixture()
    assert check_provenance(evidence, config).shape == (4, 2, 2)

@pytest.mark.parametrize('corruption', ['self', 'weights', 'nan', 'label_flag', 'bounds'])
def test_fusion_provenance_rejects(corruption):
    evidence, config = fusion_fixture()
    if corruption == 'self':
        evidence['prior_calibration_folds'] = config['folds']
    if corruption == 'weights':
        evidence['weights'][0][0] = [0.5, 0.5]
    if corruption == 'nan':
        evidence['weights'][0][0][0] = float('nan')
    if corruption == 'label_flag':
        evidence['same_fold_labels_used_to_fit_weights'] = True
    if corruption == 'bounds':
        evidence['bands'][0][1] = 9
    with pytest.raises((ValueError, AssertionError)):
        check_provenance(evidence, config)

def test_csv_values_not_just_headers(tmp_path):
    index = pd.date_range('2025-10-01', periods=960, freq='15min')
    pred = np.broadcast_to([120.0, 300.0], (960, 96, 2)).copy()
    export_pair(tmp_path, index, pred, {'protocol': {'block_start_offset_minutes': 0}}, 'unit')
    assert max(compare_csv_prediction(tmp_path, index, pred).values()) == 0
    frame = pd.read_csv(tmp_path / 's_result.csv')
    frame.iloc[0, 1] = 999
    frame.to_csv(tmp_path / 's_result.csv', index=False)
    with pytest.raises(ValueError):
        compare_csv_prediction(tmp_path, index, pred)

@pytest.mark.parametrize('family', ['tcn', 'tide_style'])
def test_checkpoint_resume_exact_epoch_boundary(bundle, tiny_neural, tmp_path, family):
    cfg = deepcopy(tiny_neural)
    cfg.update(family=family, epochs=2, dropout=0.15, max_train_origins=96)
    rows = sequence_origins(bundle, pd.Timestamp('2025-06-08'), cfg)
    norm = fit_normalization(bundle, rows, 96)
    arrays = arrays_for(bundle, rows, 96, norm)
    first, state = fit_epochs(arrays, None, cfg, 2, tmp_path / 'continued', 'refit', time.time() + 60, epochs=1)
    assert state['epoch'] == 1
    del first
    resumed, resumed_state = fit_epochs(arrays, None, cfg, 2, tmp_path / 'continued', 'refit', time.time() + 60, epochs=2)
    fresh, fresh_state = fit_epochs(arrays, None, cfg, 2, tmp_path / 'fresh', 'refit', time.time() + 60, epochs=2)
    assert resumed_state['trace'] == fresh_state['trace']
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, fresh.state_dict()[key], rtol=0, atol=0)
