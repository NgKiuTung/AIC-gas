"""私有AIC赛事研究：真实卷积、密集解码、MAPE和融合标签可用性测试。"""
import numpy as np
import pandas as pd
import pytest
import torch
from gasbench.networks import CausalBlock, build_network, masked_mape, masked_competition_mape
from gasbench.ensemble import BANDS, apply_weights, matured_folds, weights_for
from gasbench.audit import independent_score
from gasstage.metrics import score
from gasstage.submission import export_pair, validate_pair

@pytest.mark.parametrize('dilation', [1, 2, 4, 8])
def test_causal_convolution(dilation):
    torch.set_num_threads(2)
    torch.manual_seed(1)
    block = CausalBlock(4, dilation, 0).eval()
    x = torch.randn(2, 4, 40)
    changed = x.clone()
    changed[:, :, 20:] = 10000.0
    torch.testing.assert_close(block(x)[:, :, :20], block(changed)[:, :, :20], rtol=0, atol=0)

@pytest.mark.parametrize('family', ['tcn', 'tide_style'])
@pytest.mark.parametrize('history', [96, 288])
def test_forecaster_shapes_and_backprop(family, history):
    torch.set_num_threads(2)
    cfg = {'hidden': 8, 'levels': 2, 'projection': 4, 'dropout': 0}
    model = build_network(family, 81, 276, history, cfg)
    x = torch.randn(2, history, 81)
    current = torch.randn(2, 276)
    future = torch.randn(2, 96, 6)
    anchor = torch.ones(2, 96, 2)
    pred = model(x, current, future, anchor)
    torch.testing.assert_close(pred, anchor)
    loss = masked_mape(pred, anchor + 0.1)
    loss.backward()
    assert pred.shape == (2, 96, 2) and torch.isfinite(loss)

def test_masked_mape_matches_numpy():
    y = np.ones((3, 96, 2))
    p = y.copy()
    p[:, :8, 0] = 1.1
    y[0, 0, 0] = np.nan
    base = score(y, p)
    actual = masked_mape(torch.tensor(p), torch.tensor(y)).item()
    weighted = masked_competition_mape(torch.tensor(p), torch.tensor(y)).item()
    assert actual == pytest.approx(base['combined_mape'], abs=1e-07)
    assert weighted == pytest.approx((base['short_mape'] + 2 * base['long_mape']) / 3, abs=1e-07)

def test_independent_metrics():
    rng = np.random.default_rng(3)
    y = rng.uniform(50, 250, (8, 96, 2))
    p = y + rng.normal(size=y.shape)
    independent = independent_score(y, p)
    assert independent['combined_mape'] == pytest.approx(score(y, p)['combined_mape'], abs=1e-14)

@pytest.mark.parametrize('stamp,expected', [('2025-08-01', 1), ('2025-07-10', 0), ('2025-07-12', 1)])
def test_calibration_maturity(stamp, expected):
    folds = [{'name': 'july', 'start': '2025-07-01', 'end': '2025-07-10 23:45'}]
    assert len(matured_folds(folds, pd.Timestamp(stamp))) == expected

def test_fusion_prefers_better_expert():
    y = np.ones((8, 96, 2)) * 100
    p = np.stack([y + 20, y])
    w, status = weights_for(p, y, 0, 1e-05)
    out = apply_weights(p, w)
    assert np.max(np.abs(out - y)) < 0.01 and np.allclose(w.sum(axis=-1), 1)

def test_fusion_no_labels_uses_reference():
    p = np.ones((2, 3, 96, 2))
    y = np.full((3, 96, 2), np.nan)
    w, _ = weights_for(p, y, 0, 0.001)
    assert np.all(w[:, :, 0] == 1)

def test_submission_complete(tmp_path):
    idx = pd.date_range('2025-10-01', periods=960, freq='15min')
    pred = np.broadcast_to(np.array([100.0, 300.0]), (960, 96, 2)).copy()
    cfg = {'protocol': {'block_start_offset_minutes': 0}}
    r = export_pair(tmp_path, idx, pred, cfg, 'unit')
    assert (tmp_path / 'results_only.zip').exists()
    assert validate_pair(tmp_path, idx)['rows'] == 960
