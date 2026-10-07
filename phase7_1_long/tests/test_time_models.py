"""私有AIC赛事研究：共享样本、真实模型族训练、时间隔离和加载重放。"""
from dataclasses import replace
import copy, time
import numpy as np
import pandas as pd
import pytest
from gasbench.backends import active_columns, fit_backend, predict_backend, load_backend, save_backend, EXTENSIONS
from gasbench.samples import proxy_samples, direct_samples, sequence_origins, training_template, known_future
from gasbench.tree import TreePredictor
from gasbench.neural import NeuralPredictor
from gasbench.factory import load_model, predict_model
from gasbench.audit import poisoned_bundle
from gasstage.targets import future_truth, eligible_training, history_only

def test_train_purge(bundle, tiny_tree):
    rows = sequence_origins(bundle, pd.Timestamp('2025-06-01'), tiny_tree)
    assert bundle.features.index[rows[-1]] + pd.Timedelta(days=1) <= pd.Timestamp('2025-06-01')

def test_truth_hidden_samples(bundle, tiny_tree):
    cutoff = pd.Timestamp('2025-06-01')
    bad = poisoned_bundle(bundle, cutoff)
    a = direct_samples(bundle, cutoff, tiny_tree)[-1]['sample_hashes']
    b = direct_samples(bad, cutoff, tiny_tree)[-1]['sample_hashes']
    assert a == b

def test_proxy_shared_samples(bundle):
    cutoff = pd.Timestamp('2025-06-01')
    bad = poisoned_bundle(bundle, cutoff)
    assert proxy_samples(bundle, cutoff)[3]['sample_hashes'] == proxy_samples(bad, cutoff)[3]['sample_hashes']

def test_template_does_not_read_later_labels(bundle, tiny_tree):
    cutoff = pd.Timestamp('2025-06-01')
    rows = sequence_origins(bundle, cutoff, tiny_tree)
    bad = poisoned_bundle(bundle, cutoff)
    np.testing.assert_array_equal(training_template(bundle, rows), training_template(bad, rows))

def test_future_known_no_process(bundle):
    index = pd.date_range('2025-10-01', periods=2, freq='15min')
    value = known_future(index, bundle.contract)
    assert value.shape == (2, 96, 6) and np.isfinite(value).all()

def test_active_columns():
    x = np.c_[np.arange(32), np.ones(32), np.full(32, np.nan)]
    np.testing.assert_array_equal(active_columns(x), [0])

def test_no_active_columns():
    with pytest.raises(ValueError):
        active_columns(np.ones((32, 2)))

@pytest.mark.parametrize('family', ['lightgbm', 'catboost', 'xgboost'])
def test_real_backend_roundtrip(family, tmp_path):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 4)).astype('float32')
    y = 3 * x[:, 0]
    m = fit_backend(family, x, y, np.ones(len(y)), {'rounds': 5, 'device': 'cpu'}, 2)
    p = predict_backend(family, m, x, 2)
    path = tmp_path / ('model' + EXTENSIONS[family])
    save_backend(family, m, path)
    np.testing.assert_allclose(p, predict_backend(family, load_backend(family, path), x, 2), rtol=0, atol=0)

def test_unknown_backend():
    with pytest.raises(ValueError):
        fit_backend('unknown', np.zeros((32, 2)), np.ones(32), np.ones(32), {}, 1)

@pytest.mark.parametrize('family', ['lightgbm', 'catboost', 'xgboost'])
@pytest.mark.parametrize('kind', ['proxy', 'direct'])
def test_tree_contract_fit(family, kind, bundle, tiny_tree, tmp_path):
    cfg = {**tiny_tree, 'family': family, 'kind': kind}
    cutoff = pd.Timestamp('2025-06-01')
    path = tmp_path / 'model'
    model = TreePredictor.fit(bundle, cutoff, cfg, 2, path)
    features = bundle.features.loc[cutoff:].iloc[:3]
    pred = model.predict(features)
    np.testing.assert_allclose(pred, TreePredictor.load(path).predict(features), rtol=0, atol=0)
    assert pred.shape == (3, 96, 2) and (pred[:, :, 1] >= pred[:, :, 0]).all()
    with pytest.raises(ValueError):
        model.predict(bundle.features.loc[:cutoff].iloc[:1])

@pytest.mark.parametrize('family', ['tcn', 'tide_style'])
def test_neural_fit_and_replay(family, bundle, tiny_neural, tmp_path):
    cfg = {**tiny_neural, 'family': family}
    cutoff = pd.Timestamp('2025-06-01')
    model = NeuralPredictor.fit(bundle, cutoff, cfg, 2, tmp_path, time.time() + 120)
    idx = pd.date_range(cutoff, periods=3, freq='15min')
    pred = model.predict(bundle.features, idx)
    np.testing.assert_allclose(pred, NeuralPredictor.load(tmp_path).predict(bundle.features, idx), rtol=1e-06, atol=1e-05)
    np.testing.assert_allclose(pred, model.predict(bundle.features.loc[:idx[-1]], idx), rtol=0, atol=0)
    assert model.metadata['training']['pre_interval_labels_used'] is False
    assert pd.Timestamp(model.metadata['training']['inner_cutoff']) < cutoff
    with pytest.raises(ValueError):
        model.predict(bundle.features, pd.DatetimeIndex(['2025-05-31']))
