import numpy as np
import pandas as pd
from gas_power.features.spectral import add_causal_spectral_features


def test_spectral_features_are_finite_and_causal() -> None:
    n = 160
    timestamps = pd.date_range("2025-01-01", periods=n, freq="15min")
    values = np.sin(2 * np.pi * np.arange(n) / 32.0)
    frame = pd.DataFrame({"datetime": timestamps, "signal": values})
    baseline = add_causal_spectral_features(frame, {"test": "signal"}, window=32)
    changed = frame.copy()
    changed.loc[100:, "signal"] += 1000.0
    perturbed = add_causal_spectral_features(changed, {"test": "signal"}, window=32)
    feature_columns = [column for column in baseline if column.startswith("feat_")]
    assert np.isfinite(baseline[feature_columns].to_numpy()).all()
    np.testing.assert_allclose(
        baseline.loc[:99, feature_columns].to_numpy(),
        perturbed.loc[:99, feature_columns].to_numpy(),
    )
    assert baseline.loc[:30, "feat_spectral_window32_ready"].eq(0).all()
    assert baseline.loc[31:, "feat_spectral_window32_ready"].eq(1).all()
