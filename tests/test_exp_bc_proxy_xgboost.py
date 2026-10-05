"""XGBoost proxy persistence and inference contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features_experiments.exp_bc.b_variants import ProxyFit, fit_proxy


class XGBoostProxyTests(unittest.TestCase):
    def test_saved_proxy_replays_current_prediction(self) -> None:
        pre_index = pd.date_range("2025-01-01", periods=40, freq="15min")
        semi_index = pd.date_range("2025-05-03", periods=40, freq="15min")
        pre = pd.DataFrame(
            {"gas": np.linspace(70, 100, 40), "holder": np.linspace(30, 60, 40)},
            index=pre_index,
        )
        semi = pd.DataFrame(
            {"gas": np.linspace(80, 120, 40), "holder": np.linspace(40, 70, 40)},
            index=semi_index,
        )
        pre_y = np.column_stack((np.linspace(90, 110, 40), np.linspace(220, 280, 40)))
        semi_y = np.column_stack((np.linspace(100, 130, 40), np.linspace(250, 340, 40)))
        for engine, suffix in (("xgboost", "json"), ("lightgbm", "txt")):
            with (
                self.subTest(engine=engine),
                tempfile.TemporaryDirectory() as temporary,
            ):
                model_path = Path(temporary) / "proxy"
                fit = fit_proxy(
                    pre,
                    semi,
                    pre_y,
                    semi_y,
                    pd.Timestamp("2025-06-01"),
                    model_path,
                    lookback_days=180,
                    engine=engine,
                )
                restored = ProxyFit.load(model_path, pd.DataFrame())
                prediction = restored.current(semi.iloc[:5])
                self.assertEqual(prediction.shape, (5, 2))
                self.assertTrue(np.isfinite(prediction).all())
                np.testing.assert_allclose(
                    prediction, fit.current(semi.iloc[:5]), rtol=1e-7
                )
                metadata = json.loads(
                    (model_path / "fit.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["engine"], engine)
                self.assertEqual(metadata["rows"], 80)
                self.assertEqual(metadata["pre_rows"], 40)
                self.assertTrue((model_path / f"target_0.{suffix}").exists())
                self.assertTrue((model_path / f"target_1.{suffix}").exists())


if __name__ == "__main__":
    unittest.main()
