"""Boundary and label checks for the independent EXP-B/C data builder."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from features_experiments.exp_bc.infer_october import enforce_ramp
from features_experiments.exp_bc.oof_residual import FOLDS
from features_experiments.exp_bc.physical_features import (
    HIGHFREQ_VARIABLES,
    SCREENING_END,
    add_labels,
    availability_grid,
    common_features,
    highfreq_features,
)


class PhysicalFeatureTests(unittest.TestCase):
    def test_feature_screening_ends_before_first_oof_validation(self) -> None:
        self.assertLess(pd.Timestamp(SCREENING_END), FOLDS[0][1])

    def test_pre_delay_and_semifinal_boundary(self) -> None:
        pre_index = pd.date_range(
            "2025-05-02 23:30", periods=2, freq="15min", name="datetime"
        )
        semi_index = pd.date_range(
            "2025-05-03 00:00", periods=2, freq="min", name="datetime"
        )
        pre = pd.DataFrame(
            {
                "gas": [100.0, 200.0],
                "generator_1": [1.0, 2.0],
                "generator_all": [3.0, 4.0],
                "dataset_phase": "preliminary_released_eval",
            },
            index=pre_index,
        )
        semi = pd.DataFrame(
            {
                "gas": [300.0, 400.0],
                "generator_1": [5.0, 6.0],
                "generator_all": [7.0, 8.0],
                "dataset_phase": "semifinal_train",
            },
            index=semi_index,
        )
        raw, current = availability_grid(pre, semi)
        self.assertTrue(np.isnan(current.loc["2025-05-02 23:30", "gas"]))
        self.assertEqual(current.loc["2025-05-02 23:45", "gas"], 100.0)
        self.assertEqual(current.loc["2025-05-02 23:59", "gas"], 100.0)
        self.assertEqual(current.loc["2025-05-03 00:00", "gas"], 300.0)
        self.assertEqual(raw.loc["2025-05-03 00:01", "gas"], 400.0)
        origins = pd.DatetimeIndex(
            [pd.Timestamp("2025-05-03 00:00")], name="reference_time"
        )
        features, _ = common_features(raw, current, origins, ["gas"], set())
        self.assertEqual(features.iloc[0]["feat_gas__current"], 300.0)
        self.assertEqual(features.iloc[0]["feat_gas__lag_15m"], 100.0)

    def test_interval_mean_and_proxy_labels_remain_distinct(self) -> None:
        pre_index = pd.date_range(
            "2025-01-01 00:00", periods=3, freq="15min", name="datetime"
        )
        semi_index = pd.date_range(
            "2025-05-03 00:00", periods=30, freq="min", name="datetime"
        )
        pre = pd.DataFrame(
            {"generator_1": [10.0, 20.0, 30.0], "generator_all": [100.0, 200.0, 300.0]},
            index=pre_index,
        )
        semi = pd.DataFrame(
            {
                "generator_1": np.arange(30, dtype=float) + 1,
                "generator_all": np.arange(30, dtype=float) + 101,
            },
            index=semi_index,
        )
        pre_features = pd.DataFrame(
            index=pd.DatetimeIndex(
                [pd.Timestamp("2025-01-01 00:15")], name="reference_time"
            )
        )
        semi_features = pd.DataFrame(
            index=pd.DatetimeIndex(
                [pd.Timestamp("2025-05-03 00:00")], name="reference_time"
            )
        )
        result = add_labels(pre, semi, pre_features, semi_features)
        pre_row = result.loc[result["dataset_phase"].eq("preliminary")].iloc[0]
        semi_row = result.loc[result["dataset_phase"].eq("semifinal")].iloc[0]
        self.assertEqual(pre_row["proxy_current_generator_1"], 10.0)
        self.assertEqual(pre_row["label_h15_generator_1"], 30.0)
        self.assertEqual(pre_row["label_semantics"], "endpoint_proxy_unverified")
        self.assertAlmostEqual(semi_row["label_h15_generator_1"], 8.0)
        self.assertEqual(semi_row["proxy_current_generator_1"], 1.0)
        self.assertEqual(semi_row["label_semantics"], "complete_15x1min_mean")

    def test_future_process_perturbation_does_not_change_origin_features(self) -> None:
        pre_index = pd.date_range("2025-05-02 23:30", periods=2, freq="15min")
        semi_index = pd.date_range("2025-05-03 00:00", periods=90, freq="min")
        pre = pd.DataFrame(
            {
                "gas": [100.0, 110.0],
                "generator_1": [1.0, 2.0],
                "generator_all": [3.0, 4.0],
                "dataset_phase": "preliminary",
            },
            index=pre_index,
        )
        semi = pd.DataFrame(
            {
                "gas": np.arange(90, dtype=float),
                "generator_1": 1.0,
                "generator_all": 2.0,
                "dataset_phase": "semifinal",
            },
            index=semi_index,
        )
        for source in HIGHFREQ_VARIABLES:
            semi[source] = np.arange(90, dtype=float) + 1
        altered = semi.copy()
        altered.loc[altered.index > semi_index[15], "gas"] = 1e9
        altered.loc[altered.index > semi_index[15], "blast_furnace_1"] = 1e9
        origin = pd.DatetimeIndex([semi_index[15]])
        left_raw, left_current = availability_grid(pre, semi)
        right_raw, right_current = availability_grid(pre, altered)
        left_common, _ = common_features(left_raw, left_current, origin, ["gas"], set())
        right_common, _ = common_features(
            right_raw, right_current, origin, ["gas"], set()
        )
        pd.testing.assert_frame_equal(left_common, right_common)
        left_highfreq, _ = highfreq_features(semi, origin, set())
        right_highfreq, _ = highfreq_features(altered, origin, set())
        pd.testing.assert_frame_equal(left_highfreq, right_highfreq)

    def test_ramp_guard_enforces_aggregate_1_percent_limit(self) -> None:
        prediction = np.array([[[100.0, 200.0], [180.0, 400.0], [190.0, 200.0]]])
        guarded = enforce_ramp(prediction)
        self.assertTrue((np.abs(np.diff(guarded, axis=1)) <= [30.0, 66.0]).all())
        self.assertTrue((guarded[..., 1] >= guarded[..., 0]).all())


if __name__ == "__main__":
    unittest.main()
