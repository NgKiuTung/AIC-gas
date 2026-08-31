import numpy as np
import pandas as pd
from gas_power.data.causal_preprocessing import causal_fill, merge_raw_tables
from gas_power.features.domain_interactions import add_domain_interaction_features
from gas_power.features.inference import build_inference_feature_frame, select_model_features


def test_causal_fill_prefix_is_independent_of_future_values() -> None:
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min")
    prefix = pd.Series([10.0, np.nan, np.nan, np.nan, 20.0])
    future_a = pd.concat([prefix, pd.Series([30.0, 40.0, 50.0])], ignore_index=True)
    future_b = pd.concat([prefix, pd.Series([3000.0, 4000.0, 5000.0])], ignore_index=True)
    filled_a, _ = causal_fill(future_a, pd.Series(timestamps))
    filled_b, _ = causal_fill(future_b, pd.Series(timestamps))
    np.testing.assert_allclose(filled_a.iloc[:5], filled_b.iloc[:5])


def test_merge_raw_tables_marks_a_missing_source_row() -> None:
    timestamps = pd.date_range("2025-01-01", periods=3, freq="15min")
    tables = {
        "gas": pd.DataFrame({"datetime": timestamps.delete(1), "gas_x": [1.0, 3.0]}),
        "holder": pd.DataFrame({"datetime": timestamps, "holder_x": [1.0, 2.0, 3.0]}),
        "user": pd.DataFrame({"datetime": timestamps, "user_x": [1.0, 2.0, 3.0]}),
        "load": pd.DataFrame({"datetime": timestamps, "load_x": [1.0, 2.0, 3.0]}),
    }
    merged = merge_raw_tables(tables)
    assert merged.loc[1, "feat_source_missing_gas"] == 1
    assert np.isnan(merged.loc[1, "gas_x"])


def test_inference_builder_uses_only_current_and_past() -> None:
    periods = 700
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="15min")
    base = np.arange(periods, dtype=float) + 100.0
    frame = pd.DataFrame(
        {
            "datetime": timestamps,
            "feat_generator_1_filled": base,
            "feat_generator_all_filled": base + 50,
            "feat_blast_furnace_observed_sum": base + 200,
            "feat_air_heater_observed_sum": base * 0.1,
            "feat_blast_furnace_user_observed_sum": base * 0.2,
            "feat_converter_user_observed_sum": base * 0.05,
            "into_gas_mixed_blast_furnace": base * 0.03,
            "generator_use_blast_furnace_gas": base * 0.2,
            "converter_1": base * 0.4,
            "into_gas_mixed_converter": base * 0.02,
            "generator_use_converter_gas": base * 0.03,
            "generator_use_coke_gas": base * 0.04,
            "blast_furnace_gas_holder_2": base + 500,
            "coke_oven_1": base + 10,
            "into_gas_mixed_coke": base * 0.02,
        }
    )
    for column in (
        "converter_user1", "blast_furnace_user1", "blast_furnace_user2", "blast_furnace_user3",
        "blast_furnace_user4", "converter_user2", "air_heater_2", "air_heater_4", "air_heater_5",
        "blast_furnace_5",
    ):
        frame[column] = base * 0.01
    for column in ("feat_missing_x", "feat_source_missing_x", "feat_outlier_generator_1"):
        frame[column] = 0
    first = build_inference_feature_frame(frame)
    mutated = frame.copy()
    mutated.loc[691:, "feat_generator_1_filled"] += 1_000_000
    second = build_inference_feature_frame(mutated)
    schema = ["feat_p50_current", "feat_lag672_p50_current", "feat_smooth_p50_span96_ewm"]
    np.testing.assert_allclose(
        select_model_features(first, schema).iloc[690], select_model_features(second, schema).iloc[690]
    )
    assert not any(column.startswith("label_") for column in first)


def test_domain_interactions_have_expected_values() -> None:
    frame = pd.DataFrame(
        {
            "feat_blast_furnace_observed_sum": [100.0],
            "feat_air_heater_observed_sum": [10.0],
            "feat_blast_furnace_user_observed_sum": [20.0],
            "into_gas_mixed_blast_furnace": [5.0],
            "generator_use_blast_furnace_gas": [25.0],
            "generator_use_coke_gas": [30.0],
            "generator_use_converter_gas": [0.0],
            "blast_furnace_gas_holder_2": [100_000.0],
        }
    )
    output = add_domain_interaction_features(frame)
    assert output.loc[0, "feat_interact_holder_bfg_balance"] == 0.20
    assert output.loc[0, "feat_generation_fuel_structure_hhi"] == 0.5


def test_domain_interaction_handles_zero_supply() -> None:
    frame = pd.DataFrame(
        {
            "feat_blast_furnace_observed_sum": [0.0],
            "feat_air_heater_observed_sum": [0.0],
            "feat_blast_furnace_user_observed_sum": [0.0],
            "into_gas_mixed_blast_furnace": [1.0],
            "generator_use_blast_furnace_gas": [10.0],
            "generator_use_coke_gas": [0.0],
            "generator_use_converter_gas": [0.0],
            "blast_furnace_gas_holder_2": [100_000.0],
        }
    )
    output = add_domain_interaction_features(frame)
    assert np.isfinite(output["feat_interact_holder_bfg_balance"]).all()
