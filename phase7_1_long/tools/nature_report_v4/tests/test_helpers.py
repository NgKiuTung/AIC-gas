"""Focused regression tests for v4 report parsing and semantic mappings."""
from pathlib import Path
import sys

import pandas as pd
import pytest

TOOL_ROOT = Path(__file__).resolve().parents[1]
# NOTE: tests are archived inside the tool root, so import from that root.
sys.path.insert(0, str(TOOL_ROOT))

from naturev4.features import operator_group, source_group, source_operator_table  # noqa: E402
from naturev4.fig_quality import _raw_missingness  # noqa: E402
from naturev4.fig_selection import _selection_map  # noqa: E402


@pytest.mark.parametrize(
    ("operator", "source", "expected"),
    [
        ("observed", "blast_furnace_1", "state snapshot"),
        ("mean_60m", "blast_furnace_1", "rolling statistics"),
        ("diff_15m", "blast_furnace_1", "dynamics"),
        ("valid_count / scheduled_count in past hour", "blast_furnace_1", "coverage QC"),
        ("feat_day_sin", "deterministic_calendar_or_source_schedule", "known future"),
    ],
)
def test_operator_group(operator, source, expected):
    assert operator_group(operator, source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("blast_furnace_1", "gas production"),
        ("air_heater_3", "air heaters"),
        ("into_gas_mixed_coke", "mixed-gas inflow"),
        ("blast_furnace_gas_holder_1", "gas holder"),
        ("converter_user3", "industrial users"),
        ("generator_use_coke_gas", "generator fuel use"),
        ("deterministic_calendar_or_source_schedule", "known future"),
    ],
)
def test_source_group(source, expected):
    assert source_group(source) == expected


def test_source_operator_table_preserves_feature_count():
    frame = pd.DataFrame(
        {
            "source": ["blast_furnace_1", "blast_furnace_1", "deterministic_calendar_or_source_schedule"],
            "operator": ["observed", "mean_15m", "feat_day_sin"],
        }
    )
    assert int(source_operator_table(frame).to_numpy().sum()) == 3


def test_selection_map_uses_authoritative_screen_picks():
    obj = {
        "long": {
            "screen_picks": {
                "g1__h01_08": "reference_58",
                "g1__h09_24": "reference_58",
                "g1__h25_48": "reference_58",
                "g1__h49_96": "reference_58",
                "gall__h01_08": "xgb_wide_180d_d50_t12",
                "gall__h09_24": "xgb_wide_180d_d50_t12",
                "gall__h25_48": "xgb_wide_180d_d50_t12",
                "gall__h49_96": "xgb_wide_180d_d50_t12",
            }
        }
    }
    mapping = _selection_map(obj)
    assert mapping[(0, 0)] == "reference_58"
    assert mapping[(1, 3)] == "xgb_wide_180d_d50_t12"


def test_raw_missingness_uses_inventory_denominator():
    profile = pd.DataFrame(
        {
            "field": ["x", "x"],
            "missing": [10, 20],
            "source": ["a.csv", "b.csv"],
        }
    )
    inventory = pd.DataFrame({"file": ["a.csv", "b.csv"], "rows": [100, 200]})
    result = _raw_missingness(profile, inventory)
    assert result.loc[0, "missing_pct"] == pytest.approx(10.0)
