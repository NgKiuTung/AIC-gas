import numpy as np
import pandas as pd
import pytest
from gas_power.data.raw_inputs import combine_history_and_scoring_tables, price_lookup_from_frame


def test_price_lookup_uses_half_hour_rows_and_twelve_positional_months() -> None:
    intervals = [
        f"{hour:02d}:{minute:02d}-{hour:02d}:{(minute + 30) % 60:02d}"
        for hour in range(24)
        for minute in (0, 30)
    ]
    frame = pd.DataFrame({"interval": intervals})
    for month in range(1, 13):
        frame[f"month_{month}"] = month * 100 + np.arange(48)
    lookup = price_lookup_from_frame(frame)
    assert len(lookup) == 576
    assert lookup[(5, 17)] == 517.0


def test_combine_trims_training_at_scoring_boundary() -> None:
    history_times = pd.date_range("2025-04-30 23:30", periods=3, freq="15min")
    scoring_times = pd.date_range("2025-05-01 00:00", periods=2, freq="15min")
    training = {
        source: pd.DataFrame({"datetime": history_times, f"{source}_value": [1.0, 2.0, 999.0]})
        for source in ("gas", "holder", "user", "load")
    }
    scoring = {
        source: pd.DataFrame({"datetime": scoring_times, f"{source}_value": [10.0, 11.0]})
        for source in ("gas", "holder", "user", "load")
    }
    combined, references = combine_history_and_scoring_tables(training, scoring)
    assert references.equals(pd.DatetimeIndex(scoring_times))
    assert combined["load"]["load_value"].tolist() == [1.0, 2.0, 10.0, 11.0]


def test_combine_rejects_scoring_schema_drift() -> None:
    timestamps = pd.date_range("2025-05-01", periods=2, freq="15min")
    training = {
        source: pd.DataFrame({"datetime": timestamps - pd.Timedelta(days=1), f"{source}_value": [1.0, 2.0]})
        for source in ("gas", "holder", "user", "load")
    }
    scoring = {
        source: pd.DataFrame({"datetime": timestamps, f"{source}_value": [3.0, 4.0]})
        for source in ("gas", "holder", "user", "load")
    }
    scoring["gas"] = scoring["gas"].rename(columns={"gas_value": "unexpected"})
    with pytest.raises(ValueError):
        combine_history_and_scoring_tables(training, scoring)
