import numpy as np
import pytest
from gas_power.evaluation import mean_absolute_percentage_error


def test_mape_percent() -> None:
    actual = np.array([100.0, 200.0])
    predicted = np.array([90.0, 220.0])
    assert mean_absolute_percentage_error(actual, predicted) == pytest.approx(10.0)


def test_mape_requires_matching_shapes() -> None:
    with pytest.raises(ValueError):
        mean_absolute_percentage_error(np.ones(2), np.ones(3))

