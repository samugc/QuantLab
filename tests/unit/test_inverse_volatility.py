import pandas as pd
import pytest

from quantlab.portfolio.inverse_volatility import inverse_volatility_weights


def test_inverse_volatility_weights():
    volatility = pd.Series({
        "A": 0.10,
        "B": 0.20,
        "C": 0.40,
    })

    weights = inverse_volatility_weights(volatility)

    assert weights["A"] == pytest.approx(0.5714285714)
    assert weights["B"] == pytest.approx(0.2857142857)
    assert weights["C"] == pytest.approx(0.1428571429)

    assert weights.sum() == pytest.approx(1.0)

def test_inverse_volatility_rejects_zero_volatility():
    volatility = pd.Series({
        "A": 0.10,
        "B": 0.00,
    })

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        inverse_volatility_weights(
            volatility
        )