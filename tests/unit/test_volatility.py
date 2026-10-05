import numpy as np
import pandas as pd
import pytest

from quantlab.features.volatility import (
    annualized_volatility,
    rolling_annualized_volatility,
)


def test_annualized_volatility():
    returns = pd.Series([
        0.01,
        -0.01,
        0.02,
        -0.02,
        0.00,
    ])

    result = annualized_volatility(returns)

    expected = returns.std() * np.sqrt(252)

    assert result == pytest.approx(expected)


def test_rolling_annualized_volatility():
    returns = pd.Series([
        0.01,
        -0.01,
        0.02,
        0.00,
        -0.02,
    ])

    result = rolling_annualized_volatility(
        returns,
        window=3,
    )

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])

    assert result.iloc[2] == pytest.approx(0.2424871131)
    assert result.iloc[3] == pytest.approx(0.2424871131)
    assert result.iloc[4] == pytest.approx(0.3174901573)