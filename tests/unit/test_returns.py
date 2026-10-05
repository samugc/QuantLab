import numpy as np
import pandas as pd
import pytest

from quantlab.features.returns import (
    cumulative_return,
    log_returns,
    simple_returns,
)


def test_simple_returns():
    prices = pd.Series([100.0, 110.0, 99.0])

    returns = simple_returns(prices)

    assert pd.isna(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(0.10)
    assert returns.iloc[2] == pytest.approx(-0.10)


def test_cumulative_return():
    returns = pd.Series([0.10, -0.10])

    result = cumulative_return(returns)

    assert result == pytest.approx(-0.01)


def test_log_returns():
    prices = pd.Series([100.0, 110.0, 99.0])

    returns = log_returns(prices)

    assert pd.isna(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(np.log(1.10))
    assert returns.iloc[2] == pytest.approx(np.log(0.90))