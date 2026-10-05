import pandas as pd
import pytest

from quantlab.features.risk_free import (
    annual_yield_to_daily_return,
    cash_wealth_index,
    trailing_cash_return,
)


def test_annual_yield_to_daily_return():
    annual_yield = pd.Series([
        4.20,
        3.00,
        0.00,
    ])

    result = annual_yield_to_daily_return(annual_yield)

    assert result.iloc[0] == pytest.approx(0.042 / 252)
    assert result.iloc[1] == pytest.approx(0.03 / 252)
    assert result.iloc[2] == pytest.approx(0.0)


def test_cash_wealth_index():
    daily_returns = pd.Series([
        0.001,
        0.002,
        0.001,
    ])

    result = cash_wealth_index(daily_returns)

    assert result.iloc[0] == pytest.approx(1.001)
    assert result.iloc[1] == pytest.approx(1.003002)
    assert result.iloc[2] == pytest.approx(1.004005002)


def test_trailing_cash_return():
    wealth_index = pd.Series([
        1.00,
        1.01,
        1.02,
        1.04,
    ])

    result = trailing_cash_return(
        wealth_index,
        lookback=2,
    )

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])

    assert result.iloc[2] == pytest.approx(1.02 / 1.00 - 1.0)
    assert result.iloc[3] == pytest.approx(1.04 / 1.01 - 1.0)