import pandas as pd
import pytest

from quantlab.evaluation.metrics import (
    annualized_volatility,
    cagr,
    drawdown_series,
    maximum_drawdown,
    sharpe_ratio,
)


def test_annualized_volatility():
    returns = pd.Series([
        0.01,
        -0.01,
        0.02,
        -0.02,
    ])

    result = annualized_volatility(returns)

    expected = returns.std() * (252 ** 0.5)

    assert result == pytest.approx(expected)


def test_cagr():
    dates = pd.to_datetime([
        "2020-01-01",
        "2021-01-01",
    ])

    wealth = pd.Series(
        [1.0, 1.10],
        index=dates,
    )

    result = cagr(wealth)

    assert result == pytest.approx(0.10, abs=0.001)


def test_sharpe_ratio():
    returns = pd.Series([
        0.01,
        0.02,
        -0.01,
        0.015,
    ])

    risk_free_returns = pd.Series([
        0.001,
        0.001,
        0.001,
        0.001,
    ])

    excess = returns - risk_free_returns

    expected = (
        excess.mean()
        / excess.std()
        * (252 ** 0.5)
    )

    result = sharpe_ratio(
        returns,
        risk_free_returns,
    )

    assert result == pytest.approx(expected)


def test_drawdown_series():
    wealth = pd.Series([
        1.00,
        1.20,
        1.10,
        0.90,
        1.05,
    ])

    result = drawdown_series(wealth)

    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(1.10 / 1.20 - 1.0)
    assert result.iloc[3] == pytest.approx(0.90 / 1.20 - 1.0)
    assert result.iloc[4] == pytest.approx(1.05 / 1.20 - 1.0)


def test_maximum_drawdown():
    wealth = pd.Series([
        1.00,
        1.20,
        1.10,
        0.90,
        1.05,
    ])

    result = maximum_drawdown(wealth)

    assert result == pytest.approx(-0.25)