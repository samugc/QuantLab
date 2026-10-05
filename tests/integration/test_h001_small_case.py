import pandas as pd
import pytest

from quantlab.evaluation.metrics import (
    maximum_drawdown,
    sharpe_ratio,
)
from quantlab.features.returns import simple_returns
from quantlab.features.risk_free import (
    cash_wealth_index,
    trailing_cash_return,
)
from quantlab.portfolio.inverse_volatility import (
    inverse_volatility_weights,
)
from quantlab.signals.momentum import momentum_return
from quantlab.simulation.backtest import (
    simulate_rebalanced_portfolio,
)


def test_h001_end_to_end_small_case():
    dates = pd.to_datetime([
        "2026-01-01",
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
    ])

    prices = pd.DataFrame(
        {
            "A": [
                100.0,
                110.0,
                120.0,
                120.0,
                108.0,
                108.0,
                108.0,
            ],
            "B": [
                100.0,
                95.0,
                90.0,
                90.0,
                99.0,
                99.0,
                108.9,
            ],
        },
        index=dates,
    )

    asset_returns = pd.DataFrame(
        {
            ticker: simple_returns(prices[ticker])
            for ticker in prices.columns
        }
    )

    lookback = 2

    momentum = pd.DataFrame(
        {
            ticker: momentum_return(
                prices[ticker],
                lookback=lookback,
            )
            for ticker in prices.columns
        }
    )

    cash_daily_return = pd.Series(
        0.0,
        index=dates,
    )

    cash_wealth = cash_wealth_index(
        cash_daily_return
    )

    cash_momentum = trailing_cash_return(
        cash_wealth,
        lookback=lookback,
    )

    excess_momentum = momentum.sub(
        cash_momentum,
        axis=0,
    )

    signal_on = excess_momentum > 0.0

    first_decision_date = dates[2]
    second_decision_date = dates[4]

    # First decision:
    # A gained 20%, B lost 10%.
    assert bool(
        signal_on.loc[
            first_decision_date,
            "A",
        ]
    )

    assert not bool(
        signal_on.loc[
            first_decision_date,
            "B",
        ]
    )

    # Second decision:
    # A lost 10%, B gained 10%.
    assert not bool(
        signal_on.loc[
            second_decision_date,
            "A",
        ]
    )

    assert bool(
        signal_on.loc[
            second_decision_date,
            "B",
        ]
    )

    volatility = pd.Series({
        "A": 0.10,
        "B": 0.10,
    })

    base_weights = inverse_volatility_weights(
        volatility
    )

    assert base_weights["A"] == pytest.approx(
        0.50
    )

    assert base_weights["B"] == pytest.approx(
        0.50
    )

    first_target = base_weights.where(
        signal_on.loc[first_decision_date],
        other=0.0,
    )

    second_target = base_weights.where(
        signal_on.loc[second_decision_date],
        other=0.0,
    )

    target_weights = pd.DataFrame(
        [
            first_target,
            second_target,
        ],
        index=[
            first_decision_date,
            second_decision_date,
        ],
    )

    target_cash_weights = (
        1.0 - target_weights.sum(axis=1)
    )

    assert target_weights.loc[
        first_decision_date,
        "A",
    ] == pytest.approx(0.50)

    assert target_weights.loc[
        first_decision_date,
        "B",
    ] == pytest.approx(0.0)

    assert target_cash_weights.loc[
        first_decision_date
    ] == pytest.approx(0.50)

    assert target_weights.loc[
        second_decision_date,
        "A",
    ] == pytest.approx(0.0)

    assert target_weights.loc[
        second_decision_date,
        "B",
    ] == pytest.approx(0.50)

    assert target_cash_weights.loc[
        second_decision_date
    ] == pytest.approx(0.50)

    result = simulate_rebalanced_portfolio(
        asset_returns=asset_returns,
        target_weights=target_weights,
        cash_returns=cash_daily_return,
        target_cash_weights=target_cash_weights,
    )

    # Jan 5 decision executes at the Jan 6 close.
    # The portfolio is still in cash during the return
    # ending on Jan 6.
    assert result.index[0] == pd.Timestamp(
        "2026-01-06"
    )

    assert result.loc[
        pd.Timestamp("2026-01-06"),
        "portfolio_return",
    ] == pytest.approx(0.0)

    assert result.loc[
        pd.Timestamp("2026-01-06"),
        "gross_traded_notional",
    ] == pytest.approx(0.50)

    # The Jan 6 close target first affects the
    # Jan 6 -> Jan 7 return.
    #
    # 50% A * -10% + 50% cash * 0% = -5%.
    assert result.loc[
        pd.Timestamp("2026-01-07"),
        "portfolio_return",
    ] == pytest.approx(-0.05)

    # The Jan 7 decision executes at the Jan 8 close.
    # There is no market movement during the return
    # ending on Jan 8.
    assert result.loc[
        pd.Timestamp("2026-01-08"),
        "portfolio_return",
    ] == pytest.approx(0.0)

    # The second target first affects Jan 8 -> Jan 9.
    #
    # 50% B * +10% + 50% cash * 0% = +5%.
    assert result.loc[
        pd.Timestamp("2026-01-09"),
        "portfolio_return",
    ] == pytest.approx(0.05)

    # Manual wealth:
    #
    # 1.00 * 1.00 * 0.95 * 1.00 * 1.05 = 0.9975
    assert result.iloc[-1]["wealth"] == pytest.approx(
        0.9975
    )

    wealth_for_metrics = pd.concat([
        pd.Series(
            [1.0],
            index=[
                pd.Timestamp("2026-01-05"),
            ],
        ),
        result["wealth"],
    ])

    assert maximum_drawdown(
        wealth_for_metrics
    ) == pytest.approx(-0.05)

    zero_risk_free = pd.Series(
        0.0,
        index=result.index,
    )

    assert sharpe_ratio(
        result["portfolio_return"],
        zero_risk_free,
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )