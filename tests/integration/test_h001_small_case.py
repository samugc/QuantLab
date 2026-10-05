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

    # ---------------------------------------------------------
    # 1. Returns
    # ---------------------------------------------------------

    asset_returns = pd.DataFrame(
        {
            ticker: simple_returns(prices[ticker])
            for ticker in prices.columns
        }
    )

    # ---------------------------------------------------------
    # 2. Momentum
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 3. Cash
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 4. Inverse-volatility base allocation
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 5. Apply H001 momentum gate
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 6. Simulate with next-close execution
    # ---------------------------------------------------------

    result = simulate_rebalanced_portfolio(
        asset_returns=asset_returns,
        target_weights=target_weights,
        cash_returns=cash_daily_return,
        target_cash_weights=target_cash_weights,
    )

    # Decision on Jan 5 is executed at Jan 6 close.
    # Therefore its first realized return is Jan 7.
    assert result.index[0] == pd.Timestamp(
        "2026-01-07"
    )

    # 50% A * -10% + 50% cash * 0% = -5%
    assert result.loc[
        pd.Timestamp("2026-01-07"),
        "portfolio_return",
    ] == pytest.approx(-0.05)

    # No movement on Jan 8.
    assert result.loc[
        pd.Timestamp("2026-01-08"),
        "portfolio_return",
    ] == pytest.approx(0.0)

    # Second target was calculated Jan 7,
    # executed Jan 8 close, and first applies Jan 9.
    #
    # 50% B * +10% + 50% cash = +5%
    assert result.loc[
        pd.Timestamp("2026-01-09"),
        "portfolio_return",
    ] == pytest.approx(0.05)

    # Manual wealth:
    #
    # 1.00 * 0.95 * 1.00 * 1.05 = 0.9975
    assert result.iloc[-1]["wealth"] == pytest.approx(
        0.9975
    )

    # ---------------------------------------------------------
    # 7. Evaluation
    # ---------------------------------------------------------

    wealth_for_metrics = pd.concat([
        pd.Series(
            [1.0],
            index=[
                pd.Timestamp("2026-01-06"),
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