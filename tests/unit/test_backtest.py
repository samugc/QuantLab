import pandas as pd
import pytest

from quantlab.simulation.backtest import (
    portfolio_return,
    simulate_rebalanced_portfolio,
    wealth_index,
)


def test_portfolio_return_without_cash():
    weights = pd.Series({
        "A": 0.60,
        "B": 0.40,
    })

    asset_returns = pd.Series({
        "A": 0.02,
        "B": -0.01,
    })

    result = portfolio_return(
        weights=weights,
        asset_returns=asset_returns,
    )

    assert result == pytest.approx(0.008)


def test_portfolio_return_with_cash():
    weights = pd.Series({
        "A": 0.40,
        "B": 0.30,
    })

    asset_returns = pd.Series({
        "A": 0.02,
        "B": -0.01,
    })

    result = portfolio_return(
        weights=weights,
        asset_returns=asset_returns,
        cash_weight=0.30,
        cash_return=0.001,
    )

    expected = (
        0.40 * 0.02
        + 0.30 * -0.01
        + 0.30 * 0.001
    )

    assert result == pytest.approx(expected)


def test_wealth_index():
    returns = pd.Series([
        0.008,
        -0.005,
        0.010,
    ])

    result = wealth_index(returns)

    assert result.iloc[0] == pytest.approx(1.008)
    assert result.iloc[1] == pytest.approx(1.00296)
    assert result.iloc[2] == pytest.approx(1.0129896)


def test_next_close_execution_and_weight_drift():
    dates = pd.to_datetime([
        "2026-01-30",
        "2026-02-02",
        "2026-02-03",
        "2026-02-04",
    ])

    asset_returns = pd.DataFrame(
        {
            "A": [
                0.00,
                0.50,
                0.10,
                0.00,
            ],
            "B": [
                0.00,
                0.50,
                0.00,
                0.10,
            ],
        },
        index=dates,
    )

    target_weights = pd.DataFrame(
        {
            "A": [0.50],
            "B": [0.50],
        },
        index=pd.to_datetime([
            "2026-01-30",
        ]),
    )

    result = simulate_rebalanced_portfolio(
        asset_returns=asset_returns,
        target_weights=target_weights,
    )

    assert pd.Timestamp(
        "2026-02-02"
    ) not in result.index

    assert result.loc[
        pd.Timestamp("2026-02-03"),
        "portfolio_return",
    ] == pytest.approx(0.05)

    expected_second_return = (
        (0.50 / 1.05) * 0.10
    )

    assert result.loc[
        pd.Timestamp("2026-02-04"),
        "portfolio_return",
    ] == pytest.approx(
        expected_second_return
    )

    assert result.iloc[-1]["wealth"] == pytest.approx(
        1.10
    )


def test_portfolio_return_rejects_mismatched_assets():
    weights = pd.Series({
        "A": 1.0,
    })

    asset_returns = pd.Series({
        "B": 0.10,
    })

    with pytest.raises(
        ValueError,
        match="asset labels must match",
    ):
        portfolio_return(
            weights=weights,
            asset_returns=asset_returns,
        )


def test_simulator_rejects_nan_target_weight():
    dates = pd.to_datetime([
        "2026-01-30",
        "2026-02-02",
        "2026-02-03",
    ])

    asset_returns = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.01],
            "B": [0.0, 0.0, 0.01],
        },
        index=dates,
    )

    target_weights = pd.DataFrame(
        {
            "A": [0.50],
            "B": [float("nan")],
        },
        index=pd.to_datetime([
            "2026-01-30",
        ]),
    )

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        simulate_rebalanced_portfolio(
            asset_returns=asset_returns,
            target_weights=target_weights,
        )


def test_turnover_uses_drifted_pretrade_weights():
    dates = pd.to_datetime([
        "2026-01-30",
        "2026-02-02",
        "2026-02-03",
        "2026-02-27",
        "2026-03-02",
        "2026-03-03",
    ])

    asset_returns = pd.DataFrame(
        {
            "A": [
                0.00,
                0.00,
                0.10,
                0.00,
                0.00,
                0.00,
            ],
            "B": [
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
            ],
        },
        index=dates,
    )

    target_weights = pd.DataFrame(
        {
            "A": [
                0.50,
                0.50,
            ],
            "B": [
                0.50,
                0.50,
            ],
        },
        index=pd.to_datetime([
            "2026-01-30",
            "2026-02-27",
        ]),
    )

    result = simulate_rebalanced_portfolio(
        asset_returns=asset_returns,
        target_weights=target_weights,
    )

    # Initial deployment from 100% cash:
    #
    # buy 50% A + buy 50% B = 100% gross notional
    assert result.loc[
        pd.Timestamp("2026-02-03"),
        "gross_traded_notional",
    ] == pytest.approx(1.0)

    # After A rises 10%:
    #
    # A = 0.55 / 1.05 = 52.38095%
    # B = 0.50 / 1.05 = 47.61905%
    #
    # Rebalancing to 50/50 requires:
    #
    # sell A 2.38095%
    # buy B  2.38095%
    #
    # gross traded notional = 4.76190%
    expected_turnover = (
        abs(0.50 - (0.55 / 1.05))
        + abs(0.50 - (0.50 / 1.05))
    )

    assert result.loc[
        pd.Timestamp("2026-03-03"),
        "gross_traded_notional",
    ] == pytest.approx(
        expected_turnover
    )


def test_transaction_costs_reduce_net_return_and_wealth():
    dates = pd.to_datetime([
        "2026-01-30",
        "2026-02-02",
        "2026-02-03",
        "2026-02-27",
        "2026-03-02",
        "2026-03-03",
    ])

    asset_returns = pd.DataFrame(
        {
            "A": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "B": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        },
        index=dates,
    )

    target_weights = pd.DataFrame(
        {
            "A": [
                0.60,
                0.50,
            ],
            "B": [
                0.40,
                0.50,
            ],
        },
        index=pd.to_datetime([
            "2026-01-30",
            "2026-02-27",
        ]),
    )

    result = simulate_rebalanced_portfolio(
        asset_returns=asset_returns,
        target_weights=target_weights,
        transaction_cost_bps=10.0,
    )

    # Initial deployment:
    #
    # 0 -> 60% A
    # 0 -> 40% B
    #
    # gross traded notional = 100%
    assert result.loc[
        pd.Timestamp("2026-02-03"),
        "gross_traded_notional",
    ] == pytest.approx(1.0)

    # 10 bps = 0.001
    assert result.loc[
        pd.Timestamp("2026-02-03"),
        "transaction_cost",
    ] == pytest.approx(0.001)

    # Market return is zero, so net return is exactly -cost.
    assert result.loc[
        pd.Timestamp("2026-02-03"),
        "gross_portfolio_return",
    ] == pytest.approx(0.0)

    assert result.loc[
        pd.Timestamp("2026-02-03"),
        "portfolio_return",
    ] == pytest.approx(-0.001)

    # Second rebalance:
    #
    # 60/40 -> 50/50
    #
    # sell 10% A + buy 10% B = 20%
    assert result.loc[
        pd.Timestamp("2026-03-03"),
        "gross_traded_notional",
    ] == pytest.approx(0.20)

    # 20% * 10 bps = 0.0002
    assert result.loc[
        pd.Timestamp("2026-03-03"),
        "transaction_cost",
    ] == pytest.approx(0.0002)

    expected_final_wealth = (
        (1.0 - 0.001)
        * (1.0 - 0.0002)
    )

    assert result.iloc[-1]["wealth"] == pytest.approx(
        expected_final_wealth
    )