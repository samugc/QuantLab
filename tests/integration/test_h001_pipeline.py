from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "scripts"
)

sys.path.insert(
    0,
    str(SCRIPTS_DIR),
)

import run_h001_backtest as h001


def synthetic_market_data() -> pd.DataFrame:
    """Build deterministic market data for pipeline tests."""

    dates = pd.bdate_range(
        "2024-01-02",
        periods=340,
    )

    observations = np.arange(
        len(dates)
    )

    data = {}

    for position, ticker in enumerate(
        h001.TICKERS
    ):
        if position < 4:
            drift = 0.0005
        else:
            drift = -0.0003

        daily_returns = (
            drift
            + 0.002
            * np.sin(
                observations / 7.0
                + position
            )
        )

        data[ticker] = (
            100.0
            * np.cumprod(
                1.0 + daily_returns
            )
        )

    market_data = pd.DataFrame(
        data,
        index=dates,
    )

    market_data[
        "annual_yield_percent"
    ] = 4.0

    return market_data


def build_targets(
    market_data: pd.DataFrame,
):
    (
        returns,
        volatility,
        excess_momentum,
        signal_on,
        cash_daily_return,
    ) = h001.build_research_inputs(
        market_data
    )

    (
        baseline_weights,
        h001_weights,
        h001_cash_weights,
    ) = h001.build_monthly_targets(
        market_data=market_data,
        volatility=volatility,
        excess_momentum=excess_momentum,
        signal_on=signal_on,
    )

    return (
        returns,
        cash_daily_return,
        baseline_weights,
        h001_weights,
        h001_cash_weights,
    )


def test_actual_h001_pipeline_with_costs_cash_and_rebalances():
    market_data = (
        synthetic_market_data()
    )

    (
        returns,
        cash_daily_return,
        baseline_weights,
        h001_weights,
        h001_cash_weights,
    ) = build_targets(
        market_data
    )

    assert len(
        h001_weights
    ) >= 2

    assert (
        h001_cash_weights > 0.0
    ).any()

    assert np.allclose(
        baseline_weights.sum(
            axis=1
        ).to_numpy(),
        1.0,
        atol=1e-12,
    )

    (
        metrics,
        simulation,
    ) = h001.run_strategy(
        strategy_name="H001",
        cost_bps=10.0,
        returns=returns,
        target_weights=h001_weights,
        target_cash_weights=(
            h001_cash_weights
        ),
        cash_daily_return=(
            cash_daily_return
        ),
        market_index=market_data.index,
    )

    assert not simulation.empty

    assert (
        simulation[
            "transaction_cost"
        ] > 0.0
    ).any()

    assert (
        simulation[
            "gross_traded_notional"
        ] > 0.0
    ).sum() >= 2

    assert (
        cash_daily_return.loc[
            simulation.index
        ] > 0.0
    ).any()

    assert np.isfinite(
        simulation[
            "portfolio_return"
        ].to_numpy()
    ).all()

    assert np.isfinite(
        metrics[
            "final_wealth"
        ]
    )

    first_decision_date = (
        h001_weights.index[0]
    )

    decision_position = (
        market_data.index.get_loc(
            first_decision_date
        )
    )

    expected_execution_date = (
        market_data.index[
            decision_position + 1
        ]
    )

    assert (
        simulation.index[0]
        == expected_execution_date
    )


def test_future_data_changes_do_not_change_past_targets():
    market_data = (
        synthetic_market_data()
    )

    (
        _,
        _,
        baseline_weights,
        h001_weights,
        h001_cash_weights,
    ) = build_targets(
        market_data
    )

    assert len(
        h001_weights
    ) >= 3

    cutoff = (
        h001_weights.index[-2]
    )

    modified_market_data = (
        market_data.copy()
    )

    future_mask = (
        modified_market_data.index
        > cutoff
    )

    for position, ticker in enumerate(
        h001.TICKERS
    ):
        modified_market_data.loc[
            future_mask,
            ticker,
        ] *= (
            1.5
            + 0.1 * position
        )

    modified_market_data.loc[
        future_mask,
        "annual_yield_percent",
    ] = 15.0

    (
        _,
        _,
        modified_baseline_weights,
        modified_h001_weights,
        modified_h001_cash_weights,
    ) = build_targets(
        modified_market_data
    )

    pd.testing.assert_frame_equal(
        baseline_weights.loc[
            :cutoff
        ],
        modified_baseline_weights.loc[
            :cutoff
        ],
    )

    pd.testing.assert_frame_equal(
        h001_weights.loc[
            :cutoff
        ],
        modified_h001_weights.loc[
            :cutoff
        ],
    )

    pd.testing.assert_series_equal(
        h001_cash_weights.loc[
            :cutoff
        ],
        modified_h001_cash_weights.loc[
            :cutoff
        ],
    )
