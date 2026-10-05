from itertools import product
from pathlib import Path

import pandas as pd

from quantlab.evaluation.metrics import (
    annualized_volatility,
    cagr,
    maximum_drawdown,
    sharpe_ratio,
)
from quantlab.features.risk_free import (
    annual_yield_to_daily_return,
    cash_wealth_index,
    trailing_cash_return,
)
from quantlab.features.volatility import (
    rolling_annualized_volatility,
)
from quantlab.portfolio.exposure import (
    exposure_matched_weights,
)
from quantlab.portfolio.inverse_volatility import (
    inverse_volatility_weights,
)
from quantlab.signals.momentum import momentum_return
from quantlab.simulation.backtest import (
    simulate_rebalanced_portfolio,
)
from quantlab.simulation.rebalancing import (
    monthly_rebalance_dates,
)

from run_h001_backtest import (
    TICKERS,
    load_market_data,
)


VOLATILITY_WINDOWS = [
    40,
    60,
    90,
]

MOMENTUM_LOOKBACKS = [
    189,
    252,
    315,
]

ORIGINAL_VOLATILITY_WINDOW = 60
ORIGINAL_MOMENTUM_LOOKBACK = 252

TRANSACTION_COST_BPS = 10.0

RESULTS_DIR = Path(
    "results/h001"
)


def build_cash_returns(
    market_data: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    """
    Build daily cash returns and the corresponding wealth index.
    """

    cash_daily_return = (
        annual_yield_to_daily_return(
            market_data[
                "annual_yield_percent"
            ].shift(1)
        )
        .fillna(0.0)
    )

    cash_wealth = cash_wealth_index(
        cash_daily_return
    )

    return (
        cash_daily_return,
        cash_wealth,
    )


def build_parameterized_targets(
    market_data: pd.DataFrame,
    returns: pd.DataFrame,
    cash_wealth: pd.Series,
    volatility_window: int,
    momentum_lookback: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    """
    Build baseline and H001 targets for one parameter combination.
    """

    volatility = pd.DataFrame(
        index=market_data.index
    )

    momentum = pd.DataFrame(
        index=market_data.index
    )

    for ticker in TICKERS:
        volatility[ticker] = (
            rolling_annualized_volatility(
                returns[ticker],
                window=volatility_window,
            )
        )

        momentum[ticker] = (
            momentum_return(
                market_data[ticker],
                lookback=momentum_lookback,
            )
        )

    cash_trailing_return = (
        trailing_cash_return(
            cash_wealth,
            lookback=momentum_lookback,
        )
    )

    excess_momentum = (
        momentum.sub(
            cash_trailing_return,
            axis=0,
        )
    )

    signal_on = (
        excess_momentum > 0.0
    )

    rebalance_dates = (
        monthly_rebalance_dates(
            pd.Series(
                market_data.index
            )
        )
    )

    baseline_decisions = {}
    h001_decisions = {}
    h001_cash_decisions = {}

    for date in rebalance_dates:
        vol_today = (
            volatility.loc[
                date
            ]
        )

        excess_today = (
            excess_momentum.loc[
                date
            ]
        )

        signal_today = (
            signal_on.loc[
                date
            ]
        )

        if (
            vol_today.isna().any()
            or excess_today.isna().any()
        ):
            continue

        baseline_weights_today = (
            inverse_volatility_weights(
                vol_today
            )
        )

        h001_weights_today = (
            baseline_weights_today.where(
                signal_today,
                other=0.0,
            )
        )

        h001_cash_weight = float(
            1.0
            - h001_weights_today.sum()
        )

        baseline_decisions[
            date
        ] = baseline_weights_today

        h001_decisions[
            date
        ] = h001_weights_today

        h001_cash_decisions[
            date
        ] = h001_cash_weight

    baseline_weights = (
        pd.DataFrame.from_dict(
            baseline_decisions,
            orient="index",
        )
    )

    h001_weights = (
        pd.DataFrame.from_dict(
            h001_decisions,
            orient="index",
        )
    )

    h001_cash_weights = (
        pd.Series(
            h001_cash_decisions,
            name="cash",
        )
    )

    return (
        baseline_weights,
        h001_weights,
        h001_cash_weights,
    )


def simulate_parameter_combination(
    market_data: pd.DataFrame,
    returns: pd.DataFrame,
    cash_daily_return: pd.Series,
    cash_wealth: pd.Series,
    volatility_window: int,
    momentum_lookback: int,
) -> dict[str, pd.DataFrame]:
    """
    Run H001 and its exposure-matched control for one parameter set.
    """

    (
        baseline_weights,
        h001_weights,
        h001_cash_weights,
    ) = build_parameterized_targets(
        market_data=market_data,
        returns=returns,
        cash_wealth=cash_wealth,
        volatility_window=volatility_window,
        momentum_lookback=momentum_lookback,
    )

    (
        matched_weights,
        matched_cash_weights,
    ) = exposure_matched_weights(
        baseline_weights=baseline_weights,
        reference_weights=h001_weights,
    )

    h001_simulation = (
        simulate_rebalanced_portfolio(
            asset_returns=returns,
            target_weights=h001_weights,
            cash_returns=cash_daily_return,
            target_cash_weights=h001_cash_weights,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )
    )

    matched_simulation = (
        simulate_rebalanced_portfolio(
            asset_returns=returns,
            target_weights=matched_weights,
            cash_returns=cash_daily_return,
            target_cash_weights=matched_cash_weights,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )
    )

    return {
        "H001":
            h001_simulation,

        "ExposureMatched":
            matched_simulation,
    }


def evaluate_common_period(
    simulation: pd.DataFrame,
    cash_daily_return: pd.Series,
    market_index: pd.Index,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, float]:
    """
    Evaluate one simulation on the shared comparison period.

    Wealth is rebased to 1 so every parameter combination is evaluated
    on exactly the same sample.
    """

    period_returns = (
        simulation.loc[
            start_date:end_date,
            "portfolio_return",
        ]
    )

    if len(
        period_returns
    ) < 2:
        raise ValueError(
            "Not enough observations "
            "in common evaluation period."
        )

    period_wealth = (
        1.0
        + period_returns
    ).cumprod()

    first_date = (
        period_returns.index[0]
    )

    first_position = (
        market_index.get_loc(
            first_date
        )
    )

    if first_position == 0:
        raise ValueError(
            "Cannot infer initial wealth date."
        )

    initial_date = (
        market_index[
            first_position - 1
        ]
    )

    wealth_for_metrics = (
        pd.concat([
            pd.Series(
                [1.0],
                index=[
                    initial_date
                ],
            ),
            period_wealth,
        ])
    )

    aligned_cash_returns = (
        cash_daily_return.loc[
            period_returns.index
        ]
    )

    period_turnover = float(
        simulation.loc[
            period_returns.index,
            "gross_traded_notional",
        ].sum()
    )

    return {
        "final_wealth":
            float(
                period_wealth.iloc[-1]
            ),

        "cagr":
            cagr(
                wealth_for_metrics
            ),

        "volatility":
            annualized_volatility(
                period_returns
            ),

        "sharpe":
            sharpe_ratio(
                period_returns,
                aligned_cash_returns,
            ),

        "max_drawdown":
            maximum_drawdown(
                wealth_for_metrics
            ),

        "gross_traded_notional":
            period_turnover,

        "observations":
            len(
                period_returns
            ),
    }


def main() -> None:
    market_data = (
        load_market_data()
    )

    returns = (
        market_data[
            TICKERS
        ].pct_change()
    )

    (
        cash_daily_return,
        cash_wealth,
    ) = build_cash_returns(
        market_data
    )

    simulations = {}

    parameter_grid = list(
        product(
            VOLATILITY_WINDOWS,
            MOMENTUM_LOOKBACKS,
        )
    )

    print(
        "\nH001 parameter sensitivity"
    )
    print("-" * 90)

    print(
        "This is a robustness analysis, "
        "not a parameter search."
    )

    print(
        f"Transaction cost assumption: "
        f"{TRANSACTION_COST_BPS:.0f} bps"
    )

    print(
        "Original specification: "
        f"vol={ORIGINAL_VOLATILITY_WINDOW}, "
        f"momentum={ORIGINAL_MOMENTUM_LOOKBACK}"
    )

    print(
        f"Parameter combinations: "
        f"{len(parameter_grid)}"
    )

    # Run every parameter combination

    for (
        volatility_window,
        momentum_lookback,
    ) in parameter_grid:
        print(
            f"Running "
            f"vol={volatility_window:>3}, "
            f"momentum={momentum_lookback:>3}"
        )

        simulations[
            (
                volatility_window,
                momentum_lookback,
            )
        ] = (
            simulate_parameter_combination(
                market_data=market_data,
                returns=returns,
                cash_daily_return=(
                    cash_daily_return
                ),
                cash_wealth=cash_wealth,
                volatility_window=(
                    volatility_window
                ),
                momentum_lookback=(
                    momentum_lookback
                ),
            )
        )

    # Determine one common sample for all combinations

    all_simulations = [
        simulation
        for strategy_simulations
        in simulations.values()
        for simulation
        in strategy_simulations.values()
    ]

    common_start = max(
        simulation.index.min()
        for simulation
        in all_simulations
    )

    common_end = min(
        simulation.index.max()
        for simulation
        in all_simulations
    )

    print(
        "\nCommon evaluation sample"
    )
    print("-" * 90)

    print(
        f"Start: "
        f"{common_start.date()}"
    )

    print(
        f"End  : "
        f"{common_end.date()}"
    )

    # Validate identical dates

    reference_index = (
        all_simulations[0]
        .loc[
            common_start:
            common_end
        ]
        .index
    )

    for simulation in all_simulations[
        1:
    ]:
        comparison_index = (
            simulation
            .loc[
                common_start:
                common_end
            ]
            .index
        )

        if not (
            comparison_index.equals(
                reference_index
            )
        ):
            raise RuntimeError(
                "Sensitivity simulations "
                "do not share the same "
                "evaluation dates."
            )

    print(
        f"Trading sessions: "
        f"{len(reference_index)}"
    )

    # Evaluate every combination

    records = []

    for (
        volatility_window,
        momentum_lookback,
    ) in parameter_grid:
        strategy_simulations = (
            simulations[
                (
                    volatility_window,
                    momentum_lookback,
                )
            ]
        )

        h001_metrics = (
            evaluate_common_period(
                simulation=(
                    strategy_simulations[
                        "H001"
                    ]
                ),
                cash_daily_return=(
                    cash_daily_return
                ),
                market_index=(
                    market_data.index
                ),
                start_date=common_start,
                end_date=common_end,
            )
        )

        matched_metrics = (
            evaluate_common_period(
                simulation=(
                    strategy_simulations[
                        "ExposureMatched"
                    ]
                ),
                cash_daily_return=(
                    cash_daily_return
                ),
                market_index=(
                    market_data.index
                ),
                start_date=common_start,
                end_date=common_end,
            )
        )

        is_original = (
            volatility_window
            == ORIGINAL_VOLATILITY_WINDOW
            and momentum_lookback
            == ORIGINAL_MOMENTUM_LOOKBACK
        )

        records.append({
            "volatility_window":
                volatility_window,

            "momentum_lookback":
                momentum_lookback,

            "original":
                is_original,

            "h001_final_wealth":
                h001_metrics[
                    "final_wealth"
                ],

            "matched_final_wealth":
                matched_metrics[
                    "final_wealth"
                ],

            "h001_cagr":
                h001_metrics[
                    "cagr"
                ],

            "matched_cagr":
                matched_metrics[
                    "cagr"
                ],

            "cagr_difference":
                (
                    h001_metrics[
                        "cagr"
                    ]
                    - matched_metrics[
                        "cagr"
                    ]
                ),

            "h001_volatility":
                h001_metrics[
                    "volatility"
                ],

            "matched_volatility":
                matched_metrics[
                    "volatility"
                ],

            "h001_sharpe":
                h001_metrics[
                    "sharpe"
                ],

            "matched_sharpe":
                matched_metrics[
                    "sharpe"
                ],

            "sharpe_difference":
                (
                    h001_metrics[
                        "sharpe"
                    ]
                    - matched_metrics[
                        "sharpe"
                    ]
                ),

            "h001_max_drawdown":
                h001_metrics[
                    "max_drawdown"
                ],

            "matched_max_drawdown":
                matched_metrics[
                    "max_drawdown"
                ],

            "max_drawdown_difference":
                (
                    h001_metrics[
                        "max_drawdown"
                    ]
                    - matched_metrics[
                        "max_drawdown"
                    ]
                ),

            "h001_turnover":
                h001_metrics[
                    "gross_traded_notional"
                ],

            "matched_turnover":
                matched_metrics[
                    "gross_traded_notional"
                ],
        })

    results = pd.DataFrame(
        records
    )

    # Save reproducible sensitivity results

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "parameter_sensitivity_10bps.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    # Display robustness table

    display = pd.DataFrame({
        "Vol":
            results[
                "volatility_window"
            ],

        "Mom":
            results[
                "momentum_lookback"
            ],

        "Original":
            results[
                "original"
            ],

        "H001 CAGR %":
            results[
                "h001_cagr"
            ] * 100,

        "Matched CAGR %":
            results[
                "matched_cagr"
            ] * 100,

        "Delta CAGR pp":
            results[
                "cagr_difference"
            ] * 100,

        "H001 Sharpe":
            results[
                "h001_sharpe"
            ],

        "Matched Sharpe":
            results[
                "matched_sharpe"
            ],

        "Delta Sharpe":
            results[
                "sharpe_difference"
            ],

        "H001 Max DD %":
            results[
                "h001_max_drawdown"
            ] * 100,

        "Matched Max DD %":
            results[
                "matched_max_drawdown"
            ] * 100,

        "Delta Max DD pp":
            results[
                "max_drawdown_difference"
            ] * 100,
    })

    print(
        "\nSensitivity results"
    )
    print("-" * 150)

    print(
        display.to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.3f}"
            ),
        )
    )

    # Compact attribution summary

    attribution_summary = (
        pd.DataFrame({
            "Vol":
                results[
                    "volatility_window"
                ],

            "Mom":
                results[
                    "momentum_lookback"
                ],

            "Original":
                results[
                    "original"
                ],

            "CAGR delta pp":
                results[
                    "cagr_difference"
                ] * 100,

            "Sharpe delta":
                results[
                    "sharpe_difference"
                ],

            "Max DD delta pp":
                results[
                    "max_drawdown_difference"
                ] * 100,
        })
    )

    print(
        "\nH001 minus ExposureMatched"
    )
    print("-" * 100)

    print(
        attribution_summary.to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.3f}"
            ),
        )
    )

    # Robustness counts

    cagr_wins = int(
        (
            results[
                "cagr_difference"
            ] > 0.0
        ).sum()
    )

    sharpe_wins = int(
        (
            results[
                "sharpe_difference"
            ] > 0.0
        ).sum()
    )

    drawdown_wins = int(
        (
            results[
                "max_drawdown_difference"
            ] > 0.0
        ).sum()
    )

    joint_wins = int(
        (
            (
                results[
                    "cagr_difference"
                ] > 0.0
            )
            & (
                results[
                    "sharpe_difference"
                ] > 0.0
            )
            & (
                results[
                    "max_drawdown_difference"
                ] > 0.0
            )
        ).sum()
    )

    print(
        "\nRobustness counts"
    )
    print("-" * 90)

    print(
        "H001 CAGR > matched CAGR       : "
        f"{cagr_wins}/{len(results)}"
    )

    print(
        "H001 Sharpe > matched Sharpe   : "
        f"{sharpe_wins}/{len(results)}"
    )

    print(
        "H001 Max DD better than matched: "
        f"{drawdown_wins}/{len(results)}"
    )

    print(
        "H001 better on all three       : "
        f"{joint_wins}/{len(results)}"
    )

    print(
        "\nSaved sensitivity results"
    )
    print("-" * 90)

    print(
        output_path
    )

    print()

    print(
        "The original 60/252 specification "
        "remains fixed regardless of which "
        "row performs best."
    )


if __name__ == "__main__":
    main()
