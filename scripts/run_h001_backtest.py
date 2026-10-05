from pathlib import Path

import numpy as np
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


TICKERS = [
    "SPY",
    "IWM",
    "EFA",
    "EEM",
    "IEF",
    "LQD",
    "GLD",
    "DBC",
]

TIINGO_DIR = Path("data/raw/tiingo")
FRED_PATH = Path("data/raw/fred/DGS3MO.csv")

DATA_START_DATE = pd.Timestamp("2007-01-01")
DATA_END_DATE = pd.Timestamp("2026-09-30")

VOLATILITY_WINDOW = 60
MOMENTUM_LOOKBACK = 252

TRANSACTION_COST_SCENARIOS_BPS = [
    0.0,
    5.0,
    10.0,
    20.0,
]

SUBPERIOD_COST_BPS = 10.0

SUBPERIODS = [
    (
        "2008-2012",
        "2008-02-04",
        "2012-12-31",
    ),
    (
        "2013-2017",
        "2013-01-01",
        "2017-12-31",
    ),
    (
        "2018-2022",
        "2018-01-01",
        "2022-12-31",
    ),
    (
        "2023-2026",
        "2023-01-01",
        "2026-09-30",
    ),
]


def load_market_data() -> pd.DataFrame:
    """Load and validate local ETF and risk-free snapshots."""

    price_series = {}

    reference_calendar = None
    reference_ticker = None

    for ticker in TICKERS:
        path = TIINGO_DIR / f"{ticker}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing local Tiingo snapshot: {path}"
            )

        df = pd.read_csv(path)

        required_columns = {
            "date",
            "adjClose",
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{ticker} snapshot is missing columns: "
                f"{sorted(missing_columns)}"
            )

        dates = pd.to_datetime(
            df["date"],
            utc=True,
            errors="raise",
        ).dt.tz_localize(None)

        if dates.duplicated().any():
            raise ValueError(
                f"{ticker} snapshot contains duplicate dates."
            )

        adjusted_close = pd.to_numeric(
            df["adjClose"],
            errors="coerce",
        )

        if adjusted_close.isna().any():
            raise ValueError(
                f"{ticker} snapshot contains missing "
                "or non-numeric adjusted closes."
            )

        if not np.isfinite(
            adjusted_close.to_numpy()
        ).all():
            raise ValueError(
                f"{ticker} snapshot contains non-finite "
                "adjusted closes."
            )

        if (
            adjusted_close <= 0.0
        ).any():
            raise ValueError(
                f"{ticker} snapshot contains non-positive "
                "adjusted closes."
            )

        series = pd.Series(
            adjusted_close.to_numpy(
                dtype=float
            ),
            index=dates,
            name=ticker,
        ).sort_index()

        if reference_calendar is None:
            reference_calendar = series.index
            reference_ticker = ticker

        elif not series.index.equals(
            reference_calendar
        ):
            missing_dates = (
                reference_calendar.difference(
                    series.index
                )
            )

            extra_dates = (
                series.index.difference(
                    reference_calendar
                )
            )

            raise ValueError(
                f"{ticker} trading calendar differs from "
                f"{reference_ticker}. "
                f"Missing dates: "
                f"{list(missing_dates[:5])}; "
                f"extra dates: "
                f"{list(extra_dates[:5])}."
            )

        price_series[ticker] = series

    prices = pd.DataFrame(
        price_series
    )

    if prices.empty:
        raise ValueError(
            "ETF price panel is empty."
        )

    if prices.isna().any().any():
        raise ValueError(
            "Missing ETF prices after calendar validation."
        )

    if prices.index.min() < DATA_START_DATE:
        raise ValueError(
            "ETF snapshots contain observations before "
            f"{DATA_START_DATE.date()}."
        )

    if prices.index.max() != DATA_END_DATE:
        raise ValueError(
            "ETF snapshots must end on "
            f"{DATA_END_DATE.date()}, but end on "
            f"{prices.index.max().date()}."
        )

    if not FRED_PATH.exists():
        raise FileNotFoundError(
            f"Missing local FRED snapshot: {FRED_PATH}"
        )

    risk_free = pd.read_csv(
        FRED_PATH
    )

    required_fred_columns = {
        "date",
        "value",
    }

    missing_columns = (
        required_fred_columns
        - set(risk_free.columns)
    )

    if missing_columns:
        raise ValueError(
            "FRED snapshot is missing columns: "
            f"{sorted(missing_columns)}"
        )

    risk_free["date"] = pd.to_datetime(
        risk_free["date"],
        errors="raise",
    )

    if risk_free["date"].duplicated().any():
        raise ValueError(
            "FRED snapshot contains duplicate dates."
        )

    risk_free["value"] = pd.to_numeric(
        risk_free["value"],
        errors="coerce",
    )

    risk_free = risk_free.rename(
        columns={
            "value": "annual_yield_percent",
        }
    )

    risk_free = (
        risk_free
        .set_index("date")
        .sort_index()
    )

    market_data = prices.join(
        risk_free,
        how="left",
    )

    market_data[
        "annual_yield_percent"
    ] = market_data[
        "annual_yield_percent"
    ].ffill()

    if market_data[
        TICKERS
    ].isna().any().any():
        raise ValueError(
            "Missing ETF prices after alignment."
        )

    if market_data[
        "annual_yield_percent"
    ].isna().any():
        raise ValueError(
            "Missing risk-free yields after alignment."
        )

    if not np.isfinite(
        market_data[
            "annual_yield_percent"
        ].to_numpy()
    ).all():
        raise ValueError(
            "Risk-free yields contain non-finite values."
        )

    return market_data


def build_research_inputs(
    market_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    """Build returns, volatility, momentum and cash series."""

    returns = market_data[
        TICKERS
    ].pct_change()

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
                window=VOLATILITY_WINDOW,
            )
        )

        momentum[ticker] = (
            momentum_return(
                market_data[ticker],
                lookback=MOMENTUM_LOOKBACK,
            )
        )

    # Today's approximate cash accrual uses the
    # previously known Treasury yield.
    cash_daily_return = (
        annual_yield_to_daily_return(
            market_data[
                "annual_yield_percent"
            ].shift(1)
        )
        .fillna(0.0)
    )

    cash_wealth = (
        cash_wealth_index(
            cash_daily_return
        )
    )

    cash_trailing_return = (
        trailing_cash_return(
            cash_wealth,
            lookback=MOMENTUM_LOOKBACK,
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

    return (
        returns,
        volatility,
        excess_momentum,
        signal_on,
        cash_daily_return,
    )


def build_monthly_targets(
    market_data: pd.DataFrame,
    volatility: pd.DataFrame,
    excess_momentum: pd.DataFrame,
    signal_on: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    """Build baseline and H001 monthly targets."""

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

        # Skip warm-up dates.
        if (
            vol_today.isna().any()
            or excess_today.isna().any()
        ):
            continue

        base_weights = (
            inverse_volatility_weights(
                vol_today
            )
        )

        h001_weights_today = (
            base_weights.where(
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
        ] = base_weights

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


def add_initial_wealth(
    wealth: pd.Series,
    market_index: pd.Index,
) -> pd.Series:
    """
    Prepend initial wealth = 1 before the first realized return.
    """

    first_date = (
        wealth.index[0]
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

    return pd.concat([
        pd.Series(
            [1.0],
            index=[initial_date],
        ),
        wealth,
    ])


def evaluate_simulation(
    simulation: pd.DataFrame,
    cash_daily_return: pd.Series,
    market_index: pd.Index,
) -> dict[str, float]:
    """Compute performance and trading statistics."""

    portfolio_returns = (
        simulation[
            "portfolio_return"
        ]
    )

    wealth = (
        simulation[
            "wealth"
        ]
    )

    wealth_for_metrics = (
        add_initial_wealth(
            wealth=wealth,
            market_index=market_index,
        )
    )

    aligned_cash_returns = (
        cash_daily_return.loc[
            portfolio_returns.index
        ]
    )

    return {
        "final_wealth":
            float(
                wealth.iloc[-1]
            ),

        "total_return":
            float(
                wealth.iloc[-1]
                - 1.0
            ),

        "cagr":
            cagr(
                wealth_for_metrics
            ),

        "volatility":
            annualized_volatility(
                portfolio_returns
            ),

        "sharpe":
            sharpe_ratio(
                portfolio_returns,
                aligned_cash_returns,
            ),

        "max_drawdown":
            maximum_drawdown(
                wealth_for_metrics
            ),

        "gross_traded_notional":
            float(
                simulation[
                    "gross_traded_notional"
                ].sum()
            ),
    }


def run_strategy(
    strategy_name: str,
    cost_bps: float,
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cash_daily_return: pd.Series,
    market_index: pd.Index,
    target_cash_weights: pd.Series | None = None,
) -> tuple[
    dict[str, float],
    pd.DataFrame,
]:
    """Run and evaluate one strategy."""

    simulation = (
        simulate_rebalanced_portfolio(
            asset_returns=returns,
            target_weights=target_weights,
            cash_returns=cash_daily_return,
            target_cash_weights=target_cash_weights,
            transaction_cost_bps=cost_bps,
        )
    )

    metrics = (
        evaluate_simulation(
            simulation=simulation,
            cash_daily_return=cash_daily_return,
            market_index=market_index,
        )
    )

    metrics[
        "strategy"
    ] = strategy_name

    metrics[
        "cost_bps"
    ] = cost_bps

    return (
        metrics,
        simulation,
    )


def evaluate_subperiods(
    simulations: dict[str, pd.DataFrame],
    cash_daily_return: pd.Series,
    market_index: pd.Index,
) -> pd.DataFrame:
    """
    Evaluate strategies over predefined calendar subperiods.

    Each subperiod is rebased to initial wealth = 1 so performance
    reflects only returns realized inside that subperiod.
    """

    records = []

    for (
        period_name,
        start_date,
        end_date,
    ) in SUBPERIODS:
        for (
            strategy_name,
            simulation,
        ) in simulations.items():
            period_returns = (
                simulation.loc[
                    start_date:end_date,
                    "portfolio_return",
                ]
            )

            if len(period_returns) < 2:
                raise ValueError(
                    f"Not enough observations for "
                    f"{strategy_name} in "
                    f"{period_name}."
                )

            period_wealth = (
                1.0
                + period_returns
            ).cumprod()

            wealth_for_metrics = (
                add_initial_wealth(
                    wealth=period_wealth,
                    market_index=market_index,
                )
            )

            period_cash_returns = (
                cash_daily_return.loc[
                    period_returns.index
                ]
            )

            records.append({
                "period":
                    period_name,

                "strategy":
                    strategy_name,

                "start":
                    period_returns.index[
                        0
                    ],

                "end":
                    period_returns.index[
                        -1
                    ],

                "observations":
                    len(period_returns),

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
                        period_cash_returns,
                    ),

                "max_drawdown":
                    maximum_drawdown(
                        wealth_for_metrics
                    ),
            })

    return pd.DataFrame(
        records
    )


def print_final_targets(
    baseline_weights: pd.DataFrame,
    matched_weights: pd.DataFrame,
    matched_cash_weights: pd.Series,
    h001_weights: pd.DataFrame,
    h001_cash_weights: pd.Series,
    signal_on: pd.DataFrame,
) -> None:
    """Display the final targets of all strategies."""

    final_date = (
        baseline_weights.index[-1]
    )

    decision = pd.DataFrame({
        "baseline":
            baseline_weights.loc[
                final_date
            ],

        "matched":
            matched_weights.loc[
                final_date
            ],

        "h001":
            h001_weights.loc[
                final_date
            ],

        "signal_on":
            signal_on.loc[
                final_date
            ],
    })

    print(
        "\nFinal monthly portfolio decision"
    )
    print("-" * 90)

    print(
        f"Decision date: "
        f"{final_date.date()}"
    )

    print(decision)

    print()

    print(
        "Baseline ETF exposure : "
        f"{baseline_weights.loc[final_date].sum():.2%}"
    )

    print(
        "Matched ETF exposure  : "
        f"{matched_weights.loc[final_date].sum():.2%}"
    )

    print(
        "Matched cash exposure : "
        f"{matched_cash_weights.loc[final_date]:.2%}"
    )

    print(
        "H001 ETF exposure     : "
        f"{h001_weights.loc[final_date].sum():.2%}"
    )

    print(
        "H001 cash exposure    : "
        f"{h001_cash_weights.loc[final_date]:.2%}"
    )


def main() -> None:
    market_data = (
        load_market_data()
    )

    (
        returns,
        volatility,
        excess_momentum,
        signal_on,
        cash_daily_return,
    ) = build_research_inputs(
        market_data
    )

    (
        baseline_weights,
        h001_weights,
        h001_cash_weights,
    ) = build_monthly_targets(
        market_data=market_data,
        volatility=volatility,
        excess_momentum=excess_momentum,
        signal_on=signal_on,
    )

    (
        matched_weights,
        matched_cash_weights,
    ) = exposure_matched_weights(
        baseline_weights=baseline_weights,
        reference_weights=h001_weights,
    )

    print("\nData source")
    print("-" * 90)

    print(
        "Tiingo: local CSV snapshot"
    )

    print(
        "FRED   : local CSV snapshot"
    )

    print(
        f"Rows   : "
        f"{len(market_data)}"
    )

    print(
        f"Period : "
        f"{market_data.index.min().date()} "
        f"-> "
        f"{market_data.index.max().date()}"
    )

    print("\nResearch design")
    print("-" * 90)

    print(
        "Baseline        : fully invested inverse-volatility"
    )

    print(
        "Exposure-match  : inverse-volatility scaled to "
        "H001 risky exposure"
    )

    print(
        "H001            : inverse-volatility + "
        "momentum ON/OFF gate"
    )

    print(
        "Signal          : computed using close(t)"
    )

    print(
        "Execution       : close of next observed session"
    )

    print(
        "Transaction cost: gross traded ETF notional"
    )

    print_final_targets(
        baseline_weights=baseline_weights,
        matched_weights=matched_weights,
        matched_cash_weights=matched_cash_weights,
        h001_weights=h001_weights,
        h001_cash_weights=h001_cash_weights,
        signal_on=signal_on,
    )

    scenario_records = []

    zero_cost_simulations = {}
    subperiod_simulations = {}

    for cost_bps in (
        TRANSACTION_COST_SCENARIOS_BPS
    ):
        strategies = [
            (
                "Baseline",
                baseline_weights,
                None,
            ),
            (
                "ExposureMatched",
                matched_weights,
                matched_cash_weights,
            ),
            (
                "H001",
                h001_weights,
                h001_cash_weights,
            ),
        ]

        for (
            strategy_name,
            target_weights,
            target_cash_weights,
        ) in strategies:
            (
                metrics,
                simulation,
            ) = run_strategy(
                strategy_name=strategy_name,
                cost_bps=cost_bps,
                returns=returns,
                target_weights=target_weights,
                target_cash_weights=target_cash_weights,
                cash_daily_return=cash_daily_return,
                market_index=market_data.index,
            )

            scenario_records.append(
                metrics
            )

            if cost_bps == 0.0:
                zero_cost_simulations[
                    strategy_name
                ] = simulation

            if (
                cost_bps
                == SUBPERIOD_COST_BPS
            ):
                subperiod_simulations[
                    strategy_name
                ] = simulation

    scenario_results = (
        pd.DataFrame(
            scenario_records
        )
    )

    reference_simulation = (
        zero_cost_simulations[
            "Baseline"
        ]
    )

    print("\nBacktest period")
    print("-" * 90)

    print(
        f"Start: "
        f"{reference_simulation.index.min().date()}"
    )

    print(
        f"End  : "
        f"{reference_simulation.index.max().date()}"
    )

    print(
        f"Trading sessions: "
        f"{len(reference_simulation)}"
    )

    # 0 bps attribution comparison

    zero_cost_results = (
        scenario_results[
            scenario_results[
                "cost_bps"
            ] == 0.0
        ]
        .set_index(
            "strategy"
        )
    )

    reference_table = (
        pd.DataFrame({
            "Final wealth":
                zero_cost_results[
                    "final_wealth"
                ],

            "Total return %":
                zero_cost_results[
                    "total_return"
                ] * 100,

            "CAGR %":
                zero_cost_results[
                    "cagr"
                ] * 100,

            "Volatility %":
                zero_cost_results[
                    "volatility"
                ] * 100,

            "Sharpe":
                zero_cost_results[
                    "sharpe"
                ],

            "Max DD %":
                zero_cost_results[
                    "max_drawdown"
                ] * 100,
        })
    )

    print(
        "\n0 bps attribution comparison"
    )
    print("-" * 90)

    print(
        reference_table.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # Transaction-cost sensitivity

    sensitivity = (
        scenario_results[
            [
                "cost_bps",
                "strategy",
                "final_wealth",
                "cagr",
                "volatility",
                "sharpe",
                "max_drawdown",
            ]
        ]
        .copy()
    )

    sensitivity[
        "cagr"
    ] *= 100

    sensitivity[
        "volatility"
    ] *= 100

    sensitivity[
        "max_drawdown"
    ] *= 100

    sensitivity = (
        sensitivity.rename(
            columns={
                "cost_bps":
                    "Cost bps",

                "strategy":
                    "Strategy",

                "final_wealth":
                    "Final wealth",

                "cagr":
                    "CAGR %",

                "volatility":
                    "Volatility %",

                "sharpe":
                    "Sharpe",

                "max_drawdown":
                    "Max DD %",
            }
        )
    )

    print(
        "\nTransaction-cost sensitivity"
    )
    print("-" * 110)

    print(
        sensitivity.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # Trading activity

    turnover = (
        zero_cost_results[
            [
                "gross_traded_notional",
            ]
        ]
        .rename(
            columns={
                "gross_traded_notional":
                    "Total gross traded notional"
            }
        )
    )

    print("\nTrading activity")
    print("-" * 90)

    print(
        turnover.to_string(
            float_format=lambda x: f"{x:.3f}x NAV"
        )
    )

    # Exposure validation

    exposure_check = (
        pd.DataFrame({
            "ExposureMatched":
                matched_weights.sum(
                    axis=1
                ),

            "H001":
                h001_weights.sum(
                    axis=1
                ),
        })
    )

    maximum_exposure_difference = float(
        (
            exposure_check[
                "ExposureMatched"
            ]
            - exposure_check[
                "H001"
            ]
        )
        .abs()
        .max()
    )

    print(
        "\nExposure control validation"
    )
    print("-" * 90)

    print(
        "Maximum target risky-exposure difference: "
        f"{maximum_exposure_difference:.12f}"
    )

    print(
        "Average H001 risky exposure               : "
        f"{exposure_check['H001'].mean():.2%}"
    )

    print(
        "Average ExposureMatched risky exposure    : "
        f"{exposure_check['ExposureMatched'].mean():.2%}"
    )

    # Subperiod analysis

    subperiod_results = (
        evaluate_subperiods(
            simulations=subperiod_simulations,
            cash_daily_return=cash_daily_return,
            market_index=market_data.index,
        )
    )

    display_subperiods = (
        subperiod_results[
            [
                "period",
                "strategy",
                "final_wealth",
                "cagr",
                "volatility",
                "sharpe",
                "max_drawdown",
            ]
        ]
        .copy()
    )

    display_subperiods[
        "cagr"
    ] *= 100

    display_subperiods[
        "volatility"
    ] *= 100

    display_subperiods[
        "max_drawdown"
    ] *= 100

    display_subperiods = (
        display_subperiods.rename(
            columns={
                "period":
                    "Period",

                "strategy":
                    "Strategy",

                "final_wealth":
                    "Final wealth",

                "cagr":
                    "CAGR %",

                "volatility":
                    "Volatility %",

                "sharpe":
                    "Sharpe",

                "max_drawdown":
                    "Max DD %",
            }
        )
    )

    print(
        f"\nSubperiod analysis "
        f"({SUBPERIOD_COST_BPS:.0f} bps)"
    )
    print("-" * 110)

    print(
        display_subperiods.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # H001 attribution relative to ExposureMatched

    attribution_source = (
        subperiod_results[
            subperiod_results[
                "strategy"
            ].isin([
                "ExposureMatched",
                "H001",
            ])
        ]
    )

    attribution = (
        attribution_source.pivot(
            index="period",
            columns="strategy",
            values=[
                "cagr",
                "sharpe",
                "max_drawdown",
            ],
        )
    )

    attribution_delta = (
        pd.DataFrame({
            "CAGR difference pp":
                (
                    attribution[
                        "cagr"
                    ][
                        "H001"
                    ]
                    - attribution[
                        "cagr"
                    ][
                        "ExposureMatched"
                    ]
                )
                * 100,

            "Sharpe difference":
                (
                    attribution[
                        "sharpe"
                    ][
                        "H001"
                    ]
                    - attribution[
                        "sharpe"
                    ][
                        "ExposureMatched"
                    ]
                ),

            "Max DD difference pp":
                (
                    attribution[
                        "max_drawdown"
                    ][
                        "H001"
                    ]
                    - attribution[
                        "max_drawdown"
                    ][
                        "ExposureMatched"
                    ]
                )
                * 100,
        })
    )

    print(
        "\nH001 minus ExposureMatched"
    )
    print("-" * 90)

    print(
        attribution_delta.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )


if __name__ == "__main__":
    main()
