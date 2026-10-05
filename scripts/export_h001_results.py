from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quantlab.evaluation.metrics import drawdown_series
from quantlab.portfolio.exposure import exposure_matched_weights

from run_h001_backtest import (
    SUBPERIOD_COST_BPS,
    TRANSACTION_COST_SCENARIOS_BPS,
    build_monthly_targets,
    build_research_inputs,
    evaluate_subperiods,
    load_market_data,
    run_strategy,
)


RESULTS_DIR = Path("results/h001")
FIGURES_DIR = Path("figures/h001")

PRIMARY_COST_BPS = 10.0


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    market_data = load_market_data()

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

    strategies = {
        "Baseline": (
            baseline_weights,
            None,
        ),
        "ExposureMatched": (
            matched_weights,
            matched_cash_weights,
        ),
        "H001": (
            h001_weights,
            h001_cash_weights,
        ),
    }

    # Transaction-cost sensitivity

    cost_records = []

    primary_simulations = {}

    for cost_bps in TRANSACTION_COST_SCENARIOS_BPS:
        for (
            strategy_name,
            (
                target_weights,
                target_cash_weights,
            ),
        ) in strategies.items():
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

            cost_records.append(
                metrics
            )

            if cost_bps == PRIMARY_COST_BPS:
                primary_simulations[
                    strategy_name
                ] = simulation

    cost_results = pd.DataFrame(
        cost_records
    )

    cost_results.to_csv(
        RESULTS_DIR
        / "transaction_cost_sensitivity.csv",
        index=False,
    )

    # Primary 10 bps summary

    primary_results = (
        cost_results[
            cost_results[
                "cost_bps"
            ] == PRIMARY_COST_BPS
        ]
        .copy()
        .set_index("strategy")
    )

    primary_results.to_csv(
        RESULTS_DIR
        / "summary_10bps.csv"
    )

    # Subperiod results

    subperiod_results = evaluate_subperiods(
        simulations=primary_simulations,
        cash_daily_return=cash_daily_return,
        market_index=market_data.index,
    )

    subperiod_results.to_csv(
        RESULTS_DIR
        / "subperiods_10bps.csv",
        index=False,
    )

    # Target risky exposure

    exposure = pd.DataFrame({
        "Baseline":
            baseline_weights.sum(
                axis=1
            ),

        "ExposureMatched":
            matched_weights.sum(
                axis=1
            ),

        "H001":
            h001_weights.sum(
                axis=1
            ),
    })

    exposure.index.name = "date"

    exposure.to_csv(
        RESULTS_DIR
        / "target_risky_exposure.csv"
    )

    # Wealth series

    wealth = pd.DataFrame({
        strategy_name:
            simulation[
                "wealth"
            ]
        for (
            strategy_name,
            simulation,
        ) in primary_simulations.items()
    })

    wealth.index.name = "date"

    wealth.to_csv(
        RESULTS_DIR
        / "wealth_10bps.csv"
    )

    # Drawdown series

    drawdowns = pd.DataFrame({
        strategy_name:
            drawdown_series(
                simulation[
                    "wealth"
                ]
            )
        for (
            strategy_name,
            simulation,
        ) in primary_simulations.items()
    })

    drawdowns.index.name = "date"

    drawdowns.to_csv(
        RESULTS_DIR
        / "drawdowns_10bps.csv"
    )

    # Wealth figure

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for column in wealth.columns:
        ax.plot(
            wealth.index,
            wealth[column],
            label=column,
        )

    ax.set_title(
        "H001 Portfolio Wealth — 10 bps Transaction Costs"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Wealth Index"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "wealth_10bps.png",
        dpi=180,
    )

    plt.close(fig)

    # Drawdown figure

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for column in drawdowns.columns:
        ax.plot(
            drawdowns.index,
            drawdowns[column] * 100.0,
            label=column,
        )

    ax.set_title(
        "H001 Drawdowns — 10 bps Transaction Costs"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Drawdown (%)"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "drawdowns_10bps.png",
        dpi=180,
    )

    plt.close(fig)

    # Risky-exposure figure

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        exposure.index,
        exposure["H001"] * 100.0,
        label="H001 / ExposureMatched",
    )

    ax.axhline(
        100.0,
        linestyle="--",
        linewidth=1.0,
        label="Baseline",
    )

    ax.set_title(
        "H001 Target Risky Exposure"
    )

    ax.set_xlabel(
        "Decision Date"
    )

    ax.set_ylabel(
        "ETF Exposure (%)"
    )

    ax.set_ylim(
        0.0,
        105.0,
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "target_risky_exposure.png",
        dpi=180,
    )

    plt.close(fig)

    # Console confirmation

    print("\nH001 artifacts exported")
    print("-" * 80)

    print(
        f"Results directory: "
        f"{RESULTS_DIR}"
    )

    print(
        f"Figures directory: "
        f"{FIGURES_DIR}"
    )

    print(
        "\nPrimary results "
        f"({PRIMARY_COST_BPS:.0f} bps)"
    )

    display = pd.DataFrame({
        "CAGR %":
            primary_results[
                "cagr"
            ] * 100,

        "Volatility %":
            primary_results[
                "volatility"
            ] * 100,

        "Sharpe":
            primary_results[
                "sharpe"
            ],

        "Max DD %":
            primary_results[
                "max_drawdown"
            ] * 100,

        "Final wealth":
            primary_results[
                "final_wealth"
            ],
    })

    print(
        display.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )


if __name__ == "__main__":
    main()
