import numpy as np
import pandas as pd


def portfolio_return(
    weights: pd.Series,
    asset_returns: pd.Series,
    cash_weight: float = 0.0,
    cash_return: float = 0.0,
) -> float:
    """Compute portfolio return for one period."""

    if weights.index.has_duplicates:
        raise ValueError(
            "Weight asset labels must be unique."
        )

    if asset_returns.index.has_duplicates:
        raise ValueError(
            "Return asset labels must be unique."
        )

    if set(weights.index) != set(asset_returns.index):
        raise ValueError(
            "Weight and return asset labels must match."
        )

    if weights.isna().any():
        raise ValueError(
            "Weights contain missing values."
        )

    if asset_returns.isna().any():
        raise ValueError(
            "Asset returns contain missing values."
        )

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError(
            "Weights must contain only finite values."
        )

    if not np.isfinite(asset_returns.to_numpy()).all():
        raise ValueError(
            "Asset returns must contain only finite values."
        )

    if not np.isfinite(cash_weight):
        raise ValueError(
            "Cash weight must be finite."
        )

    if not np.isfinite(cash_return):
        raise ValueError(
            "Cash return must be finite."
        )

    aligned_returns = asset_returns.reindex(
        weights.index
    )

    risky_return = float(
        (weights * aligned_returns).sum()
    )

    return risky_return + cash_weight * cash_return


def wealth_index(
    returns: pd.Series,
    initial_wealth: float = 1.0,
) -> pd.Series:
    """Compute cumulative portfolio wealth from periodic returns."""
    return initial_wealth * (1.0 + returns).cumprod()


def simulate_rebalanced_portfolio(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cash_returns: pd.Series | None = None,
    target_cash_weights: pd.Series | None = None,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Simulate a portfolio with drifting weights between rebalances.

    A target computed using the close of date t is executed at the close
    of the next available trading session t+1.

    Holdings already in the portfolio earn the return ending on t+1.
    The rebalance and its transaction cost are then applied at the t+1
    close. The new target therefore first affects the return from t+1
    to t+2.

    Transaction costs are charged on gross traded ETF notional.
    Cash is treated as the residual funding asset and is not counted
    separately as traded notional.
    """

    if asset_returns.columns.has_duplicates:
        raise ValueError(
            "Asset return columns must be unique."
        )

    if target_weights.columns.has_duplicates:
        raise ValueError(
            "Target weight columns must be unique."
        )

    if set(target_weights.columns) != set(
        asset_returns.columns
    ):
        raise ValueError(
            "Target weight assets must match return assets."
        )

    if not asset_returns.index.is_monotonic_increasing:
        raise ValueError(
            "Asset return dates must be sorted."
        )

    if asset_returns.index.has_duplicates:
        raise ValueError(
            "Asset return dates must be unique."
        )

    if target_weights.index.has_duplicates:
        raise ValueError(
            "Target weight dates must be unique."
        )

    if target_weights.isna().any().any():
        raise ValueError(
            "Target weights contain missing values."
        )

    if not np.isfinite(
        target_weights.to_numpy()
    ).all():
        raise ValueError(
            "Target weights must contain only finite values."
        )

    if not np.isfinite(transaction_cost_bps):
        raise ValueError(
            "Transaction cost must be finite."
        )

    if transaction_cost_bps < 0.0:
        raise ValueError(
            "Transaction cost must be non-negative."
        )

    transaction_cost_rate = (
        transaction_cost_bps / 10_000.0
    )

    target_weights = target_weights[
        asset_returns.columns
    ].sort_index()

    unknown_target_dates = (
        target_weights.index.difference(
            asset_returns.index
        )
    )

    if not unknown_target_dates.empty:
        raise ValueError(
            "Target weight dates must exist in the return index."
        )

    if cash_returns is None:
        cash_returns = pd.Series(
            0.0,
            index=asset_returns.index,
        )
    else:
        cash_returns = cash_returns.reindex(
            asset_returns.index
        )

        if cash_returns.isna().any():
            raise ValueError(
                "Cash returns are missing for some simulation dates."
            )

        if not np.isfinite(
            cash_returns.to_numpy()
        ).all():
            raise ValueError(
                "Cash returns must contain only finite values."
            )

    if target_cash_weights is None:
        target_cash_weights = (
            1.0 - target_weights.sum(axis=1)
        )
    else:
        target_cash_weights = target_cash_weights.reindex(
            target_weights.index
        )

        if target_cash_weights.isna().any():
            raise ValueError(
                "Target cash weights contain missing values."
            )

        if not np.isfinite(
            target_cash_weights.to_numpy()
        ).all():
            raise ValueError(
                "Target cash weights must contain only finite values."
            )

    total_target_weight = (
        target_weights.sum(axis=1)
        + target_cash_weights
    )

    if not np.allclose(
        total_target_weight,
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Target asset weights plus cash weights must sum to 1."
        )

    current_asset_weights = pd.Series(
        0.0,
        index=asset_returns.columns,
    )

    current_cash_weight = 1.0
    current_wealth = 1.0
    active = False

    records = []
    dates = asset_returns.index

    for position, date in enumerate(dates):
        decision_date = None

        if position >= 1:
            decision_date = dates[position - 1]

        rebalance_today = (
            decision_date is not None
            and decision_date in target_weights.index
        )

        if not active and not rebalance_today:
            continue

        returns_today = asset_returns.loc[date]

        if returns_today.isna().any():
            raise ValueError(
                f"Missing asset return on {date}."
            )

        if not np.isfinite(
            returns_today.to_numpy()
        ).all():
            raise ValueError(
                f"Non-finite asset return on {date}."
            )

        cash_return_today = float(
            cash_returns.loc[date]
        )

        gross_portfolio_return = portfolio_return(
            weights=current_asset_weights,
            asset_returns=returns_today,
            cash_weight=current_cash_weight,
            cash_return=cash_return_today,
        )

        current_wealth *= (
            1.0 + gross_portfolio_return
        )

        # Holdings drift through the session before any trade
        # executed at today's close.
        asset_values = (
            current_asset_weights
            * (1.0 + returns_today)
        )

        cash_value = (
            current_cash_weight
            * (1.0 + cash_return_today)
        )

        total_value = (
            asset_values.sum()
            + cash_value
        )

        current_asset_weights = (
            asset_values / total_value
        )

        current_cash_weight = (
            cash_value / total_value
        )

        gross_traded_notional = 0.0
        transaction_cost = 0.0

        if rebalance_today:
            new_asset_weights = (
                target_weights
                .loc[decision_date]
                .astype(float)
                .copy()
            )

            new_cash_weight = float(
                target_cash_weights.loc[
                    decision_date
                ]
            )

            # Turnover is measured against the portfolio weights
            # immediately before trading at today's close.
            gross_traded_notional = float(
                (
                    new_asset_weights
                    - current_asset_weights
                )
                .abs()
                .sum()
            )

            transaction_cost = (
                gross_traded_notional
                * transaction_cost_rate
            )

            if transaction_cost >= 1.0:
                raise ValueError(
                    "Transaction cost would consume the entire portfolio."
                )

            current_wealth *= (
                1.0 - transaction_cost
            )

            current_asset_weights = (
                new_asset_weights
            )

            current_cash_weight = (
                new_cash_weight
            )

            active = True

        net_portfolio_return = (
            (1.0 + gross_portfolio_return)
            * (1.0 - transaction_cost)
            - 1.0
        )

        records.append({
            "date": date,
            "portfolio_return": net_portfolio_return,
            "gross_portfolio_return": gross_portfolio_return,
            "wealth": current_wealth,
            "cash_weight": current_cash_weight,
            "gross_traded_notional": gross_traded_notional,
            "transaction_cost": transaction_cost,
        })

    result = pd.DataFrame(records)

    if result.empty:
        return result

    return result.set_index("date")