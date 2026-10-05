import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute annualized volatility from periodic portfolio returns."""
    return float(returns.std() * np.sqrt(periods_per_year))


def cagr(
    wealth: pd.Series,
) -> float:
    """Compute compound annual growth rate from a wealth series."""
    if len(wealth) < 2:
        raise ValueError("At least two wealth observations are required.")

    start_date = wealth.index[0]
    end_date = wealth.index[-1]

    years = (end_date - start_date).days / 365.25

    if years <= 0:
        raise ValueError("Wealth index must span a positive time period.")

    initial_wealth = float(wealth.iloc[0])
    final_wealth = float(wealth.iloc[-1])

    return (final_wealth / initial_wealth) ** (1.0 / years) - 1.0


def sharpe_ratio(
    returns: pd.Series,
    risk_free_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute annualized Sharpe ratio using periodic risk-free returns."""
    excess_returns = returns - risk_free_returns

    excess_std = float(excess_returns.std())

    if excess_std == 0.0:
        raise ValueError("Excess return volatility must be greater than zero.")

    return (
        float(excess_returns.mean())
        / excess_std
        * np.sqrt(periods_per_year)
    )


def drawdown_series(
    wealth: pd.Series,
) -> pd.Series:
    """Compute drawdown relative to the previous wealth peak."""
    running_peak = wealth.cummax()

    return wealth / running_peak - 1.0


def maximum_drawdown(
    wealth: pd.Series,
) -> float:
    """Compute the maximum drawdown of a wealth series."""
    drawdowns = drawdown_series(wealth)

    return float(drawdowns.min())