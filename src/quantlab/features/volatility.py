import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute annualized volatility from periodic returns."""
    periodic_volatility = float(returns.std())

    return periodic_volatility * np.sqrt(periods_per_year)


def rolling_annualized_volatility(
    returns: pd.Series,
    window: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Compute rolling annualized volatility."""
    rolling_volatility = returns.rolling(window=window).std()

    return rolling_volatility * np.sqrt(periods_per_year)