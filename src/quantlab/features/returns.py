import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series) -> pd.Series:
    """Compute simple period-to-period returns."""
    return prices.pct_change()


def cumulative_return(returns: pd.Series) -> float:
    """Compute cumulative return from a series of simple returns."""
    growth_factor = float((1.0 + returns).prod())
    return growth_factor - 1.0


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute continuously compounded period-to-period returns."""
    return np.log(prices / prices.shift(1))