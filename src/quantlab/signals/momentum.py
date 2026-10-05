import pandas as pd


def momentum_return(
    prices: pd.Series,
    lookback: int,
) -> pd.Series:
    """Compute trailing momentum return over a fixed lookback."""
    return prices / prices.shift(lookback) - 1.0