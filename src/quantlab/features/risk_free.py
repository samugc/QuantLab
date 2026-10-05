import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def annual_yield_to_daily_return(
    annual_yield_percent: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Convert an annualized yield in percent to an approximate daily return."""
    annual_yield_decimal = annual_yield_percent / 100.0

    return annual_yield_decimal / periods_per_year


def cash_wealth_index(
    daily_returns: pd.Series,
) -> pd.Series:
    """Compute cumulative cash wealth from daily returns."""
    return (1.0 + daily_returns).cumprod()


def trailing_cash_return(
    wealth_index: pd.Series,
    lookback: int,
) -> pd.Series:
    """Compute trailing cash return over a fixed lookback."""
    return wealth_index / wealth_index.shift(lookback) - 1.0