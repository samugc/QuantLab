import pandas as pd


def monthly_rebalance_dates(
    dates: pd.Series,
) -> pd.Series:
    """Return the final observed trading date of each calendar month."""

    periods = dates.dt.to_period("M")

    rebalance_dates = dates.groupby(periods).max()

    return rebalance_dates.reset_index(drop=True)