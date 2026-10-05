import pandas as pd

from quantlab.simulation.rebalancing import monthly_rebalance_dates


def test_monthly_rebalance_dates():
    dates = pd.Series(
        pd.to_datetime([
            "2026-01-28",
            "2026-01-29",
            "2026-01-30",
            "2026-02-02",
            "2026-02-26",
            "2026-02-27",
            "2026-03-30",
            "2026-03-31",
        ])
    )

    result = monthly_rebalance_dates(dates)

    expected = pd.Series(
        pd.to_datetime([
            "2026-01-30",
            "2026-02-27",
            "2026-03-31",
        ])
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )