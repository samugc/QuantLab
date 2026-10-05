import pandas as pd
import pytest

from quantlab.signals.momentum import momentum_return


def test_momentum_return():
    prices = pd.Series([
        100.0,
        105.0,
        110.0,
        120.0,
        115.0,
    ])

    result = momentum_return(
        prices,
        lookback=3,
    )

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])

    assert result.iloc[3] == pytest.approx(0.20)
    assert result.iloc[4] == pytest.approx(115.0 / 105.0 - 1.0)