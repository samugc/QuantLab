import numpy as np
import pandas as pd


def inverse_volatility_weights(
    volatility: pd.Series,
) -> pd.Series:
    """Compute normalized inverse-volatility portfolio weights."""

    if volatility.empty:
        raise ValueError(
            "Volatility series must not be empty."
        )

    if volatility.index.has_duplicates:
        raise ValueError(
            "Volatility asset labels must be unique."
        )

    if volatility.isna().any():
        raise ValueError(
            "Volatility contains missing values."
        )

    if not np.isfinite(volatility.to_numpy()).all():
        raise ValueError(
            "Volatility must contain only finite values."
        )

    if (volatility <= 0.0).any():
        raise ValueError(
            "Volatility must be strictly positive."
        )

    inverse_volatility = 1.0 / volatility

    weights = (
        inverse_volatility
        / inverse_volatility.sum()
    )

    return weights