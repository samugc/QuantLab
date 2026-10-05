import numpy as np
import pandas as pd


NUMERICAL_TOLERANCE = 1e-12


def exposure_matched_weights(
    baseline_weights: pd.DataFrame,
    reference_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Match the aggregate risky exposure of a reference portfolio.

    The baseline portfolio keeps its relative allocation across risky
    assets while its total risky exposure is scaled to match the
    reference portfolio.

    Residual exposure is allocated to cash.
    """

    if not baseline_weights.index.equals(
        reference_weights.index
    ):
        raise ValueError(
            "Baseline and reference target dates must match."
        )

    if not baseline_weights.columns.equals(
        reference_weights.columns
    ):
        raise ValueError(
            "Baseline and reference assets must match."
        )

    if baseline_weights.isna().any().any():
        raise ValueError(
            "Baseline weights contain missing values."
        )

    if reference_weights.isna().any().any():
        raise ValueError(
            "Reference weights contain missing values."
        )

    if not np.isfinite(
        baseline_weights.to_numpy()
    ).all():
        raise ValueError(
            "Baseline weights must be finite."
        )

    if not np.isfinite(
        reference_weights.to_numpy()
    ).all():
        raise ValueError(
            "Reference weights must be finite."
        )

    baseline_exposure = (
        baseline_weights.sum(axis=1)
    )

    if not np.allclose(
        baseline_exposure.to_numpy(),
        1.0,
        atol=NUMERICAL_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError(
            "Baseline risky weights must sum to 1."
        )

    reference_exposure = (
        reference_weights.sum(axis=1)
    )

    materially_below_zero = (
        reference_exposure
        < -NUMERICAL_TOLERANCE
    )

    materially_above_one = (
        reference_exposure
        > 1.0 + NUMERICAL_TOLERANCE
    )

    if (
        materially_below_zero.any()
        or materially_above_one.any()
    ):
        raise ValueError(
            "Reference risky exposure must be between 0 and 1."
        )

    # Remove harmless floating-point noise such as
    # 1.0000000000000002 or -1e-16.
    reference_exposure = (
        reference_exposure.clip(
            lower=0.0,
            upper=1.0,
        )
    )

    # Normalize rows to remove negligible floating-point drift while
    # preserving the baseline portfolio's relative allocations.
    normalized_baseline = (
        baseline_weights.div(
            baseline_exposure,
            axis=0,
        )
    )

    matched_weights = (
        normalized_baseline.mul(
            reference_exposure,
            axis=0,
        )
    )

    matched_cash_weights = (
        1.0 - reference_exposure
    )

    return (
        matched_weights,
        matched_cash_weights,
    )