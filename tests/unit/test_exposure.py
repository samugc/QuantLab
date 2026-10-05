import pandas as pd
import pytest

from quantlab.portfolio.exposure import (
    exposure_matched_weights,
)


def test_exposure_matched_weights():
    dates = pd.to_datetime([
        "2026-01-30",
        "2026-02-27",
    ])

    baseline_weights = pd.DataFrame(
        {
            "A": [0.60, 0.40],
            "B": [0.40, 0.60],
        },
        index=dates,
    )

    reference_weights = pd.DataFrame(
        {
            "A": [0.60, 0.00],
            "B": [0.00, 0.30],
        },
        index=dates,
    )

    (
        matched_weights,
        cash_weights,
    ) = exposure_matched_weights(
        baseline_weights=baseline_weights,
        reference_weights=reference_weights,
    )

    assert matched_weights.loc[
        dates[0],
        "A",
    ] == pytest.approx(0.36)

    assert matched_weights.loc[
        dates[0],
        "B",
    ] == pytest.approx(0.24)

    assert cash_weights.loc[
        dates[0]
    ] == pytest.approx(0.40)

    assert matched_weights.loc[
        dates[1],
        "A",
    ] == pytest.approx(0.12)

    assert matched_weights.loc[
        dates[1],
        "B",
    ] == pytest.approx(0.18)

    assert cash_weights.loc[
        dates[1]
    ] == pytest.approx(0.70)

    matched_exposure = (
        matched_weights.sum(axis=1)
    )

    reference_exposure = (
        reference_weights.sum(axis=1)
    )

    assert matched_exposure.loc[
        dates[0]
    ] == pytest.approx(
        reference_exposure.loc[
            dates[0]
        ]
    )

    assert matched_exposure.loc[
        dates[1]
    ] == pytest.approx(
        reference_exposure.loc[
            dates[1]
        ]
    )


def test_exposure_matching_tolerates_floating_point_noise():
    dates = pd.to_datetime([
        "2026-01-30",
    ])

    baseline_weights = pd.DataFrame(
        {
            "A": [0.60],
            "B": [0.40],
        },
        index=dates,
    )

    # Mathematically intended to represent 100% exposure,
    # with a tiny floating-point excess.
    reference_weights = pd.DataFrame(
        {
            "A": [0.50 + 5e-13],
            "B": [0.50],
        },
        index=dates,
    )

    (
        matched_weights,
        cash_weights,
    ) = exposure_matched_weights(
        baseline_weights=baseline_weights,
        reference_weights=reference_weights,
    )

    assert matched_weights.sum(
        axis=1
    ).iloc[0] == pytest.approx(
        1.0
    )

    assert cash_weights.iloc[
        0
    ] == pytest.approx(
        0.0
    )


def test_exposure_matching_rejects_materially_invalid_exposure():
    dates = pd.to_datetime([
        "2026-01-30",
    ])

    baseline_weights = pd.DataFrame(
        {
            "A": [0.60],
            "B": [0.40],
        },
        index=dates,
    )

    reference_weights = pd.DataFrame(
        {
            "A": [0.80],
            "B": [0.40],
        },
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        exposure_matched_weights(
            baseline_weights=baseline_weights,
            reference_weights=reference_weights,
        )