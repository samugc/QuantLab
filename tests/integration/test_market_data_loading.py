from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "scripts"
)

sys.path.insert(
    0,
    str(SCRIPTS_DIR),
)

import run_h001_backtest as h001


def write_tiingo_snapshot(
    path: Path,
    dates: list[str],
) -> None:
    pd.DataFrame(
        {
            "date": dates,
            "adjClose": [
                100.0 + i
                for i in range(len(dates))
            ],
        }
    ).to_csv(
        path,
        index=False,
    )


def write_fred_snapshot(
    path: Path,
    dates: list[str],
) -> None:
    pd.DataFrame(
        {
            "date": dates,
            "value": [
                4.0
                for _ in dates
            ],
        }
    ).to_csv(
        path,
        index=False,
    )


def configure_test_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    tiingo_dir = (
        tmp_path
        / "tiingo"
    )

    fred_dir = (
        tmp_path
        / "fred"
    )

    tiingo_dir.mkdir()
    fred_dir.mkdir()

    fred_path = (
        fred_dir
        / "DGS3MO.csv"
    )

    monkeypatch.setattr(
        h001,
        "TIINGO_DIR",
        tiingo_dir,
    )

    monkeypatch.setattr(
        h001,
        "FRED_PATH",
        fred_path,
    )

    return (
        tiingo_dir,
        fred_path,
    )


def test_load_market_data_accepts_identical_etf_calendars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        tiingo_dir,
        fred_path,
    ) = configure_test_paths(
        tmp_path,
        monkeypatch,
    )

    dates = [
        "2026-09-28",
        "2026-09-29",
        "2026-09-30",
    ]

    for ticker in h001.TICKERS:
        write_tiingo_snapshot(
            tiingo_dir
            / f"{ticker}.csv",
            dates,
        )

    write_fred_snapshot(
        fred_path,
        dates,
    )

    result = (
        h001.load_market_data()
    )

    assert len(result) == 3

    assert result.index.equals(
        pd.to_datetime(
            dates
        )
    )

    assert not result[
        h001.TICKERS
    ].isna().any().any()

    assert result.index.max() == pd.Timestamp(
        "2026-09-30"
    )


def test_load_market_data_rejects_missing_etf_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        tiingo_dir,
        fred_path,
    ) = configure_test_paths(
        tmp_path,
        monkeypatch,
    )

    dates = [
        "2026-09-28",
        "2026-09-29",
        "2026-09-30",
    ]

    for ticker in h001.TICKERS:
        ticker_dates = dates

        if ticker == "EEM":
            ticker_dates = [
                "2026-09-28",
                "2026-09-30",
            ]

        write_tiingo_snapshot(
            tiingo_dir
            / f"{ticker}.csv",
            ticker_dates,
        )

    write_fred_snapshot(
        fred_path,
        dates,
    )

    with pytest.raises(
        ValueError,
        match="trading calendar differs",
    ):
        h001.load_market_data()


def test_load_market_data_rejects_incomplete_terminal_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        tiingo_dir,
        fred_path,
    ) = configure_test_paths(
        tmp_path,
        monkeypatch,
    )

    dates = [
        "2026-09-28",
        "2026-09-29",
    ]

    for ticker in h001.TICKERS:
        write_tiingo_snapshot(
            tiingo_dir
            / f"{ticker}.csv",
            dates,
        )

    write_fred_snapshot(
        fred_path,
        dates,
    )

    with pytest.raises(
        ValueError,
        match="must end on 2026-09-30",
    ):
        h001.load_market_data()