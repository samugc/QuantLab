from pathlib import Path

import requests

from quantlab.data.fred import fetch_fred_series
from quantlab.data.tiingo import fetch_daily_prices


TICKERS = [
    "SPY",
    "IWM",
    "EFA",
    "EEM",
    "IEF",
    "LQD",
    "GLD",
    "DBC",
]

DATA_START_DATE = "2007-01-01"
END_DATE = "2026-09-30"

TIINGO_DIR = Path("data/raw/tiingo")
FRED_DIR = Path("data/raw/fred")

OVERWRITE = False


def download_tiingo_ticker(
    ticker: str,
) -> bool:
    """
    Download and save one Tiingo ticker.

    Returns True when the ticker is available locally.
    Returns False when the Tiingo rate limit is reached.
    """

    output_path = TIINGO_DIR / f"{ticker}.csv"

    if output_path.exists() and not OVERWRITE:
        print(f"{ticker}: already exists, skipping")
        return True

    print(f"{ticker}: downloading")

    try:
        df = fetch_daily_prices(
            ticker=ticker,
            start_date=DATA_START_DATE,
            end_date=END_DATE,
        )

    except requests.HTTPError as error:
        response = error.response

        if response is None:
            raise

        status_code = response.status_code

        if status_code != 429:
            raise

        retry_after = response.headers.get(
            "Retry-After"
        )

        rate_limit_reset = response.headers.get(
            "X-RateLimit-Reset"
        )

        rate_limit_remaining = response.headers.get(
            "X-RateLimit-Remaining"
        )

        print()
        print("Tiingo rate limit reached.")
        print("-" * 60)
        print(f"Ticker               : {ticker}")
        print(f"HTTP status          : {status_code}")
        print(f"Retry-After          : {retry_after}")
        print(f"X-RateLimit-Reset    : {rate_limit_reset}")
        print(f"X-RateLimit-Remaining: {rate_limit_remaining}")
        print()
        print(
            "Existing local snapshots have been preserved."
        )
        print(
            "Stop here and retry later after the rate limit resets."
        )

        return False

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"{ticker}: saved "
        f"{len(df)} rows -> {output_path}"
    )

    return True


def download_fred() -> None:
    """Download and save the FRED risk-free proxy."""

    output_path = FRED_DIR / "DGS3MO.csv"

    if output_path.exists() and not OVERWRITE:
        print("DGS3MO: already exists, skipping")
        return

    print("DGS3MO: downloading")

    df = fetch_fred_series(
        series_id="DGS3MO",
        start_date=DATA_START_DATE,
        end_date=END_DATE,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"DGS3MO: saved "
        f"{len(df)} rows -> {output_path}"
    )


def main() -> None:
    TIINGO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FRED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nDownloading H001 data snapshot")
    print("-" * 60)

    for ticker in TICKERS:
        success = download_tiingo_ticker(
            ticker
        )

        if not success:
            print()
            print("Download stopped because of Tiingo rate limiting.")
            return

    download_fred()

    print()
    print("H001 data snapshot complete.")


if __name__ == "__main__":
    main()