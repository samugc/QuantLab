import os

import pandas as pd
import requests
from dotenv import load_dotenv


TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"


def fetch_daily_prices(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily market data from Tiingo."""

    load_dotenv()

    api_key = os.getenv("TIINGO_API_KEY")

    if api_key is None:
        raise RuntimeError("TIINGO_API_KEY no encontrada")

    url = f"{TIINGO_BASE_URL}/{ticker}/prices"

    params = {
        "startDate": start_date,
        "endDate": end_date,
    }

    headers = {
        "Authorization": f"Token {api_key}",
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        raise ValueError(f"No se recibieron datos para {ticker}")

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df