import pandas as pd


FRED_CSV_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download a time series from FRED as CSV."""

    url = (
        f"{FRED_CSV_BASE_URL}"
        f"?id={series_id}"
        f"&cosd={start_date}"
        f"&coed={end_date}"
    )

    df = pd.read_csv(url)

    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = df.sort_values("date").reset_index(drop=True)

    return df