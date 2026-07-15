import os
import pandas as pd
import logging
from fredapi import Fred

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FRED_API_KEY = "a0b2b4d6fbefab1d61711db07705f1e4"
EXTERNAL_DIR = "data/external"
SERIES = {
    "unrate": "UNRATE",
    "fed_funds": "FEDFUNDS",
    "cpi": "CPIAUCSL",
}

def fetch_fred_indicators():
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    fred = Fred(api_key=FRED_API_KEY)

    df = None
    for name, series_id in SERIES.items():
        logger.info(f"Fetching {name} ({series_id})...")
        series = fred.get_series(series_id)
        s = series.reset_index()
        s.columns = ["date", name]
        s["date"] = pd.to_datetime(s["date"])
        if df is None:
            df = s
        else:
            df = df.merge(s, on="date", how="outer")

    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    out_path = os.path.join(EXTERNAL_DIR, "fred_indicators.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Columns: {list(df.columns)}")
    return out_path

if __name__ == "__main__":
    fetch_fred_indicators()
