"""
ingest.py
=========
Pulls the latest daily OHLCV data for a set of tickers via yfinance and
upserts it into the 'prices' table in Postgres. Designed to be run once a
day (locally while testing, then via GitHub Actions on a schedule).

Usage:
    python ingest.py                # pulls last 5 days (safety buffer) for all tickers
    python ingest.py --full-history # one-time backfill of 20 years of history
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
TICKERS = os.environ.get("TICKERS", "AAPL,MSFT,GOOGL,AMZN,NVDA").split(",")


def get_engine():
    if not DATABASE_URL:
        sys.exit("ERROR: DATABASE_URL not set. Copy .env.example to .env and fill it in.")
    return create_engine(DATABASE_URL)


def fetch_prices(ticker: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data for one ticker from Yahoo Finance."""
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if df.empty:
        print(f"  [warn] no data returned for {ticker}")
        return pd.DataFrame()

    df = df.reset_index()
    # yfinance sometimes returns MultiIndex columns when multiple tickers are
    # passed at once; flatten just in case a single-ticker call still does this
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["ticker"] = ticker
    return df[["ticker", "date", "open", "high", "low", "close", "volume"]]


def upsert_prices(engine, df: pd.DataFrame, chunk_size: int = 200, max_retries: int = 3):
    """
    Insert rows in small batches. Each batch is retried with a short
    backoff if the connection drops mid-transfer (flaky wifi, or a
    cold-starting serverless DB) — so a transient blip costs a few
    seconds, not the whole run.
    """
    if df.empty:
        return 0

    insert_sql = text("""
        INSERT INTO prices (ticker, date, open, high, low, close, volume)
        VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
        ON CONFLICT (ticker, date) DO NOTHING
    """)

    rows = df.to_dict(orient="records")
    total_inserted = 0
    num_chunks = (len(rows) + chunk_size - 1) // chunk_size

    for chunk_num, i in enumerate(range(0, len(rows), chunk_size), start=1):
        chunk = rows[i:i + chunk_size]

        for attempt in range(1, max_retries + 1):
            try:
                with engine.begin() as conn:
                    result = conn.execute(insert_sql, chunk)
                total_inserted += result.rowcount
                break
            except OperationalError as e:
                if attempt == max_retries:
                    print(f"    [error] chunk {chunk_num}/{num_chunks} failed after "
                          f"{max_retries} attempts: {e}")
                    raise
                wait = 2 ** attempt
                print(f"    [retry] chunk {chunk_num}/{num_chunks} attempt {attempt} "
                      f"failed, retrying in {wait}s...")
                time.sleep(wait)

    return total_inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-history", action="store_true",
                         help="Backfill 5 years of history instead of just the last few days")
    args = parser.parse_args()

    period = "20y" if args.full_history else "5d"
    engine = get_engine()

    total_inserted = 0
    for ticker in TICKERS:
        ticker = ticker.strip()
        print(f"Fetching {ticker} ({period})...")
        df = fetch_prices(ticker, period)
        inserted = upsert_prices(engine, df)
        print(f"  -> {inserted} new row(s) inserted")
        total_inserted += inserted

    print(f"\nDone. {total_inserted} total new rows inserted at {datetime.utcnow().isoformat()}Z")


if __name__ == "__main__":
    main()
