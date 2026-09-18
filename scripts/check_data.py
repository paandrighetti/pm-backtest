"""Sanity check of the ingested data before running the report. Usage: python scripts/check_data.py"""

import duckdb
import pandas as pd

prices = duckdb.sql(
    "SELECT count(*) AS rows, count(DISTINCT market_id) AS markets, "
    "to_timestamp(min(ts)) AS first_ts, to_timestamp(max(ts)) AS last_ts "
    "FROM read_parquet('data/prices/*.parquet')"
).df()
print("price points:")
print(prices.to_string(index=False))

m = pd.read_parquet("data/markets.parquet")
print(f"\nmarkets: {len(m)}, with outcome: {int(m['outcome'].notna().sum())}")
print("market_type:", m["market_type"].value_counts(dropna=False).to_dict())
print("markets per event:", m.groupby("n_in_event").size().head(8).to_dict())
print("end dates:", pd.to_datetime(m["end_ts"], unit="s").min(), "->", pd.to_datetime(m["end_ts"], unit="s").max())
