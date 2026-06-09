"""Fetch full AMZN history and overwrite amazon_stock.csv."""
import yfinance as yf
import pandas as pd

df = yf.Ticker("AMZN").history(period="max")
df.index = pd.to_datetime(df.index).tz_localize(None)
df.index.name = "date"
df = df[["Open", "High", "Low", "Close", "Volume"]]
df.columns = ["open", "high", "low", "close", "volume"]
df["adj_close"] = df["close"]
df.to_csv("amazon_stock.csv")
print(f"Updated through {df.index[-1].date()}, {len(df)} rows")
