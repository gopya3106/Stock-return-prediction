"""Step 1: download daily closing prices from Yahoo Finance and save them."""
import os

import pandas as pd
import yfinance as yf

import warnings
warnings.filterwarnings("ignore")   # yfinance re-enables warnings on import, so silence them after it

from config import TICKERS, START_DATE, END_DATE, RAW_PATH

os.makedirs("data", exist_ok=True)

frames = {}
for name, symbol in TICKERS.items():
    hist = yf.Ticker(symbol).history(start=START_DATE, end=END_DATE, auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"No data returned for {symbol} - check the ticker or your internet")
    close = hist["Close"]
    close.index = close.index.tz_localize(None).normalize()   # drop timezones so dates line up
    frames[name] = close
    print(f"{name:7s} {symbol:9s} {len(close):5d} rows  {close.index.min().date()} -> {close.index.max().date()}")

prices = pd.DataFrame(frames)

# Keep only days when the target stock traded (US trading calendar)
prices = prices[prices["MSFT"].notna()]
prices.index.name = "Date"
prices.to_csv(RAW_PATH)
print(f"\nSaved {prices.shape[0]} rows x {prices.shape[1]} columns to {RAW_PATH}")
print("Missing values per column:\n", prices.isna().sum())
