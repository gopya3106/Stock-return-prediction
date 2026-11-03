"""Step 3: build the target (Y) and features (X) for the regression."""
import numpy as np
import pandas as pd

from config import RAW_PATH, FEATURES_PATH, LATEST_PATH, TARGET, HORIZON, TARGET_LAGS

prices = pd.read_csv(RAW_PATH, index_col="Date", parse_dates=True).ffill().dropna()
logp = np.log(prices)

# Y = MSFT log return over the NEXT 5 trading days (shifted back so each row predicts the future)
Y = (logp[TARGET].shift(-HORIZON) - logp[TARGET]).rename(f"{TARGET}_fwd_ret")

# X part 1: past 5-day log return of every correlated asset
others = [c for c in prices.columns if c != TARGET]
X_assets = logp[others].diff(HORIZON).add_suffix("_ret")

# X part 2: MSFT's own past returns over several windows
X_lags = pd.concat({f"{TARGET}_ret_{l}d": logp[TARGET].diff(l) for l in TARGET_LAGS}, axis=1)

data = pd.concat([X_assets, X_lags, Y], axis=1)

# The most recent row has features but no known future -> keep it for "expected return next week"
latest = data.drop(columns=Y.name).dropna().iloc[[-1]]
latest.to_csv(LATEST_PATH)

# Sample once a week so the 5-day windows do not overlap, then drop incomplete rows
data = data.iloc[::HORIZON].dropna()
data.to_csv(FEATURES_PATH)

print(f"Feature table: {data.shape[0]} weekly rows, {data.shape[1] - 1} features")
print("Columns:", list(data.columns))
print(f"Date range: {data.index.min().date()} -> {data.index.max().date()}")
print(f"Latest feature row (for next-week forecast): {latest.index[0].date()}")
