"""Shared settings for every script in the project."""

START_DATE = "2010-01-01"
END_DATE = None          # None = up to today

TARGET = "MSFT"          # stock whose weekly return we predict

# Yahoo Finance symbols for every series we need (column name -> ticker)
TICKERS = {
    "MSFT": "MSFT",
    "IBM": "IBM",
    "GOOGL": "GOOGL",
    "USDJPY": "JPY=X",       # yen per US dollar
    "GBPUSD": "GBPUSD=X",    # dollars per pound
    "SP500": "^GSPC",
    "DJIA": "^DJI",
    "VIX": "^VIX",
}

HORIZON = 5              # trading days in one "week"
TARGET_LAGS = [5, 15, 30, 60]   # past MSFT returns used as features (days)
TEST_SIZE = 0.2          # last 20% of the timeline is the test set
RANDOM_STATE = 42

RAW_PATH = "data/raw_prices.csv"
FEATURES_PATH = "data/features.csv"
LATEST_PATH = "data/latest_features.csv"
