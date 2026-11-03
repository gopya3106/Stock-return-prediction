"""Step 2: exploratory data analysis. Saves every chart to figures/."""
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

from config import RAW_PATH, HORIZON

os.makedirs("figures", exist_ok=True)
prices = pd.read_csv(RAW_PATH, index_col="Date", parse_dates=True)

# 1. Data quality
print("Shape:", prices.shape)
print("\nMissing values before filling:\n", prices.isna().sum())
prices = prices.ffill().dropna()      # FX/index holidays differ from US stocks

# 2. Prices over time (normalised to 100 so they fit on one chart)
(prices / prices.iloc[0] * 100).plot(figsize=(12, 6), logy=True)
plt.title("Growth of 100 invested (log scale)")
plt.ylabel("Value")
plt.tight_layout(); plt.savefig("figures/01_normalised_prices.png", dpi=120); plt.close()

# 3. Weekly log returns
weekly = np.log(prices).diff(HORIZON).iloc[::HORIZON].dropna()
print("\nSummary of weekly log returns:\n", weekly.describe().T.round(4))
print("\nSkew:\n", weekly.skew().round(2))
print("\nExcess kurtosis (fat tails if > 0):\n", weekly.kurt().round(2))

# 4. Distribution of the target
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(weekly["MSFT"], bins=60, kde=True, ax=ax)
ax.set_title("MSFT weekly log returns")
plt.tight_layout(); plt.savefig("figures/02_msft_return_hist.png", dpi=120); plt.close()

# 5. Correlation heatmap (the key EDA chart for this project)
plt.figure(figsize=(9, 7))
sns.heatmap(weekly.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
plt.title("Correlation of weekly returns")
plt.tight_layout(); plt.savefig("figures/03_correlation_heatmap.png", dpi=120); plt.close()

# 6. Scatter of MSFT vs each asset
others = [c for c in weekly.columns if c != "MSFT"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.flat, others):
    ax.scatter(weekly[col], weekly["MSFT"], s=6, alpha=0.5)
    ax.set_xlabel(col); ax.set_ylabel("MSFT")
for ax in axes.flat[len(others):]:
    ax.axis("off")
plt.suptitle("MSFT weekly return vs other assets")
plt.tight_layout(); plt.savefig("figures/04_scatter.png", dpi=120); plt.close()

# 7. Rolling volatility of MSFT (annualised)
daily = np.log(prices["MSFT"]).diff()
(daily.rolling(63).std() * np.sqrt(252)).plot(figsize=(12, 4))
plt.title("MSFT rolling 3-month volatility (annualised)")
plt.tight_layout(); plt.savefig("figures/05_rolling_vol.png", dpi=120); plt.close()

# 8. Stationarity: prices vs returns (ADF test, p < 0.05 means stationary)
print("\nADF test p-values")
print(f"  MSFT price  : {adfuller(prices['MSFT'])[1]:.4f}")
print(f"  MSFT return : {adfuller(weekly['MSFT'])[1]:.4f}")

print("\nCharts saved in figures/")
