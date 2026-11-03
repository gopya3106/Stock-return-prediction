Stock Return Prediction (MSFT) with Supervised Learning

Predicting the next-week return of Microsoft (MSFT) from its own past returns and the returns of correlated assets, and comparing five machine-learning frameworks on the same task: XGBoost, Keras, PyTorch + fastai, PyTorch + TabNet and PyTorch + Lightning Flash.

Problem
Type: supervised regression
Target (Y): MSFT log return over the next 5 trading days
Features (X):
5-day log returns of correlated assets: IBM, Alphabet (GOOGL), USD/JPY, GBP/USD, S&P 500, Dow Jones, VIX
MSFT's own past returns over 5, 15, 30 and 60 trading days
Data: daily closing prices from Yahoo Finance, January 2010 to November 2026, sampled weekly (834 weekly observations)
Split: chronological, no shuffling. Train: Mar 2010 – Jun 2023 (667 weeks); test: Jun 2023 – Oct 2026 (167 weeks)
Method
Download prices, fill currency-market holidays forward
EDA: return distributions, fat tails, correlation heatmap, rolling volatility, ADF stationarity test
Build weekly target and features
Train each model on the training period; neural networks use the last 20% of the training period for early stopping
Evaluate on the test period with RMSE, MAE, R² and directional accuracy, against two baselines:
predicting the training-period average return
always predicting "up"
Use each model's prediction for the most recent week as its expected return
Results (test period)
Model	RMSE	MAE	R²	Directional accuracy
Baseline (mean / "always up")	0.0374	0.0256	0	56.3%
PyTorch + fastai	0.0367	0.0252	0.035	59.3%
Keras	0.0373	0.0259	0.005	61.1%
PyTorch + Lightning Flash	0.0373	0.0263	0.004	61.1%
XGBoost	0.0373	0.0257	0.003	56.3%
PyTorch + TabNet	0.0490	0.0316	-0.719	47.3%

Full table: results/comparison.csv. Charts: figures/.

Weekly stock returns are mostly noise, so every model lands close to the baseline. fastai is the only one ahead on every metric, by a small margin over 167 test weeks.

Project structure
File	Step
config.py	Shared settings (tickers, dates, horizon, test size)
utils.py	Chronological train/test split, scaling, metrics, saving results
01_get_data.py	Download daily prices from Yahoo Finance
02_eda.py	Exploratory data analysis, charts in figures/
03_features.py	Build weekly target and features
04_xgboost.py	Model 1: XGBoost
05_keras.py	Model 2: Keras neural network
06_fastai.py	Model 3: PyTorch + fastai
07_tabnet.py	Model 4: PyTorch + TabNet
08_lightning_flash.py	Model 5: PyTorch + Lightning Flash (separate environment)
09_compare.py	Comparison table and charts for all models
notebooks/	The same steps as Jupyter notebooks (01–09)
figures/	Charts produced by the scripts / notebooks
results/	Metrics (JSON), test predictions (CSV), comparison.csv

data/ is not stored in the repo; 01_get_data.py (or 01_get_data.ipynb) downloads it.

How to run (Windows cmd)

Main environment (Python 3.12):

py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python 01_get_data.py
python 02_eda.py
python 03_features.py
python 04_xgboost.py
python 05_keras.py
python 06_fastai.py
python 07_tabnet.py
deactivate

Lightning Flash environment (Python 3.10, because Lightning Flash is archived):

py -3.10 -m venv .venv-flash
.venv-flash\Scripts\activate
pip install -r requirements-flash.txt
python 08_lightning_flash.py
deactivate

Comparison (back in the main environment):

.venv\Scripts\activate
python 09_compare.py
Notebooks

The same pipeline can be run as notebooks instead of scripts:

.venv\Scripts\activate
jupyter notebook

Open notebooks/ and run each notebook in order with Run → Run All Cells. 08_lightning_flash.ipynb uses the "Python 3.10 (flash)" kernel, registered once with:

.venv-flash\Scripts\activate
pip install ipykernel
python -m ipykernel install --user --name flash --display-name "Python 3.10 (flash)"
