# Stock Return Prediction (MSFT) with Supervised Learning

<!-- Add your own project description here. -->

## Project structure

| File | Step |
|---|---|
| `config.py` | Shared settings (tickers, dates, horizon, test size) |
| `utils.py` | Chronological train/test split, scaling, metrics, saving results |
| `01_get_data.py` | Download daily prices from Yahoo Finance |
| `02_eda.py` | Exploratory data analysis, charts in `figures/` |
| `03_features.py` | Build weekly target and features |
| `04_xgboost.py` | Model 1: XGBoost |
| `05_keras.py` | Model 2: Keras neural network |
| `06_fastai.py` | Model 3: PyTorch + fastai |
| `07_tabnet.py` | Model 4: PyTorch + TabNet |
| `08_lightning_flash.py` | Model 5: PyTorch + Lightning Flash (separate environment) |
| `09_compare.py` | Comparison table and charts for all models |

## How to run (Windows cmd)

Main environment (Python 3.12):

```
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
```

Lightning Flash environment (Python 3.10, because Lightning Flash is archived):

```
py -3.10 -m venv .venv-flash
.venv-flash\Scripts\activate
pip install -r requirements-flash.txt
python 08_lightning_flash.py
deactivate
```

Comparison (back in the main environment):

```
.venv\Scripts\activate
python 09_compare.py
```

Outputs: `results/` (metrics JSON, predictions CSV, `comparison.csv`) and `figures/` (charts).
