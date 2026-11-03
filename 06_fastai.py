"""Step 6: Model 3 - PyTorch with fastai (tabular_learner)."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.tabular.all import (
    EarlyStoppingCallback, Normalize, RegressionBlock, SaveModelCallback,
    TabularPandas, rmse, set_seed, tabular_learner,
)

from config import RANDOM_STATE, TARGET
from utils import load_data, evaluate, save_results

set_seed(RANDOM_STATE, reproducible=True)
X_train, X_test, y_train, y_test, X_latest = load_data()
y_col = f"{TARGET}_fwd_ret"
features = list(X_train.columns)

# fastai wants one DataFrame plus a list of which rows are train vs validation
# Target in PERCENT: raw weekly returns (~0.03) are too small for fastai's defaults to learn well
SCALE = 100
train_df = X_train.copy()
train_df[y_col] = y_train * SCALE
n_valid = int(len(train_df) * 0.2)                      # last 20% of training period
splits = (list(range(len(train_df) - n_valid)), list(range(len(train_df) - n_valid, len(train_df))))

to = TabularPandas(train_df.reset_index(drop=True), procs=[Normalize], cont_names=features,
                   y_names=y_col, y_block=RegressionBlock(), splits=splits)
dls = to.dataloaders(bs=32, shuffle=True)

learn = tabular_learner(dls, layers=[64, 32], metrics=rmse)

start = time.time()
with learn.no_bar():
    learn.fit_one_cycle(50, lr_max=1e-3,
                        cbs=[EarlyStoppingCallback(monitor="valid_loss", patience=10),
                             SaveModelCallback(monitor="valid_loss", fname="fastai_best")])
train_seconds = time.time() - start


def predict(X):
    dl = learn.dls.test_dl(X.reset_index(drop=True))
    with learn.no_bar():
        preds, _ = learn.get_preds(dl=dl)
    return preds.numpy().ravel() / SCALE          # back to decimal returns


y_pred = predict(X_test)
metrics = evaluate(y_test, y_pred, y_train)
expected = predict(X_latest)
save_results("fastai", metrics, y_test, y_pred, expected, train_seconds)

learn.recorder.plot_loss()
plt.title("fastai training vs validation loss")
plt.tight_layout(); plt.savefig("figures/fastai_loss.png", dpi=120); plt.close()
