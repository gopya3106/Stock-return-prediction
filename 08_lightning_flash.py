"""Step 8: Model 5 - PyTorch with Lightning Flash (TabularRegressor).

Lightning Flash is archived (no longer maintained) and only installs on older
Python/PyTorch. Run this script from the separate Python 3.10 environment
described in README / requirements-flash.txt.
"""
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import flash
from flash.tabular import TabularRegressionData, TabularRegressor
from pytorch_lightning.callbacks import EarlyStopping

from config import RANDOM_STATE, TARGET
from utils import load_data, evaluate, save_results, validation_split

torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

X_train, X_test, y_train, y_test, X_latest = load_data()
X_tr, X_val, y_tr, y_val = validation_split(X_train, y_train)
y_col = f"{TARGET}_fwd_ret"
features = list(X_train.columns)
SCALE = 100          # train on returns in PERCENT (raw ~0.03 values are too small to learn well)


def frame(X, y=None):
    df = X.reset_index(drop=True).copy()
    if y is not None:
        df[y_col] = y.values * SCALE
    return df


# Flash normalises the numeric columns itself, using the training data statistics
datamodule = TabularRegressionData.from_data_frame(
    numerical_fields=features, target_field=y_col,
    train_data_frame=frame(X_tr, y_tr), val_data_frame=frame(X_val, y_val),
    batch_size=32,
)

# "category_embedding" = a plain feed-forward network from pytorch-tabular
model = TabularRegressor.from_data(datamodule, backbone="category_embedding",
                                   learning_rate=1e-3)
trainer = flash.Trainer(max_epochs=50, enable_progress_bar=False, logger=False,
                        enable_checkpointing=False,
                        callbacks=[EarlyStopping(monitor="valid_loss", patience=10)])

start = time.time()
trainer.fit(model, datamodule=datamodule)
train_seconds = time.time() - start


def predict(X):
    dm = TabularRegressionData.from_data_frame(
        predict_data_frame=frame(X), parameters=datamodule.parameters, batch_size=256)
    batches = trainer.predict(model, datamodule=dm, output="preds")
    return np.concatenate([np.ravel(np.asarray(b, dtype=float)) for b in batches]) / SCALE


y_pred = predict(X_test)
metrics = evaluate(y_test, y_pred, y_train)
expected = predict(X_latest)
save_results("lightning_flash", metrics, y_test, y_pred, expected, train_seconds)
