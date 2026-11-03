"""Step 7: Model 4 - PyTorch with TabNet (attention-based network for tables)."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor

from config import RANDOM_STATE
from utils import load_data, evaluate, save_results, scale, validation_split

X_train, X_test, y_train, y_test, X_latest = load_data()
X_train_s, X_test_s, X_latest_s = scale(X_train, X_test, X_latest)
X_tr, X_val, y_tr, y_val = validation_split(X_train_s, y_train)

model = TabNetRegressor(
    n_d=8, n_a=8,          # width of the decision / attention layers
    n_steps=3,             # number of sequential "attention" steps
    gamma=1.3,
    seed=RANDOM_STATE,
    verbose=10,
)

start = time.time()
model.fit(
    X_tr.values, y_tr.values.reshape(-1, 1),          # TabNet needs a 2-D target
    eval_set=[(X_val.values, y_val.values.reshape(-1, 1))],
    eval_name=["valid"], eval_metric=["rmse"],
    max_epochs=200, patience=20,
    batch_size=128, virtual_batch_size=64,
)
train_seconds = time.time() - start

y_pred = model.predict(X_test_s.values).ravel()
metrics = evaluate(y_test, y_pred, y_train)
expected = model.predict(X_latest_s.values).ravel()
save_results("tabnet", metrics, y_test, y_pred, expected, train_seconds)

# TabNet's built-in feature importance (from its attention masks) - its transparency advantage
imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
imp.plot.barh(figsize=(8, 5), title="TabNet feature importance (attention masks)")
plt.tight_layout(); plt.savefig("figures/tabnet_importance.png", dpi=120); plt.close()
