"""Step 4: Model 1 - XGBoost regression, tuned with time-series cross-validation."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from config import RANDOM_STATE
from utils import load_data, evaluate, save_results

X_train, X_test, y_train, y_test, X_latest = load_data()

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [2, 3, 5],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

start = time.time()
search = GridSearchCV(
    XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1),
    param_grid,
    cv=TimeSeriesSplit(n_splits=5),         # folds respect time order
    scoring="neg_root_mean_squared_error",
)
search.fit(X_train, y_train)
train_seconds = time.time() - start
model = search.best_estimator_
print("Best params:", search.best_params_)

y_pred = model.predict(X_test)
metrics = evaluate(y_test, y_pred, y_train)
expected = model.predict(X_latest)

save_results("xgboost", metrics, y_test, y_pred, expected, train_seconds,
             extra={"best_params": search.best_params_})

# Feature importance (transparency)
imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
imp.plot.barh(figsize=(8, 5), title="XGBoost feature importance")
plt.tight_layout(); plt.savefig("figures/xgboost_importance.png", dpi=120); plt.close()

# Predicted vs actual on the test period
pd.DataFrame({"Actual": y_test, "Predicted": y_pred}, index=y_test.index).plot(figsize=(12, 4))
plt.title("XGBoost: predicted vs actual weekly return (test set)")
plt.tight_layout(); plt.savefig("figures/xgboost_pred_vs_actual.png", dpi=120); plt.close()
