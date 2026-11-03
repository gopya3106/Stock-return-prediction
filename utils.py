"""Helpers shared by all five model scripts, so every framework is judged the same way."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import FEATURES_PATH, LATEST_PATH, TEST_SIZE, TARGET


def load_data():
    """Return X_train, X_test, y_train, y_test, X_latest using a chronological split (no shuffling)."""
    data = pd.read_csv(FEATURES_PATH, index_col="Date", parse_dates=True)
    latest = pd.read_csv(LATEST_PATH, index_col="Date", parse_dates=True)
    y_col = f"{TARGET}_fwd_ret"
    X, y = data.drop(columns=y_col), data[y_col]
    split = int(len(data) * (1 - TEST_SIZE))
    print(f"Train: {X.index[0].date()} -> {X.index[split - 1].date()} ({split} rows)")
    print(f"Test : {X.index[split].date()} -> {X.index[-1].date()} ({len(X) - split} rows)")
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], latest[X.columns]


def evaluate(y_true, y_pred, y_train=None):
    """Regression metrics + directional accuracy (+ a naive baseline that always predicts the train mean)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred).ravel()
    out = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Directional_Accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }
    if y_train is not None:
        base = np.full_like(y_true, float(np.mean(y_train)))
        out["Baseline_RMSE"] = float(np.sqrt(mean_squared_error(y_true, base)))
        out["Baseline_MAE"] = float(mean_absolute_error(y_true, base))
    # Accuracy of a "always predict up" rule = share of up-weeks in the test period
    out["Baseline_Directional_Accuracy"] = float(np.mean(y_true > 0))
    return out


def scale(X_train, *others):
    """Standardise features using statistics from the TRAINING set only (avoids leakage)."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    to_df = lambda X: pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    return [to_df(X_train)] + [to_df(X) for X in others]


def validation_split(X_train, y_train, frac=0.2):
    """Hold out the LAST part of the training period for early stopping (time order kept)."""
    cut = int(len(X_train) * (1 - frac))
    return X_train.iloc[:cut], X_train.iloc[cut:], y_train.iloc[:cut], y_train.iloc[cut:]


def save_results(model_name, metrics, y_test, y_pred, expected_return, train_seconds, extra=None):
    """Write metrics + predictions to results/ so the comparison step can read them later."""
    os.makedirs("results", exist_ok=True)
    weekly = float(np.ravel(expected_return)[0])
    record = {
        "model": model_name,
        **metrics,
        "train_seconds": round(train_seconds, 2),
        "expected_weekly_log_return": weekly,
        "expected_weekly_return_pct": (np.exp(weekly) - 1) * 100,
        "expected_annualised_return_pct": (np.exp(weekly * 52) - 1) * 100,
        **(extra or {}),
    }
    with open(f"results/metrics_{model_name}.json", "w") as f:
        json.dump(record, f, indent=2)
    pd.DataFrame({"actual": np.asarray(y_test), "predicted": np.ravel(y_pred)},
                 index=y_test.index).to_csv(f"results/predictions_{model_name}.csv")
    for k, v in record.items():
        print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")
    return record
