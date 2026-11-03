"""Step 5: Model 2 - Keras neural network (multi-layer perceptron).

Keras 3 can run on TensorFlow, PyTorch or JAX. We use the PyTorch backend so the
whole project needs only one deep-learning install. To use TensorFlow instead,
`pip install tensorflow` and change "torch" below to "tensorflow".
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import time

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RANDOM_STATE
from utils import load_data, evaluate, save_results, scale, validation_split

keras.utils.set_random_seed(RANDOM_STATE)

X_train, X_test, y_train, y_test, X_latest = load_data()
X_train_s, X_test_s, X_latest_s = scale(X_train, X_test, X_latest)
X_tr, X_val, y_tr, y_val = validation_split(X_train_s, y_train)

model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),                      # one output = predicted return
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
model.summary()

start = time.time()
history = model.fit(
    X_tr.values, y_tr.values,
    validation_data=(X_val.values, y_val.values),
    epochs=300, batch_size=32, verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
)
train_seconds = time.time() - start
print(f"Stopped after {len(history.history['loss'])} epochs")

y_pred = model.predict(X_test_s.values, verbose=0).ravel()
metrics = evaluate(y_test, y_pred, y_train)
expected = model.predict(X_latest_s.values, verbose=0).ravel()
save_results("keras", metrics, y_test, y_pred, expected, train_seconds)

# Learning curve: shows over-fitting if validation loss rises while training loss falls
pd.DataFrame(history.history).plot(figsize=(8, 4), logy=True)
plt.title("Keras training vs validation loss"); plt.xlabel("Epoch")
plt.tight_layout(); plt.savefig("figures/keras_loss.png", dpi=120); plt.close()
