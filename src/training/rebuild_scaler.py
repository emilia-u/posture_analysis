"""
Rebuilds models/scaler.pkl (the StandardScaler used before the model).

scaler.pkl is not committed to git, so a fresh clone does not have it. Without it the
server feeds unscaled values to the model and every frame is classified as bad posture.

This repeats exactly the data prep from train_model.ipynb (same split, random_state=42),
so the result is identical to the original scaler. The model itself is not retrained.

Usage (from the repo root):  python src/training/rebuild_scaler.py
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "data", "posture_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "posture_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

LANDMARKS_TO_USE = ["nose", "left_shoulder", "right_shoulder"]
FEATURE_COLS = [f"{lm}_{ax}" for lm in LANDMARKS_TO_USE for ax in ["x", "y", "z"]]

df = pd.read_csv(CSV_PATH)
X = df[FEATURE_COLS].copy()
y = df["label"]
X = X - df[df["label"] == 0][FEATURE_COLS].mean()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)

model = joblib.load(MODEL_PATH)
acc = accuracy_score(y_test, model.predict(scaler.transform(X_test)))
print(f"Test accuracy with rebuilt scaler: {acc:.4f} (expected about 0.9853)")
if acc < 0.95:
    raise SystemExit("Accuracy too low: the scaler does not match the model. Not saving.")

joblib.dump(scaler, SCALER_PATH)
print(f"Saved {SCALER_PATH}")
