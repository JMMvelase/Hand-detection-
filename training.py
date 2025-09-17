import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# --- Config ---
CSV_PATH = "sasl_dataset/sasl_landmarks.csv"
MODEL_PATH = "sasl_model.pkl"

# --- Load dataset ---
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Check your columns
expected_cols = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label", "hand"]
if not all(col in df.columns for col in expected_cols):
    raise KeyError("CSV columns do not match expected landmarks columns.")

# --- Features & labels ---
X = df[[f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]]
y = df["label"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Train model ---
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# --- Evaluate ---
acc = clf.score(X_test, y_test)
print(f"✅ Model trained. Test accuracy: {acc:.2f}")

# --- Save model ---
joblib.dump(clf, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
