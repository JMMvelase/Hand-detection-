import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ------------------------------
# Config
# ------------------------------
CSV_PATH = "sasl_dataset/sasl_landmarks.csv"
MODEL_PATH = "sasl_model.pkl"

# ------------------------------
# Normalization function (same as in real-time inference)
# ------------------------------
def normalize_landmarks(row):
    """
    Normalize one row of landmarks:
    - Center on wrist (landmark 0)
    - Scale by max distance to make size invariant
    """
    num_points = len(row) // 2
    # Auto-detect number of landmarks
    x_cols = [c for c in df.columns if c.startswith("x")]
    y_cols = [c for c in df.columns if c.startswith("y")]
    num_points = len(x_cols)
    
    xs = np.array([row[c] for c in x_cols])
    ys = np.array([row[c] for c in y_cols])
    
    # Use wrist as origin
    wrist_x, wrist_y = xs[0], ys[0]
    xs -= wrist_x
    ys -= wrist_y

    # Scale by max distance
    max_val = max(np.max(np.abs(xs)), np.max(np.abs(ys)))
    if max_val > 0:
        xs /= max_val
        ys /= max_val

    return np.concatenate([xs, ys])


# ------------------------------
# Load dataset
# ------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

if "label" not in df.columns:
    raise KeyError("CSV must contain a 'label' column.")

# Extract features
num_points = 21
X = []
for _, row in df.iterrows():
    normed = normalize_landmarks(row)
    X.append(normed)

X = np.array(X)
y = df["label"]

# ------------------------------
# Train/test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Train model
# ------------------------------
print("📊 Training RandomForest on normalized landmarks...")
model = RandomForestClassifier(
    n_estimators=300, 
    max_depth=None, 
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"✅ Accuracy on test set: {acc:.2f}")

# ------------------------------
# Save model
# ------------------------------
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
