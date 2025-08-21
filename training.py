import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

DATA_PATH = "sasl_dataset/sasl_landmarks.csv"
MODEL_PATH = "sasl_model.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Columns in CSV:", df.columns.tolist())

if "label" not in df.columns:
    raise KeyError("Column 'label' not found in CSV. Please check your data collection and CSV header.")

X = df.drop(["label", "hand"], axis=1, errors="ignore")
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {acc:.2f}")

joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
