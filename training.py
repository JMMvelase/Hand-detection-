import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("sasl_dataset/sasl_landmarks.csv")

# Features = all x,y columns, Label = letter
X = df.drop("label", axis=1)
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
acc = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {acc:.2f}")

# Save model
joblib.dump(model, "sasl_model.pkl")
