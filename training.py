import cv2
import mediapipe as mp
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# === Setup ===
LETTERS = [chr(i) for i in range(65, 91)]  # A-Z
IMG_DIR = "sasl_images"   # reference images (A.jpg, B.jpg, ...)
DATA_DIR = "sasl_dataset"
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "sasl_landmarks.csv")
MODEL_PATH = "sasl_model.pkl"

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Create CSV if it doesn't exist
if not os.path.isfile(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label", "hand"]
        writer.writerow(header)

# === Helper function ===
def extract_landmarks(hand_landmarks, shape):
    h, w, _ = shape
    x_list = [lm.x * w for lm in hand_landmarks.landmark]
    y_list = [lm.y * h for lm in hand_landmarks.landmark]
    return x_list, y_list

# === Camera setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

letter_index = 0
print("📷 SASL Trainer: SPACE = capture, N = next letter, ESC = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Load reference image
    letter = LETTERS[letter_index]
    ref_img_path = os.path.join(IMG_DIR, f"{letter}.jpg")
    if os.path.exists(ref_img_path):
        ref_img = cv2.imread(ref_img_path)
        ref_img = cv2.resize(ref_img, (150, 150))
        frame[10:160, -160:-10] = ref_img

    # Draw landmarks
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # On-screen instructions
    cv2.putText(frame, f"Letter: {letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "SPACE=Capture | N=Next Letter | ESC=Quit",
                (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("SASL Trainer", frame)
    key = cv2.waitKey(1) & 0xFF

    # === Capture data ===
    if key == 32 and results.multi_hand_landmarks:  # SPACE
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x_list, y_list = extract_landmarks(hand_landmarks, frame.shape)
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness_label = results.multi_handedness[i].classification[0].label
            else:
                handedness_label = "Unknown"
            row = x_list + y_list + [letter, handedness_label]
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"✅ Saved {letter} ({handedness_label})")

    # === Next letter ===
    if key == ord("n"):
        letter_index = (letter_index + 1) % len(LETTERS)
        print(f"➡️ Now practicing {LETTERS[letter_index]}")

    # === Quit ===
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# === Train model after capturing ===
if os.path.exists(CSV_PATH):
    print("\n📊 Training model...")
    df = pd.read_csv(CSV_PATH)

    if "label" not in df.columns:
        raise KeyError("Column 'label' not found in CSV.")

    X = df.drop(["label", "hand"], axis=1, errors="ignore")
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"✅ Model trained with accuracy: {acc:.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
