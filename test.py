import cv2
import mediapipe as mp
import joblib
import os

MODEL_PATH = "sasl_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def extract_landmarks(hand_landmarks, shape):
    h, w, _ = shape
    x_list = [lm.x * w for lm in hand_landmarks.landmark]
    y_list = [lm.y * h for lm in hand_landmarks.landmark]
    return x_list + y_list

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("🖐 SASL Recognizer: ESC = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = extract_landmarks(hand_landmarks, frame.shape)
            if len(features) == 42:  # 21 x, 21 y
                pred = model.predict([features])[0]
                cv2.putText(frame, f"Prediction: {pred}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Invalid hand landmarks", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, "ESC=Quit", (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("SASL Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
