import cv2
import mediapipe as mp
import joblib

# Load model
model = joblib.load("sasl_model.pkl")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def extract_landmarks(hand_landmarks, shape):
    h, w, _ = shape
    x_list = [lm.x * w for lm in hand_landmarks.landmark]
    y_list = [lm.y * h for lm in hand_landmarks.landmark]
    return x_list + y_list

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            features = extract_landmarks(hand_landmarks, frame.shape)
            pred = model.predict([features])[0]

            # Show prediction
            cv2.putText(frame, f"Prediction: {pred}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("SASL Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
