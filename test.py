import cv2
import mediapipe as mp
import joblib
import pyttsx3
import time

# Load trained model
model = joblib.load("sasl_model.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Initialize TTS engine
engine = pyttsx3.init()
last_pred = None
last_time = 0

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

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Predict letter
            features = extract_landmarks(hand_landmarks, frame.shape)
            pred = model.predict([features])[0]

            # Get hand orientation
            handedness_label = results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'

            # Prepare display and TTS text
            display_text = f"{handedness_label} hand: {pred}"

            # Only speak new predictions or after cooldown
            current_time = time.time()
            if display_text != last_pred or current_time - last_time > 2:  # 2 sec cooldown
                engine.say(display_text)
                engine.runAndWait()
                last_pred = display_text
                last_time = current_time

            # Show on webcam
            cv2.putText(frame, display_text, (10, 40 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("SASL Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
