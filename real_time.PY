import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# -----------------------------
# SETTINGS
# -----------------------------
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 42
WORDS = ["hello", "thanks"]  # same order as training

# Load trained model
model = load_model("sign_model.h5")  # save your model after training using model.save("sign_model.h5")

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# Buffer to hold the last SEQUENCE_LENGTH frames
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Extract landmarks
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Pad/truncate landmarks to fixed size
        landmarks = landmarks[:NUM_LANDMARKS]
        if len(landmarks) < NUM_LANDMARKS:
            landmarks += [0] * (NUM_LANDMARKS - len(landmarks))

        # Add to buffer
        sequence_buffer.append(landmarks)

        # If buffer is full → predict
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            seq_array = np.expand_dims(sequence_buffer, axis=0)  # shape (1, SEQUENCE_LENGTH, NUM_LANDMARKS)
            prediction = model.predict(seq_array, verbose=0)[0]
            pred_word = WORDS[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Show prediction
            cv2.putText(frame, f"{pred_word} ({confidence:.2f})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Sign Recognition", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
