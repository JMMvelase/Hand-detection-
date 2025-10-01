import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = "sign_words"
WORDS = ["hello", "thanks", "help", "me", "please"]   # your words here
SEQUENCE_LENGTH = 30          # frames per sequence
NUM_LANDMARKS = 42            # x,y for 21 landmarks

os.makedirs(DATA_DIR, exist_ok=True)
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:
    for word in WORDS:
        print(f"\nReady to record word: {word}")
        word_dir = os.path.join(DATA_DIR, word)
        os.makedirs(word_dir, exist_ok=True)
        seq_count = 0
        while True:
            print(f"\nPress 's' to START recording {word} sequence {seq_count+1}")
            print("Press 'n' to move to NEXT word, or 'q' to quit.")
            # Wait for user to press 's', 'n', or 'q'
            key = None
            while True:
                ret, frame = cap.read()
                cv2.putText(frame, f"Press 's' to start {word} ({seq_count+1}) | 'n': next word | 'q': quit",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.imshow("Recording", frame)
                k = cv2.waitKey(10) & 0xFF
                if k == ord('s'):
                    key = 's'
                    break
                elif k == ord('n'):
                    key = 'n'
                    break
                elif k == ord('q'):
                    key = 'q'
                    break
            if key == 'n':
                print(f"Moving to next word...")
                break
            if key == 'q':
                print("Quitting data capture.")
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

            # Countdown
            for countdown in range(3,0,-1):
                ret, frame = cap.read()
                cv2.putText(frame, f"Starting in {countdown}...",
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                cv2.imshow("Recording", frame)
                cv2.waitKey(1000)

            print("Recording now... Press 'q' to stop early.")
            sequence = []
            while len(sequence) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Extract landmarks
                landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y])

                # Ensure every frame has exactly NUM_LANDMARKS values
                landmarks = landmarks[:NUM_LANDMARKS]  # truncate if longer
                if len(landmarks) < NUM_LANDMARKS:
                    landmarks += [0]*(NUM_LANDMARKS - len(landmarks))  # pad with zeros

                sequence.append(landmarks)

                # Draw landmarks for feedback
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.putText(frame, f"{word} Seq:{seq_count+1}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("Recording", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Stopped recording early.")
                    break

            # Save the sequence if it has enough frames
            if len(sequence) >= SEQUENCE_LENGTH:
                sequence = np.array(sequence)  # shape (SEQUENCE_LENGTH, 42)
                np.save(os.path.join(word_dir, f"{seq_count}.npy"), sequence)
                print(f"Saved {word} sequence {seq_count+1}")
                seq_count += 1
            else:
                print(f"Sequence too short, not saved.")

cap.release()
cv2.destroyAllWindows()
