import cv2
import mediapipe as mp
import joblib
import pyttsx3
import time
import os
from threading import Thread
from queue import Queue
import numpy as np
import mediapipe as mp
import pyttsx3
import time
from tensorflow.keras.models import load_model

class HandRecognizer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Load the trained model
        self.model = load_model("asl_alphabet_model.h5", compile=False)
        
        # Define labels
        self.labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize variables
        self.last_prediction = None
        self.last_speak_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def extract_hand_image(self, frame, hand_landmarks):
        """Extract and preprocess hand image for the model."""
        # Get hand bounding box
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Extract hand region
        hand_img = frame[y_min:y_max, x_min:x_max]
        
        # Preprocess for model
        hand_img = cv2.resize(hand_img, (64, 64))  # Model input size
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        hand_img = hand_img.astype('float32') / 255.0
        hand_img = np.expand_dims(hand_img, axis=0)
        
        return hand_img, (x_min, y_min, x_max, y_max)
    
    def process_frame(self, frame):
        # Convert to RGB for hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(rgb_frame)
        
        # Draw FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        else:
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract and preprocess hand image
                hand_img, (x_min, y_min, x_max, y_max) = self.extract_hand_image(frame, hand_landmarks)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Make prediction
                prediction = self.model.predict(hand_img)
                predicted_class = np.argmax(prediction[0])
                prediction_label = self.labels[predicted_class]
                confidence = prediction[0][predicted_class]
                hand_type = handedness.classification[0].label
                
                # Display prediction
                text = f"{hand_type} hand: {prediction_label} ({confidence:.2f})"
                y_pos = 70 + idx * 40
                cv2.putText(frame, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Speak prediction if it's new and confident enough
                current_time = time.time()
                if (confidence > 0.7 and  # Only speak if confident
                    prediction_label != self.last_prediction and 
                    current_time - self.last_speak_time > 2):  # 2 second cooldown
                    self.engine.say(f"{hand_type} hand shows {prediction_label}")
                    self.engine.runAndWait()
                    self.last_prediction = prediction_label
                    self.last_speak_time = current_time
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("👋 Hand Recognizer started! Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Show frame
            cv2.imshow("Hand Recognition", processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    recognizer = HandRecognizer()
    recognizer.run()
