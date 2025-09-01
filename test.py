import cv2
import mediapipe as mp
import joblib
import pyttsx3
import time
import os
from gtts import gTTS
from playsound import playsound
from threading import Thread
from queue import Queue


# ------------------------------
# Initialization Functions
# ------------------------------
def load_model(model_path="sasl_model.pkl"):
    """Load the trained sign language model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def init_tts():
    """Initialize pyttsx3 TTS with custom settings (SAPI5 for Windows)."""
    engine = pyttsx3.init(driverName="sapi5")
    voices = engine.getProperty("voices")
    if len(voices) > 1:
        engine.setProperty("voice", voices[1].id)  # use 2nd voice if available
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 0.9)
    return engine


def speak_text(text):
    """Convert text to speech using gTTS + playsound."""
    try:
        tts = gTTS(text=text, lang="en")
        filename = "temp_speech.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
        return True
    except Exception as e:
        print(f"[TTS Error] {e}")
        return False


# ------------------------------
# Helper Functions
# ------------------------------
def extract_landmarks(hand_landmarks, frame_shape):
    """Extract & normalize hand landmarks from MediaPipe output."""
    h, w, _ = frame_shape
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    return x_coords + y_coords


def draw_info(frame, fps, speech_enabled):
    """Overlay FPS and speech status on the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    status = "Speech: ON" if speech_enabled else "Speech: OFF"
    cv2.putText(frame, status, (frame.shape[1] - 120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def calculate_fps(frame_count, start_time):
    """Calculate frames per second."""
    elapsed = time.time() - start_time
    if elapsed == 0:
        return 0, start_time, frame_count
    fps = frame_count / elapsed
    return fps, time.time(), 0  # reset counter every second


# ------------------------------
# Main Application
# ------------------------------
def main():
    try:
        # Load model & initialize tools
        model = load_model()
        engine = init_tts()
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        # Configure MediaPipe for better performance
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,  # Reduced for better performance
            min_tracking_confidence=0.5,
            model_complexity=0  # Use fastest model
        )
        
        # Initialize speech queue for async TTS
        speech_queue = Queue()
        def speech_worker():
            while True:
                text = speech_queue.get()
                if text == "STOP":
                    break
                speak_text(text)
                speech_queue.task_done()
        
        # Start speech thread
        speech_thread = Thread(target=speech_worker, daemon=True)
        speech_thread.start()

        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Webcam not accessible")

        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Reduced resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG format

        print("👋 SASL Recognizer started!")
        print("Controls: ESC = Quit, S = Toggle speech")

        # Variables
        last_pred, last_time = None, 0
        speech_enabled, speech_failed = True, False
        frame_count, fps_start = 0, time.time()
        fps = 0.0
        process_this_frame = True  # Frame processing flag
        target_fps = 30
        frame_time = 1.0 / target_fps

        # Main loop
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame grab failed")
                break
                
            # Skip frame if we're running behind
            if not process_this_frame:
                process_this_frame = True
                continue

            # Resize frame for faster processing
            frame = cv2.resize(frame, (480, 360))
            frame = cv2.flip(frame, 1)
            
            # Process frame
            if process_this_frame:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                frame_count += 1
            else:
                results = None
                
            # Calculate processing time and adjust frame skip
            process_time = time.time() - loop_start
            if process_time > frame_time:
                process_this_frame = False  # Skip next frame if we're running slow
            else:
                process_this_frame = True

            # Update FPS once per second
            if time.time() - fps_start >= 1.0:
                fps, fps_start, frame_count = calculate_fps(frame_count, fps_start)

            predictions = []

            # Process detected hands
            if results.multi_hand_landmarks and results.multi_handedness:
                for i, (landmarks, handedness) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)):

                    # Draw landmarks with color per hand
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)
                    mp_drawing.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2))

                    # Predict
                    features = extract_landmarks(landmarks, frame.shape)
                    pred = model.predict([features])[0]
                    label = handedness.classification[0].label
                    text = f"{label} hand: {pred}"
                    predictions.append(text)

                    # Display prediction
                    cv2.putText(frame, text, (10, 40 + 30 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Handle speech asynchronously
                if speech_enabled and predictions:
                    combined_pred = " | ".join(predictions)
                    if (combined_pred != last_pred and 
                        time.time() - last_time > 2):  # Minimum 2-second gap between speeches
                        if speech_queue.empty():  # Only queue if not already speaking
                            speech_queue.put(combined_pred)
                            last_pred = combined_pred
                            last_time = time.time()

            # Draw overlays
            draw_info(frame, fps, speech_enabled)
            cv2.imshow("SASL Recognizer", frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("s"):
                speech_enabled = not speech_enabled
                print(f"🔊 Speech {'enabled' if speech_enabled else 'disabled'}")

    except Exception as e:
        print(f"[Error] {e}")

    finally:
        print("\n🔒 Closing SASL Recognizer...")
        try:
            # Stop speech thread
            speech_queue.put("STOP")
            speech_thread.join(timeout=1)
            
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            engine.stop()
        except:
            pass


if __name__ == "__main__":
    main()
