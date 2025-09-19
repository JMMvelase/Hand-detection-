import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands and Face Detection
mpHands = mp.solutions.hands
mpFace = mp.solutions.face_detection
hands = mpHands.Hands(max_num_hands=4)
face_detection = mpFace.FaceDetection()
drawLines = mp.solutions.drawing_utils

# Timer variables
hand_open_start_time = None
hand_open_duration = 5  # seconds
# When True the program will close the camera automatically after `hand_open_duration` seconds
# of continuous open-hand detection. Default is False to avoid unexpected camera shutdowns.
auto_close_on_open = False

def is_hand_open(hand_landmarks, img_height):
    """
    Determine if the hand is open by checking the relative positions of the finger tips
    compared to their respective base joints.
    """
    # Landmarks index: 0 - Wrist, 4 - Thumb tip, 3 - Thumb IP, 8 - Index finger tip, 6 - Index finger PIP,
    # 12 - Middle finger tip, 10 - Middle finger PIP, 16 - Ring finger tip, 14 - Ring finger PIP,
    # 20 - Pinky tip, 18 - Pinky PIP

    # Helper function to get the y coordinate
    def get_y_coord(landmark):
        return landmark.y * img_height

    thumb_tip_y = get_y_coord(hand_landmarks.landmark[4])
    thumb_ip_y = get_y_coord(hand_landmarks.landmark[3])
    index_tip_y = get_y_coord(hand_landmarks.landmark[8])
    index_pip_y = get_y_coord(hand_landmarks.landmark[6])
    middle_tip_y = get_y_coord(hand_landmarks.landmark[12])
    middle_pip_y = get_y_coord(hand_landmarks.landmark[10])
    ring_tip_y = get_y_coord(hand_landmarks.landmark[16])
    ring_pip_y = get_y_coord(hand_landmarks.landmark[14])
    pinky_tip_y = get_y_coord(hand_landmarks.landmark[20])
    pinky_pip_y = get_y_coord(hand_landmarks.landmark[18])

    # Check if the tip of each finger is higher (y-coordinate is smaller) than its respective PIP joint
    thumb_open = thumb_tip_y < thumb_ip_y
    index_open = index_tip_y < index_pip_y
    middle_open = middle_tip_y < middle_pip_y
    ring_open = ring_tip_y < ring_pip_y
    pinky_open = pinky_tip_y < pinky_pip_y

    return thumb_open and index_open and middle_open and ring_open and pinky_open

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image for hands
    hand_result = hands.process(img_rgb)
    # Process the image for face detection
    face_result = face_detection.process(img_rgb)

    # Draw face detection results
    if face_result.detections:
        for detection in face_result.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            
    # Function to get the hand vector
    def get_hand_vector(hand_landmarks, img_shape):
        h, w, _ = img_shape
        return [ (lm.x * w, lm.y * h) for lm in hand_landmarks.landmark ]

    # Draw hand landmarks and check if the hand is open
    hand_open = False
    if hand_result.multi_hand_landmarks:
        for i, hand_lms in enumerate(hand_result.multi_hand_landmarks):
            drawLines.draw_landmarks(img, hand_lms, mpHands.HAND_CONNECTIONS)
            if is_hand_open(hand_lms, img.shape[0]):
                cv2.putText(img, f'Hand {i+1} Open', (10, 70 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                hand_open = True
            else:
                cv2.putText(img, f'Hand {i+1} Closed', (10, 70 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Update the timer
    if hand_open:
        if hand_open_start_time is None:
            hand_open_start_time = time.time()
        elif time.time() - hand_open_start_time >= hand_open_duration:
            if auto_close_on_open:
                print(f"Hand has been open for {hand_open_duration} seconds. Closing the camera.")
                break
            else:
                # Auto-close is disabled: notify once and reset timer so we don't spam the log
                print(f"Hand has been open for {hand_open_duration} seconds — auto-close disabled. Continuing.")
                hand_open_start_time = None
    else:
        hand_open_start_time = None

    # Show the image
    cv2.imshow("Image", img)

    # Break the loop on space key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        break
    # Toggle auto-close at runtime with 'c'
    if key == ord('c'):
        auto_close_on_open = not auto_close_on_open
        print(f"Auto-close on open hand: {'enabled' if auto_close_on_open else 'disabled'}")

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
