"""
Real-time Hand Tracking using MediaPipe and OpenCV.

This script captures video from the default webcam, processes each frame
to detect and track hands using MediaPipe's Hand Landmark model, and
displays the results with hand landmarks and connections drawn over the
video feed.
"""

import cv2
import mediapipe as mp

# --- MediaPipe Solutions Imports ---
# Prefer importing the specific solutions/classes directly under a clear alias
# for better readability and to avoid long path names.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Configuration Constants ---
# Use constants for configuration values for easy adjustment and clarity
WEBCAM_INDEX = 0  # Default webcam index
WINDOW_NAME = "MediaPipe Hand Tracking"
EXIT_KEY = 'q'
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 0 # 0 for fast inference, 1 or 2 for higher accuracy (slower)


def run_hand_tracking_on_webcam():
    """
    Initializes webcam, sets up the MediaPipe Hand tracker, and runs the
    main loop for real-time hand detection and visualization.
    """
    print(f"👋 Starting hand tracking (Press '{EXIT_KEY}' to quit)...")

    # 1. Initialize Video Capture
    cam = cv2.VideoCapture(WEBCAM_INDEX)
    if not cam.isOpened():
        print(f"🚨 Error: Could not open webcam at index {WEBCAM_INDEX}.")
        return

    # 2. Initialize MediaPipe Hands Detector
    # The 'with' statement ensures resources (like the underlying TensorFlow graph)
    # are properly released when the context is exited.
    with mp_hands.Hands(
        model_complexity=MODEL_COMPLEXITY,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        # 3. Main Processing Loop
        while cam.isOpened():
            # Read frame from camera
            success, frame = cam.read()

            if not success:
                # This could happen if the camera is disconnected or has an internal error.
                print("⚠️ Warning: Empty frame received. Skipping frame.")
                # We should attempt to wait for a moment before continuing/breaking.
                # For now, let's just continue the loop.
                continue

            # a. Performance Optimization: Mark frame as not writeable to pass by reference.
            # This is a MediaPipe recommendation to improve performance.
            frame.flags.writeable = False

            # b. Color Conversion (BGR to RGB)
            # MediaPipe expects RGB input, while OpenCV captures in BGR.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # c. Process the frame
            results = hands.process(frame_rgb)

            # d. Preparation for Drawing (Mark frame as writeable again)
            # Must be writeable for cv2.putText and mp_drawing.draw_landmarks.
            frame.flags.writeable = True

            # e. Draw Hand Landmarks
            if results.multi_hand_landmarks:
                # 
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the 21 hand landmarks and connections
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        # Use default styles for a clean visualization
                        landmark_drawing_spec=\
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=\
                            mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # f. Display the Result
            # cv2.flip(frame, 1) flips the image horizontally
            # (mirror effect)
            # which is typical for webcam feeds and intuitive for
            # hand tracking.
            cv2.imshow(WINDOW_NAME, cv2.flip(frame, 1))

            # g. Exit Condition
            # Check for the exit key 'q' (0xFF is a mask for 8-bit systems)
            if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
                break

    # 4. Cleanup Resources
    print("🎬 Stopping video stream and closing window...")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hand_tracking_on_webcam()