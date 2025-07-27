import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Screen size
screen_width, screen_height = pyautogui.size()
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False  # CAREFUL with this!

# Smoothing parameters
SMOOTHING_WINDOW = 5
VELOCITY_SMOOTHING = 0.2
ACCELERATION_FACTOR = 1.5
cursor_history = deque(maxlen=SMOOTHING_WINDOW)
prev_cursor_pos = None
velocity = [0, 0]

# Kalman filter class
class SimpleKalman:
    def __init__(self):
        self.estimate = None
        self.estimate_error = 1.0
        self.process_noise = 0.01
        self.measurement_noise = 0.1

    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            return self.estimate
        pred_error = self.estimate_error + self.process_noise
        kalman_gain = pred_error / (pred_error + self.measurement_noise)
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.estimate_error = (1 - kalman_gain) * pred_error
        return self.estimate

x_filter = SimpleKalman()
y_filter = SimpleKalman()

# Distance function
def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Enhanced smooth movement function
def enhanced_smooth_move(target_x, target_y):
    global prev_cursor_pos, velocity

    filtered_x = x_filter.update(target_x)
    filtered_y = y_filter.update(target_y)

    if prev_cursor_pos is None:
        prev_cursor_pos = (filtered_x, filtered_y)
        pyautogui.moveTo(int(filtered_x), int(filtered_y))
        return

    current_velocity = [
        (filtered_x - prev_cursor_pos[0]) * ACCELERATION_FACTOR,
        (filtered_y - prev_cursor_pos[1]) * ACCELERATION_FACTOR
    ]

    velocity = [
        VELOCITY_SMOOTHING * current_velocity[0] + (1 - VELOCITY_SMOOTHING) * velocity[0],
        VELOCITY_SMOOTHING * current_velocity[1] + (1 - VELOCITY_SMOOTHING) * velocity[1]
    ]

    smooth_x = filtered_x + velocity[0]
    smooth_y = filtered_y + velocity[1]

    smooth_x = np.clip(smooth_x, 0, screen_width)
    smooth_y = np.clip(smooth_y, 0, screen_height)

    pyautogui.moveTo(int(smooth_x), int(smooth_y))
    prev_cursor_pos = (smooth_x, smooth_y)

# Main loop
def main():
    try:
        while cap.isOpened():
            start_time = time.time()
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    try:
                        landmarks = hand_landmarks.landmark

                        if len(landmarks) > 8:
                            index_tip = (int(landmarks[8].x * frame_width), int(landmarks[8].y * frame_height))
                            thumb_tip = (int(landmarks[4].x * frame_width), int(landmarks[4].y * frame_height))

                            x_normalized = landmarks[8].x
                            y_normalized = landmarks[8].y

                            # Virtual control box (map small region to full screen)
                            x_min, x_max = 0.3, 0.7
                            y_min, y_max = 0.2, 0.8

                            x_clamped = np.clip(x_normalized, x_min, x_max)
                            y_clamped = np.clip(y_normalized, y_min, y_max)

                            x_scaled = (x_clamped - x_min) / (x_max - x_min)
                            y_scaled = (y_clamped - y_min) / (y_max - y_min)

                            screen_x = x_scaled * screen_width
                            screen_y = y_scaled * screen_height

                            screen_x = np.clip(screen_x, 0, screen_width)
                            screen_y = np.clip(screen_y, 0, screen_height)

                            enhanced_smooth_move(screen_x, screen_y)

                            cv2.circle(frame, index_tip, 8, (0, 255, 0), -1)
                            cv2.circle(frame, thumb_tip, 8, (0, 0, 255), -1)

                            if distance(index_tip, thumb_tip) < 30:
                                pyautogui.click()
                                cv2.putText(frame, "CLICK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    except Exception as e:
                        print(f"Landmark processing error: {e}")
                        continue

            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Virtual Mouse', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
