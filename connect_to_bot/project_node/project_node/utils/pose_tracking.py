import mediapipe as mp
import cv2
import numpy as np
import json

class PoseTracking:
    def __init__(self, fps=30):
        # --- MediaPipe Hands Only ---
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawer = mp.solutions.drawing_utils

        # --- Gesture recognizer ---
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path="/home/rosdev/ros2_ws/src/project_node/project_node/utils/asl_letters_only.task",
                delegate=BaseOptions.Delegate.GPU,
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
        )
        self.gesture_recognizer = GestureRecognizer.create_from_options(
            self.gesture_options
        )

        # Frame state
        self.frame = None
        self.frame_rgb = None
        self.frame_counter = 0
        self.fps = fps

        # Hand smoothing state
        self.smoothing_factor = 0.1
        self.prev_left_hand = None
        self.prev_right_hand = None

        # Gesture states
        self.left_gesture = None
        self.right_gesture = None
        self.last_left = None
        self.last_right = None
        self.left_stable_counter = 0
        self.right_stable_counter = 0


    # -------------------------------
    # Landmark smoothing
    # -------------------------------
    def smooth_landmarks(self, landmarks, prev):
        coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
        if prev is not None:
            coords = (self.smoothing_factor * np.array(prev) +
                      (1 - self.smoothing_factor) * coords)
        return coords.tolist()

    def apply_smoothed(self, proto, coords):
        for i, lm in enumerate(proto.landmark):
            lm.x, lm.y, lm.z = coords[i]
        return proto


    # -------------------------------
    # Hand detection & drawing
    # -------------------------------
    def hand_tracking(self):
        result = self.hands.process(self.frame_rgb)

        if not result.multi_hand_landmarks:
            return

        for hand_lms, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = hand_info.classification[0].label  # "Left" or "Right"

            if label == "Left":
                smoothed = self.smooth_landmarks(hand_lms.landmark, self.prev_left_hand)
                self.prev_left_hand = smoothed
            else:
                smoothed = self.smooth_landmarks(hand_lms.landmark, self.prev_right_hand)
                self.prev_right_hand = smoothed

            # Apply smoothing
            hand_lms = self.apply_smoothed(hand_lms, smoothed)

            # Draw
            self.drawer.draw_landmarks(
                self.frame, hand_lms,
                mp.solutions.hands.HAND_CONNECTIONS,
                self.drawer.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                self.drawer.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )


    # -------------------------------
    # Gesture recognition
    # -------------------------------
    def gesture_tracking(self, flip_hands=True):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=self.frame_rgb
        )

        timestamp_ms = int(self.frame_counter * 1000 / self.fps)
        result = self.gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)

        if not (result.gestures and result.handedness):
            return

        for gesture_list, handedness_list in zip(result.gestures, result.handedness):
            gesture = gesture_list[0]
            handedness = handedness_list[0]

            gesture_name = gesture.category_name
            score = gesture.score
            hand_label = handedness.category_name  # "Left" / "Right"

            if flip_hands:
                hand_label = "Right" if hand_label == "Left" else "Left"

            if score < 0.7:
                continue

            if hand_label == "Left":
                if gesture_name != self.last_left:
                    self.left_stable_counter = 0
                else:
                    self.left_stable_counter += 1
                    if self.left_stable_counter > 3:
                        self.left_gesture = gesture_name
                self.last_left = gesture_name

            else:
                if gesture_name != self.last_right:
                    self.right_stable_counter = 0
                else:
                    self.right_stable_counter += 1
                    if self.right_stable_counter > 3:
                        self.right_gesture = gesture_name
                self.last_right = gesture_name


    # -------------------------------
    # Main update
    # -------------------------------
    def run_tracking(self, frame):
        self.frame = frame
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_counter += 1

        self.hand_tracking()
        self.gesture_tracking()

        return self.frame
