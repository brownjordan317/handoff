import mediapipe as mp
import cv2
import numpy as np

class PoseTracking():
    def __init__(self, fps=30):
        # Holistic
        self.mp_holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            refine_face_landmarks=False
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Gesture recognizer
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path="asl_letters_only.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
        )
        self.gesture_recognizer = GestureRecognizer.create_from_options(self.gesture_options)

        # State
        self.frame = None
        self.frame_rgb = None
        self.frame_counter = 0
        self.fps = fps

        # Gesture state
        self.left_gesture = None
        self.right_gesture = None
        self.last_left = None
        self.left_stable_counter = 0
        self.last_right = None 
        self.right_stable_counter = 0

        # Smoothing + persistence
        self.smoothing_factor = 0.1

        self.prev_pose_landmarks = None
        self.prev_left_hand_landmarks = None
        self.prev_right_hand_landmarks = None

        self.last_pose = None
        self.pose_miss_counter = 0
        self.last_left_hand = None
        self.left_hand_miss_counter = 0
        self.last_right_hand = None
        self.right_hand_miss_counter = 0

    def smooth_landmarks(self, landmarks, prev_landmarks):
        coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
        if prev_landmarks is not None:
            coords = (self.smoothing_factor * np.array(prev_landmarks) +
                      (1 - self.smoothing_factor) * coords)
        return coords.tolist()

    def apply_smoothed_to_proto(self, landmarks_proto, smoothed_coords):
        for i, lm in enumerate(landmarks_proto.landmark):
            lm.x, lm.y, lm.z = smoothed_coords[i]
        return landmarks_proto

    def holistic_tracking(self):
        result = self.mp_holistic.process(self.frame_rgb)

        if result.pose_landmarks:
            smoothed = self.smooth_landmarks(result.pose_landmarks.landmark,
                                             self.prev_pose_landmarks)
            result.pose_landmarks = self.apply_smoothed_to_proto(result.pose_landmarks, smoothed)
            self.prev_pose_landmarks = smoothed
            self.last_pose = result.pose_landmarks
            self.pose_miss_counter = 0

            self.mp_drawing.draw_landmarks(
                self.frame, result.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            )
        else:
            if self.last_pose:
                self.pose_miss_counter += 1
                if self.pose_miss_counter < 5:
                    self.mp_drawing.draw_landmarks(
                        self.frame, self.last_pose, mp.solutions.holistic.POSE_CONNECTIONS
                    )

        if result.left_hand_landmarks:
            smoothed = self.smooth_landmarks(result.left_hand_landmarks.landmark,
                                             self.prev_left_hand_landmarks)
            result.left_hand_landmarks = self.apply_smoothed_to_proto(result.left_hand_landmarks, smoothed)
            self.prev_left_hand_landmarks = smoothed
            self.last_left_hand = result.left_hand_landmarks
            self.left_hand_miss_counter = 0

            self.mp_drawing.draw_landmarks(
                self.frame, result.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
        else:
            if self.last_left_hand:
                self.left_hand_miss_counter += 1
                # if self.left_hand_miss_counter < 5:
                #     self.mp_drawing.draw_landmarks(
                #         self.frame, self.last_left_hand, mp.solutions.holistic.HAND_CONNECTIONS
                #     )

        if result.right_hand_landmarks:
            smoothed = self.smooth_landmarks(result.right_hand_landmarks.landmark,
                                             self.prev_right_hand_landmarks)
            result.right_hand_landmarks = self.apply_smoothed_to_proto(result.right_hand_landmarks, smoothed)
            self.prev_right_hand_landmarks = smoothed
            self.last_right_hand = result.right_hand_landmarks
            self.right_hand_miss_counter = 0

            self.mp_drawing.draw_landmarks(
                self.frame, result.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
        else:
            if self.last_right_hand:
                self.right_hand_miss_counter += 1
                # if self.right_hand_miss_counter < 5:
                #     self.mp_drawing.draw_landmarks(
                #         self.frame, self.last_right_hand, mp.solutions.holistic.HAND_CONNECTIONS
                #     )

    def gesture_tracking(self):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.frame_rgb)
        timestamp_ms = int(self.frame_counter * 1000 / self.fps)
        result = self.gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)

        if result.gestures and result.handedness:
            for gesture_list, handedness_list in zip(result.gestures, result.handedness):
                gesture = gesture_list[0]
                handedness = handedness_list[0]

                gesture_name = gesture.category_name
                score = gesture.score
                hand_label = handedness.category_name

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

                print(f"{hand_label} hand: {gesture_name} ({score:.2f})")

    def run_tracking(self, frame):
        self.frame = frame
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_counter += 1

        self.holistic_tracking()
        self.gesture_tracking()
        return frame
