import mediapipe as mp
import cv2

class PoseTracking():
    def __init__(self):
        # Holistic model (pose + hands + face)
        self.mp_holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Gesture recognizer (same as your code)
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path="asl_letters_only.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )
        self.gesture_recognizer = GestureRecognizer.create_from_options(
            self.gesture_options
        )

        self.frame = None
        self.frame_rgb = None
        self.frame_counter = 0

        self.left_gesture = None
        self.right_gesture = None

    def holistic_tracking(self):
        result = self.mp_holistic.process(self.frame_rgb)

        # Pose
        if result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                self.frame,
                result.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0),
                                            thickness=2,
                                            circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0),
                                            thickness=2,
                                            circle_radius=2),
            )

        # Left hand
        if result.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                self.frame,
                result.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 255),
                                            thickness=2,
                                            circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255),
                                            thickness=2,
                                            circle_radius=2),
            )

        # Right hand
        if result.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                self.frame,
                result.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 255),
                                            thickness=2,
                                            circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255),
                                            thickness=2,
                                            circle_radius=2),
            )

        # Face (optional, comment out if you don’t need it)
        if result.face_landmarks:
            self.mp_drawing.draw_landmarks(
                self.frame,
                result.face_landmarks,
                mp.solutions.holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(color=(80, 110, 10),
                                            thickness=1,
                                            circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(80, 256, 121),
                                            thickness=1,
                                            circle_radius=1),
            )

    def gesture_tracking(self):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=self.frame_rgb)
        result = self.gesture_recognizer.recognize_for_video(
            mp_image,
            self.frame_counter
        )

        if result.gestures and result.handedness:
            for gesture_list, handedness_list in zip(result.gestures,
                                                     result.handedness):
                gesture = gesture_list[0]
                handedness = handedness_list[0]

                gesture_name = gesture.category_name
                score = gesture.score
                hand_label = handedness.category_name  # "Left" or "Right"

                if hand_label == "Left":
                    self.left_gesture = gesture_name
                else:
                    self.right_gesture = gesture_name

                print(f"{hand_label} hand: {gesture_name} ({score:.2f})")

    def run_tracking(self, frame):
        self.frame = frame
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_counter += 1

        # Holistic landmarks (pose + hands + face)
        self.holistic_tracking()

        # Gesture recognition
        self.gesture_tracking()

        return frame


if __name__ == '__main__':
    tracker = PoseTracking()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.run_tracking(frame)
        cv2.imshow('Holistic + Gestures', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
