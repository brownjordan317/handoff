import cv2
from ultralytics import YOLO
import numpy as np
from tabulate import tabulate
import torch

from project_node.utils.pose_tracking import PoseTracking
from project_node.utils.hand_controls import HandControls

class PersonTracking():
    def __init__(self, 
                 scale_percent, 
                 conf_level, 
                 fps,
                 patience
                 ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.scale_percent = scale_percent
        self.conf_level = conf_level
        
        self.turn_direction = None
        self.mc_number = None
        self.dist_x = None

        self.locked_id = None
        self.center_counts = {}
        self.fps = fps
        self.lock_frames = patience * self.fps  # 2 seconds to lock
        self.patience = patience * self.fps
        self.missing_frames = 0   # counts how many frames locked target is missing
        self.state = "unlocked"

        self.class_IDS = [0]  # person only
        self.frame_num = 1
        self.model = YOLO('yolo11m')

        self.pose_tracker = PoseTracking()
        self.hand_controls = HandControls()
        self.drive_command = None

    def draw_boxj(self, frame, track_id, box, color):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    f"ID {track_id}:{conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

        if not self.state == "locked":
            # Draw center reticle
            cv2.rectangle(frame, 
                        (self.x_min, self.y_min - 10), 
                        (self.x_max, self.y_max + 10), 
                        (125, 125, 125), 
                        2)

    def resize_frame(self, frame):
        frame = cv2.flip(frame, 1)
        width = int(frame.shape[1] * self.scale_percent / 100)
        height = int(frame.shape[0] * self.scale_percent / 100)
        return cv2.resize(frame, 
                          (width, height), 
                          interpolation=cv2.INTER_AREA)

    def find_people(self, frame):
        results = self.model.track(
            frame,
            conf=self.conf_level,
            classes=self.class_IDS,
            persist=True,
            verbose=False,
            device=self.device,
        )
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        return results[0].boxes

    def check_lock(self, frame, center_x, center_y, track_id):
        if self.is_in_center(frame, center_x, center_y):
            self.center_counts[track_id] = self.center_counts.get(track_id, 
                                                                  0) + 1
        else:
            self.center_counts[track_id] = 0

        if self.center_counts[track_id] >= self.lock_frames:
            self.locked_id = track_id

    def handle_locking(self, frame, center_x, center_y, track_id):
        if self.locked_id is None:
            if self.is_in_center(frame, center_x, center_y):
                self.check_lock(frame, center_x, center_y, track_id)
                return "candidate"
            else:
                self.mc_number = 0.0
                return "unlocked"
        elif self.locked_id == track_id:
            self.dist_x = center_x - self.image_center[0]
            thr = self.image_center[0] - self.x_min
            max_val = self.image_center[1]
            dist = abs(self.dist_x)

            if dist <= thr:
                self.mc_number = 0
            else:
                self.mc_number = 0.3 + 1 * ((dist - thr) / (max_val - thr))
                self.mc_number = min(self.mc_number, 1.3)

            if self.dist_x < 0:
                self.turn_direction = "left"
            else:
                self.turn_direction = "right"

            # data = [[track_id, self.dist_x, 
            #          self.mc_number, self.turn_direction]]
            # headers = ["Locked ID", "X Distance", 
            #            "Motor Speed", "Turn Direction"]

            # print("\033[H\033[J", end="")  # ANSI clear
            # print(tabulate(data, headers=headers, tablefmt="fancy_grid"))
            
            return "locked"
        self.mc_number = 0.0
        return "unlocked"
    
    def expand_box(self, box, frame_shape, 
                   expand_x_ratio=0.05, 
                   expand_y_ratio=0.1):
        h, w, _ = frame_shape
        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1

        expand_x = int(box_w * expand_x_ratio)
        expand_y = int(box_h * expand_y_ratio)

        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(w, x2 + expand_x)
        y2 = min(h, y2 + expand_y)
        return x1, y1, x2, y2

    def apply_target_mask(self, frame, crop, box):
        x1, y1, x2, y2 = box
        frame[y1:y2, x1:x2] = crop
        return frame

    def process_locked_target(self, frame, box):
        expanded_box = self.expand_box(box, frame.shape)
        x1, y1, x2, y2 = expanded_box

        locked_crop = frame[y1:y2, x1:x2]
        locked_crop = self.pose_tracker.run_tracking(locked_crop)

        return self.apply_target_mask(frame, locked_crop, expanded_box)


    def update_lock_state(self, seen_ids):
        if self.locked_id is not None:
            if self.locked_id not in seen_ids:
                self.missing_frames += 1
                # wait up to self.patience frames before unlocking
                if self.missing_frames >= self.patience:
                    self.locked_id = None
                    self.state = "unlocked"
                    self.missing_frames = 0
            else:
                # reset counter if seen again
                self.missing_frames = 0

    def draw_detections(self, frame, boxes_obj):
        seen_ids = set()

        # Case 1: Not locked yet draw all boxes, update lock state
        if self.locked_id is None:
            for box in boxes_obj:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                track_id = int(box.id[0]) if box.id is not None else -1
                seen_ids.add(track_id)

                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                self.state = self.handle_locking(
                    frame, 
                    center_x, 
                    center_y, 
                    track_id
                )

                # Draw candidate/unlocked boxes
                color = {
                    "candidate": (0, 255, 255),  # yellow when in lock region
                    "locked": (0, 255, 0),       # (rare, just transitioned)
                }.get(self.state, (0, 0, 255))   # red otherwise

                self.draw_boxj(frame, track_id, box, color)

        # Case 2: Locked process ONLY the locked ID
        else:
            for box in boxes_obj:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                track_id = int(box.id[0]) if box.id is not None else -1
                seen_ids.add(track_id)

                # Skip all non-locked IDs immediately
                if track_id != self.locked_id:
                    continue  

                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                self.state = self.handle_locking(
                    frame, 
                    center_x, 
                    center_y, 
                    track_id
                )

                # Process pose tracking for locked target only
                frame = self.process_locked_target(frame, (x1, y1, x2, y2))

                # Draw only the locked target’s box in green
                self.draw_boxj(frame, track_id, box, (0, 255, 0))
                break  # stop after processing the locked target

        # If locked target disappears reset after patience
        self.update_lock_state(seen_ids)
        return frame


    def is_in_center(self, frame, center_x, center_y, margin=0.1):
        if not hasattr(self, "h") or not hasattr(self, "w"):
            self.h, self.w = frame.shape[:2]
        if not hasattr(self, "x_min"):
            self.x_min = int(self.w * (0.5 - margin / 2))
        if not hasattr(self, "x_max"):
            self.x_max = int(self.w * (0.5 + margin / 2))
        if not hasattr(self, "y_min"):
            self.y_min = int(self.h * (0.5 - margin / 2))
        if not hasattr(self, "y_max"):
            self.y_max = int(self.h * (0.5 + margin / 2))
        return self.x_min <= center_x <= self.x_max and \
               self.y_min <= center_y <= self.y_max

    def detect_pedestrians(self, frame):
        frame = self.resize_frame(frame)
        if not hasattr(self, 'image_center'):
            self.image_center = (frame.shape[1] // 2, 
                                 frame.shape[0] // 2)

        boxes = self.find_people(frame)
        if boxes is None:
            self.update_lock_state(set())
            return frame

        frame = self.draw_detections(frame, boxes)

        self.hand_controls.call_controls(
            self.pose_tracker.left_gesture, 
            self.pose_tracker.right_gesture
        )

        self.drive_command = self.hand_controls.drive_command
        return frame

if __name__ == '__main__':
    detector = PersonTracking(100, 0.50, 15, 3)
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    frame_skip = 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter + frame_skip)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        frame = detector.detect_pedestrians(frame)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        detector.frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
