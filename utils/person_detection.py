import cv2
from ultralytics import YOLO
import numpy as np
import torch
from collections import deque

class PersonDetection():
    def __init__(self, 
                 scale_percent, 
                 conf_level, 
                 ):
        self.scale_percent = scale_percent
        self.conf_level = conf_level


        self.locked_id = None
        self.center_counts = {}
        self.fps = 30
        self.lock_frames = 2 * self.fps  # 2 seconds to lock
        self.trajectories = {}

        self.class_IDS = [0]  # person only
        self.frame_num = 1
        self.model = YOLO('yolov8n.pt')

    def resize_frame(self, frame):
        width = int(frame.shape[1] * self.scale_percent / 100)
        height = int(frame.shape[0] * self.scale_percent / 100)
        return cv2.resize(frame, 
                          (width, height), 
                          interpolation=cv2.INTER_AREA)

    def track_pedestrians(self, frame):
        results = self.model.track(frame,
                                   conf=self.conf_level,
                                   classes=self.class_IDS,
                                   persist=True,
                                   verbose=False,
                                   device="cpu")
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        return results[0].boxes

    def check_lock(self, frame, center_x, center_y, track_id):
        if self.is_in_center(frame, center_x, center_y):
            self.center_counts[track_id] = self.center_counts.get(track_id
                                                                  , 0) + 1
        else:
            self.center_counts[track_id] = 0

        if self.center_counts[track_id] >= self.lock_frames:
            self.locked_id = track_id

    def update_trajectories(self, track_id, center):
        if track_id not in self.trajectories:
            self.trajectories[track_id] = deque(maxlen=200)
        self.trajectories[track_id].append(center)

    def handle_locking(self, frame, center_x, center_y, track_id):
        """
        Returns the state of the track:
        - "locked"
        - "candidate"
        - "unlocked"
        """
        if self.locked_id is None:
            if self.is_in_center(frame, center_x, center_y):
                self.check_lock(frame, center_x, center_y, track_id)
                return "candidate"
            else:
                return "unlocked"
        elif self.locked_id == track_id:
            dist_x = center_x - self.image_center[0]
            print(f"Locked ID {track_id} | X distance from center: {dist_x}")
            return "locked"
        return "unlocked"

    def draw_box_and_traj(self, frame, track_id, box, color):
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

        # Draw trajectory points
        for (cx, cy) in self.trajectories[track_id]:
            cv2.circle(frame, (cx, cy), 3, color, -1)

    def recover_lost_lock(self, active_ids):
        if not hasattr(self, "lost_counter"):
            self.lost_counter = 0

        if self.locked_id is not None and self.locked_id not in active_ids:
            self.lost_counter += 1

            # Nearest-ID recovery
            if self.trajectories.get(self.locked_id):
                last_x, last_y = self.trajectories[self.locked_id][-1]
                min_dist, new_lock = float("inf"), None

                for tid in active_ids:
                    cx, cy = self.trajectories[tid][-1]
                    dist = np.hypot(cx - last_x, cy - last_y)
                    if dist < min_dist:
                        min_dist, new_lock = dist, tid

                if new_lock is not None and min_dist < 100:  # px threshold
                    print(f"Re-locking from {self.locked_id} to \
                          {new_lock} (nearest match)")
                    self.locked_id = new_lock
                    self.lost_counter = 0
                    return

            # Unlock if gone too long
            if self.lost_counter > 90:  # patience (frames)
                print(f"Lost lock on {self.locked_id}, resetting...")
                self.locked_id = None
                self.lost_counter = 0
        else:
            self.lost_counter = 0

    def draw_detections(self, frame, boxes_obj):
        active_ids = set()
        
        for box in boxes_obj:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            track_id = int(box.id[0]) if box.id is not None else -1
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Skip everyone except locked target
            if self.locked_id is not None and track_id != self.locked_id:
                continue

            # Update trajectories
            self.update_trajectories(track_id, (center_x, center_y))
            active_ids.add(track_id)

            # Handle lock logic and get state
            state = self.handle_locking(frame, center_x, center_y, track_id)

            # Assign colors by state
            if state == "locked":
                color = (0, 255, 0)   # green
            elif state == "candidate":
                color = (0, 255, 255) # yellow
            else:
                color = (0, 0, 255)   # red

            # Draw box + trajectory
            self.draw_box_and_traj(frame, track_id, box, color)

            if not state == "locked":
                # Draw lock box
                cv2.rectangle(frame, 
                            (self.x_min, self.y_min - 10), 
                            (self.x_max, self.y_max + 10), 
                            (125, 125, 125), 
                            2)

        # Handle lost lock recovery
        self.recover_lost_lock(active_ids)

        return frame

    def is_in_center(self, frame, center_x, center_y, margin=0.25):
        if not hasattr(self, "h") or not hasattr(self, "w"):
            self.h, self.w = frame.shape[:2]
        if not hasattr(self, "x_min"):
            self.x_min = int(self.w * (0.5 - margin / 2))
        if not hasattr(self, "x_max"):
            self.x_max = int(self.w * (0.5 + margin / 2))
        self.y_min = 0
        self.y_max = self.h
        return self.x_min <= center_x <= self.x_max and \
               self.y_min <= center_y <= self.y_max

    def detect_pedestrians(self, frame):
        frame = self.resize_frame(frame)
        if not hasattr(self, 'image_center'):
            self.image_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        boxes = self.track_pedestrians(frame)
        if boxes is None:
            return frame

        frame = self.draw_detections(frame, boxes)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return frame


if __name__ == '__main__':
    detector = PersonDetection(25, 0.50)
    cap = cv2.VideoCapture('rtsp://10.226.36.234:8080/h264_opus.sdp')
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
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        detector.frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
