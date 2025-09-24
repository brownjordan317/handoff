import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np

class PersonDetection():
    def __init__(self, verbose, scale_percent, 
                 conf_level, thr_centers, 
                 frame_max, patience, alpha):
        self.verbose = verbose
        self.scale_percent = scale_percent
        self.conf_level = conf_level
        self.thr_centers = thr_centers # threshold ID for persistence
        self.frame_max = frame_max
        self.patience = patience
        self.alpha = alpha

        self.locked_id = None
        self.center_counts = {}  # Track how long each ID stayed in center
        self.fps = 15            # assume 30 FPS, can set dynamically
        self.lock_frames = 2 * self.fps  # frames needed to lock (5 seconds)

        self.class_IDS = [0] # person class only
        self.centers_old = {}
        self.lastKey = ''
        self.frame_num = 1
        self.model = YOLO('yolov8x.pt')

    def resize_frame(self, frame):
        """
        Resizes a given frame to a percentage of its original size.

        Parameters:
        frame (numpy.ndarray): The input frame to be resized.
        scale_percent (int): The percentage of the original frame size to 
        scale to.

        Returns:
        numpy.ndarray: The resized frame.
        """
        width = int(frame.shape[1] * self.scale_percent / 100)
        height = int(frame.shape[0] * self.scale_percent / 100)
        return cv2.resize(frame, 
                          (width, height), 
                          interpolation=cv2.INTER_AREA)

    def filter_tracks(self, centers, patience):
        """
        Filter out tracks which have not been seen in the last 'patience' 
        frames.

        Parameters:
        centers (dict): Dictionary of tracks, where each key is a track ID and
            the value is a dictionary of frame numbers to track centers.
        patience (int): The number of frames in which a track must be seen to 
            be considered active.

        Returns:
        dict: A dictionary of active tracks, where each key is a track ID and
            the value is a dictionary of frame numbers to track centers.
        """
        filter_dict = {}
        for k, i in centers.items():
            d_frames = i.items()
            filter_dict[k] = dict(list(d_frames)[-patience:])
        return filter_dict

    def update_tracking(self,
                        centers_old,
                        obj_center, 
                        thr_centers, 
                        lastKey, 
                        frame_num, 
                        frame_max):
        """
        Updates the tracking of detected objects.

        Parameters:
        centers_old (dict): Dictionary of tracks, where each key is a track ID
        and the value is a dictionary of frame numbers to track centers.
        obj_center (list): The center of the object currently being tracked.
        thr_centers (int): The maximum distance between the current object 
            center and the previous object center to be considered the same 
            object.
        lastKey (str): The ID of the object in the previous frame.
        frame_num (int): The current frame number.
        frame_max (int): The maximum number of frames to look back when 
            checking for previous object centers.
        """
        is_new = 0
        lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) 
                   for k, center in centers_old.items()]
        lastpos = [(i[0], i[2]) for i in lastpos 
                   if abs(i[1] - frame_num) <= frame_max]

        # Check distance from last known centers
        previous_pos = [(k, obj_center) for k, centers in lastpos 
                            if np.linalg.norm(np.array(centers) 
                                            - np.array(obj_center)) 
                                            < thr_centers]

        if previous_pos:
            id_obj = previous_pos[0][0]
            centers_old[id_obj][frame_num] = obj_center
        else:
            if lastKey:
                last = lastKey.split('D')[1]
                id_obj = 'ID' + str(int(last)+1)
            else:
                id_obj = 'ID0'
            is_new = 1
            centers_old[id_obj] = {frame_num: obj_center}
            lastKey = id_obj

        self.centers_old = centers_old
        self.id_obj = id_obj
        self.is_new = is_new
        self.lastKey = lastKey

    def predict_pedestrians(self, frame):
        """
        Predict pedestrians in a given frame using the YOLO model.

        Parameters:
            frame (numpy.ndarray): The input frame to detect pedestrians in.

        Returns:
            numpy.ndarray: The detected pedestrians in the frame.
        """
        y_hat = self.model.predict(frame,
                                conf=self.conf_level,
                                classes=self.class_IDS,
                                device="cpu",
                                verbose=False)

        if len(y_hat[0].boxes) == 0:
            return None
        return y_hat[0].boxes

    def format_detections(self, boxes_obj):
        """
        Formats the output of the YOLO model into a pandas DataFrame.

        Parameters:
            boxes_obj (torch.Tensor): The output of the YOLO model.

        Returns:
            pandas.DataFrame: A DataFrame containing the bounding box 
            coordinates, confidence scores, and class labels.
        """
        boxes = boxes_obj.xyxy.cpu().numpy()
        conf = boxes_obj.conf.cpu().numpy()
        classes = boxes_obj.cls.cpu().numpy()

        return pd.DataFrame(
            np.concatenate(
                [boxes, conf.reshape(-1, 1), classes.reshape(-1, 1)], 
                 axis=1
            ),
            columns=["xmin", "ymin", 
                     "xmax", "ymax", 
                     "conf", "class"],
        )

    def check_lock(self, frame, center_x, center_y):
        """
        Checks if an object is currently in the center of the frame and
        updates the tracking counts accordingly. If the object has been in
        the center for a certain number of frames (defined by lock_frames),
        the object is considered locked (i.e. self.locked_id is set to the
        object's ID).
        """
        if self.is_in_center(frame, center_x, center_y):
            self.center_counts[self.id_obj] = \
                self.center_counts.get(self.id_obj, 0) + 1
        else:
            self.center_counts[self.id_obj] = 0

        if self.center_counts[self.id_obj] >= self.lock_frames:
            self.locked_id = self.id_obj

    def draw_bbox(self, frame, xmin, ymin, xmax, ymax):
        """
        Draws a bounding box around the detected object in the frame.

        Parameters:
            frame (numpy.ndarray): The frame to draw the bounding box on.
            xmin (int): The minimum x-coordinate of the bounding box.
            ymin (int): The minimum y-coordinate of the bounding box.
            xmax (int): The maximum x-coordinate of the bounding box.
            ymax (int): The maximum y-coordinate of the bounding box.
        """
        color = (0, 255, 0) if self.locked_id == self.id_obj else (0, 0, 255)
        thickness = 3 if self.locked_id == self.id_obj else 2
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)

    def draw_trajectory(self, frame):
        """
        Draws a trajectory of the object's past positions in the frame.

        Parameters:
            frame (numpy.ndarray): The frame to draw the trajectory on.

        Notes:
            The color of the trajectory is green if the object is currently
            locked, and blue otherwise.
        """
        color = (0, 255, 0) if self.locked_id == self.id_obj else (0, 0, 255)
        for cx, cy in self.centers_old[self.id_obj].values():
            cv2.circle(frame,
                    (cx, cy),
                    int(5 * self.scale_percent / 100),
                    color,
                    -1)

    def draw_label(self, frame, xmin, ymin, conf):  
        """
        Draws a label on the detected object in the frame.

        Parameters:
            frame (numpy.ndarray): The frame to draw the label on.
            xmin (int): The minimum x-coordinate of the bounding box.
            ymin (int): The minimum y-coordinate of the bounding box.
            conf (float): The confidence score of the detected object.

        Notes:
            The color of the label is green if the object is currently locked,
            and blue otherwise.
        """
        color = (0, 255, 0) if self.locked_id == self.id_obj else (0, 0, 255)
        cv2.putText(
            frame,
            f"{self.id_obj}:{np.round(float(conf), 2)}",
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.8 * self.scale_percent / 100,
            color,
            1,
        )

    def draw_detections(self, frame, positions_frame):  
        """
        Draws the detected objects in a given frame.

        Parameters:
            frame (numpy.ndarray): The frame to draw the detected objects on.
            positions_frame (pandas.DataFrame): A DataFrame containing the 
                bounding box coordinates, confidence scores, and class labels 
                of the detected objects.

        Returns:
            numpy.ndarray: The frame with the detected objects drawn.

        Notes:
            If the object is currently locked, it will skip drawing all other 
            IDs. If the object is not currently locked, it will check if this 
            ID should be locked. After checking for locking, it will draw the 
            bounding box, trajectory, and label for each
            detected object.
        """
        for _, row in positions_frame.iterrows():
            xmin, ymin, xmax, ymax, _, _ = row.astype("int")
            center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

            # Update track ID
            self.update_tracking(
                self.centers_old,
                (center_x, center_y),
                self.thr_centers,
                self.lastKey,
                self.frame_num,
                self.frame_max,
            )

            # If locked, skip all other IDs
            if self.locked_id is not None and self.id_obj != self.locked_id:
                self.dist_from_center = self.image_center[0] - center_x
                print(f"X distance from center: {self.dist_from_center}")
                continue

            # If not locked yet → check if this ID should be locked
            if self.locked_id is None:
                self.check_lock(frame, center_x, center_y)

            # Draw bounding box, trajectory, and label
            self.draw_bbox(frame, xmin, ymin, xmax, ymax)
            self.draw_trajectory(frame)
            self.draw_label(frame, xmin, ymin, row["conf"])

        return frame

    def is_in_center(self, frame, center_x, center_y, margin=0.2):
        """
        Checks if a given center point is within the center of a frame
        (defined as the region within a certain margin of the frame's
        center point).

        Parameters:
        frame (numpy.ndarray): The frame to check against.
        center_x (int): The x-coordinate of the center point to check.
        center_y (int): The y-coordinate of the center point to check.
        margin (float): The margin as a fraction of the frame's size
            (default is 0.2).

        Returns:
        bool: True if the center point is within the center of the frame,
            False otherwise.
        """
        h, w = frame.shape[:2]
        if not hasattr(self, "x_min"):
            self.x_min = int(w*(0.5 - margin/2))
        if not hasattr(self, "x_max"):
            self.x_max = int(w*(0.5 + margin/2))
        if not hasattr(self, "y_min"):
            self.y_min = int(h*(0.5 - margin/2))
        if not hasattr(self, "y_max"):
            self.y_max = int(h*(0.5 + margin/2))
        return self.x_min <= center_x <= self.x_max and \
            self.y_min <= center_y <= self.y_max

    def detect_pedestrains(self, frame):
        """
        Detects pedestrians in a given frame and overlays tracking 
        information.

        Parameters:
            frame (numpy.ndarray): The input frame to detect pedestrians in.

        Returns:
            numpy.ndarray: The frame with bounding boxes and trajectories 
            drawn.
        """
        frame = self.resize_frame(frame)
        if not hasattr(self, 'image_center'):
            self.image_center = (frame.shape[1] // 2, 
                                 frame.shape[0] // 2)
        detections = self.predict_pedestrians(frame)
        if detections is None:
            return frame

        positions_frame = self.format_detections(detections)

        frame = self.draw_detections(frame, positions_frame)
        self.centers_old = self.filter_tracks(self.centers_old, self.patience)

        return frame

if __name__ == '__main__':
    verbose = False
    scale_percent = 25
    conf_level = 0.25
    thr_centers = 50  # threshold for ID persistence
    frame_max = 10
    patience = 100
    alpha = 0.3
    frame_skip = 5   # process every nth frame

    detector = PersonDetection(verbose, scale_percent, conf_level, 
                               thr_centers, frame_max, patience, alpha)
    cap = cv2.VideoCapture('rtsp://10.226.36.234:8080/h264_opus.sdp')

    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # skip frames that are not multiples of frame_skip
        if frame_counter % frame_skip != 0:
            continue

        frame = detector.detect_pedestrains(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        detector.frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

