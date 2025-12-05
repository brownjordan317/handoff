import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

from lidar_person_detection.dr_spaam.detector import Detector

from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray

TIMER = 1.0 / 20.0  # 20 Hz

CHECKPOINT_FILE = "/home/rosdev/ros2_ws/src/lidar_person_detection/lidar_person_detection/checkpoints/ckpt_jrdb_ann_ft_dr_spaam_e20.pth"

CLS_THRESH = 0.99

# Detection angle window (degrees) matches FOV of PiCAM
ANGLE_MIN_DEG = -230
ANGLE_MAX_DEG = -130
ANGLE_MIN = np.deg2rad(ANGLE_MIN_DEG)
ANGLE_MAX = np.deg2rad(ANGLE_MAX_DEG)

# New: Patience Filter Settings
PATIENCE_IN = 3      # must appear in 3 consecutive frames
PATIENCE_OUT = 5     # removed after missing for 5 frames
CELL_SIZE = 0.5      # meters – merge detections into spatial bins


class DetectorNode(Node):
    def __init__(self):
        super().__init__("dr_spaam_detector")

        self.detector = Detector(
            CHECKPOINT_FILE,
            model="DR-SPAAM",
            gpu=True,
            stride=1,
            panoramic_scan=True
        )
        self.detector.set_laser_fov(360.0)

        self.laser_scan = None

        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data
        )

        # Publishers
        self.pose_pub = self.create_publisher(PoseArray, "/person_detections", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/person_markers", 10)

        # New: dictionary for stable detection filtering
        self.stable_detections = {}  # key: (ix,iy), value: dict

        self.timer = self.create_timer(TIMER, self.main_callback)
        self.get_logger().info("DR-SPAAM Detector Node Initialized")

    # ------------------ Helpers ------------------
    def scan_callback(self, msg):
        self.laser_scan = msg

    def angle_in_range(self, angle, amin, amax):
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        amin = np.arctan2(np.sin(amin), np.cos(amin))
        amax = np.arctan2(np.sin(amax), np.cos(amax))

        if amin <= amax:
            return amin <= angle <= amax
        return angle >= amin or angle <= amax

    def convert_xy_to_polar(self, dets_xy):
        r = np.linalg.norm(dets_xy, axis=1)
        theta = np.arctan2(dets_xy[:, 1], dets_xy[:, 0])
        return r, theta

    def grid_key(self, x, y):
        return (int(x / CELL_SIZE), int(y / CELL_SIZE))

    # ------------------ Main Loop ------------------
    def main_callback(self):
        if self.laser_scan is None:
            return

        now = self.get_clock().now().to_msg()

        ranges = np.array(self.laser_scan.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = 0.0
        ranges[ranges < 0.01] = 0.01

        try:
            dets_xy, det_classes, instance_mask = self.detector(ranges)
        except Exception as e:
            self.get_logger().error(f"Detector error: {e}")
            return

        mask = det_classes > CLS_THRESH
        dets_xy = dets_xy[mask]
        det_classes = det_classes[mask]

        r_vals, th_vals = self.convert_xy_to_polar(dets_xy)

        # Track hits for this frame
        seen_keys = set()

        # Update detection entries
        for (x, y), r, th, conf in zip(dets_xy, r_vals, th_vals, det_classes):

            if not self.angle_in_range(th, ANGLE_MIN, ANGLE_MAX):
                continue

            key = self.grid_key(x, y)
            seen_keys.add(key)

            if key not in self.stable_detections:
                self.stable_detections[key] = {
                    "count_in": 1,
                    "count_out": 0,
                    "xy": (x, y),
                    "conf": conf
                }
            else:
                self.stable_detections[key]["count_in"] += 1
                self.stable_detections[key]["count_out"] = 0
                self.stable_detections[key]["xy"] = (x, y)
                self.stable_detections[key]["conf"] = conf

        # Increment "missing" counts and delete stale detections
        to_delete = []
        for key, entry in self.stable_detections.items():
            if key not in seen_keys:
                entry["count_out"] += 1
                if entry["count_out"] >= PATIENCE_OUT:
                    to_delete.append(key)

        for key in to_delete:
            del self.stable_detections[key]

        # ------------------ PoseArray Output ------------------
        pose_array = PoseArray()
        pose_array.header.frame_id = "laser"
        pose_array.header.stamp = now

        marker_array = MarkerArray()
        marker_id = 0

        for key, entry in self.stable_detections.items():
            if entry["count_in"] < PATIENCE_IN:
                continue  # not stable enough

            x, y = entry["xy"]
            conf = entry["conf"]

            # Pose output
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = float(conf)
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

            # Marker output
            marker = Marker()
            marker.header.frame_id = "laser"
            marker.header.stamp = now
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.0

            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3

            marker.color.r = float(1.0 - conf)
            marker.color.g = float(conf)
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.lifetime.sec = 1
            marker_array.markers.append(marker)
            marker_id += 1

        # Publish final filtered outputs
        self.pose_pub.publish(pose_array)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
