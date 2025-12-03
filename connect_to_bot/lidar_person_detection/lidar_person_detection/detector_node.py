
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray

from lidar_person_detection.dr_spaam.detector import Detector

TIMER = 1.0 / 10.0  # 10 Hz
CHECKPOINT_FILE = '/home/rosdev/ros2_ws/src/lidar_person_detection/lidar_person_detection/checkpoints/ckpt_jrdb_ann_ft_dr_spaam_e20.pth'
CLS_THRESH = 0.7

class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        self.detector = Detector(
            CHECKPOINT_FILE, 
            model="DR-SPAAM", 
            gpu=True, 
            stride=1, 
            panoramic_scan=False
        )
        self.detector.set_laser_fov(360.0)  # Set FOV to 360 degrees
        
        self.detections=[]

        self.laser_scan = None

        self.create_subs()
        self.main_timer = self.create_timer(
            timer_period_sec=TIMER,
            callback=self.main_callback
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, 
            "/person_markers", 
            10)
    
    def create_subs(self):
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data
        )
    
    def scan_callback(self, msg):
        # Process the LaserScan message
        # self.get_logger().info('Received LaserScan data')
        self.laser_scan = msg

    def image_callback(self, msg):
        # Process the Image message
        # self.get_logger().info('Received Image data')
        # Image processing logic goes here
        pass

    def convert_detections_to_polar(self, dets_xy):
        dets_r = np.sqrt(dets_xy[:, 0] ** 2 + dets_xy[:, 1] ** 2)
        dets_phi = np.arctan2(dets_xy[:, 1], dets_xy[:, 0])
        return zip(dets_r, dets_phi)

    def main_callback(self):
        if self.laser_scan is not None:
            # Perform detection using the laser scan data
            # self.get_logger().info('Performing detection on LaserScan data')
            ranges = np.array(self.laser_scan.ranges, dtype=np.float32)
            ranges = np.array(self.laser_scan.ranges, dtype=np.float32)

            # Replace invalid values
            ranges[~np.isfinite(ranges)] = 0.0        # replace nan/inf with 0
            ranges[ranges < 0.01] = 0.01             # min range (prevents log/ratio issues)

            dets_xy, dets_cls, instance_mask = self.detector(ranges)
            cls_mask = dets_cls > CLS_THRESH
            dets_xy = dets_xy[cls_mask]
            dets_cls = dets_cls[cls_mask]
            self.get_logger().info(f'Detections: {len(dets_xy)} persons detected')
            dets_rtheta = self.convert_detections_to_polar(dets_xy)
            for r, theta in dets_rtheta:
                self.get_logger().info(f'Detection: {r:.2f} m at {np.rad2deg(theta):.2f} degrees')

            self.publish_markers(dets_xy)

            # self.get_logger().info(f'Detection Classes: {dets_cls}')

    def publish_markers(self, dets_xy):
        marker_array = MarkerArray()
        marker_id = 0

        # Clear old markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # Create a marker for each detection
        for x, y in dets_xy:
            # Sphere marker
            m = Marker()
            m.header.frame_id = "base_link"   # or "laser" depending on your TF
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "person_detection"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.0

            m.scale.x = 0.3   # sphere diameter
            m.scale.y = 0.3
            m.scale.z = 0.3

            # red sphere
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0

            marker_array.markers.append(m)

            # Text marker (label)
            txt = Marker()
            txt.header.frame_id = "base_link"
            txt.header.stamp = self.get_clock().now().to_msg()
            txt.ns = "person_labels"
            txt.id = marker_id + 1000
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD

            txt.pose.position.x = float(x)
            txt.pose.position.y = float(y)
            txt.pose.position.z = 0.5    # above sphere

            txt.scale.z = 0.3  # text height

            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 1.0

            txt.text = f"({x:.1f}, {y:.1f})"

            marker_array.markers.append(txt)

            marker_id += 1

        self.marker_pub.publish(marker_array)



def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()