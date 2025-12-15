from project_node.utils.person_tracking import PersonTracking

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String

import numpy as np

IMAGE_RECV_TOPIC = "/camera/image_raw"
# IMAGE_RECV_TOPIC = "/shiv_1/camera/image"
IMAGE_PUB_TOPIC = "jordansProject/detections/image"
IMAGE_SCALE_PERCENT = 100
FPS = 30.0
PATIENCE = 3
TRAILING_DISTANCE = 0.5

class FollowerNode(Node):
    def __init__(self):
        super().__init__('follower_node')
        self.set_variables()
        self.initialize_ros2()
        self.intitialize_detector()   

    def set_variables(self):
        self.bridge = CvBridge()
        self.image = None
        self.target = None

    def create_subs(self):
        """
        Create ROS2 subscriptions to the image and pose array topics.
        """
        self.image_sub = \
            self.create_subscription(Image, 
                                     topic = IMAGE_RECV_TOPIC, 
                                     callback = self.set_image, 
                                     qos_profile = \
                                        rclpy.qos.qos_profile_sensor_data)
        
        self.pose_array_sub = \
            self.create_subscription(PoseArray, 
                                     topic = "/person_detections", 
                                     callback = self.set_pose_array, 
                                     qos_profile = \
                                        rclpy.qos.qos_profile_sensor_data)

    def create_pubs(self):
        """
        Create ROS2 publishers to the predicted image, motion, and hand 
        detection topics.
        """
        self.pred_image_pub = self.create_publisher(Image,
                                                   topic = IMAGE_PUB_TOPIC,
                                                   qos_profile = 1)
        
        self.motion_pub = self.create_publisher(Twist, 
                                                   '/cmd_vel', 
                                                   10)
        
        self.hand_detecion_pub = self.create_publisher(String,
                                                   topic = "hand_detections",
                                                   qos_profile = 1)

    def initialize_ros2(self):
        """
        Initialize ROS2 subscriptions and publishers, and start the matching 
        timer.

        This method creates subscriptions to the image and pose array topics,
        and publishers to the predicted image, motion, and hand detection 
        topics. It also starts the matching timer, which runs the matching 
        algorithm at a rate of {FPS} Hz.

        """
        self.create_subs()
        self.create_pubs()
        
        self.matching_timer = self.create_timer(timer_period_sec = 1.0 / FPS, 
                                                callback = self.run)
        
    def intitialize_detector(self):
        """
        Initialize the person tracking detector with the given parameters.

        Parameters
        ----------
        scale_percent : int
            The scale percentage of the image to detect people.
        conf_level : float
            The confidence level of the detector.
        fps : int
            The frames per second of the detector.
        patience : int
            The number of frames to wait for a person to be detected before
            locking on to the person.

        Returns
        -------
        None
        """
        self.detector = PersonTracking(scale_percent = IMAGE_SCALE_PERCENT, 
                                        conf_level = 0.25,
                                        fps = FPS,
                                        patience = PATIENCE)

    def set_image(self, msg: Image):
        """
        Set the current image to the given ROS2 Image message.

        Parameters
        ----------
        msg : Image
            The ROS2 Image message to set as the current image.

        Returns
        -------
        None
        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, 
                                             desired_encoding = "bgr8")
        self.image = cv_image

    def calc_best_target(self, msg):
        """
        Compute the best target based on the given poses.

        If no poses are given, then the target is set to None.
        Otherwise, the function extracts the x and y coordinates of the poses, 
        computes the angles in degrees (0–360), and computes the distances of 
        the poses from the origin.The function then masks out the detections 
        that are not within the angle window of 165 to 205 degrees. If no 
        detections are left after masking, then the target is set to None. 
        Otherwise, the function finds the index of the closest target within 
        the mask, and saves the final target pose.

        Parameters
        ----------
        msg : PoseArray
            The ROS2 PoseArray message containing the poses of the detections.

        Returns
        -------
        None
        """
        if len(msg.poses) == 0:
            self.target = None
            return

        # Extract x and y into NumPy arrays
        xs = np.array([p.position.x for p in msg.poses])
        ys = np.array([p.position.y for p in msg.poses])

        # Compute angles in degrees (0–360)
        angles = np.degrees(np.arctan2(ys, xs))
        angles = np.where(angles < 0, angles + 360, angles)

        # Compute distances
        dists = np.sqrt(xs**2 + ys**2)

        # Mask: only keep detections in angle window
        mask = (angles >= 165) & (angles <= 205)

        if not np.any(mask):
            self.target = None
            return

        # Index of closest target *within mask*
        idx_in_mask = np.argmin(dists[mask])
        true_idx = np.arange(len(msg.poses))[mask][idx_in_mask]

        # Save final target pose
        self.target = msg.poses[true_idx]

    def set_pose_array(self, msg: PoseArray):
        """
        Set the last pose array and calculate the best target based on the 
        given poses.

        Parameters
        ----------
        msg : PoseArray
            The ROS2 PoseArray message containing the poses of the detections.
        """
        self.last_pose_array = msg
        self.calc_best_target(msg)

    def publish_pred_image(self, image):
        """
        Publish the detected image to the predicted image topic.

        Parameters
        ----------
        image : numpy.ndarray
            The detected image to publish.

        Returns
        -------
        None
        """
        msg = self.bridge.cv2_to_imgmsg(image, 
                                        encoding = "bgr8")
        self.pred_image_pub.publish(msg)

    def run(self):
        """
        Publish the detected image and control command based on the detected 
        pedestrian if the image is not None.

        If the detected pedestrian is in the center of the frame, calculate 
        the turn speed based on the distance from the center of the frame to 
        the center of the bounding box. Then, publish a Twist message with the
        calculated turn speed to the spot turn topic.
        """
        if self.image is not None:
            image = self.detector.detect_pedestrians(self.image)
            # cv2.imshow('image', image)
            # cv2.waitKey(1)
            self.publish_pred_image(
                image
                )
            
            twist_msg = Twist()

            self.hand_detecion_pub.publish(
                String(
                    data=f"Left: {self.detector.pose_tracker.left_gesture}" + 
                    f", Right: {self.detector.pose_tracker.right_gesture}"
                )
            )

            if self.detector.turn_direction is not None \
                and self.detector.mc_number is not None \
                    and self.detector.dist_x is not None\
                        and self.detector.state == "locked":
                
                turn_speed = self.detector.mc_number if \
                    self.detector.turn_direction == "right" else \
                        -self.detector.mc_number
                # clip turn speed between -1 and 1
                # turn_speed = max(min(turn_speed, 1.0), -1.0)
                twist_msg.angular.z = float(turn_speed)
                self.motion_pub.publish(twist_msg)
            else:
                twist_msg.angular.z = 0.0
            self.motion_pub.publish(twist_msg)
            if twist_msg.angular.z == 0.0 and self.detector.state == "locked":
                if self.target is not None:
                    distance = -(self.target.position.x + 0.25)

                    if distance > TRAILING_DISTANCE:
                        twist_msg.linear.x = 0.7
                        print("TRAILING")
                    else:
                        twist_msg.linear.x = 0.0
                        print("STOP")
                    self.motion_pub.publish(twist_msg)

            if self.detector.hand_controls.drive_command is not None and \
                self.detector.state == "locked":
                drive_cmd = self.detector.hand_controls.drive_command
                if drive_cmd == "Unlock":
                    self.detector.unlock()
                    self.detector.hand_controls.drive_command = None
                    print("UNLOCKED")
                    return

                if drive_cmd == "FORWARD":
                    twist_msg.linear.x = 0.5
                elif drive_cmd == "LEFT":
                    twist_msg.angular.z = 1.0
                elif drive_cmd == "RIGHT":
                    twist_msg.angular.z = -1.0
                else:  # STOP or unrecognized command
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                
                self.motion_pub.publish(twist_msg)
                

def main(args=None):
    rclpy.init(args=args)
    node = FollowerNode()
    rclpy.spin(node)
    rclpy.shutdown()