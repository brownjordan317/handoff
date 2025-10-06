from project_node.utils.person_tracking import PersonTracking

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

IMAGE_RECV_TOPIC = "shiv_1/camera/image"
IMAGE_PUB_TOPIC = "jordansProject/detections/image"
FPS = 15
PATIENCE = 3

class FollowerNode(Node):
    def __init__(self):
        super().__init__('follower_node')
        self.set_variables()
        self.initialize_ros2()
        self.intitialize_detector()   

    def set_variables(self):
        self.bridge = CvBridge()
        self.image = None

    def create_subs(self):
        self.image_sub = \
            self.create_subscription(Image, 
                                     topic = IMAGE_RECV_TOPIC, 
                                     callback = self.set_image, 
                                     qos_profile = \
                                        rclpy.qos.qos_profile_sensor_data)

    def create_pubs(self):
        self.pred_image_pub = self.create_publisher(Image,
                                                   topic = IMAGE_PUB_TOPIC,
                                                   qos_profile = 1)
        
        self.spot_turn_pub = self.create_publisher(Twist, 
                                                   '/cmd_vel', 
                                                   10)

    def initialize_ros2(self):
        self.create_subs()
        self.create_pubs()
        
        self.matching_timer = self.create_timer(timer_period_sec = 1 / FPS, 
                                                callback = self.run)
        
    def intitialize_detector(self):
        self.detector = PersonTracking(scale_percent = 25, 
                                        conf_level = 0.25,
                                        fps = FPS,
                                        patience = PATIENCE)

    def set_image(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 
                                             desired_encoding = "rgb8")
        self.image = cv_image

    def publish_pred_image(self, image):
        msg = self.bridge.cv2_to_imgmsg(image, 
                                        encoding = "rgb8")
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
            self.publish_pred_image(
                self.detector.detect_pedestrians(self.image)
                )

            if self.detector.turn_direction is not None \
                and self.detector.mc_number is not None \
                    and self.detector.dist_x is not None:
                twist_msg = Twist()
                turn_speed = self.detector.mc_number if \
                    self.detector.turn_direction == "right" else \
                        -self.detector.mc_number
                twist_msg.angular.z = turn_speed
                self.spot_turn_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FollowerNode()
    rclpy.spin(node)
    rclpy.shutdown()