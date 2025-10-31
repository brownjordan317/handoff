from project_node.utils.person_tracking import PersonTracking

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

IMAGE_RECV_TOPIC = "/camera/image_raw"
IMAGE_PUB_TOPIC = "jordansProject/detections/image"
FPS = 30
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
                                             desired_encoding = "bgr8")
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
            image = self.detector.detect_pedestrians(self.image)
            cv2.imshow('image', image)
            cv2.waitKey(1)
            # self.publish_pred_image(
            #     image
            #     )
            
            twist_msg = Twist()
            if self.detector.turn_direction is not None \
                and self.detector.mc_number is not None \
                    and self.detector.dist_x is not None\
                        and self.detector.state == "locked":
                
                turn_speed = -self.detector.mc_number if \
                    self.detector.turn_direction == "right" else \
                        self.detector.mc_number
                # clip turn speed between -1 and 1
                # turn_speed = max(min(turn_speed, 1.0), -1.0)
                twist_msg.angular.z = float(turn_speed)
                self.spot_turn_pub.publish(twist_msg)
            else:
                twist_msg.angular.z = 0.0
            self.spot_turn_pub.publish(twist_msg)

            # if self.detector.hand_controls.drive_command is not None:
            #     drive_cmd = self.detector.hand_controls.drive_command
            #     if drive_cmd == "FORWARD":
            #         twist_msg.linear.x = 0.5
            #     elif drive_cmd == "LEFT":
            #         twist_msg.angular.z = 1.0
            #     elif drive_cmd == "RIGHT":
            #         twist_msg.angular.z = -1.0
            #     else:  # STOP or unrecognized command
            #         twist_msg.linear.x = 0.0
            #         twist_msg.angular.z = 0.0
                
            #     self.spot_turn_pub.publish(twist_msg)
                

def main(args=None):
    rclpy.init(args=args)
    node = FollowerNode()
    rclpy.spin(node)
    rclpy.shutdown()