"""
This script initializes a ROS node that publishes video frames from the 
default webcam (/dev/video0) at a rate of 30 Hz. The video frames are resized 
to 320x240 pixels before being published to the '/camera/image_raw' topic. 
This allows for real-time video streaming within a ROS environment without 
needing to connect to a picam or other hardware.
"""


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class WebcamPublisher(Node):
    def __init__(self):
        """
        Initialize the WebcamPublisher node.

        This node publishes a stream of images from the default
        webcam (/dev/video0) at a rate of 30 Hz.
        """
        super().__init__('webcam_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 1)
        self.timer = self.create_timer(1/30, self.timer_callback)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # 0 = /dev/video0
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open webcam /dev/video0")

    def timer_callback(self):
        """
        Timer callback to publish video frames from the default webcam
        (/dev/video0).

        This function reads a frame from the webcam, resizes it to 320x240,
        converts it to a ROS Image message, and publishes it to the
        '/camera/image_raw' topic.

        If the frame cannot be read, a warning is logged.
        """
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        if not ret:
            self.get_logger().warn("Failed to grab frame")
            return
        # cv2.imshow('Webcam', frame)
        # cv2.waitKey(1)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing video frame')

def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
