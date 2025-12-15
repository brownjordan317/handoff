# <u>H</u>and-<u>A</u>ction <u>N</u>avigation & <u>D</u>etection <u>O</u>ptimized <u>F</u>ollowing <u>F</u>ramework (HANDOFF)
 
This project presents the design and implementation of a low-cost, human-following robot capable of interpreting hand gestures to control its behavior. The system integrates multiple sensing modalities, including a forward-facing camera for person detection, a 2D LiDAR for distance estimation and obstacle avoidance, and vision-based human pose tracking for gesture recognition. YOLOv12 is employed for real-time person detection, while Google MediaPipe facilitates hand and body gesture recognition, allowing intuitive and responsive robot control. Sensor fusion of RGB and LiDAR data ensures accurate tracking even under partial occlusion or lateral movement. Gesture commands are mapped to key robot actions such as driving, target switching, and state control, enabling safe and interactive operation. The platform leverages affordable hardware, including a Raspberry Pi 5, a Pi Camera, and an RPLiDAR sensor, providing a cost-effective alternative to commercial systems. Experimental results demonstrate reliable target tracking, effective gesture-based control, and robust operation in dynamic environments, highlighting the system’s potential for applications in assistive robotics, interactive robotic pets, and automated service platforms. 

## Robot Hardware Specifications

| **Subsystem** | **Component**           | **Specification**                  |
|---------------|--------------------------|------------------------------------|
| **Compute**   | Raspberry Pi 5           | 8 GB RAM, Active Cooler            |
| **Lidar**     | RPLidar A1        | 2D Laser Scanner (360°)            |
| **Vision**    | Pi Camera Module 3       | V2 (8 MP), CSI Interface           |
| **Actuation** | ServoCity Motors         | 163 RPM Mini Econ Gear             |
| **Driver**    | Pololu Driver            | TB67H420FTG Dual Driver            |
| **Odometry**  | Magnetic Encoders        | AS5600 (12-bit Precision)          |
| **Power**     | 3S UPS Module            | 18650 Li-Ion Array                 |

## Construction of the Robot
For construction of the robot please refer to the GitHub repository hosted by [John Merila](github.com/JohnMerila/UND_EDUBot). The repository covers all required hardware and software components. It also has instructions for assembly and configuration of the robot along with a parts list.

## Target Following

The robot is able to follow a target by using the following strategy:
1. Use YOLO to detect the target

![Person Detection](/readme_files/detection_and_locking.gif)

2. Use the person detection to center the robots camera on the target 

![Angular Follower](readme_files/angular_following.gif)

3. Use the LiDAR to measure the distance to the target \***Calculated using [DR-SPAAM](https://github.com/VisualComputingInstitute/DR-SPAAM-Detector)**

4. Move the robot to the target using the distance measurement and the position of the target in the camera frame 

![Linear Follower](readme_files/linear_following.gif)

The robot will follow the target as long as it is in the field of view of the camera. If the target is not in the field of view of the camera, the robot will stop following the target.

## Gesture Recognition

The robot is able to use gestures to control its behavior. Gestures are recognized using a custom model based around [Googles MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer) framework. The specific gestures that the robot is able to recognize are the American Sign Language (alphabet) minus J and Z.

![Gesture Recognition](readme_files/hand_asl.gif)

## Using the Gesture Recognition for Robot Controls
Once the robot can recognize a gesture, it can be used to control the robot. 

### State Control
The gesture controls can be used to control the state of the robot. One such state switch is using the gesture to unlock the robot from a locked state.

![State Control](readme_files/target_switching.gif)