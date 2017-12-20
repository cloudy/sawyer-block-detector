# sawyer_block_detector
Rethink Robotics Sawyer color cube detector


Current implementation will detect blue and green 3d printed blocks on a face-down camera and track their movements.



Setup 

Source ROS configuration: `source /opt/ros/kinetic/setup.zsh`

Source devel workspace: `cd ~/ros_ws && source devel/setup.zsh`

Launch camera node: `roslaunch path/to/umd/launchfile.launch`

Run cv.py: `python vision/cv.py`



TODO:

Transformation to robot world frame

Collect positional data from each block and joint angles of sawyer robotic arm into ROSBAG


