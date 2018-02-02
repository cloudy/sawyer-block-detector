#!/usr/bin/env python

import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import sys
import cv2
import math
import numpy as np
from collections import deque


# Pull images from ROS node and process them
class BridgeImage:

    def __init__(self):
    
        self.bridge = CvBridge()
        self.im_pub = rospy.Publisher("processed_image", Image, queue_size=1)
        self.im_sub = rospy.Subscriber("image_raw", Image, self.callback)
        self.initframe = True
        self.flag = '1'
    
    def callback(self, data):

        try:
            cv_im = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        while self.flag == '1':
            cv2.imshow("processed_image", cv_im)
            cv2.waitKey(1)
            self.flag = raw_input("Input val:")
            print self.flag
        print "Exiting..."

        rospy.signal_shutdown("BC Y NOT")
        sys.exit(0)

        try:
            self.im_pub.publish(self.bridge.cv2_to_imgmsg(cv_im, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main(args):
    bi = BridgeImage()
    rospy.init_node('bridge_image', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Exiting...")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

