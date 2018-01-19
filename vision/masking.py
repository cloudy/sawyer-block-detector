#!/usr/bin/env python

#roslib.load_manifest('INSERT NAME HERE') # FOR FUTURE WORK
import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import sys
import cv2
import math
import numpy as np

# Color values in HSV
GREENLOWER = np.array([80, 60, 60])
GREENUPPER = np.array([100 ,255, 255])

BLUELOWER = np.array([105, 100, 100])
BLUEUPPER = np.array([120, 255, 255])

# Determines noise clear for morph
KERNELOPEN = np.ones((5,5))
KERNELCLOSE = np.ones((5,5))

# Font details for display windows
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)

# Pull images from ROS node and process them
class BridgeImage:

    def __init__(self):
    
        self.bridge = CvBridge()
        self.im_pub = rospy.Publisher("processed_image", Image, queue_size=10)
        self.im_sub = rospy.Subscriber("image_raw", Image, self.callback)
        self.initframe = True
        
    # Process each image
    def callback(self, data):

        try:
            cv_im = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        self.block_detector(cv_im)

        cv2.imshow("processed_image", cv_im)
        cv2.waitKey(3)

        try:
            self.im_pub.publish(self.bridge.cv2_to_imgmsg(cv_im, "bgr8"))
        except CvBridgeError as e:
            print(e)

    # Perform 3d printed block detection with each image
    def block_detector(self, im):
        
        # Convert image to HSV
        imHSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        # Mask out image and use morphology to clear of image 'debris'
        mask_green = cv2.inRange(imHSV, GREENLOWER, GREENUPPER)
        mask_green_open = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, KERNELOPEN)
        mask_green_close = cv2.morphologyEx(mask_green_open, cv2.MORPH_CLOSE, KERNELCLOSE)
    
        mask_blue = cv2.inRange(imHSV, BLUELOWER, BLUEUPPER)
        mask_blue_open = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, KERNELOPEN)
        mask_blue_close = cv2.morphologyEx(mask_blue_open, cv2.MORPH_CLOSE, KERNELCLOSE)
    
        # Combine masks to keep all blocks in single view, distinction between blocks lost
        mask_final = mask_green_close + mask_blue_close
   
        res = cv2.bitwise_and(im, im, mask = mask_final)
    
        _, conts, h = cv2.findContours(mask_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
        cv2.drawContours(im, conts, -1, (255, 255, 0), 1)
        # Display various views 
        cv2.imshow("Green Mask", mask_green_close)
        #cv2.imshow("Blue Mask", mask_blue_close)
        #cv2.imshow("Combined Mask", mask_final)
        cv2.imshow("Result", res)
        cv2.waitKey(10)




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

