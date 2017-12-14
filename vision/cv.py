#!/usr/bin/env python

#roslib.load_manifest('INSERT NAME HERE') # FOR FUTURE WORK
import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import sys
import rospy
import cv2
import numpy as np

lowerBound = np.array([33,80,40])
upperBound = np.array([102,255,255])

kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

class bridge_image:

    def __init__(self):
    
        self.bridge = CvBridge()
        self.im_pub = rospy.Publisher("processed_image", Image, queue_size=10)
        self.im_sub = rospy.Subscriber("image_raw", Image, self.callback)
    
    def callback(self, data):

        try:
            cv_im = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        detector(cv_im)

        #(rows, cols, chans) = cv_im.shape
        
        cv2.imshow("processed_image", cv_im)
        cv2.waitKey(3)

        try:
            self.im_pub.publish(self.bridge.cv2_to_imgmsg(cv_im, "bgr8"))
        except CvBridgeError as e:
            print(e)

def detector(im):
    imHSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(imHSV, lowerBound, upperBound)

    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    maskFinal = maskClose

    _, conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(im, conts, -1, (255,0,0), 3)
    for i in range(len(conts)):
        x,y,w,h = cv2.boundingRect(conts[i])
        cv2.rectangle(im, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(im, str(i+1), (x, y+h), fontface, fontscale, fontcolor)

    cv2.imshow("mc", maskClose)
    cv2.imshow("mo", maskOpen)
    cv2.imshow("mask", mask)
    cv2.imshow("cam", im)
    cv2.waitKey(10)




def main(args):
    bi = bridge_image()
    rospy.init_node('bridge_image', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Exiting...")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

