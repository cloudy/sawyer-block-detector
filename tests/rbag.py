#!/usr/bin/env python
import roslib, rospy
from rosbag import bag
from sensor_msgs.msg import JointState, Image
from intera_core_msgs.msg import IODeviceStatus
from cv_bridge import CvBridge, CvBridgeError

import sys
import cv2
import math
import numpy as np

OUTPUTFILE = 'pos.bag'
ROBOTTOPIC = '/joint_states_slow' #'/robot/joint_states'
CAMTOPIC = '/image_raw'

CUFFTOPIC = '/io/robot/cuff/state'

class SawyerCuff:
    
    def __init__(self, cufftopic):
    
        self.sub_cuff = rospy.Subscriber(cufftopic, IODeviceStatus, self.callback)
        self.isPressed = False
    
    def callback(self,iostate):
        val = int(iostate.signals[0].data[1])
        print(val)
        if val > 0:
            self.isPressed = True

        self.isPressed = False

class DataCollector:

    def __init__(self, bagname, bottopic, camtopic):
        
        self.bridge = CvBridge()
        self.initframe = True
        
        self.sub_js = rospy.Subscriber(bottopic, JointState, self.callbackjoint)
        self.sub_cam = rospy.Subscriber(camtopic, Image, self.callbackcam)

        self.bag = bag.Bag(bagname, 'w')

    def callbackjoint(self, js):
        try:
            self.bag.write('/bagrec/', js)
            print("JointState written")
            #print(js)
        except:
            print("Error:", sys.exc_info()[0])
            raise
    
    def callbackcam(self, data):
        
        try:
            im = self.bridge.imgmsg_to_cv2(data, "bgr8")
            print("Camera written")
        except CvBridgeError as e:
            print(e)
            raise

        self.blockdetector(im)

        
        try:
            self.bag.write('/camrec/', data)
        except:
            print("Error:", sys.exc_info()[0])
            raise

    
    def blockdetector(self, im):
        cv2.imshow("proc img", im)
        cv2.waitKey(3)


    def __del__(self):
        self.bag.close()


def main():
   
    cuff = SawyerCuff(CUFFTOPIC)

    #dc = DataCollector(OUTPUTFILE, ROBOTTOPIC, CAMTOPIC)
    rospy.init_node('data_collector', anonymous=True, disable_signals=True)
    
    while cuff.isPressed == False:
        continue

    print("Stage 1")

    while cuff.isPressed == False:
        continue

    print("Stage 2")


    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Exiting...")

    del dc
    cv2.destroyAllWindows()     



if __name__ == "__main__":
    main()

