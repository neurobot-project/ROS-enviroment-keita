#!/usr/bin/env python3
import os 
import numpy as np
import sys
sys.path.append(os.path.basename(__file__))
import cv2

from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt 
import PIL

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

import rospy 
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image


class Preprocess:
    def __init__(self):
        self.init_mobile_net()
        self.h = 480
        self.w = 640
        self.depth_img = None
        self.color_img = None

        self.object_list = [] # list for cache object pos and label [[label,box],...]

    def init_mobile_net(self):
        print(sys.path)
        self.label_path = '/home/keita/catkin_ws/src/preprocess/scripts/models/voc-model-labels.txt'
        self.model_path = '/home/keita/catkin_ws/src/preprocess/scripts/models/mb2-ssd-lite-mp-0_686.pth'

        self.class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.num_classes = len(self.class_names)
        self.net = create_mobilenetv2_ssd_lite(len(self.class_names), is_test=True)
        self.net.load(self.model_path)
        self.predictor = create_mobilenetv2_ssd_lite_predictor(self.net, candidate_size=200)

    def ros_color_to_arr(self,ros_depth_data):
        h = ros_depth_data.height
        w = ros_depth_data.width
        ros_img = np.fromstring(ros_depth_data.data, np.uint8)

        r = ros_img[0::3].reshape(h,w)
        g = ros_img[1::3].reshape(h,w)
        b = ros_img[2::3].reshape(h,w)
        img = np.dstack((np.dstack((r, g)), b))
        return img
    
    def ros_depth_to_arr(self,ros_img_data):      
        h = ros_img_data.height
        w = ros_img_data.width      
        ros_img = np.fromstring(ros_img_data.data, np.uint8)
        lower = ros_img[0::2]
        upper = ros_img[1::2] # 8
        img = upper*256+lower
        img = img.reshape(h, w)
        return img
    
    

    def callback_depth(self,depth_sub):
        self.depth_img = self.ros_depth_to_arr(depth_sub)
        
        

    def callback_color(self,color_sub):
        self.color_img = self.ros_color_to_arr(color_sub)
        boxes, labels, probs = self.predictor.predict(self.color_img, 10, 0.4)
        
        self.object_list = []
        self.object_pos_list = []
        depth_arr = self.depth_img.copy()
        for box, label_num in zip(boxes,labels):
            label = self.class_names[label_num]
            self.object_list.append([label,box])

            x = int((box[0]+box[2])/2)
            y = int((box[1]+box[3])/2)
            z = int(self.cal_mean_distance(box,depth_arr))
            self.object_pos_list.append([label,x,y,z])
        print(self.object_pos_list)


    def cal_mean_distance(self,box,depth_arr):
        '''
        box [[x0,y0,x1,y1]]
        '''
        x0 = max(int(box[0]),0)
        y0 = max(int(box[1]),0)
        x1 = min(int(box[2]),self.w)
        y1 = min(int(box[3]),self.h)

        z = np.median(depth_arr[y0:y1,x0:x1])
        return z


    def listener(self):
        rospy.init_node('preprocess',anonymous=True)
        depth_sub = message_filters.Subscriber('camera/depth/image_rect_raw', Image)
        color_sub = message_filters.Subscriber('camera/color/image_raw', Image)
        depth_sub.registerCallback(self.callback_depth)
        color_sub.registerCallback(self.callback_color)
        
        
        # ts = message_filters.TimeSynchronizer([color_sub, depth_sub], 10)
        # ts.registerCallback(self.callback)
        rospy.spin()


if __name__ == "__main__":
    preprocess = Preprocess()
    preprocess.listener()