#!/usr/bin/env python3
import rospy
import message_filters
from std_msgs.msg import String
from nav_msgs.msg import Odometry

class SubscirbePos:
    def __init__(self):
        self.posX = 0
        self.posY = 0
        self.posZ = 0

    def callback(self,data):
        print(data.pose.pose.position)
        pass
    
    def listener(self):
        rospy.init_node('sub_position',anonymous=True)
        pos_sub = message_filters.Subscriber('odom',Odometry)

        pos_sub.registerCallback(self.callback)
        rospy.spin()

if __name__ == "__main__":
    subscribe_pos = SubscirbePos()
    subscribe_pos.listener()