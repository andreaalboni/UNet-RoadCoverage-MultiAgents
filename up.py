#!/usr/bin/env python3
import rospy
import time
from std_msgs.msg import Bool, Empty
from geometry_msgs.msg import TwistStamped

if __name__ == '__main__':
    rospy.init_node("up_node")
    rospy.loginfo("Node has been started")

    arm = rospy.Publisher("/bridge/arm", Bool, queue_size=10)
    up = rospy.Publisher("/autopilot/start", Empty, queue_size=10)
    stayup = rospy.Publisher("/autopilot/velocity_command", TwistStamped, queue_size=1)

    time.sleep(10)

    #sending the message
    rospy.loginfo("Ready, set...")
    arm.publish(True)
    rospy.loginfo("Up they go")
    up.publish()
    rospy.loginfo("In air")
    #------------------

    time.sleep(10)

    msg = TwistStamped()
    msg.header.frame_id = "world"
    msg.twist.linear.z = 1.0
    for i in range(250):    
        stayup.publish(msg)
        rospy.sleep(0.1)