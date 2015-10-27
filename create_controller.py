__author__ = 'tushar'


#!/usr/bin/env python
"""
ROS based interface for the iRobot Create 2, with localization using AprilTag detection.
Written for CIS390 at the University of Pennsylvania
"""
import roslib
import numpy as np
import numpy.matlib
import sys
import rospy
import cv2

from std_msgs.msg import (
    Header,
    UInt16,
)

from geometry_msgs.msg import (
    PoseArray,
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    Twist,
)

from utility import *

class CreateController(object):
    def __init__(self):
        """
        ROS STUFF
        """
        name = rospy.get_param("~myname")
        self._fresh=False
        self._no_detection=True
        self._pub = rospy.Publisher("/"+name+"/cmd_vel",Twist,queue_size=10)
        rospy.Subscriber("/"+name+"/tag_detections_pose",PoseArray,self._pose_callback)

    def _pose_callback(self,posearray):
        """
        ROS STUFF
        """
        if not posearray.poses:
            self._marker_t = None
            self._marker_R = None
            self._fresh = True
            self._no_detection = True
            return
        (self._marker_t, self._marker_R) = get_t_R(posearray.poses[0])
        angle = 12*np.pi/180
        R = np.dot(np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]]),np.array([[1,0,0,0],[0,np.cos(angle),-np.sin(angle),0],[0,np.sin(angle),np.cos(angle),0],[0,0,0,1]]))
        self._marker_t = np.dot(R,self._marker_t)
        self._marker_R = np.dot(R,self._marker_R)
        self._fresh = True
        self._no_detection = False

    def get_marker_pose(self):
        """
        If a detection is found, returns the robot x,y and theta in the marker frame, as well
        as a boolean to determine if a new detection has arrived.
        If a new detection has arrived but no tags were found, returns all None.
        """
        if self._no_detection:
            return None, None, None, None
        T = np.copy(self._marker_t)
        T[1]-=0.09
        homR = self._marker_R
        homT = np.array([[1,0,0,T[0,0]],[0,1,0,T[1,0]],[0,0,1,T[2,0]],[0,0,0,1]])
        marker_hom = np.dot(homR,homT)
        Rot = np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]])
        marker_hom = np.dot(Rot,marker_hom)
        angle = np.arctan2(marker_hom[1,0],marker_hom[0,0])
        dx = marker_hom[0,3]
        dy = marker_hom[1,3]
        fresh = self._fresh
        self._fresh=False
        return dx,dy,angle, fresh

    def command_velocity(self,vx,wz):
        """
        Commands the robot to move with linear velocity vx and angular
        velocity wz
        """
        twist=Twist()
        twist.linear.x = vx
        twist.angular.z = wz
        self._pub.publish(twist)

    def command_create(self):
        MAX_SPEED=0.1
        k_p = 0.5
        k_a = 0.5
        k_b = 0.5
        """
        YOUR CODE HERE
        This function is called at 60Hz. At each iteration, check if a fresh measurement has come in.
        If so, use your controller to move the create according to the robot pose.
        """

        (x, y, theta, fresh) = self.get_marker_pose()
        if fresh:
            rho = np.sqrt(x*x + y*y)
            beta = -atan(y/x)
            alpha = -theta - beta
            v = k_p * rho
            omega = k_a*alpha + k_b * beta
            self.command_velocity(v, omega)

        return

def main(args):
    rospy.init_node('create_controller')
    controller = CreateController()
    r = rospy.Rate(60)
    while not rospy.is_shutdown():
        controller.command_create()
        r.sleep()
    # Done, stop robot
    twist = Twist()
    controller._pub.publish(twist)

if __name__ == "__main__":
    try:
        main(sys.argv)
    except rospy.ROSInterruptException: pass