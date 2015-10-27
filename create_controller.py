#!/usr/bin/env python
"""
ROS based interface for the iRobot Create 2, with localization using AprilTag detection.
Written for CIS390 at the University of Pennsylvania
Updated Oct 22, 2015
"""
import roslib
import numpy as np
import numpy.matlib
import sys
import rospy
import cv2
import math

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
            self._robot_t = None
            self._robot_R = None
            self._fresh = True
            self._no_detection = True
            return
        (marker_t, marker_R) = get_t_R(posearray.poses[0])
        # Compensate for tilt of camera
        angle = 12*np.pi/180
        camera_tilt=np.array([[1,0,0,0],[0,np.cos(angle),-np.sin(angle),0],[0,np.sin(angle),np.cos(angle),0],[0,0,0,1]])
        marker_t = np.dot(camera_tilt,marker_t)
        marker_R = np.dot(camera_tilt,marker_R)
        # Put everything in homogeneous form
        camera2marker = marker_R
        camera2marker[0:3,3] = marker_t[0:3].T
        print camera2marker
        # Transform from camera to robot
        camera2robot=np.linalg.inv(np.array([[0,0,1,0],[-1,0,0,-0.09],[0,-1,0,0],[0,0,0,1]]))
        # Transform from final pose to marker
        final2marker = np.array([[0,0,-1,0],[-1,0,0,0],[0,1,0,0],[0,0,0,1]])
        # Put it all together
        # Want: final to robot
        final_hom = np.dot(np.dot(final2marker,np.linalg.inv(camera2marker)),camera2robot)
        print final_hom
        self._robot_t = np.copy(final_hom[0:3,3])
        self._robot_R = final_hom
        self._robot_R[0:3,3] = 0
        print self._robot_t
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
        # Angle is yaw
        angle = np.arctan2(self._robot_R[1,0],self._robot_R[0,0])
        dx = self._robot_t[0]
        dy = self._robot_t[1]
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
        """
        YOUR CODE HERE
        This function is called at 60Hz. At each iteration, check if a fresh measurement has come in.
        If so, use your controller to move the create according to the robot pose.
        """

        MAX_SPEED=1
        x, y, theta, fresh = self.get_marker_pose()
        kp=0.5
        ka=0.5
        kb=0
        if fresh is None:
            self.command_velocity(0, 0)
        if not fresh:
            return

        rho = np.sqrt(x*x + y*y)
        beta = -math.atan2(-y, -x)

        alpha = -beta - theta
        if alpha < -np.pi:
            alpha += 2 * np.pi
        if alpha > np.pi:
            alpha -= 2 * np.pi

        v = kp * rho
        w = ka * alpha + kb * beta

        self.command_velocity(min(v, MAX_SPEED), w)


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