#!/usr/bin/env python
"""
ROS based interface for the iRobot Create 2, with localization using AprilTag detection.
Written for CIS390 at the University of Pennsylvania
Updated Nov 19, 2015
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

from apriltags_ros.msg import (
    AprilTagDetectionArray,
    AprilTagDetection
)

from utility import *

class CreateController(object):
    def __init__(self,world_map_init):
        """
        ROS STUFF
        """
        name = rospy.get_param("~myname")
        self._fresh=False
        self._no_detection=True
        self._pub = rospy.Publisher("/"+name+"/cmd_vel",Twist,queue_size=10)
        rospy.Subscriber("/"+name+"/tag_detections_pose",PoseArray,self._pose_callback)
        self.x_t = np.array([[0.,0.,0.]]).T
        self.v=0.05
        self.omega=0
        self.world_map = world_map_init

    def _pose_callback(self,posemsgarray):
        """
        ROS STUFF
        """
        if not posemsgarray.poses:
            self._robot_t = None
            self._robot_R = None
            self._detections = None
            self._fresh = True
            self._no_detection = True
            return

        self._detections=[]

        for i in range(0,len(posemsgarray.poses)):
            x = posemsgarray.poses[i].position.x
            y = posemsgarray.poses[i].position.y
            theta = posemsgarray.poses[i].orientation.x
            id = posemsgarray.poses[i].orientation.y
            self._detections.append([x,y,theta,id])
        self._fresh = True
        self._no_detection = False

    def get_measurements(self):
        """
        Returns a tuple of detections and a boolean fresh. If no detection is found,
        fresh is False and the detections are none. If detections are found, they are
        returned as a list of landmark poses ([x,y,theta,id]) in list form.
        """
        if self._no_detection:
            return None, None
        fresh = self._fresh
        self._fresh = False
        return self._detections, fresh

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
        """
        YOUR CODE HERE
        This function is called at 60Hz. At each iteration, check if a fresh measurement has come in.
        If so, process the measurements with your particle filter and then move to generate new measurements.
        """
        return


def main(args):
    rospy.init_node('create_controller')
    # EXAMPLE WORLD MAP, REPLACE WITH TRUE MAP
    world_map = [[0.0,0.0,0,1],[1.,0.,np.pi/2,1],[-2.,1.,0.,2]]
    controller = CreateController(world_map)
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
