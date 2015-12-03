#!/usr/bin/python
"""
2D Unicycle model-based robot simulator for a particle filter implementation
for CIS 390 Fall 2015 at the University of Pennsylvania
"""

from matplotlib import pyplot as plt
import numpy as np
import time
import math
from Graph import Graph
from ShortestPath import astar
from copy import deepcopy
import roslib
import rospy
from geometry_msgs.msg import (
    PoseArray,
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    Twist,
)

class CreateSim(object):
    def __init__(self,world_map_init,occupancy_map_init):
        """
        Initialize starting pose and uncertainty parameters here.
        Inputs:
            world_map_init: list of lists [x,y,theta,id], each representing
             a landmark in the world map. Note that there is an ambiguity
             between landmarks such that multiple landmarks may have the
             same id number.
        """
        self.done=False
        self.x_t = np.array([[0.0,0.0,0.0]]).T
        # Particles to plot - list of (x,y,theta,weight)
        self.particles = []
        # self.particles = [(0.5,0.5,0,1),(0.5,-0.5,0,0.5)]; # Example
        # Map stored as array of (x,y,theta) for the april tags
        self.world_map = world_map_init
        self.occupancy_map = occupancy_map_init
        
        self.iteration = 0
        self.graph = self.generate_graph(occupancy_map_init)

        name = rospy.get_param("~myname")
        self.filter_est_pub = rospy.Publisher("/"+name+"/filter_est",PoseStamped,queue_size=10)
        rospy.Subscriber("/"+name+"/cmd_vel",Twist,self.propagate_particles)
        rospy.Subscriber("/"+name+"/tag_detections_pose",PoseArray,self._pose_callback)
        
    def propogate_particles(self,cmd):
        v = cmd.linear.x
        w = cmd.angular.z

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
        Returns a list of lists of visible landmarks (x,y,theta) and a fresh boolean (always true)
        """
        fresh = self._fresh
        self._fresh = False
        return self._detections,fresh

    def publish_pose(self):
        """
        Publishes self.x_t to the robot
        """
        pose = Pose()
        pose.position.x = self.x_t[0,0]
        pose.position.y = self.x_t[1,0]
        pose.orientation.w = self.x_t[2,0]
        posestamped = PoseStamped()
        posestamped.header = Header()
        posestamped.header.stamp = rospy.Time.now()
        posestamped.pose = pose
        self.filter_est_pub.publish(posestamped)

def main():
    """
    Modify simulation parameters here. In particular, the world map,
    starting position, and max iterations to simulate
    """
    world_map = [[0.27,0.27,-np.pi/2,1],[0.81,0.27,-np.pi/2,2],[1.89,0.27,-np.pi/2,3],[2.43,0.27,-np.pi/2,4],[0.81,0.81,0,5],[1.89,0.81,np.pi,5],[0.27,1.35,np.pi/2,6],[0.81,1.35,0,7],[1.89,1.35,np.pi,8],[1.89,1.89,np.pi,9],[0.81,2.43,-np.pi/2,10],[1.35,2.43,-np.pi/2,11]]
    pos_init = np.array([[1.25,-1,np.pi/2]]).T

    # No changes needed after this point
    sim = CreateSim(world_map, occupancy_map, pos_init)

    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        sim.publish_pose()
        rate.sleep()

if __name__ == "__main__":
    main()

