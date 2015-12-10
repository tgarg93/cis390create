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
    def __init__(self,world_map_init,occupancy_map_init,x_gt_init):
        """
        Initialize starting pose and uncertainty parameters here.
        Inputs:
            world_map_init: list of lists [x,y,theta,id], each representing
             a landmark in the world map. Note that there is an ambiguity
             between landmarks such that multiple landmarks may have the
             same id number.
            x_gt_init: 3x1 numpy array that represents the true starting
             position of the robot
        """
        self.done=False
        self._fresh = False
        self.x_t = np.array([[0.0,0.0,0.0]]).T
        self.x_gt = x_gt_init

        self.world_map = world_map_init
        self.occupancy_map = occupancy_map_init
        
        self.iteration = 0
        self.graph = self.generate_graph(occupancy_map_init)
        self.v = 0
        self.omega = 0
        
        self.shortest_path = astar(self.graph,1,28)[0]
        self.checkpoint = 0
        self.CHECKPOINT_RADIUS = .05
        
        self._pub = rospy.Publisher("/"+name+"/cmd_vel",Twist,queue_size=10)
        name = rospy.get_param("~myname","pi4")
        rospy.Subscriber("/"+name+"/filter_est",PoseStamped,self._pose_callback)
        
def generate_graph(self,occupancy_map):
        nodes = []
        edges = []
        map_num = deepcopy(occupancy_map)
        

        for x in range(0,len(occupancy_map)):
            for y in range(0,len(occupancy_map[0])):
                if not occupancy_map[x][y]:
                    nodes.append([(x*0.4572+0.2286),-((y-2)*0.4572+0.2286)])
                    map_num[x][y]=len(nodes)-1
                    edge = []
                    if (y)>0 and not occupancy_map[x][y-1]:
                        edge.append(map_num[x][y-1])
                    if x>0 and not occupancy_map[x-1][y]:
                        edge.append(map_num[x-1][y])
                    edges.append(edge)

        for i in range(0,len(edges)):
            for j in range(0,len(edges[i])):
                if i not in edges[edges[i][j]]:
                    edges[edges[i][j]].append(i)
                    
        return Graph(nodes,edges)

    def _pose_callback(self,posemsg):
        """
        ROS STUFF
        """
        self.x_t = np.array([[posemsgarray.poses[i].position.x],[posemsgarray.poses[i].position.y],[posemsgarray.poses[i].orientation.w]])
        
        self._fresh = True

    def get_measurements(self):
        """
        Returns a list of lists of visible landmarks (x,y,theta) and a fresh boolean (always true)
        """
        fresh = self._fresh
        self._fresh = False
        return self.x_t,fresh

    def command_velocity(self,vx,wz):
        """
        Commands the robot to move with linear velocity vx and angular
        velocity wz
        """
        twist=Twist()
        twist.linear.x = vx
        twist.angular.z = wz
        self._pub.publish(twist)

    def get_checkpoint_position(self):
        nodes = self.graph.nodes
        index = self.shortest_path[self.checkpoint]
        return nodes[index]

    def command_create(self):
        """ 
        YOUR CODE HERE
        """
        MAX_SPEED=0.2
        (self.x_t,fresh) = self.get_measurements()
        if not fresh:
            return

         # ============= CONTROLLER =============

        # Relative distance from where you are to checkpoint
        checkpoint_pos = self.get_checkpoint_position()
        dx = self.x_t[0,0] - checkpoint_pos[0]
        dy = self.x_t[1,0] - checkpoint_pos[1]

        # Check/update if you are near checkpoint
        if np.linalg.norm([dx, dy]) < self.CHECKPOINT_RADIUS:
            self.checkpoint += 1
            if self.checkpoint == len(self.shortest_path):
                self.done = True
                return
            checkpoint_pos = self.get_checkpoint_position()
            dx = self.x_t[0,0] - checkpoint_pos[0]
            dy = self.x_t[1,0] - checkpoint_pos[1]

        # Calculate v and omega
        kp = 1
        ka = 10
        kb = 0
        rho = np.sqrt(dx * dx + dy * dy)
        beta = -math.atan2(-dy, -dx)
        alpha = -beta - self.x_t[2,0]
        if alpha < -np.pi:
            alpha += 2 * np.pi
        if alpha > np.pi:
            alpha -= 2 * np.pi

        self.v = kp * rho
        self.omega = ka * alpha + kb * beta

        # Move
        self.command_velocity(self.v, self.omega)
        
        return


def main():
    """
    Modify simulation parameters here. In particular, the world map,
    starting position, and max iterations to simulate
    """
    # Real map
    occupancy_map = [[1,1,1,1,1,1,1,1],
                     [1,0,0,0,0,1,0,1],
                     [1,0,0,0,0,1,0,1],
                     [0,0,1,0,0,0,0,1],
                     [0,1,1,0,0,0,0,1],
                     [0,0,1,0,0,0,0,1],
                     [1,0,0,0,0,1,0,1],
                     [1,0,0,0,0,1,0,1],
                     [1,1,1,1,1,1,1,1]]
    world_map = [[    0.0,    0.0,     0.0, 4],
                 [    0.0,-1.8288,     0.0, 2],
                 [ 0.9398,-0.8128,     0.0, 1],
                 [ 1.4859, 0.8636, np.pi/2, 0],
                 [ 1.4859,-2.3876,-np.pi/2, 3],
                 [ 2.4892,    0.0,     0.0, 5],
                 [ 2.4284, 1.8288,     0.0, 6],
                 [ 3.5687,-0.8128,     0.0, 7]]
    pos_init = np.array([[-2.4892, -0.8128, 0.0]]).T

    # No changes needed after this point
    rospy.init_node('FilterRobot')

    sim = CreateSim(world_map, occupancy_map, pos_init)

    rate = rospy.Rate(60)

    while not rospy.is_shutdown() and not sim.done:
        sim.command_create()
        rate.sleep()

if __name__ == "__main__":
    main()

