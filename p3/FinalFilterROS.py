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
        self.num_particles = 50
        self.init_particles()
        self.iteration = 0
        self.world_map = world_map_init
        self.occupancy_map = occupancy_map_init
        
        self.graph = self.generate_graph(occupancy_map_init)

        name = rospy.get_param("~myname")
        self.filter_est_pub = rospy.Publisher("/"+name+"/filter_est",PoseStamped,queue_size=10)
        rospy.Subscriber("/"+name+"/cmd_vel",Twist,self.propagate_particles)
        rospy.Subscriber("/"+name+"/tag_detections_pose",PoseArray,self._pose_callback)
       

    # Generate particles at a uniform distribution
    def init_particles(self):

        # determine reasonable bounding box given april tags
        min_x = np.min([i[0] for i in self.world_map]) - 1
        max_x = np.max([i[0] for i in self.world_map]) + 1
        min_y = np.min([i[1] for i in self.world_map]) - 1
        max_y = np.max([i[1] for i in self.world_map]) + 1
        range_x = max_x - min_x
        range_y = max_y - min_y

        # gaussian distribution centered at initial position
        rand_x = np.random.normal(self.x_t[0], 0.5, self.num_particles)

        # gaussian distribution centered at initial position
        rand_y = np.random.normal(self.x_t[1], 0.5, self.num_particles)

        # uniform distribution between [-pi, pi]
        rand_angle = np.random.normal(self.x_t[2], 0.5, self.num_particles)

        # weights
        weights = [1.0 / self.num_particles] * self.num_particles

        # create particles
        self.particles = zip(rand_x, rand_y, rand_angle, weights)
        self.particles = [list(i) for i in self.particles]

    def noise(self, mean, var):
        return np.random.normal(mean, var, len(self.particles))

    def propogate_particles(self,cmd):
        v = cmd.linear.x
        w = cmd.angular.z
        updated_x = [i[0] + v * self.dt * np.cos(i[2]) for i in self.particles] + self.noise(0, 0.04)
        updated_y = [i[1] + v * self.dt * np.sin(i[2]) for i in self.particles] + self.noise(0, 0.04)
        updated_angle = [i[2] + w * self.dt for i in self.particles] + self.noise(0, 0.02)
        weights = [i[3] for i in self.particles]
        self.particles = zip(updated_x, updated_y, updated_angle, weights)
        self.particles = [list(i) for i in self.particles]

    # implements the maximum likelihood function
    def likelihood(self, x, y, theta):
        a = 1.0
        b = 1.0
        c = 5.0
        numerator = (a * (x ** 2)) + (b * (y ** 2)) + (c * (theta ** 2))
        return math.exp(-numerator / 2.0)

    # See pseudocode in section 3.3 of project specs
    def reweight_particles(self, measurements):
        for i in range(0, len(self.particles)):
            w = []
            for t_r in measurements:
                wi = 0
                x_t_r = t_r[0]
                y_t_r = t_r[1]
                theta_t_r = t_r[2]
                
                # Find the closest match to the tag we see
                for tag in self.world_map:
                    if tag[3] == t_r[3]:
                        # particle from world frame
                        x_p_w = self.particles[i][0]
                        y_p_w = self.particles[i][1]
                        theta_p_w = self.particles[i][2]
                        cos_p_w = np.cos(theta_p_w)
                        sin_p_w = np.sin(theta_p_w)

                        # tag from world frame
                        x_t_w = tag[0]
                        y_t_w = tag[1]
                        theta_t_w = tag[2]
                        cos_t_w = np.cos(theta_t_w)
                        sin_t_w = np.sin(theta_t_w)

                        # define matrices
                        p_w = np.array([[cos_p_w, -sin_p_w, x_p_w], [sin_p_w, cos_p_w, y_p_w], [0, 0, 1]])
                        t_w = np.array([[cos_t_w, -sin_t_w, x_t_w], [sin_t_w, cos_t_w, y_t_w], [0, 0, 1]])

                        # calculate tag position from particle frame
                        t_p = np.linalg.solve(p_w, t_w)
                        x_t_p = t_p[0][2]
                        y_t_p = t_p[1][2]
                        theta_t_p = math.atan2(t_p[1][0], t_p[0][0])

                        wi = max(wi, self.likelihood(x_t_p - x_t_r, y_t_p - y_t_r, theta_t_p - theta_t_r))

                w.append(wi)
            self.particles[i][3] = np.sum(np.array(w))

        # normalize weights
        w_all = sum([x[3] for x in self.particles])
        for particle in self.particles:
            particle[3] /= w_all

        # new position using weighted average of particles
        self.x_t[0] = [sum(i[0] * i[3] for i in self.particles)]
        self.x_t[1] = [sum(i[1] * i[3] for i in self.particles)]
        self.x_t[2] = [sum(i[2] * i[3] for i in self.particles)]

    def resample(self):
        S = []
        c = []
        w_sum = 0

        # compute cumulative distribution
        for particle in self.particles:
            c.append(w_sum)
            w_sum += particle[3]

        for i in range(0, len(self.particles)):
            u = np.random.random_sample()
            # find largest j such that c_j <= u
            for j in range(len(c) - 1, -1, -1):
                if c[j] <= u:
                    break
            self.particles[j][0:3] += np.random.normal(0, 0.02, 3)
            self.particles[j][3] = 1.0 / len(self.particles)
            S.append(self.particles[j])

        self.particles = S

    def print_particles(self):
        for particle in self.particles:
            print particle

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
        # move after 50th iteration
        if sim.iteration > 50:
            sim.publish_pose()
        (meas, fresh) = sim.get_measurements()

        # reweight if measurement is fresh
        if fresh:
            sim.iteration += 1
            sim.reweight_particles(meas)
            if sim.iteration % 6 == 0:
                sim.resample()
        rate.sleep()

if __name__ == "__main__":
    main()

