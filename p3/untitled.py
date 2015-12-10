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

from std_msgs.msg import Header

class CreateSim(object):
    def __init__(self,world_map_init,x_init):
        """
        Initialize starting pose and uncertainty parameters here.
        Inputs:
            world_map_init: list of lists [x,y,theta,id], each representing
             a landmark in the world map. Note that there is an ambiguity
             between landmarks such that multiple landmarks may have the
             same id number.
        """
        self.done=False
        self._fresh = False
        self._detections = None
        self.x_t = x_init
        self.particles = []
        self.num_particles = 100
        self.world_map = world_map_init
        self.iteration = 0
        self.init_particles()
        self.dt = None
        self.last_time = None
        self.count = 0


             self.turn = False

        name = rospy.get_param("~myname","pi4")
        self.filter_est_pub = rospy.Publisher("/"+name+"/filter_est",PoseStamped,queue_size=10)
        rospy.Subscriber("/"+name+"/cmd_vel",Twist,self.propogate_particles)
        rospy.Subscriber("/"+name+"/tag_detections_pose",PoseArray,self._pose_callback)


    # Generate particles at a uniform distribution
    def init_particles(self):

        # gaussian distribution centered at initial position
        rand_x = np.random.normal(self.x_t[0][0], 0.5, self.num_particles)
        rand_y = np.random.normal(self.x_t[1][0], 0.5, self.num_particles)
        rand_angle = np.random.normal(self.x_t[2][0], 0.5, self.num_particles)

        # weights
        weights = [1.0 / self.num_particles] * self.num_particles

        # create particles
        self.particles = zip(rand_x, rand_y, rand_angle, weights)
        self.particles = [list(i) for i in self.particles]

        self.updated_robot_position()

    def updated_robot_position(self):
        # new position using weighted average of particles
        self.x_t[0] = [sum(i[0] * i[3] for i in self.particles)]
        self.x_t[1] = [sum(i[1] * i[3] for i in self.particles)]
        self.x_t[2] = [sum(i[2] * i[3] for i in self.particles)]

    def noise(self, mean, var):
        return np.random.normal(mean, var, len(self.particles))

    def propogate_particles(self,cmd):
        if self.dt is None:
            self.last_time = time.time()
        self.dt = time.time() - self.last_time
        self.last_time = time.time()
        v = cmd.linear.x
        w = cmd.angular.z
        updated_x = [i[0] + v * self.dt * np.cos(i[2]) for i in self.particles] + self.noise(0, 0.04)
        updated_y = [i[1] + v * self.dt * np.sin(i[2]) for i in self.particles] + self.noise(0, 0.04)
        updated_angle = [i[2] + w * self.dt for i in self.particles] + self.noise(0, 0.02)
        weights = [i[3] for i in self.particles]


    # implements the maximum likelihood function
    def likelihood(self, x, y, theta):
        a = .1
        b = .1
        c = .5
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
                        # calculate tag position from particle frame
                        t_p = np.linalg.solve(p_w, t_w)
                        x_t_p = t_p[0][2]
                        y_t_p = t_p[1][2]
                        theta_t_p = math.atan2(t_p[1][0], t_p[0][0])

                        wi = max(wi, self.likelihood(x_t_p - x_t_r, y_t_p - y_t_r, theta_t_p - theta_t_r))

                w.append(wi)
            self.particles[i][3] = np.prod(np.array(w))

        # normalize weights
        w_all = sum([x[3] for x in self.particles])
        temp = np.array(self.particles)
        temp[:, 3] /= w_all
        self.particles = temp.tolist()
        self.updated_robot_position()

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
            self.particles[j][0:3] += np.random.normal(0, 0.05, 3)
            self.particles[j][3] = 1.0 / len(self.particles)
            S.append(self.particles[j])

        self.particles = S

    def get_checkpoint_position(self):
        nodes = self.graph.nodes
        def get_checkpoint_position(self):
        nodes = self.graph.nodes
        index = self.shortest_path[self.checkpoint]
        return nodes[index]

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
            self._detections = []
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
        #print self._detections
        return self._detections,fresh

    def publish_pose(self):
        """
        Publishes self.x_t to the robot
        def publish_pose(self):
        """
        Publishes self.x_t to the robot
        """
        pose = Pose()
        if self.turn:
            pose.position.x = 0
            pose.position.y = 0
            pose.orientation.w = 0
        else:
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
    world_map = [[    0.0,    0.0,     0.0, 1],
                 [    0.0,-1.8288,     0.0, 2],
                 [ 0.9398,-0.8128,     0.0, 3],
                 [ 1.4859, 0.8636,-np.pi/2, 4],
                 [ 1.4859,-2.3876,np.pi/2, 5],
                 [ 2.4892,    0.0,     0.0, 7],
                 [ 2.4284,-1.8288,     0.0, 8],
                 [ 3.5687,-0.8128,     0.0, 6]]
    pos_init = np.array([[-2.4892, -0.8128, 0.0]]).T

    # No changes needed after this point
    rospy.init_node('Particle_filter')
    sim = CreateSim(world_map, pos_init)

    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        """
        TODO: This loop runs at 60Hz. Call your particle filter functions here (i.e. check
        for measurements, and if so reweight and potentially resample). You do not need to
        call your function to propagate particles here. That is taken care of automatically.
        """
        (meas, fresh) = sim.get_measurements()

# reweight if measurement is fresh
        if fresh and meas is not None and len(meas) != 0:
            # =========== PARTICLE FILTER ===========
            sim.iteration += 1
            sim.reweight_particles(meas)
            if sim.iteration % 6 == 0:
                sim.resample()
            sim.count = 0
            sim.turn = False

        elif meas is not None and len(meas) == 0 and fresh:
            sim.count += 1
            if sim.count >= 5:
                sim.turn = True
                print "YATZEE"

        if sim.iteration >= 50:
            sim.publish_pose()

        rate.sleep()

if __name__ == "__main__":
    main()
