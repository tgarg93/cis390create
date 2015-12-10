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

class CreateSim(object):
    def __init__(self,world_map_init,occupancy_map_init,x_gt_init):
        """
        Initialize starting pose and uncertainty parameters here.
        Inputs:
            world_map_init: 
                list of lists [x,y,theta,id], each representing
                a landmark in the world map. Note that there is an ambiguity
                between landmarks such that multiple landmarks may have the
                same id number.
            occupancy_map_init: 
                map of 1s and 0s, 1 = occupied and 0 = free
            x_gt_init: 
                3x1 numpy array that represents the true starting
                position of the robot
        """
        self.done=False
        self.x_gt = x_gt_init
        self.v = 0
        self.omega = 0
        self.x_t = np.array([[0.0,0.0,0.0]]).T
        self.dt=1.0/60
        self.THETA_MAX = np.pi/2 # Angle at which we can see markers
        self.particles = [] # Particles to plot - list of (x,y,theta,weight)
        self.num_particles = 50
        self.world_map = world_map_init
        self.occupancy_map = occupancy_map_init
        self.iteration = 0
        self.graph = self.generate_graph(occupancy_map_init)
        self.shortest_path = astar(self.graph, 0, len(self.graph.nodes) - 1)[0]
        self.checkpoint = 0 # Index of point in shortest path that we are heading towards
        self.CHECKPOINT_RADIUS = .1
        self.init_particles()
        
    # ==================== HELPER FUNCTIONS ====================

    # Update x_t based on weighted average of particles
    def updated_robot_position(self):
        self.x_t[0] = [sum(i[0] * i[3] for i in self.particles)]
        self.x_t[1] = [sum(i[1] * i[3] for i in self.particles)]
        self.x_t[2] = [sum(i[2] * i[3] for i in self.particles)]

    # Gaussian with mean and var
    def noise(self, mean, var):
        return np.random.normal(mean, var, len(self.particles))

    # implements the maximum likelihood function
    def likelihood(self, x, y, theta):
        a = 1.0
        b = 1.0
        c = 5.0
        numerator = (a * (x ** 2)) + (b * (y ** 2)) + (c * (theta ** 2))
        return math.exp(-numerator / 2.0)

    # pretty self-explanatory
    def get_checkpoint_position(self):
        nodes = self.graph.nodes
        index = self.shortest_path[self.checkpoint]
        return nodes[index]

    # pretty self-explanatory
    def print_particles(self):
        for particle in self.particles:
            print particle
        print ""

    # ==================== MAIN FUNCTIONS ====================

    # Generate particles at a uniform distribution
    def init_particles(self):

        # gaussian distribution centered at initial position
        rand_x = np.random.normal(self.x_gt[0][0], 0.5, self.num_particles)
        rand_y = np.random.normal(self.x_gt[1][0], 0.5, self.num_particles)
        rand_angle = np.random.normal(self.x_gt[2][0], 0.5, self.num_particles)

        # weights
        weights = [1.0 / self.num_particles] * self.num_particles

        # create particles
        self.particles = zip(rand_x, rand_y, rand_angle, weights)
        self.particles = [list(i) for i in self.particles]

        # initialize x_t
        self.updated_robot_position()

    # This adds some noise to x, y, and angle
    def propogate_particles(self, v, w):
        updated_x = [i[0] + v * self.dt * np.cos(i[2]) for i in self.particles] + self.noise(0, 0.02)
        updated_y = [i[1] + v * self.dt * np.sin(i[2]) for i in self.particles] + self.noise(0, 0.02)
        updated_angle = [i[2] + w * self.dt for i in self.particles] + self.noise(0, 0.01)
        weights = [i[3] for i in self.particles]
        self.particles = zip(updated_x, updated_y, updated_angle, weights)
        self.particles = [list(i) for i in self.particles]

    # Assigns new weights to particles based on likelihood. 
    # Notation: x_t_r = x position of tag from robot's frame.
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
            self.particles[i][3] = np.prod(np.array(w))

        # normalize weights
        w_all = sum([x[3] for x in self.particles])
        temp = np.array(self.particles)
        temp[:, 3] /= w_all
        self.particles = temp.tolist()

        # recompute x_t
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
            self.particles[j][0:3] += np.random.normal(0, 0.02, 3)
            self.particles[j][3] = 1.0 / len(self.particles)
            S.append(self.particles[j])

        self.particles = S

    # ==================== GIVEN FUNCTIONS ====================

    def get_measurements(self):
        """
        Returns a list of lists of visible landmarks (x,y,theta) and a fresh boolean (always true)
        """
        # Create pose matrix for the robot
        th = self.x_gt[2,0]
        H_WR = np.array([[np.cos(th), -np.sin(th), self.x_gt[0,0]],
                         [np.sin(th), np.cos(th), self.x_gt[1,0]],
                         [0., 0., 1.]])
        # Get measurements to the robot frame
        meas = []
        for i in range(len(self.world_map)):
            x_new = np.linalg.solve(H_WR,np.array([self.world_map[i][0], self.world_map[i][1], 1]))
            theta_new = self.world_map[i][2] - th
            if theta_new > np.pi:
                theta_new -= 2*np.pi
            if theta_new < -np.pi:
                theta_new += 2*np.pi
            if np.absolute(np.arccos(x_new[0]/np.sqrt(x_new[0]**2 + x_new[1]**2))) < self.THETA_MAX:
                meas.append([x_new[0],x_new[1],theta_new,self.world_map[i][3]])
        return meas,True

    def generate_graph(self,occupancy_map):
        nodes = []
        edges = []
        map_num = deepcopy(occupancy_map)
        
        for y in range(0,len(occupancy_map)):
            for x in range(0,len(occupancy_map[0])):
                if not occupancy_map[y][x]:
                    nodes.append([x*0.27+0.135,y*0.27+0.135])
                    map_num[y][x]=len(nodes)-1
                    edge = []
                    if x>0 and not occupancy_map[y][x-1]:
                        edge.append(map_num[y][x-1])
                    if y>0 and not occupancy_map[y-1][x]:
                        edge.append(map_num[y-1][x])
                    edges.append(edge)

        for i in range(0,len(edges)):
            for j in range(0,len(edges[i])):
                if i not in edges[edges[i][j]]:
                    edges[edges[i][j]].append(i)
                    
        return Graph(nodes,edges)

    def command_velocity(self,vx,wz):
        """
        Simulate the robot's motion using Euler integration. Noise added
        to the commands. Does not update measured state x_t.
        """
        # This part computes robot's motion based on the groundtruth
        self.x_gt[0,0]+=self.dt*vx*np.cos(self.x_gt[2,0])
        self.x_gt[1,0]+=self.dt*vx*np.sin(self.x_gt[2,0])
        self.x_gt[2,0]+=self.dt*wz
        if self.x_gt[2,0]>np.pi:
            self.x_gt[2,0]-=2*np.pi
        if self.x_gt[2,0]<-np.pi:
            self.x_gt[2,0]+=2*np.pi
        return

    def command_create(self):
        MAX_SPEED=0.2
        (meas,fresh) = self.get_measurements() # always going to be fresh

        # =========== PARTICLE FILTER ===========

        self.propogate_particles(self.v, self.omega)
        self.reweight_particles(meas)
        if self.iteration % 6 == 0:
            self.resample()
        if self.iteration < 50: # Only start moving after 50 iterations
            return


        # ============= CONTROLLER =============

        # Relative distance from where you are to checkpoint
        checkpoint_pos = self.get_checkpoint_position()
        dx = self.x_t[0][0] - checkpoint_pos[0]
        dy = self.x_t[1][0] - checkpoint_pos[1]

        # Check/update if you are near checkpoint
        if np.linalg.norm([dx, dy]) < self.CHECKPOINT_RADIUS:
            self.checkpoint += 1
            print "=============================="
            print "Checkpoint!"
            print "Iteration =\t", self.iteration

            if self.checkpoint == len(self.shortest_path):
                self.done = True
                print "Arrived at goal!"
                return
                
            checkpoint_pos = self.get_checkpoint_position()
            dx = self.x_t[0][0] - checkpoint_pos[0]
            dy = self.x_t[1][0] - checkpoint_pos[1]
            print "Actual position =\t", [round(self.x_gt[0][0],3), round(self.x_gt[1][0],3)]
            print "Estimated position =\t", [round(self.x_t[0][0],3), round(self.x_t[1][0],3)]
            print "Next checkpoint position =\t", checkpoint_pos
            print "Distance to next checkpoint =\t", np.linalg.norm([dx, dy])
            print "=============================="
            print ""

        # Calculate v and omega
        kp = 1
        ka = 10
        kb = 0
        rho = np.sqrt(dx * dx + dy * dy)
        beta = -math.atan2(-dy, -dx)
        alpha = -beta - self.x_t[2][0]
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
    max_iters=4000

    occupancy_map = [[1,1,1,1,0,0,1,1,1,1],
                     [1,1,1,1,0,0,1,1,1,1],
                     [1,1,1,1,0,0,1,1,1,1],
                     [1,1,1,1,0,0,1,1,1,1],
                     [1,1,1,1,0,0,1,1,1,1],
                     [1,1,1,1,0,0,1,1,1,1],
                     [0,0,0,0,0,0,1,1,1,1],
                     [0,0,0,0,0,0,1,1,1,1],
                     [0,0,1,1,1,1,1,1,1,1],
                     [0,0,1,1,1,1,1,1,1,1]]
    world_map = [[0.27,0.27,-np.pi/2,1],
                 [0.81,0.27,-np.pi/2,2],
                 [1.89,0.27,-np.pi/2,3],
                 [2.43,0.27,-np.pi/2,4],
                 [0.81,0.81,0,5],
                 [1.89,0.81,np.pi,5],
                 [0.27,1.35,np.pi/2,6],
                 [0.81,1.35,0,7],
                 [1.89,1.35,np.pi,8],
                 [1.89,1.89,np.pi,9],
                 [0.81,2.43,-np.pi/2,10],
                 [1.35,2.43,-np.pi/2,11]]
    pos_init = np.array([[1.25,-1,np.pi/2]]).T
    '''
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
    '''
    # No changes needed after this point
    sim = CreateSim(world_map, occupancy_map, pos_init)
    print "Graph nodes:"
    print sim.graph.nodes
    print "Shortest Path:"
    print sim.shortest_path
    fig = plt.figure(1,figsize=(5,5),dpi=90)
    ax=fig.add_subplot(111)
    plt.hold(True)
    for i in range(len(sim.world_map)):
        ax.arrow(sim.world_map[i][0],sim.world_map[i][1],0.2*np.cos(sim.world_map[i][2]),0.2*np.sin(sim.world_map[i][2]),color=[0.,1.,0.],head_width=0.05,head_length=0.01)
    
    while not sim.done and sim.iteration < max_iters:
        if sim.iteration % 5 == 0:
            ax.plot(sim.x_gt[0,0],sim.x_gt[1,0],'gx')
        sim.iteration += 1
        sim.command_create()

    plt.hold(True)
    if sim.iteration == max_iters:
        print "Max iters reached"

    ax.arrow(sim.x_t[0,0],sim.x_t[1,0],0.1*np.cos(sim.x_t[2,0]),0.1*np.sin(sim.x_t[2,0]),head_width=0.01,head_length=0.08)
    ax.plot(sim.x_t[0,0],sim.x_t[1,0],'rx')
    ax.plot(sim.x_gt[0,0],sim.x_gt[1,0],'gx')
    ax.arrow(sim.x_gt[0,0],sim.x_gt[1,0],0.1*np.cos(sim.x_gt[2,0]),0.1*np.sin(sim.x_gt[2,0]),head_width=0.01,head_length=0.08)

    plt.ylim((-1.5,5))
    plt.xlim((-1.5,5))
    plt.draw()
    plt.show()

if __name__ == "__main__":
    main()

