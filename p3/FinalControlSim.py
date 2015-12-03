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
            world_map_init: list of lists [x,y,theta,id], each representing
             a landmark in the world map. Note that there is an ambiguity
             between landmarks such that multiple landmarks may have the
             same id number.
            x_gt_init: 3x1 numpy array that represents the true starting
             position of the robot
        """
        self.done=False
        self.x_gt = x_gt_init
        self.v = 0
        self.omega = 0
        self.x_t = np.array([[0.0,0.0,0.0]]).T
        self.dt=1.0/60
        self.THETA_MAX = np.pi/2 # Angle at which we can see markers
        # Particles to plot - list of (x,y,theta,weight)
        self.particles = []
        # self.particles = [(0.5,0.5,0,1),(0.5,-0.5,0,0.5)]; # Example
        # Map stored as array of (x,y,theta) for the april tags
        self.world_map = world_map_init
        self.occupancy_map = occupancy_map_init
        
        self.iteration = 0
        self.graph = self.generate_graph(occupancy_map_init)
        self.checkpoint = 0
        
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
        vx += np.random.normal(0,0.05*self.dt*vx,1)
        wx += np.random.normal(0,0.05*self.dt*abs(wz),1)
        # This part computes robot's motion based on the groundtruth
        self.x_gt[0,0]+=self.dt*vx*np.cos(self.x_gt[2,0])
        self.x_gt[1,0]+=self.dt*vx*np.sin(self.x_gt[2,0])
        self.x_gt[2,0]+=self.dt*wz
        if self.x_gt[2,0]>np.pi:
            self.x_gt[2,0]-=2*np.pi
        if self.x_gt[2,0]<-np.pi:
            self.x_gt[2,0]+=2*np.pi
        return

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
            th_w_t = self.world_map[i][2]
            H_WT = np.array([[np.cos(th_w_t), -np.sin(th_w_t), self.world_map[i][0]],
                            [np.sin(th_w_t), np.cos(th_w_t), self.world_map[i][1]],
                            [0., 0., 1.]])
            H_RT = np.linalg.solve(H_WR,H_WT)
            x_new = H_RT[0:2,2]
            theta_new = math.atan2(H_RT[1,0], H_RT[0,0])
            if theta_new > np.pi:
                theta_new -= 2*np.pi
            if theta_new < -np.pi:
                theta_new += 2*np.pi
            if np.absolute(np.arccos(x_new[0]/(math.sqrt(x_new[0]**2 + x_new[1]**2)))) < self.THETA_MAX:
                meas.append([x_new[0],x_new[1],theta_new,self.world_map[i][3]])
        return meas,True

    def command_create(self):
        """ 
        YOUR CODE HERE
        """
        MAX_SPEED=0.2
        (meas,fresh) = self.get_measurements()
        """
        The particle filter estimate might take a lot of time. If you'd like,
        you can uncomment the line below to run it constantly at the beginning to 
        allow the filter to converge, and then only run it every 60th iteration
        """
        #if self.iteration < 50 or self.iteration % 50:
            # Run particle filter here

        # Only start moving after 50 iterations
        #if self.iteration<50:
            # return
        
        #self.iteration+=1
        
        #self.command_velocity(self.v, self.omega)
        return

def main():
    """
    Modify simulation parameters here. In particular, the world map,
    starting position, and max iterations to simulate
    """
    max_iters=2000
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

    # No changes needed after this point
    sim = CreateSim(world_map, occupancy_map, pos_init)

    fig = plt.figure(1,figsize=(5,5),dpi=90)
    ax=fig.add_subplot(111)
    plt.hold(True)
    for i in range(len(sim.world_map)):
        ax.arrow(sim.world_map[i][0],sim.world_map[i][1],0.2*np.cos(sim.world_map[i][2]),0.2*np.sin(sim.world_map[i][2]),color=[0.,1.,0.],head_width=0.05,head_length=0.01)
    while not sim.done and sim.iteration < max_iters:
        if sim.iteration % 5 == 0:
            ax.plot(sim.x_gt[0,0],sim.x_gt[1,0],'gx')
        sim.iteration += 1
        if sim.iteration % 100 == 0:
            print sim.iteration
        sim.command_create()
    plt.hold(True)
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

