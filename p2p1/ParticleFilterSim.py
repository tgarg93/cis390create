#!/usr/bin/python
"""
2D Unicycle model-based robot simulator for a particle filter implementation
for CIS 390 Fall 2015 at the University of Pennsylvania
"""

from matplotlib import pyplot as plt
import numpy as np
import time
import math

class CreateSim(object):
    def __init__(self,world_map_init,x_gt_init):
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
        self.v = 0.05
        self.omega = 0
        self.x_t = np.array([[0.0,0.0,0.0]]).T
        self.dt=1.0/60
        self.plot = True # Set to true if you want to update plot
        self.THETA_MAX = np.pi/2 # Angle at which we can see at
        # Particles to plot - list of (x,y,theta,weight)
        self.particles = []
        # self.particles = [(0.5,0.5,0,1),(0.5,-0.5,0,0.5)]; # Example
        # Map stored as array of (x,y,theta) for the april tags
        self.world_map = world_map_init
        self.num_particles = 10
        self.init_particles()

    # Generate particles at a uniform distribution
    def init_particles(self):
        # determine reasonable bounding box given april tags
        min_x = np.min([i[0] for i in self.world_map]) - 1
        max_x = np.max([i[0] for i in self.world_map]) + 1
        min_y = np.min([i[1] for i in self.world_map]) - 1
        max_y = np.max([i[1] for i in self.world_map]) + 1
        range_x = max_x - min_x
        range_y = max_y - min_y

        # uniform distribution between [min_x, max_x]
        rand_x = [i + min_x for i in np.random.random_sample(self.num_particles) * range_x]

        # uniform distribution between [min_y, max_y]
        rand_y = [i + min_y for i in np.random.random_sample(self.num_particles) * range_y]

        # uniform distribution between [0, 2 pi]
        rand_angle = np.random.random_sample(self.num_particles) * 2 * np.pi

        # weights
        weights = [1 / self.num_particles] * self.num_particles

        # create particles
        self.particles = zip(rand_x, rand_y, rand_angle, weights)


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
            theta_new = self.world_map[i][2] + th
            if theta_new > np.pi:
                theta_new -= 2*np.pi
            if theta_new < -np.pi:
                theta_new += 2*np.pi
            if np.absolute(np.arccos(x_new[0]/(x_new[0]**2 + x_new[1]**2))) < self.THETA_MAX:
                meas.append([x_new[0],x_new[1],theta_new,self.world_map[i][3]])
        return meas,True

    # See equation in section 3.2 of project specs
    def propogate_particles(self, v, w):
        updated_x = [i[0] + v * self.dt * np.cos(i[2]) for i in self.particles]
        updated_y = [i[1] + v * self.dt * np.sin(i[2]) for i in self.particles]
        updated_angle = [i[2] + w * self.dt for i in self.particles]
        weights = [i[3] for i in self.particles]
        self.particles = zip(updated_x, updated_y, updated_angle, weights)

    def likelihood(self, x, y, theta):
        a = 1
        b = 1
        c = 1
        numerator = (a * (x ** 2)) + (b * (y ** 2)) + (c * (theta ** 2))
        return math.exp(-numerator / 2)

    # See pseudocode in section 3.3 of project specs
    def reweight_particles(self, measurements):
        self.particles = list(self.particles)

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
                        t_p = np.dot(np.linalg.inv(p_w), t_w)
                        x_t_p = t_p[0][2]
                        y_t_p = t_p[1][2]
                        theta_t_p = math.atan2(t_p[1][0], t_p[0][0])

                        wi = max(wi, self.likelihood(x_t_p - x_t_r, y_t_p - y_t_r, theta_t_p - theta_t_r))
                w.append(wi)
            particle = list(self.particles[i])
            particle[3] = np.prod(np.array(w))
            self.particles[i] = tuple(particle)

    def command_create(self):
        """ 
        YOUR CODE HERE
        """
        MAX_SPEED=0.1
        (meas,fresh) = self.get_measurements()
        kp=0.5
        ka=0.5
        kb=0
        v=0.5
        w=0
        self.propogate_particles(v, w)
        self.reweight_particles(meas)
        return

def main():
    """
    Modify simulation parameters here. In particular, the world map,
    starting position, and max iterations to simulate
    """
    max_iters=50
    world_map = [[0.0,0.0,np.pi/2,1],[1.,0.,np.pi/2,1],[-2.,1.,0.,2]]
    # Other ones of varying difficulty 
    # world_map = [[0.0,0.0,np.pi/2,1],[1.,0.,np.pi/2,2],
    #                   [-2.,0.,0.,1],[-2.,1.,0.,2]]
    # world_map = [[0.0,0.0,np.pi/2,1],[1.,0.,np.pi/2,1],
    #                    [2.,-1.,0.,2],[2.,-2.,0.,1],
    #                    [1.,-3.,-np.pi/2,2],[0.,-3.,-np.pi/2,4],
    #                    [-1.,-2.,np.pi,3],[-1.,-1.,np.pi,3]])
    pos_init = np.array([[1,1,-3*np.pi/4]]).T

    # No changes needed after this point
    sim = CreateSim(world_map, pos_init)
    fig = plt.figure(1,figsize=(5,5),dpi=90)
    ax=fig.add_subplot(111)
    plt.ylim((-2,2))
    plt.xlim((-2,2))
    ax.plot(sim.x_t[0,0],sim.x_t[1,0],'rx')
    plt.hold(True)
    for i in range(len(sim.world_map)):
        ax.arrow(sim.world_map[i][0],sim.world_map[i][1],0.2*np.cos(sim.world_map[i][2]),0.2*np.sin(sim.world_map[i][2]),color=[0.,1.,0.],head_width=0.05,head_length=0.01)
    iteration = 0
    while not sim.done and iteration < max_iters:
        sim.command_create()
        if sim.plot:
            ax.plot(sim.x_t[0,0],sim.x_t[1,0],'rx')
            ax.plot(sim.x_gt[0,0],sim.x_gt[1,0],'gx')
            ax.arrow(sim.x_gt[0,0],sim.x_gt[1,0],0.1*np.cos(sim.x_gt[2,0]),0.1*np.sin(sim.x_gt[2,0]),head_width=0.01,head_length=0.08)
            for i in range(len(sim.particles)):
                ax.plot(sim.particles[i][0],sim.particles[i][1],'bo',ms=5*sim.particles[i][3])
            plt.draw()
            plt.show(block=False)
        iteration += 1
    print "Max iters reached"
    plt.show()

if __name__ == "__main__":
    main()

