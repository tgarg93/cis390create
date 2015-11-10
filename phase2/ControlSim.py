#!/usr/bin/python
"""
2D Unicycle model-based robot simulator with measurement and process noise
for CIS 390 Fall 2015 at the University of Pennsylvania
"""

from matplotlib import pyplot as plt
import numpy as np
import time

class CreateSim(object):
    def __init__(self):
        """
        Initialize starting pose and uncertaqinty parameters here. Goal is always the origin.
        """
        self.done=False
        self.x_gt = np.array([[1,1,-3*np.pi/4-0.2]]).T
        self.v = 0.05
        self.omega = 0
        self.Q_t = np.eye(3)
        self.R_t = np.eye(3)
        self.x_t = np.array([[0,0,0]]).T
        self.P_t = np.array([[10.**6,0,0],[0,10.**6,0],[0,0,10.**6]])
        self.dt=1.0/60
        self.noise = 0.01

    def get_marker_pose(self):
        """
        Functions are the same as on the Creates, although there is no
        simulation for fresh detections or field of view constraints
        """
        x_noisy = self.x_gt + np.array([np.random.normal(0, self.noise, 3)]).T
        return x_noisy,True

    def F_matrix(self,dt,v,theta_t):
        """
        Same as on the Creates.
        """	
        return np.array([[1, 0, -dt*v*np.sin(theta_t)],
                         [0, 1,  dt*v*np.cos(theta_t)],
                         [0, 0,                     1]])

    def command_velocity(self,vx,wz):
        """
        Simulate the robot's motion using Euler integration
        """
        vx_noisy = vx + np.random.normal(0, self.noise)
        wz_noisy = wz + np.random.normal(0, self.noise)
        self.x_t[0,0]+=self.dt*vx_noisy*np.cos(self.x_t[2,0])
        self.x_t[1,0]+=self.dt*vx_noisy*np.sin(self.x_t[2,0])
        self.x_t[2,0]+=self.dt*wz_noisy
        if self.x_t[2,0]>np.pi:
            self.x_t[2,0]-=2*np.pi
        if self.x_t[2,0]<-np.pi:
            self.x_t[2,0]+=2*np.pi
        # This part computes robot's motion based on the groundtruth
        self.x_gt[0,0]+=self.dt*vx*np.cos(self.x_gt[2,0])
        self.x_gt[1,0]+=self.dt*vx*np.sin(self.x_gt[2,0])
        self.x_gt[2,0]+=self.dt*wz
        if self.x_gt[2,0]>np.pi:
            self.x_gt[2,0]-=2*np.pi
        if self.x_gt[2,0]<-np.pi:
            self.x_gt[2,0]+=2*np.pi
        if np.sqrt(self.x_t[0,0]**2+self.x_t[1,0]**2)<0.01:
            self.done=True
        return
        
    def command_create(self):
        """ 
        YOUR CODE HERE
        """
        MAX_SPEED=0.1
        (z_t,fresh) = self.get_marker_pose()
        kp=0.5
        ka=0.5
        kb=0

        v = 0.05
        w = 0

        if fresh:
            # Update step
            K = np.dot(self.P_t,np.linalg.inv(self.P_t + self.R_t))
            self.x_t = self.x_t + np.dot(K, (z_t - self.x_t))
            self.P_t = np.dot((np.identity(3) - K),self.P_t)
            self.command_velocity(v, w)
        else:
            # Prediction step
            self.x_t[0] = self.x_t[0] + (v * dt * np.cos(self.x_t[2]))
            self.x_t[1] = self.x_t[1] + (v * dt * np.sin(self.x_t[2]))
            self.x_t[2] = self.x_t[2] + (w * dt)
            F = self.F_matrix(dt, v, x_t[2])
            self.P_t = np.dot(np.dot(F,self.P_t), np.transpose(F)) + self.Q_t

        return

def main():
    sim = CreateSim()
    fig = plt.figure(1,figsize=(5,5),dpi=90)
    ax=fig.add_subplot(111)
    plt.ylim((-2,2))
    plt.xlim((-2,2))
    ax.plot(sim.x_t[0,0],sim.x_t[1,0],'rx')
    plt.hold(True)
    iter = 0
    while not sim.done and iter<1000:
        sim.command_create()
        if (iter%10)==0:
            ax.plot(sim.x_t[0,0],sim.x_t[1,0],'rx')
	    ax.plot(sim.x_gt[0,0],sim.x_gt[1,0],'gx')
            ax.arrow(sim.x_t[0,0],sim.x_t[1,0],0.05*np.cos(sim.x_t[2,0]),0.05*np.sin(sim.x_t[2,0]),head_width=0.005,head_length=0.01)
            plt.draw()
            plt.show(block=False)
        iter+=1
    if sim.done:
        print "Goal reached!"
    else:
        print "Max iters reached"
    plt.show()

if __name__ == "__main__":
    main()

