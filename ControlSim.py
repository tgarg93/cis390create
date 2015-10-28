#!/usr/bin/python
"""
2D Unicycle model-based robot simulator for CIS 390 Fall 2015 at the
University of Pennsylvania
"""

from matplotlib import pyplot as plt
import numpy as np
import time
import math

class CreateSim(object):
    def __init__(self):
        """
        Initialize starting pose here. Goal is always the origin.
        """
        self.done=False
        self.x=1
        self.y=1
        self.theta=-3*np.pi/4-0.2
        self.dt=1.0/60

    def get_marker_pose(self):
        """
        Functions are the same as on the Creates, although there is no
        simulation for fresh detections or field of view constraints
        """
        return self.x,self.y,self.theta,True

    def command_velocity(self,vx,wz):
        """
        Simulate the robot's motion using Euler integration
        """
        self.x+=self.dt*vx*np.cos(self.theta)
        self.y+=self.dt*vx*np.sin(self.theta)
        self.theta+=self.dt*wz
        if self.theta>np.pi:
            self.theta-=2*np.pi
        if self.theta<-np.pi:
            self.theta+=2*np.pi
        if np.sqrt(self.x**2+self.y**2)<0.01:
            self.done=True
        return

    def command_create(self, goal_x, goal_y):
        """
        YOUR CODE HERE
        kp, ka and kb are gains on rho, alpha and beta
        For the first project, kb should always be set to 0, as we do not
        require alignment of the create's pose with the marker, and otherwise
        you will most likely lose the marker from your field of view.
        Set kp and ka as you see fit.
        """
        MAX_SPEED=1
        x, y, theta, fresh = self.get_marker_pose()
        x -= goal_x
        y -= goal_y
        kp=0.5
        ka=0.5
        kb=0
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

        self.command_velocity(v, w)
        return

def main():
    sim = CreateSim()
    fig = plt.figure(1,figsize=(5,5),dpi=90)
    ax=fig.add_subplot(111)
    plt.ylim((-2,2))
    plt.xlim((-2,2))
    ax.plot(sim.x,sim.y,'rx')
    plt.hold(True)
    iter = 0
    while not sim.done and iter<1000:
        sim.command_create()
        if (iter%10)==0:
            ax.plot(sim.x,sim.y,'rx')
            ax.arrow(sim.x,sim.y,0.05*np.cos(sim.theta),0.05*np.sin(sim.theta),head_width=0.005,head_length=0.01)
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
