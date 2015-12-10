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
        self.v = 0.05
        self.omega = 0
        self.dt=1.0/60
        self.x_gt = x_gt_init # real robot position
        self.x_t = np.array([[0.0,0.0,0.0]]).T # estimated robot position
        self.plot = True # Set to true if you want to update plot
        self.THETA_MAX = np.pi/2 # Angle at which we can see at
        self.particles = []
        self.num_particles = 5
        self.world_map = world_map_init
        self.iteration = 0
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

    # Estimates likelihood that particle is at correct position
    def likelihood(self, x, y, theta):
        a = 1.0
        b = 1.0
        c = 5.0
        numerator = (a * (x ** 2)) + (b * (y ** 2)) + (c * (theta ** 2))
        return math.exp(-numerator / 2.0)

    # ==================== MAIN FUNCTIONS ====================

    # Generate particles at a uniform distribution
    def init_particles(self):

        # gaussian distribution centered at initial position
        rand_x = np.random.normal(self.x_gt[0], 0.5, self.num_particles)
        rand_y = np.random.normal(self.x_gt[1], 0.5, self.num_particles)
        rand_angle = np.random.normal(self.x_gt[2], 0.5, self.num_particles)

        # weights
        weights = [1.0 / self.num_particles] * self.num_particles

        # create particles
        self.particles = zip(rand_x, rand_y, rand_angle, weights)
        self.particles = [list(i) for i in self.particles]

        # initialize x_t
        self.updated_robot_position()

    # This adds some noise to x, y, and angle
    def propogate_particles(self, v, w):
        updated_x = [i[0] + v * self.dt * np.cos(i[2]) for i in self.particles] + self.noise(0, 0.04)
        updated_y = [i[1] + v * self.dt * np.sin(i[2]) for i in self.particles] + self.noise(0, 0.04)
        updated_angle = [i[2] + w * self.dt for i in self.particles] + self.noise(0, 0.02)
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
        """ 
        YOUR CODE HERE
        """
        MAX_SPEED=0.1
        (meas,fresh) = self.get_measurements()
        kp=0.5
        ka=0.5
        kb=0
        v=0.1
        w=0
        self.iteration += 1
        self.propogate_particles(v, w)
        self.reweight_particles(meas)
        if self.iteration % 6 == 0:
            self.resample()
        self.command_velocity(v, w)
        return

def main():

    """
    Modify simulation parameters here. In particular, the world map,
    starting position, and max iterations to simulate
    """
    max_iters=10000
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
    # Other ones of varying difficulty 
    #world_map = [[0.0,0.0,np.pi/2,0],[1.,0.,np.pi/2,1],[-2.,1.,0.,2]]
    # world_map = [[0.0,0.0,np.pi/2,1],[1.,0.,np.pi/2,2],
    #                   [-2.,0.,0.,1],[-2.,1.,0.,2]]
    # world_map = [[0.0,0.0,np.pi/2,1],[1.,0.,np.pi/2,1], [2.,-1.,0.,2],[2.,-2.,0.,1], [1.,-3.,-np.pi/2,2],[0.,-3.,-np.pi/2,4], [-1.,-2.,np.pi,3],[-1.,-1.,np.pi,3]]
    #pos_init = np.array([[1,1,-3*np.pi/4]]).T

    # You probably won't need this if you're embedding things in a tkinter plot...
    plt.ion()

    fig = plt.figure(1,figsize=(10,10),dpi=90)
    ax = fig.add_subplot(111)
    x_t = np.array([[1.25,-1,np.pi/2]]).T
    x = 1
    y = 1
    z = [.96, .97, .98, .99, 1, 1.01, 1.02, 1.03, 1.04]
    i = 0
    line1, = ax.plot(x, y, 'rx')
    while True:
        i += 1
        i %= len(z)
        line1.set_ydata(z[i])
        fig.canvas.draw()
    
    '''
    for i in range(0, 1000):
        x_t[0][0] += .01
        x_t[1][0] += .01
        line1.set_xdata([x_t[0][0]])
        line1.set_ydata([x_t[1][0]])
        fig.canvas.draw()
        return
    '''

    """
    Modify simulation parameters here. In particular, the world map,
    starting position, and max iterations to simulate
    """
    
    '''
    x_t = [[0], [0], [0]]

    fig = plt.figure(1,figsize=(5,5),dpi=90)
    ax=fig.add_subplot(111)
    plt.ylim((-2,2))
    plt.xlim((-2,2))
    for i in range(0, 1):
        fig.clear()
        ax.plot(x_t[0][0],x_t[1][0],'rx')
        ax.plot(x_t[0][0] - .5 ,x_t[1][0],'rx')
        x_t[0][0] += .01
        x_t[1][0] += .01
    #plt.draw()

    plt.show()
    '''
    
    '''
    plt.hold(True)
    for i in range(len(sim.world_map)):
        ax.arrow(sim.world_map[i][0],sim.world_map[i][1],0.2*np.cos(sim.world_map[i][2]),0.2*np.sin(sim.world_map[i][2]),color=[0.,1.,0.],head_width=0.05,head_length=0.01)
    iteration = 0
    while not sim.done and iteration<max_iter:
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
    '''

if __name__ == "__main__":
    main()