import re
from casadi.casadi import linspace
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from scipy.integrate import odeint

class A():
    def __init__(self, x):
        self.opti = ca.Opti()
        self.a = x[0]
        self.b = x[1]
        self.NN = [self.a, self.b]

    def m(self, q):
        print(q)
        m11 = q[0]
        m12 = q[1]
        m21 = m12
        m22 = q[0]
        
        return [[m11, m12], [m21, m22]]

    def c(self, q, dq):
        c1 = -q[0] + dq[0]
        c2 = -q[1] + dq[1]

        return [c1, c2]

    def g(self, q):
        g1 = -q[0]
        g2 = -q[1]
        return [g1, g2]

def updatestate(q0, dq0, u1):
    print(q0, dq0, u1)
    robot = A([1, 2])
    m = robot.m(q0)
    m = np.asarray(m)
    m_inv = np.linalg.inv(m)

    print(m)
    print(m_inv)

    def odefun(y, t):
        q1, q2, dq1, dq2 = y
        c = robot.c([q1, q2], [dq1, dq2])
        g = robot.g([q1, q2])

        dydt1 = [dq1, dq2,
                u1[0] - m_inv[0][0]*(c[0]+g[0]) - m_inv[0][1]*(c[1]+g[1]),
                u1[1] - m_inv[1][0]*(c[0]+g[0]) - m_inv[1][1]*(c[1]+g[1])]

        dydt = [dq1, dq2, -dq1+2*q1, -dq2+2*q2]
        print("--------------------------")
        print(dydt1)
        print(dydt)

        return dydt1

    q_init = []
    q_init.extend(q0)
    q_init.extend(dq0)
    print(q_init)
    t = [0.0,0.2, 0.4, 0.6, 0.8, 1.0]
    t = [0.0,1.0]
    sol = odeint(odefun, q_init, t)
    return sol

class ab():
    def __init__(self):
        print("class A the first step")
        pass

    def b(self):
        print("class A the second step")

def visual():
    from numpy import sin, cos
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as integrate
    import matplotlib.animation as animation
    from collections import deque

    G = 9.8  # acceleration due to gravity, in m/s^2
    L1 = 1.0  # length of pendulum 1 in m
    L2 = 1.0  # length of pendulum 2 in m
    L = L1 + L2  # maximal length of the combined pendulum
    M1 = 1.0  # mass of pendulum 1 in kg
    M2 = 1.0  # mass of pendulum 2 in kg
    t_stop = 5  # how many seconds to simulate
    history_len = 500  # how many trajectory points to display


    def derivs(state, t):

        dydx = np.zeros_like(state)
        dydx[0] = state[1]

        delta = state[2] - state[0]
        den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
        dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                    + M2 * G * sin(state[2]) * cos(delta)
                    + M2 * L2 * state[3] * state[3] * sin(delta)
                    - (M1+M2) * G * sin(state[0]))
                / den1)

        dydx[2] = state[3]

        den2 = (L2/L1) * den1
        dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                    + (M1+M2) * G * sin(state[0]) * cos(delta)
                    - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                    - (M1+M2) * G * sin(state[2]))
                / den2)

        return dydx

    # create a time array from 0..t_stop sampled at 0.02 second steps
    dt = 0.02
    t = np.arange(0, t_stop, dt)

    # th1 and th2 are the initial angles (degrees)
    # w10 and w20 are the initial angular velocities (degrees per second)
    th1 = 120.0
    w1 = 0.0
    th2 = -10.0
    w2 = 0.0

    # initial state
    state = np.radians([th1, w1, th2, w2])

    # integrate your ODE using scipy.integrate.
    y = integrate.odeint(derivs, state, t)

    x1 = L1*sin(y[:, 0])
    y1 = -L1*cos(y[:, 0])

    x2 = L2*sin(y[:, 2]) + x1
    y2 = -L2*cos(y[:, 2]) + y1

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        if i == 0:
            history_x.clear()
            history_y.clear()

        history_x.appendleft(thisx[2])
        history_y.appendleft(thisy[2])

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (i*dt))
        return line, trace, time_text

    print(len(y))
    ani = animation.FuncAnimation(
        fig, animate, len(y), interval=dt*1000, blit=True)
    
    plt.show()


def visual2():
    import sys
    import numpy as np
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # Pendulum rod lengths (m), bob masses (kg).
    L1, L2 = 1, 1
    m1, m2 = 1, 1
    # The gravitational acceleration (m.s-2).
    g = 9.81

    def deriv(y, t, L1, L2, m1, m2):
        """Return the first derivatives of y = theta1, z1, theta2, z2."""
        theta1, z1, theta2, z2 = y

        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

        theta1dot = z1
        z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
                (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
        theta2dot = z2
        z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
                m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
        return theta1dot, z1dot, theta2dot, z2dot

    def calc_E(y):
        """Return the total energy of the system."""

        th1, th1d, th2, th2d = y.T
        V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
        T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
                2*L1*L2*th1d*th2d*np.cos(th1-th2))
        return T + V

    # Maximum time, time point spacings and the time grid (all in s).
    tmax, dt = 30, 0.01
    t = np.arange(0, tmax+dt, dt)
    # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
    y0 = np.array([3*np.pi/7, 0, 3*np.pi/4, 0])

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

    # Check that the calculation conserves total energy to within some tolerance.
    EDRIFT = 0.05
    # Total energy from the initial conditions
    E = calc_E(y0)
    if np.max(np.sum(np.abs(calc_E(y) - E))) > EDRIFT:
        sys.exit('Maximum energy drift of {} exceeded.'.format(EDRIFT))

    # Unpack z and theta as a function of time
    theta1, theta2 = y[:,0], y[:,2]

    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    # Plotted bob circle radius
    r = 0.05
    # Plot a trail of the m2 bob's position for the last trail_secs seconds.
    trail_secs = 1
    # This corresponds to max_trail time points.
    max_trail = int(trail_secs / dt)

    def make_plot(i):
        # Plot and save an image of the double pendulum configuration for time
        # point i.
        # The pendulum rods.
        ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
        # Circles representing the anchor point of rod 1, and bobs 1 and 2.
        c0 = Circle((0, 0), r/2, fc='k', zorder=10)
        c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
        c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
        ax.add_patch(c0)
        ax.add_patch(c1)
        ax.add_patch(c2)

        # The trail will be divided into ns segments and plotted as a fading line.
        ns = 20
        s = max_trail // ns

        for j in range(ns):
            imin = i - (ns-j)*s
            if imin < 0:
                continue
            imax = imin + s + 1
            # The fading looks better if we square the fractional length along the
            # trail.
            alpha = (j/ns)**2
            ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                    lw=2, alpha=alpha)

        # Centre the image on the fixed anchor point, and ensure the axes are equal
        ax.set_xlim(-L1-L2-r, L1+L2+r)
        ax.set_ylim(-L1-L2-r, L1+L2+r)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')
        # plt.savefig('frames/_img{:04d}.png'.format(i//di), dpi=72)
        plt.show()
        plt.cla()


    # Make an image every di time points, corresponding to a frame rate of fps
    # frames per second.
    # Frame rate, s-1
    fps = 10
    di = int(1/fps/dt)
    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)

    for i in range(0, t.size, di):
        print(i // di, '/', t.size // di)
        make_plot(i)

def stepresponse():
    pass

if __name__ == "__main__":
    q0 = [1.0, 0.0]
    dq0 = [0.0, 0.0]
    u1 = [0.0, 0.0]
    q1 = np.array([[1, 21,5], [1, 5, 7]])
    b = q1.tolist()
    print(list(q1), b[0][1], q0[1])
    
    # sol = updatestate(q0, dq0, u1)
    # print("===================")
    # print(sol)
    # print(sol[-1,0:2])

    # temp = A()
    # # temp.b()
    # print(np.linspace(0, 4, 4))
    # visual2()

