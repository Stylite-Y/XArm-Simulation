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
    
if __name__ == "__main__":
    q0 = [1.0, 0.0]
    dq0 = [0.0, 0.0]
    u1 = [0.0, 0.0]
    
    sol = updatestate(q0, dq0, u1)
    print("===================")
    print(sol)
    print(sol[-1,0:2])

    # temp = A()
    # # temp.b()
    # print(np.linspace(0, 4, 4))
    pass

