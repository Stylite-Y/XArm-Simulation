import os
import pickle
import datetime
import numpy as np
import scipy.linalg
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import casadi as ca
from casadi import sin as s
from casadi import cos as c

class Bipedal_hybrid():
    def __init__(self):
        self.opti = ca.Opti()
        self.qmax = 0.75*np.pi
        self.dqmax = 64
        self.umax = 27

        # time and collection defination related parameter
        self.N = 2
        self.m = [1.0, 1.0]
        self.I = [0.0075, 0.0075]
        self.l = [0.15, 0.15]
        self.I_ = [self.m[i]*self.l[i]**2+self.I[i] for i in range(2)]

        self.L = [0.3, 0.3]

        # * define variable
        self.q = [self.opti.variable(2) for _ in range(self.N)]
        self.dq = [self.opti.variable(2) for _ in range(self.N)]
        # self.ddq = [(self.dq[i+1]-self.dq[i]) /
        #                 0.001 for i in range(self.N-1)]

        # ! set the last u to be zero at constraint
        self.u = [self.opti.variable(2) for _ in range(self.N)]

        pass

    def MassMatrix(self, q):
        m0 = self.m[0]
        m1 = self.m[1]
        lc0 = self.l[0]
        lc1 = self.l[1]
        L0 = self.L[0]
        L1 = self.L[1]
        I0 = self.I[0]
        I1 = self.I[1]

        M11 = I0 + I1 + m0*lc0**2+m1*(L0**2+lc1**2+2*L0*lc1*c(q[1]))

        M12 = I1 + m1*(lc1**2+L0*lc1*c(q[1]))
        M21 = M12
        M22 = I1 + m1*lc1**2

        return [[M11, M12],
                [M21, M22]]

x = ca.MX.sym('x',2,2)
y = x[0][0]
print(y)
robot = Bipedal_hybrid()
massM = robot.MassMatrix(robot.q[1])
massMX = ca.MX(massM)
M2 = massM@massM
print(M2)
# qinv = ca.inv(robot.q)
# print(massM)
print(robot.q)
# help(ca.MatrixCommon)
Minv = ca.inv(massM)
print(Minv)
print(type(robot.q))
pass