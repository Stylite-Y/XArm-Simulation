'''
1. 双足机器人轨迹优化
2. 将接触的序列预先制定
3. 格式化配置参数输入和结果输出
4. 混合动力学系统，系统在机器人足底和地面接触的时候存在切换
5. 加入双臂的运动
8. 2022.10.18:
        - 将目标函数归一化
9. 2023.03.01:
        - 将质量m, 加速比协同优化
10. 2023.03.02:
        - 四连杆动力学
11. 2023.03.06:
        - 同时优化三个减速比和两个质量
        - sol1: gamma = [2.0, 2.9, 4.2]
                mm2 = [3.0, 3.75]
'''

from ast import In, walk
import os
import yaml
import datetime
import pickle
import casadi as ca
from casadi import sin as s
from casadi import cos as c
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import time
from ruamel.yaml import YAML
from math import acos, atan2, sqrt, sin, cos
from DataProcess import DataProcess
from scipy import signal
import matplotlib.animation as animation


class Bipedal_hybrid():
    def __init__(self, Im, cfg):
        self.opti = ca.Opti()
        # load config parameter
        # self.CollectionNum = cfg['Controller']['CollectionNum']

        # self.Pf = [0.2, 0.2, 0.2]
        # self.Vf = [0.8, 0.8, 0.8]
        # self.Ff = [0.05, 0.05, 0.05]
        # self.Pwf = [0.05, 0.05, 0.05]

        # time and collection defination related parameter
        self.T = cfg['Controller']['Period']
        # self.dt = cfg['Controller']['dt']
        # self.N = int(self.T / self.dt)
        self.Im = Im
        self.N = cfg['Controller']['CollectionNum']
        self.dt = self.T / self.N
        
        # mass and geometry related parameter
        # self.m = cfg['Robot']['Mass']['mass']
        # self.I = cfg['Robot']['Mass']['inertia']
        self.mm1 = [10, 15]
        self.mm2 = self.opti.variable(2)
        self.gam = self.opti.variable(3)
        # self.gamhip = self.opti.variable(1)
        # self.gamhip = [1.5]

        self.L = [cfg['Robot']['Geometry']['L_leg'],
                  cfg['Robot']['Geometry']['L_body'],             
                  cfg['Robot']['Geometry']['L_arm'],
                  cfg['Robot']['Geometry']['L_farm']]
        
        self.I = [self.mm1[0]*self.L[0]**2/12, self.mm1[1]*self.L[1]**2/12, 
                self.mm2[0]*self.L[2]**2/12, self.mm2[1]*self.L[3]**2/12]
        self.l = cfg['Robot']['Mass']['massCenter']
        # self.I_ = [self.m[i]*self.l[i]**2+self.I[i] for i in range(4)]

        # motor parameter
        self.motor_cs = cfg['Robot']['Motor']['CriticalSpeed']
        self.motor_ms = cfg['Robot']['Motor']['MaxSpeed']
        self.motor_mt = cfg['Robot']['Motor']['MaxTorque']

        # evironemnt parameter
        self.mu = cfg['Environment']['Friction_Coeff']
        self.g = cfg['Environment']['Gravity']
        self.damping = cfg['Robot']['damping']

        # boundary parameter
        self.bound_fy = cfg['Controller']['Boundary']['Fy']
        self.bound_fx = cfg['Controller']['Boundary']['Fx']
        # self.F_LB = [self.bound_fx[0], self.bound_fy[0]]
        # self.F_UB = [self.bound_fx[1], self.bound_fy[1]]

        # t_f = self.gamhip[0]
        tor_k = 12
        self.qmax = [0.5*np.pi, 0.9*np.pi, 1.2*np.pi, 0.9*np.pi]
        self.dqmax = [36, 54/self.gam[0], 54/self.gam[1], 54/self.gam[2]]
        self.umax = [tor_k, 36*self.gam[0], 36*self.gam[1], 36*self.gam[2]]

        self.u_LB = [-tor_k] + [-self.motor_mt]*3
        self.u_UB = [tor_k] + [self.motor_mt]*3

        # shank, thigh, body, arm, forearm
        self.q_LB = [-np.pi/10, -np.pi/30, 0, -np.pi*0.9] 
        self.q_UB = [np.pi/2, np.pi*0.9, 1.2*np.pi, 0]   

        self.dq_LB = [-36] + [-self.motor_ms]*3   # arm 

        self.dq_UB = [36] + [self.motor_ms]*3 # arm 

        # * define variable
        self.q = [self.opti.variable(4) for _ in range(self.N)]
        self.dq = [self.opti.variable(4) for _ in range(self.N)]
        self.ddq = [(self.dq[i+1]-self.dq[i]) /
                        self.dt for i in range(self.N-1)]

        # ! set the last u to be zero at constraint
        self.u = [self.opti.variable(4) for _ in range(self.N)]

        # ! Note, the last force represents the plused force at the contact moment
        # self.F = [self.opti.variable(2) for _ in range(self.N)]

        pass

    def MassMatrix(self, q):
        m0 = self.mm1[0]
        m1 = self.mm1[1]
        m2 = self.mm2[0]
        m3 = self.mm2[1]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        lc3 = self.l[3]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]
        I0 = self.I[0]
        I1 = self.I[1]
        I2 = self.I[2]
        I3 = self.I[3]
        gam0 = self.gam[0]
        gam1 = self.gam[1]
        gam2 = self.gam[2]
        Im = self.Im

        M11 = I0 + I1 + I2 + I3 + L0**2*(m1+m2+m3)+L1**2*(m2+m3)+L2**2*(m3) +\
            lc0**2*m0+lc1**2*m1+lc2**2*m2+lc3**2*m3+\
            2*L0*(L1*m2*c(q[1])+L1*m3*c(q[1])+L2*m3*c(q[1]+q[2])+lc1*m1*c(q[1])+lc2*m2*c(q[1]+q[2])+lc3*m3*c(q[1]+q[2]+q[3]))+\
            2*L1*(L2*m3*c(q[2])+lc2*m2*c(q[2])+lc3*m3*c(q[2]+q[3]))+\
            2*L2*lc3*m3*c(q[3])

        M12 = I1 + I2 + I3 + L1**2*(m2+m3)+L2**2*(m3) +\
            lc1**2*m1+lc2**2*m2+lc3**2*m3+\
            1*L0*(L1*m2*c(q[1])+L1*m3*c(q[1])+L2*m3*c(q[1]+q[2])+lc1*m1*c(q[1])+lc2*m2*c(q[1]+q[2])+lc3*m3*c(q[1]+q[2]+q[3]))+\
            2*L1*(L2*m3*c(q[2])+lc2*m2*c(q[2])+lc3*m3*c(q[2]+q[3]))+\
            2*L2*lc3*m3*c(q[3])

        M13 = I2 + I3 + L2**2*(m3) +\
            lc2**2*m2+lc3**2*m3+\
            1*L0*(L2*m3*c(q[1]+q[2])+lc2*m2*c(q[1]+q[2])+lc3*m3*c(q[1]+q[2]+q[3]))+\
            1*L1*(L2*m3*c(q[2])+lc2*m2*c(q[2])+lc3*m3*c(q[2]+q[3]))+\
            2*L2*lc3*m3*c(q[3])

        M14 = I3+\
            lc3**2*m3+\
            1*L0*(lc3*m3*c(q[1]+q[2]+q[3]))+\
            1*L1*(lc3*m3*c(q[2]+q[3]))+\
            1*L2*lc3*m3*c(q[3])
        
        M21 = M12
        M22 = Im*gam0**2 + I1 + I2 + I3+ L1**2*(m2+m3)+L2**2*(m3) +\
            lc1**2*m1+lc2**2*m2+lc3**2*m3+\
            2*L1*(L2*m3*c(q[2])+lc2*m2*c(q[2])+lc3*m3*c(q[2]+q[3]))+\
            2*L2*lc3*m3*c(q[3])
        M23 = I2 + I3+ L2**2*(m3) +\
            lc2**2*m2+lc3**2*m3+\
            1*L1*(L2*m3*c(q[2])+lc2*m2*c(q[2])+lc3*m3*c(q[2]+q[3]))+\
            2*L2*lc3*m3*c(q[3])
        M24 = I3+\
            lc3**2*m3+\
            1*L1*(lc3*m3*c(q[2]+q[3]))+\
            1*L2*lc3*m3*c(q[3])

        M31 = M13
        M32 = M23
        M33 = Im*gam1**2 + I2 + I3+ L2**2*(m3) +\
            lc2**2*m2+lc3**2*m3+\
            2*L2*lc3*m3*c(q[3])
        M34 = I3+\
            lc3**2*m3+\
            1*L2*lc3*m3*c(q[3])

        M41 = M14
        M42 = M24
        M43 = M34
        M44 = I3+lc3**2*m3+Im*gam2**2

        return [[M11, M12, M13, M14],
                [M21, M22, M23, M24],
                [M31, M32, M33, M34],
                [M41, M42, M43, M44]]

    def coriolis(self, q, dq):
        m0 = self.mm1[0]
        m1 = self.mm1[1]
        m2 = self.mm2[0]
        m3 = self.mm2[1]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        lc3 = self.l[3]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]

        C0 = -2*L0*(L1*m2*s(q[1])+L1*m3*s(q[1])+L2*m3*s(q[1]+q[2])+lc1*m1*s(q[1])+lc2*m2*s(q[1]+q[2])+lc3*m3*s(q[1]+q[2]+q[3])) * dq[0]*dq[1] \
            -2*(L0*L2*m3*s(q[1]+q[2])+L0*lc2*m2*s(q[1]+q[2])+L0*lc3*m3*s(q[1]+q[2]+q[3])+ L1*L2*m3*s(q[2])+L1*lc2*m2*s(q[2])+L1*lc3*m3*s(q[2]+q[3]))*dq[0]*dq[2] \
            -2*lc3*m3*(L0*s(q[1]+q[2]+q[3])+L1*s(q[2]+q[3])+L2*s(q[3]))*dq[0]*dq[3] \
            -1*L0*(L1*m2*s(q[1])+L1*m3*s(q[1])+L2*m3*s(q[1]+q[2])+lc1*m1*s(q[1])+lc2*m2*s(q[1]+q[2])+lc3*m3*s(q[1]+q[2]+q[3])) * dq[1]*dq[1] \
            -2*(L0*L2*m3*s(q[1]+q[2])+L0*lc2*m2*s(q[1]+q[2])+L0*lc3*m3*s(q[1]+q[2]+q[3])+ L1*L2*m3*s(q[2])+L1*lc2*m2*s(q[2])+L1*lc3*m3*s(q[2]+q[3]))*dq[1]*dq[2] \
            -2*lc3*m3*(L0*s(q[1]+q[2]+q[3])+L1*s(q[2]+q[3])+L2*s(q[3]))*dq[1]*dq[3] \
            -1*(L0*L2*m3*s(q[1]+q[2])+L0*lc2*m2*s(q[1]+q[2])+L0*lc3*m3*s(q[1]+q[2]+q[3])+ L1*L2*m3*s(q[2])+L1*lc2*m2*s(q[2])+L1*lc3*m3*s(q[2]+q[3]))*dq[2]*dq[2] \
            -2*lc3*m3*(L0*s(q[1]+q[2]+q[3])+L1*s(q[2]+q[3])+L2*s(q[3]))*dq[2]*dq[3] \
            -1*lc3*m3*(L0*s(q[1]+q[2]+q[3])+L1*s(q[2]+q[3])+L2*s(q[3]))*dq[3]*dq[3] \
        # C1 = C1 - 0.2 * dq[0] 
        
        C1 = L0*(L1*m2*s(q[1])+L1*m3*s(q[1])+L2*m3*s(q[1]+q[2])+lc1*m1*s(q[1])+lc2*m2*s(q[1]+q[2])+lc3*m3*s(q[1]+q[2]+q[3])) * dq[0]*dq[0] \
            -2*L1*(L2*m3*s(q[2])+lc2*m2*s(q[2])+lc3*m3*s(q[2]+q[3]))*dq[0]*dq[2] \
            -2*lc3*m3*(L1*s(q[2]+q[3])+L2*s(q[3]))*dq[0]*dq[3] \
            -2*L1*(L2*m3*s(q[2])+lc2*m2*s(q[2])+lc3*m3*s(q[2]+q[3]))*dq[1]*dq[2] \
            -2*lc3*m3*(L1*s(q[2]+q[3])+L2*s(q[3]))*dq[1]*dq[3] \
            -1*L1*(L2*m3*s(q[2])+lc2*m2*s(q[2])+lc3*m3*s(q[2]+q[3]))*dq[2]*dq[2] \
            -2*lc3*m3*(L1*s(q[2]+q[3])+L2*s(q[3]))*dq[2]*dq[3] \
            -1*lc3*m3*(L1*s(q[2]+q[3])+L2*s(q[3]))*dq[3]*dq[3]
        # C2 = C2 - 0.2 * dq[1] 

        C2 = (L0*L2*m3*s(q[1]+q[2])+L0*lc2*m2*s(q[1]+q[2])+L0*lc3*m3*s(q[1]+q[2]+q[3])+ L1*L2*m3*s(q[2])+L1*lc2*m2*s(q[2])+L1*lc3*m3*s(q[2]+q[3]))*dq[0]*dq[0] \
            +2*L1*(L2*m3*s(q[2])+lc2*m2*s(q[2])+lc3*m3*s(q[2]+q[3]))*dq[0]*dq[1] \
            -2*L2*lc3*m3*s(q[3])*dq[0]*dq[3] \
            +1*L1*(L2*m3*s(q[2])+lc2*m2*s(q[2])+lc3*m3*s(q[2]+q[3]))*dq[1]*dq[1] \
            -2*L2*lc3*m3*s(q[3])*dq[1]*dq[3] \
            -2*L2*lc3*m3*s(q[3])*dq[2]*dq[3] \
            -1*L2*lc3*m3*s(q[3])*dq[3]*dq[3] \
        # C3 = C3 - 0.2 * dq[2] 

        C3 = lc3*m3*(L0*s(q[1]+q[2]+q[3])+L1*s(q[2]+q[3])+L2*s(q[3]))*dq[0]*dq[0] \
            +2*lc3*m3*(L1*s(q[2]+q[3])+L2*s(q[3]))*dq[0]*dq[1] \
            +2*lc3*m3*(L2*s(q[3]))*dq[0]*dq[2] \
            +1*lc3*m3*(L1*s(q[2]+q[3])+L2*s(q[3]))*dq[1]*dq[1] \
            +2*lc3*m3*(L2*s(q[3]))*dq[1]*dq[2] \
            +1*lc3*m3*(L2*s(q[3]))*dq[2]*dq[2] \

        return [C0, C1, C2, C3]

    def gravity(self, q):
        m0 = self.mm1[0]
        m1 = self.mm1[1]
        m2 = self.mm2[0]
        m3 = self.mm2[1]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        lc3 = self.l[3]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]

        G1 = -(L0*m1*s(q[0]) + L0*m2*s(q[0]) + L0*m3*s(q[0]) + \
            L1*m2*s(q[0]+q[1]) + L1*m3*s(q[0]+q[1]) + L2*m3*s(q[0]+q[1]+q[2]) + \
            lc0*m0*s(q[0]) + lc1*m1*s(q[0]+q[1]) + lc2*m2*s(q[0]+q[1]+q[2]) + lc3*m3*s(q[0]+q[1]+q[2]+q[3]))
        
        G2 = -(L1*m2*s(q[0]+q[1]) + L1*m3*s(q[0]+q[1]) + L2*m3*s(q[0]+q[1]+q[2]) + \
            lc1*m1*s(q[0]+q[1]) + lc2*m2*s(q[0]+q[1]+q[2]) + lc3*m3*s(q[0]+q[1]+q[2]+q[3]))

        G3 = -(L2*m3*s(q[0]+q[1]+q[2]) + \
            lc2*m2*s(q[0]+q[1]+q[2]) + lc3*m3*s(q[0]+q[1]+q[2]+q[3]))
        
        G4 = -(lc3*m3*s(q[0]+q[1]+q[2]+q[3]))

        return [G1*self.g, G2*self.g, G3*self.g, G4*self.g]

        pass

    def inertia_force(self, q, acc):
        mm = self.MassMatrix(q)
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2]+mm[i][3]*acc[3] for i in range(4)]
        
        return inertia_force

    def inertia_force2(self, q, acc):
        # region calculate inertia force, split into two parts
        mm = self.MassMatrix(q)
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2]+mm[i][3]*acc[3] for i in range(4)]
        inertia_main = [mm[i][i]*acc[i] for i in range(4)]
        inertia_coupling = [inertia_force[i]-inertia_main[i] for i in range(4)]
        # endregion
        return inertia_main, inertia_coupling

    def SupportForce(self, q, dq, ddq):
        m0 = self.mm1[0]
        m1 = self.mm1[1]
        m2 = self.mm2[0]
        m3 = self.mm2[1]
        l0 = self.l[0]
        l1 = self.l[1]
        l2 = self.l[2]
        l3 = self.l[3]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]
        # acceleration cal
        
        Fx3 = 0.5*L3*m3*ddq[3]*c(q[0]+q[1]+q[2]+q[3])-0.5*L3*m3*dq[3]**2*s(q[0]+q[1]+q[2]+q[3])
        Fy3 = m3*self.g-0.5*L3*m3*ddq[3]*s(q[0]+q[1]+q[2]+q[3])-0.5*L3*m3*dq[3]**2*c(q[0]+q[1]+q[2]+q[3])
        Fx2 = 0.5*L2*m2*ddq[2]*c(q[0]+q[1]+q[2])-0.5*L2*m2*dq[2]**2*s(q[0]+q[1]+q[2]) - Fx3
        Fy2 = m2*self.g-0.5*L2*m2*ddq[2]*s(q[0]+q[1]+q[2])-0.5*L2*m2*dq[2]**2*c(q[0]+q[1]+q[2]) - Fy3
        Fx1 = 0.5*L1*m1*ddq[1]*c(q[0]+q[1])-0.5*L1*m1*dq[1]**2*s(q[0]+q[1]) - Fx2
        Fy1 = m1*self.g-0.5*L1*m1*ddq[1]*s(q[0]+q[1])-0.5*L1*m1*dq[1]**2*c(q[0]+q[1]) - Fy2
        AccFx = 0.5*L0*m0*ddq[0]*c(q[0])-0.5*L0*m0*dq[0]**2*s(q[0]) - Fx1
        AccFy = m0*self.g-0.5*L0*m0*ddq[0]*s(q[0])-0.5*L0*m0*dq[0]**2*c(q[0]) - Fy1

        AccF = [AccFx, AccFy]

        return AccF
        pass

    def SupportForce2(self, mm2, q, dq, ddq):
        m0 = self.mm1[0]
        m1 = self.mm1[1]
        m2 = mm2[0][0]
        m3 = mm2[0][1]
        l0 = self.l[0]
        l1 = self.l[1]
        l2 = self.l[2]
        l3 = self.l[3]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]
        # acceleration cal
        
        Fx3 = 0.5*L3*m3*ddq[3]*c(q[0]+q[1]+q[2]+q[3])-0.5*L3*m3*dq[3]**2*s(q[0]+q[1]+q[2]+q[3])
        Fy3 = m3*self.g-0.5*L3*m3*ddq[3]*s(q[0]+q[1]+q[2]+q[3])-0.5*L3*m3*dq[3]**2*c(q[0]+q[1]+q[2]+q[3])
        Fx2 = 0.5*L2*m2*ddq[2]*c(q[0]+q[1]+q[2])-0.5*L2*m2*dq[2]**2*s(q[0]+q[1]+q[2]) - Fx3
        Fy2 = m2*self.g-0.5*L2*m2*ddq[2]*s(q[0]+q[1]+q[2])-0.5*L2*m2*dq[2]**2*c(q[0]+q[1]+q[2]) - Fy3
        Fx1 = 0.5*L1*m1*ddq[1]*c(q[0]+q[1])-0.5*L1*m1*dq[1]**2*s(q[0]+q[1]) - Fx2
        Fy1 = m1*self.g-0.5*L1*m1*ddq[1]*s(q[0]+q[1])-0.5*L1*m1*dq[1]**2*c(q[0]+q[1]) - Fy2
        AccFx = 0.5*L0*m0*ddq[0]*c(q[0])-0.5*L0*m0*dq[0]**2*s(q[0]) - Fx1
        AccFy = m0*self.g-0.5*L0*m0*ddq[0]*s(q[0])-0.5*L0*m0*dq[0]**2*c(q[0]) - Fy1

        AccF = [AccFx, AccFy]

        return AccF
        pass


    def SupportForce3(self, q, dq, ddq):
        m0 = self.mm1[0]
        m1 = self.mm1[1]
        m2 = self.mm2[0]
        m3 = self.mm2[1]
        l0 = self.l[0]
        l1 = self.l[1]
        l2 = self.l[2]
        l3 = self.l[3]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]
        # acceleration cal
        ddx0 = -l0*s(q[0])*dq[0]**2 + l0*c(q[0])*ddq[0]
        ddy0 = -l0*c(q[0])*dq[0]**2 - l0*s(q[0])*ddq[0]
        ddx1 = -L0*s(q[0])*dq[0]**2 - l1*s(q[0]+q[1])*(dq[0]+dq[1])**2 + \
                L0*c(q[0])*ddq[0] + l1*c(q[0]+q[:1])*(ddq[0]+ddq[1])
        ddy1 = -L0*c(q[0])*dq[0]**2 - l1*c(q[0]+q[1])*(dq[0]+dq[1])**2 - \
                L0*s(q[0])*ddq[0] - l1*s(q[0]+q[1])*(dq[0]+dq[1])
        ddx2 = -L0*s(q[0])*dq[0]**2 - L1*s(q[0]+q[1])*(dq[0]+dq[1])**2 - l2*s(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])**2 + \
                L0*c(q[0])*ddq[0] + L1*c(q[0]+q[1])*(ddq[0]+ddq[1]) + l2*c(q[0]+q[1]+q[2])*(ddq[0]+ddq[1]+ddq[2])
        ddy2 = -L0*c(q[0])*dq[0]**2 - L1*c(q[0]+q[1])*(dq[0]+dq[1])**2 - l2*c(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])**2 - \
                L0*s(q[0])*ddq[0] - L1*s(q[0]+q[1])*(ddq[0]+ddq[1]) - l2*s(q[0]+q[1]+q[2])*(ddq[0]+ddq[1]+ddq[2])
        ddx3 = -L0*s(q[0])*dq[0]**2 - L1*s(q[0]+q[1])*(dq[0]+dq[1])**2 - L2*s(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])**2 - l3*s(q[0]+q[1]+q[2]+q[3])*(dq[0]+dq[1]+dq[2]+dq[3])**2 + \
                L0*c(q[0])*ddq[0] + L1*c(q[0]+q[1])*(ddq[0]+ddq[1]) + L2*c(q[0]+q[1]+q[2])*(ddq[0]+ddq[1]+ddq[2])+ l3*c(q[0]+q[1]+q[2]+q[3])*(ddq[0]+ddq[1]+ddq[2]+ddq[3])
        ddy3 = -L0*c(q[0])*dq[0]**2 - L1*c(q[0]+q[1])*(dq[0]+dq[1])**2 - L2*c(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])**2 - l3*c(q[0]+q[1]+q[2]+q[3])*(dq[0]+dq[1]+dq[2]+dq[3])**2 - \
                L0*s(q[0])*ddq[0] - L1*s(q[0]+q[1])*(ddq[0]+ddq[1]) - L2*s(q[0]+q[1]+q[2])*(ddq[0]+ddq[1]+ddq[2]) - l3*s(q[0]+q[1]+q[2]+q[3])*(ddq[0]+ddq[1]+ddq[2]+ddq[3])
        
        AccFx = -(m0*ddx0 + m1*ddx1 + m2*ddx2 + m3*ddx3)
        AccFy = -(m0*ddy0 + m1*ddy1 + m2*ddy2 + m3*ddy3) - (m0*self.g + m1*self.g + m2*self.g + m3*self.g)

        AccF = [AccFx, AccFy]

        return AccF
        pass


    @staticmethod
    def get_posture(q):
        L = [0.9, 0.5, 0.4, 0.4]
        lsx = np.zeros(2)
        lsy = np.zeros(2)
        ltx = np.zeros(2)
        lty = np.zeros(2)
        lax = np.zeros(2)
        lay = np.zeros(2)
        lafx = np.zeros(2)
        lafy = np.zeros(2)
        lsx[0] = 0
        lsx[1] = lsx[0] + L[0]*np.sin(q[0])
        lsy[0] = 0
        lsy[1] = lsy[0] + L[0]*np.cos(q[0])

        ltx[0] = 0 + L[0]*np.sin(q[0])
        ltx[1] = ltx[0] + L[1]*np.sin(q[0]+q[1])
        lty[0] = 0 + L[0]*np.cos(q[0])
        lty[1] = lty[0] + L[1]*np.cos(q[0]+q[1])

        lax[0] = 0 + L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1])
        lax[1] = lax[0] + L[2]*np.sin(q[0]+q[1]+q[2])
        lay[0] = 0 + L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1])
        lay[1] = lay[0] + L[2]*np.cos(q[0]+q[1]+q[2])

        lafx[0] = 0 + L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1])+L[2]*np.sin(q[0]+q[1]+q[2])
        lafx[1] = lafx[0] + L[3]*np.sin(q[0]+q[1]+q[2]+q[3])
        lafy[0] = 0 + L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1])+L[2]*np.cos(q[0]+q[1]+q[2])
        lafy[1] = lafy[0] + L[3]*np.cos(q[0]+q[1]+q[2]+q[3])
        return [lsx, lsy, ltx, lty, lax, lay, lafx, lafy]

    @staticmethod
    def get_motor_boundary(speed, MaxTorque=36, CriticalSpeed=27, MaxSpeed=53):
        upper = MaxTorque - (speed-CriticalSpeed) / \
            (MaxSpeed-CriticalSpeed)*MaxTorque
        upper = np.clip(upper, 0, MaxTorque)
        lower = -MaxTorque + (speed+CriticalSpeed) / \
            (-MaxSpeed+CriticalSpeed)*MaxTorque
        lower = np.clip(lower, -MaxTorque, 0)
        return upper, lower

    pass


class nlp():
    def __init__(self, legged_robot, cfg, armflag = True):
        # load parameter
        self.cfg = cfg
        self.armflag = armflag
        self.trackingCoeff = cfg["Optimization"]["CostCoeff"]["trackingCoeff"]
        self.velCoeff = cfg["Optimization"]["CostCoeff"]["VelCoeff"]
        self.powerCoeff = cfg["Optimization"]["CostCoeff"]["powerCoeff"]
        self.forceCoeff = cfg["Optimization"]["CostCoeff"]["forceCoeff"]
        self.smoothCoeff = cfg["Optimization"]["CostCoeff"]["smoothCoeff"]
        self.impactCoeff = cfg["Optimization"]["CostCoeff"]["ImpactCoeff"]
        self.forceRatio = cfg["Environment"]["ForceRatio"]
        max_iter = cfg["Optimization"]["MaxLoop"]

        self.cost = self.Cost(legged_robot)
        legged_robot.opti.minimize(self.cost)

        self.ceq = self.getConstraints(legged_robot)
        legged_robot.opti.subject_to(self.ceq)

        p_opts = {"expand": True, "error_on_fail": False}
        s_opts = {"max_iter": max_iter}
        legged_robot.opti.solver("ipopt", p_opts, s_opts)
        self.initialGuess(legged_robot)
        pass

    def initialGuess(self, walker):
        init = walker.opti.set_initial
        # region: sol1
        init(walker.gam[0], 1.6)
        init(walker.gam[1], 4.0)
        init(walker.gam[2], 4.0)
        init(walker.mm2[0], 5.0)
        init(walker.mm2[1], 4.0)
        # endregion
        for i in range(walker.N):
            for j in range(4):
                if j == 2:
                    init(walker.q[i][j], np.pi)
                    init(walker.dq[i][j], 0)
                else:
                    init(walker.q[i][j], 0)
                    init(walker.dq[i][j], 0)
            pass

    def Cost(self, walker):
        # region aim function of optimal control
        power = 0
        force = 0
        VelTar = 0
        PosTar = 0
        Ptar = [0, 0, np.pi, 0.0]

        # Pf = walker.Pf
        # Vf = walker.Vf
        # Ff = walker.Ff
        # Pwf = walker.Pwf
        Pf = [0.3]*4
        Vf = [0.6]*4
        Ff = [0.1]*4
        # Pf = [0.3]*4
        # Vf = [0.4]*4
        # Ff = [0.3]*4
        # Pwf = [0.05, 0.05, 0.05]

        qmax = walker.qmax
        dqmax = walker.dqmax
        umax = walker.umax
        # endregion
        
        for i in range(walker.N):
            for k in range(4):
                # power += ((walker.dq[i][k]*walker.u[i][k]) / (dqmax[k]*umax[k]))**2 * walker.dt*Pwf[k]
                force += (walker.u[i][k] / umax[k])**2 * walker.dt * Ff[k]  

                VelTar += (walker.dq[i][k]/dqmax[k])**2 * walker.dt * Vf[k]
                PosTar += ((walker.q[i][k] - Ptar[k])/qmax[k])**2 * walker.dt * Pf[k]              
                pass
            pass
        
        for j in range(4):
            VelTar += (walker.dq[-1][j]/dqmax[j])**2 * Vf[j] * 20
            PosTar += ((walker.q[-1][j] - Ptar[j])/qmax[j])**2 * Pf[j] * 100
       
        u = walker.u

        smooth = 0
        AM = [100, 400, 100]
        for i in range(walker.N-1):
            for k in range(4):
                smooth += ((u[i+1][k]-u[i][k])/10)**2
                pass
            pass

        res = 0
        # res = (res + power*self.powerCoeff) if (self.powerCoeff > 1e-6) else res
        # res = (res + VelTar*self.velCoeff) if (self.velCoeff > 1e-6) else res
        # res = (res + PosTar*self.trackingCoeff) if (self.trackingCoeff > 1e-6) else res
        # res = (res + force*self.forceCoeff) if (self.forceCoeff > 1e-6) else res
        # res = (res + smooth*self.smoothCoeff) if (self.smoothCoeff > 1e-6) else res
        res = (res + PosTar)
        res = (res + VelTar)
        # res = (res + power)
        res = (res + force)

        return res

    def getConstraints(self, walker):
        ceq = []
        # region dynamics constraints
        # continuous dynamics
        #! 约束的数量为 (6+6）*（NN1-1+NN2-1）
        for j in range(walker.N):
            if j < (walker.N-1):
                ceq.extend([walker.q[j+1][k]-walker.q[j][k]-walker.dt/2 *
                            (walker.dq[j+1][k]+walker.dq[j][k]) == 0 for k in range(4)])
                inertia = walker.inertia_force(
                    walker.q[j], walker.ddq[j])
                coriolis = walker.coriolis(
                    walker.q[j], walker.dq[j])
                gravity = walker.gravity(walker.q[j])
                # ceq.extend([inertia[0]+gravity[0]+coriolis[0] == 0])
                # ceq.extend([inertia[k+1]+gravity[k+1]+coriolis[k+1] -
                #             walker.u[j][k] == 0 for k in range(4)])
                ceq.extend([inertia[k]+gravity[k]+coriolis[k] -
                            walker.u[j][k] == 0 for k in range(4)])

            if not self.armflag:
                ceq.extend([walker.q[j][2] == np.pi])
                ceq.extend([walker.dq[j][2] == 0])
                ceq.extend([walker.q[j][3] == 0])
                ceq.extend([walker.dq[j][3] == 0])
            
            pass

        # endregion

        ceq.extend([walker.mm2[0]-0.8*walker.mm2[1] >= 0])

        # region leg locomotion constraint
        # for i in range(walker.N-1):
        #     AccF = walker.SupportForce(walker.q[i], walker.dq[i], walker.ddq[i])
        #     Fx = AccF[0]
        #     Fy = AccF[1]
        #     ceq.extend([Fy >= 0])
        #     ceq.extend([Fy <= 4000])
        #     ceq.extend([Fx <= 4000])
        #     ceq.extend([Fx >= -4000])
        #     ceq.extend([Fy*walker.mu - Fx >= 0])  # 摩擦域条件
        #     ceq.extend([Fy*walker.mu + Fx >= 0])  # 摩擦域条件
        # endregion

        gamma = [1.0, walker.gam[0], walker.gam[1], walker.gam[2]]

        # region boundary constraint
        for temp_q in walker.q:
            ceq.extend([walker.opti.bounded(walker.q_LB[j],
                        temp_q[j], walker.q_UB[j]) for j in range(4)])
            pass
        for temp_dq in walker.dq:
            ceq.extend([walker.opti.bounded(walker.dq_LB[j]/gamma[j],
                        temp_dq[j], walker.dq_UB[j]/gamma[j]) for j in range(4)])
            pass
        for temp_u in walker.u:
            ceq.extend([walker.opti.bounded(walker.u_LB[j]*gamma[j],
                        temp_u[j], walker.u_UB[j]*gamma[j]) for j in range(4)])
            pass
        # endregion

        # region motor external characteristic curve
        cs = []
        ms = []
        mt = []
        for k in range(4):
            cs.append(walker.motor_cs/gamma[k])
            ms.append(walker.motor_ms/gamma[k])
            mt.append(walker.motor_mt*gamma[k])
        for j in range(len(walker.u)):
            ceq.extend([walker.u[j][k]-ca.fmax(mt[k] - (walker.dq[j][k] -
                                                        cs[k])/(ms[k]-cs[k])*mt[k], 0) <= 0 for k in range(4)])
            ceq.extend([walker.u[j][k]-ca.fmin(-mt[k] + (walker.dq[j][k] +
                                                            cs[k])/(-ms[k]+cs[k])*mt[k], 0) >= 0 for k in range(4)])
            pass

        ceq.extend([walker.opti.bounded(1.0,
                        walker.gam[0], 2.0)])
        ceq.extend([walker.opti.bounded(1.0,
                        walker.gam[1], 10.0)])
        ceq.extend([walker.opti.bounded(1.0,
                        walker.gam[2], 10.0)])
        ceq.extend([walker.opti.bounded(3.0,
                        walker.mm2[0], 10.0)])
        ceq.extend([walker.opti.bounded(1.0,
                        walker.mm2[1], 10.0)])
        # endregion

        theta = np.pi/30
        ceq.extend([walker.q[0][0]==theta])
        ceq.extend([walker.q[0][1]==theta*0.2])
        ceq.extend([walker.q[0][2]==np.pi])
        ceq.extend([walker.q[0][3]==0.0])

        ceq.extend([walker.dq[0][0]==0])
        ceq.extend([walker.dq[0][1]==0])
        ceq.extend([walker.dq[0][2]==0])
        ceq.extend([walker.dq[0][3]==0])

        # region smooth constraint
        for j in range(len(walker.u)-1):
            ceq.extend([(walker.u[j][k]-walker.u
                        [j+1][k])**2 <= 50 for k in range(4)])
            pass
        # endregion

        return ceq

    def solve_and_output(self, robot, flag_save=True, StorePath="./"):
        # solve the nlp and stroge the solution
        q = []
        dq = []
        ddq = []
        u = []
        t = []
        gamma = []
        mm2 = []
        try:
            sol1 = robot.opti.solve()
            gamma.append(sol1.value(robot.gam))
            mm2.append(sol1.value(robot.mm2))
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([sol1.value(robot.q[j][k]) for k in range(4)])
                dq.append([sol1.value(robot.dq[j][k])
                            for k in range(4)])
                if j < (robot.N-1):
                    ddq.append([sol1.value(robot.ddq[j][k])
                                for k in range(4)])
                    u.append([sol1.value(robot.u[j][k])
                                for k in range(4)])
                else:
                    ddq.append([sol1.value(robot.ddq[j-1][k])
                                for k in range(4)])
                    u.append([sol1.value(robot.u[j-1][k])
                                for k in range(4)])
                pass
            pass
        except:
            value = robot.opti.debug.value
            gamma.append(value(robot.gam))
            mm2.append(value(robot.mm2))
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([value(robot.q[j][k])
                            for k in range(4)])
                dq.append([value(robot.dq[j][k])
                            for k in range(4)])
                if j < (robot.N-1):
                    ddq.append([value(robot.ddq[j][k])
                                for k in range(4)])
                    u.append([value(robot.u[j][k])
                                for k in range(4)])
                else:
                    ddq.append([value(robot.ddq[j-1][k])
                                for k in range(4)])
                    u.append([value(robot.u[j-1][k])
                                for k in range(4)])
                pass
            pass
        finally:
            q = np.asarray(q)
            dq = np.asarray(dq)
            ddq = np.asarray(ddq)
            u = np.asarray(u)
            t = np.asarray(t).reshape([-1, 1])

            return q, dq, ddq, u, t, gamma, mm2


class SolutionData():
    def __init__(self, old_solution=None):
        if old_solution is None:
            self.q = None
            self.dq = None
            self.ddq = None
            self.u = None
            self.t = None
            self.N = 0
            self.dt = None
            self.sudden_id = None
            pass
        else:
            self.q = old_solution[:, 0:5]
            self.dq = old_solution[:, 5:10]
            self.ddq = old_solution[:, 10:15]
            self.u = old_solution[:, 15:20]
            self.t = old_solution[:, 20]

            index = 0
            while(self.t[index] != self.t[index+1]):
                index += 1
                pass
            self.sudden_id = index
            self.ddq[index] = (self.dq[index+1] -
                               self.dq[index]) / (self.t[1]-self.t[0])
            self.ddq[-1] = (self.dq[0]-self.dq[-1])/(self.t[1]-self.t[0])
            self.f[index] = self.f[index] / (self.t[1]-self.t[0])/0.2
            self.f[-1] = self.f[-1] / (self.t[1]-self.t[0])/0.2
            self.N = len(self.t)
            self.dt = np.ones_like(self.t) * (self.t[1]-self.t[0])
            self.dt[index] = (self.t[1]-self.t[0])/5
            self.dt[-1] = (self.t[1]-self.t[0])/5
            pass
        pass
    pass


def main():
    # region optimization trajectory for bipedal hybrid robot system
    vis_flag = True
    save_flag = True
    # armflag = False
    armflag = True
    # endregion

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    ani_path = StorePath + "/data/animation/"
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # seed = None
    seed = StorePath + str(todaytime) + "_sol.npy"
    # region load config file
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Biped_balance_Gam_M_opt.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # endregion

    # region create robot and NLP problem
    Im = 5e-4
    robot = Bipedal_hybrid(Im, cfg)
    # nonlinearOptimization = nlp(robot, cfg, seed=seed)
    # nonlinearOptimization = nlp(robot, cfg)
    nonlinearOptimization = nlp(robot, cfg, armflag)
    # endregion
    q, dq, ddq, u, t, gamma, mm2 = nonlinearOptimization.solve_and_output(
        robot, flag_save=save_flag, StorePath=StorePath)
    
    print("="*50)
    print("gamma:", gamma)
    print("="*50)
    print("m:", mm2[0][0],mm2[0][1])
    print("="*50)

    # region: support force cal
    Fx = np.array([0.0])
    Fy = np.array([0.0])
    for i in range(robot.N-1):
        AccF = robot.SupportForce2(mm2,q[i], dq[i], ddq[i])
        tempx = AccF[0]
        tempy = AccF[1]
        Fx = np.concatenate((Fx, [tempx]))
        Fy = np.concatenate((Fy, [tempy]))
        if i == robot.N-2:
            Fx = np.concatenate((Fx, [tempx]))
            Fy = np.concatenate((Fy, [tempy]))
    Fx = Fx[1:]
    Fy = Fy[1:]
    F = np.concatenate(([Fx], [Fy]), axis=1)
    b,a = signal.butter(3, 0.12, 'lowpass')
    Fy2 = signal.filtfilt(b, a, Fy)
    # endregion

    #region: costfun cal
    # Ptar = [0, 0, np.pi]
    # Pcostfun = 0.0
    # Vcostfun = 0.0
    # Fcostfun = 0.0
    # Power = 0.0
    # for i in range(robot.N):
    #     for k in range(3):
    #         if i < robot.N-1:
    #             Pcostfun += ((q[i][k] - Ptar[k])/robot.qmax)**2 * robot.dt * robot.Pf[k]
    #             Fcostfun += (u[i][k] / robot.umax)**2 * robot.dt * robot.Ff[k]  
    #             Vcostfun += ((dq[i][k])/robot.dqmax)**2 * robot.dt * robot.Vf[k]
    #             Power += ((dq[i][k] * u[i][k])/(robot.qmax*robot.umax))**2 * robot.dt * robot.Pwf[k]
    #         else:
    #             Pcostfun += ((q[i][k] - Ptar[k])/robot.qmax)**2 * robot.dt * robot.Pf[k] * 100
    #             Vcostfun += ((dq[i][k])/robot.dqmax)**2 * robot.dt * robot.Vf[k]* 20

    # print(Pcostfun, Vcostfun, Fcostfun, Power)
    # endregion

    theta = np.pi/40
    F = 0
    visual = DataProcess(cfg, robot, theta, q, dq, ddq, u, F, t, save_dir, save_flag)
    if save_flag:
        SaveDir = visual.DataSave(save_flag, 0, 0)

    if vis_flag:
        visual.animationFourLink(0, save_flag)

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib as mpl
        from matplotlib.patches import ConnectionPatch
        plt.style.use("science")
        params = {
            'text.usetex': True,
            'image.cmap': 'inferno',
            'lines.linewidth': 1.5,
            'font.size': 15,
            'axes.labelsize': 15,
            'axes.titlesize': 22,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 15,
        }

        plt.rcParams.update(params)
        cmap = mpl.cm.get_cmap('viridis')
        fig = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=False)
        fig2 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=False)
        gs = fig.add_gridspec(1, 1)
        gm = fig2.add_gridspec(1, 1)
        g_data = gs[0].subgridspec(3, 4  , wspace=0.3, hspace=0.33)
        ax_m = fig2.add_subplot(gm[0])

        # gs = fig.add_gridspec(2, 1, height_ratios=[2,1],
        #                       wspace=0.3, hspace=0.33)
        # g_data = gs[1].subgridspec(3, 6, wspace=0.3, hspace=0.33)

        # ax_m = fig.add_subplot(gs[0])
        ax_p = [fig.add_subplot(g_data[0, i]) for i in range(4)]
        ax_v = [fig.add_subplot(g_data[1, i]) for i in range(4)]
        ax_u = [fig.add_subplot(g_data[2, i]) for i in range(4)]

        # vel = [robot.foot_vel(q[i, :], dq[i, :]) for i in range(len(q[:, 0]))]

        # plot robot trajectory here
        ax_m.axhline(y=0, color='k')
        num_frame = 6
        for tt in np.linspace(0, robot.T, num_frame):
            idx = np.argmin(np.abs(t-tt))
            # print(idx)
            pos = Bipedal_hybrid.get_posture(q[idx, :])
            ax_m.plot(pos[0], pos[1], 'o-', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1)
            ax_m.plot(pos[2], pos[3], 'o:', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1)
            ax_m.plot(pos[4], pos[5], 'o-', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1)
            ax_m.plot(pos[6], pos[7], 'o-', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1)
            patch = patches.Rectangle(
                (pos[2][0]-0.02, pos[3][0]-0.05), 0.04, 0.1, alpha=tt/robot.T*0.8+0.2, lw=0, color=cmap(tt/robot.T))
            ax_m.add_patch(patch)
            ax_m.axis('equal')
            pass

        ax_m.set_ylabel('z(m)')
        ax_m.set_xlabel('x(m)')
        ax_m.xaxis.set_tick_params()
        ax_m.yaxis.set_tick_params()

        [ax_v[i].plot(t, dq[:, i]) for i in range(4)]
        ax_v[0].set_ylabel('Velocity(m/s)')
        [ax_p[i].plot(t, q[:, i]) for i in range(4)]
        ax_p[0].set_ylabel('Position(m)')

        # ax_u[0].plot(t[1:], Fx)
        # ax_u[0].plot(t[1:], Fy)
        ax_u[0].plot(t, u[:, 0])
        ax_u[0].set_xlabel('ankle')
        ax_u[0].set_ylabel('Torque (N.m)')
        ax_u[1].plot(t, u[:, 1])
        ax_u[1].set_xlabel('waist')
        ax_u[2].plot(t, u[:, 2])
        ax_u[2].set_xlabel('shoulder')
        ax_u[3].plot(t, u[:, 3])
        ax_u[3].set_xlabel('shoulder')
        # [ax_u[j].set_title(title_u[j]) for j in range(4)]

        fig3 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=False)
        plt.plot(t, Fx, label = 'Fx')
        plt.plot(t, Fy, label = 'Fy')
        plt.plot(t, Fy2, label = 'Fy')
        plt.xlabel("time (s)")
        plt.ylabel("Force (N)")
        plt.legend()

        if save_flag:
            savename1 =  SaveDir + "Traj.jpg"
            savename3 =  SaveDir + "Fy.jpg"
            savename2 =  SaveDir + "Pos-Vel-uF.jpg"
            fig.savefig(savename2)
            fig2.savefig(savename1)
            fig3.savefig(savename3)
        

        plt.show()


        pass
    # F = [Fx, Fy]
    # return u, Fy2, t, Pcostfun, Vcostfun, Fcostfun, Power

if __name__ == "__main__":
    main()
    pass
