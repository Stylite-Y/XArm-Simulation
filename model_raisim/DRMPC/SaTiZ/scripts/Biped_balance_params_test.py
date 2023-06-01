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
11. 2023.03.06:
        - 同时优化三个减速比和两个质量
12. 2023.03.06:
        - 测试三种不同工况下的质心、调节时间、力不同
13. 2023.05.24:
        - 角动量计算函数: Momentum() and Momentum()
        - 曲线带箭头绘制函数: add_arrow_to_line2D()
14. 2023.05.26:
        - 相空间绘制: PhasePlot()
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
        ## model test
        # self.mm2 = [3.5, 4.4]
        # self.gam = [2.0, 3.2, 4.0]
        ## traj-opt test
        # self.mm2 = [3.0, 3.75]
        # self.gam = [2.0, 2.9, 4.2]
        ## noarm test
        self.mm2 = [3.0, 3.75]
        self.gam = [2.0, 3.0, 4.0]


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
        self.q_LB = [-np.pi/10, -np.pi/30, 0, -np.pi*0.8] 
        self.q_UB = [np.pi/2, np.pi*0.9, 4*np.pi/3, 0]   

        self.dq_LB = [-self.motor_ms]*4   # arm 

        self.dq_UB = [self.motor_ms]*4 # arm 

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
    def get_compos(q):
        L = [0.9, 0.5, 0.4, 0.4]
        l = [0.5, 0.25, 0.2, 0.2]
        lsx = np.zeros(2)
        lsy = np.zeros(2)
        ltx = np.zeros(2)
        lty = np.zeros(2)
        lax = np.zeros(2)
        lay = np.zeros(2)
        lafx = np.zeros(2)
        lafy = np.zeros(2)
        lsx[0] = 0
        lsx[1] = lsx[0] + l[0]*np.sin(q[0])
        lsy[0] = 0
        lsy[1] = lsy[0] + l[0]*np.cos(q[0])

        ltx[0] = 0 + L[0]*np.sin(q[0])
        ltx[1] = ltx[0] + l[1]*np.sin(q[0]+q[1])
        lty[0] = 0 + L[0]*np.cos(q[0])
        lty[1] = lty[0] + l[1]*np.cos(q[0]+q[1])

        lax[0] = 0 + L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1])
        lax[1] = lax[0] + l[2]*np.sin(q[0]+q[1]+q[2])
        lay[0] = 0 + L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1])
        lay[1] = lay[0] + l[2]*np.cos(q[0]+q[1]+q[2])

        lafx[0] = 0 + L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1])+L[2]*np.sin(q[0]+q[1]+q[2])
        lafx[1] = lafx[0] + l[3]*np.sin(q[0]+q[1]+q[2]+q[3])
        lafy[0] = 0 + L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1])+L[2]*np.cos(q[0]+q[1]+q[2])
        lafy[1] = lafy[0] + l[3]*np.cos(q[0]+q[1]+q[2]+q[3])
        return [lsx, lsy, ltx, lty, lax, lay, lafx, lafy]

    def get_compos2(q):
        L = [0.9, 0.5, 0.4, 0.4]
        l = [0.5, 0.25, 0.2, 0.2]
        x1 = l[0]*np.sin(q[0])
        y1 = l[0]*np.cos(q[0])
        
        x2 = L[0]*np.sin(q[0]) + l[1]*np.sin(q[0]+q[1])
        y2 = L[0]*np.cos(q[0]) + l[1]*np.cos(q[0]+q[1])

        x3 = L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1]) + l[2]*np.sin(q[0]+q[1]+q[2])
        y3 = L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1])+ l[2]*np.cos(q[0]+q[1]+q[2])

        x4 = L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1])+L[2]*np.sin(q[0]+q[1]+q[2]) +\
            l[3]*np.sin(q[0]+q[1]+q[2]+q[3])
        y4 = L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1])+L[2]*np.cos(q[0]+q[1]+q[2]) +\
            l[3]*np.cos(q[0]+q[1]+q[2]+q[3])

        return [x1, y1, x2, y2, x3, y3, x4, y4]

    @staticmethod
    def get_comvel(q, dq):
        L = [0.9, 0.5, 0.4, 0.4]
        l = [0.5, 0.25, 0.2, 0.2]
        v1x = l[0]*np.cos(q[0])*dq[0]
        v1y = -l[0]*np.sin(q[0])*dq[0]

        v2x = L[0]*np.cos(q[0])*dq[0] + l[1]*np.cos(q[0]+q[1])*(dq[0]+dq[1])
        v2y = -L[0]*np.sin(q[0])*dq[0] - l[1]*np.sin(q[0]+q[1])*(dq[0]+dq[1])

        v3x = L[0]*np.cos(q[0])*dq[0] + L[1]*np.cos(q[0]+q[1])*(dq[0]+dq[1]) +\
                 l[2]*np.cos(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])
        v3y = -L[0]*np.sin(q[0])*dq[0] - L[1]*np.sin(q[0]+q[1])*(dq[0]+dq[1]) -\
                 l[2]*np.sin(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])

        v4x = L[0]*np.cos(q[0])*dq[0] + L[1]*np.cos(q[0]+q[1])*(dq[0]+dq[1]) +\
              L[2]*np.cos(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2]) + \
              l[3]*np.cos(q[0]+q[1]+q[2]+q[3])*(dq[0]+dq[1]+dq[2]+dq[3])
        v4y = -L[0]*np.sin(q[0])*dq[0] - L[1]*np.sin(q[0]+q[1])*(dq[0]+dq[1]) -\
               L[2]*np.sin(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2]) - \
               l[3]*np.sin(q[0]+q[1]+q[2]+q[3])*(dq[0]+dq[1]+dq[2]+dq[3])

        return [v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y]


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
        try:
            sol1 = robot.opti.solve()
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

            return q, dq, ddq, u, t


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
    armflag = False
    # armflag = True
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
    q, dq, ddq, u, t = nonlinearOptimization.solve_and_output(
        robot, flag_save=save_flag, StorePath=StorePath)
    
    print("="*50)
    print("gamma:", robot.gam)
    print("="*50)
    print("m:", robot.mm2)
    print("="*50)

    # region: support force cal
    # Fx = np.array([0.0])
    # Fy = np.array([0.0])
    # for i in range(robot.N-1):
    #     AccF = robot.SupportForce2(robot.mm2,q[i], dq[i], ddq[i])
    #     tempx = AccF[0]
    #     tempy = AccF[1]
    #     Fx = np.concatenate((Fx, [tempx]))
    #     Fy = np.concatenate((Fy, [tempy]))
    #     if i == robot.N-2:
    #         Fx = np.concatenate((Fx, [tempx]))
    #         Fy = np.concatenate((Fy, [tempy]))
    # Fx = Fx[1:]
    # Fy = Fy[1:]
    # F = np.concatenate(([Fx], [Fy]), axis=1)
    # b,a = signal.butter(3, 0.12, 'lowpass')
    # Fy2 = signal.filtfilt(b, a, Fy)
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

    # 质心位置求解
    com_x = []
    com_y = []
    mass = [robot.mm1[0], robot.mm1[1], robot.mm2[0], robot.mm2[1]]
    for i in range(robot.N):
        pos = Bipedal_hybrid.get_compos(q[i, :])
        comtmp_x = (mass[0]*pos[0][1]+mass[1]*pos[2][1]+mass[2]*pos[4][1]+mass[3]*pos[6][1])/(mass[0]+mass[1]+mass[2]+mass[3])
        comtmp_y = (mass[0]*pos[1][1]+mass[1]*pos[3][1]+mass[2]*pos[5][1]+mass[3]*pos[7][1])/(mass[0]+mass[1]+mass[2]+mass[3])
        com_x.append(comtmp_x)
        com_y.append(comtmp_y)
        pass

    # 做功和功率计算
    W_k = 0
    W_w = 0
    I_k = 0
    I_w = 0
    I_s = 0
    I_e = 0
    P_k = []
    P_w = []
    for i in range(robot.N):
        I_k += u[i][0]*robot.dt
        I_w += u[i][1]*robot.dt
        I_s += u[i][2]*robot.dt
        I_e += u[i][3]*robot.dt
        P1 = u[i][0]*dq[i][0]
        P2 = u[i][1]*dq[i][1]
        W_k += P1 * robot.dt
        W_w += P2 * robot.dt
        P_k.append(P1)
        P_w.append(P2)
        pass

    # print("="*50)
    # print("com_x: ", com_x)

    theta = np.pi/40
    F = 0
    visual = DataProcess(cfg, robot, theta, q, dq, ddq, u, F, t, save_dir, save_flag)
    if save_flag:
        SaveDir = visual.DataSave(save_flag, com_x, com_y, W_k, W_w, P_k, P_w, I_k, I_w, I_s, I_e)

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
        plt.plot(t, com_x, label = 'X COM')
        plt.plot(t, com_y, label = 'Y COM')
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

def COMPos():
    
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" +"2023-03-07" + "/"
    # save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name1 = "model_test/model.pkl"
    name2 = "traj_test/traj.pkl"
    name3 = "noarm/noarm.pkl"

    f1 = open(save_dir+name1,'rb')
    data1 = pickle.load(f1)

    f2 = open(save_dir+name2,'rb')
    data2 = pickle.load(f2)

    f3 = open(save_dir+name3,'rb')
    data3 = pickle.load(f3)

    com_x1 = data1['com_x']
    com_y1 = data1['com_y']
    dq1 = data1['dq']
    u1 = data1['u']
    W_k1 = round(data1['W_k'], 2)
    W_w1 = round(data1['W_w'], 2)
    I_k1 = round(data1['I_k'], 2)
    I_w1 = round(data1['I_w'], 2)
    I_s1 = round(data1['I_s'], 2)
    I_e1 = round(data1['I_e'], 2)
    P_k1 = data1['P_k']
    P_w1 = data1['P_w']
    t1 = data1['t']

    com_x2 = data2['com_x']
    com_y2 = data2['com_y']
    dq2 = data2['dq']
    u2 = data2['u']
    W_k2 = round(data2['W_k'], 2)
    W_w2 = round(data2['W_w'], 2)
    I_k2 = round(data2['I_k'], 2)
    I_w2 = round(data2['I_w'], 2)
    I_s2 = round(data2['I_s'], 2)
    I_e2 = round(data2['I_e'], 2)
    P_k2 = data2['P_k']
    P_w2 = data2['P_w']
    t2 = data2['t']

    com_x3 = data3['com_x']
    com_y3 = data3['com_y']
    dq3 = data3['dq']
    u3 = data3['u']
    W_k3 = round(data3['W_k'], 2)
    W_w3 = round(data3['W_w'], 2)
    I_k3 = round(data3['I_k'], 2)
    I_w3 = round(data3['I_w'], 2)
    I_s3 = round(data3['I_s'], 2)
    I_e3 = round(data3['I_e'], 2)
    P_k3 = data3['P_k']
    P_w3 = data3['P_w']
    t3 = data3['t']

    u11 = I_k1 / 2.0
    u12 = I_w1 / 2.0
    u21 = I_k2 / 2.0
    u22 = I_w2 / 2.0
    u31 = I_k3 / 2.0
    u32 = I_w3 / 2.0

    F11 = 0
    F12 = 0
    Ws_k = [0, 0, 0]
    Ws_w = [0, 0, 0]
    for i in range(len(t1)):
        F11 += u3[i][0]
        F12 += u3[i][1]
        # Ws_k[0] += np.abs(P_k1[i]*2.0/500)
        # Ws_k[1] += np.abs(P_k2[i]*2.0/500)
        # Ws_k[2] += np.abs(P_k3[i]*2.0/500)
        # Ws_w[0] += np.abs(P_w1[i]*2.0/500)
        # Ws_w[1] += np.abs(P_w2[i]*2.0/500)
        # Ws_w[2] += np.abs(P_w3[i]*2.0/500)
        Ws_k[0] += np.abs(u1[i][0]*dq1[i][0]*2.0/500)
        Ws_k[1] += np.abs(u2[i][0]*dq2[i][0]*2.0/500)
        Ws_k[2] += np.abs(u3[i][0]*dq3[i][0]*2.0/500)
        Ws_w[0] += np.abs(u1[i][1]*dq1[i][1]*2.0/500)
        Ws_w[1] += np.abs(u2[i][1]*dq2[i][1]*2.0/500)
        Ws_w[2] += np.abs(u3[i][1]*dq3[i][1]*2.0/500)
    F11 = F11 / 500
    F12 = F12 / 500
    print(F11, F12)
    print(W_k1, W_k2, W_k3)
    print(W_w1, W_w2, W_w3)
    print(Ws_k)
    print(Ws_w)
  
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib as mpl
    from matplotlib.patches import ConnectionPatch
    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 15,
        'legend.fontsize': 20,
        'axes.labelsize': 20,
        'lines.linewidth': 2,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 8,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }
    plt.rcParams.update(params)

    fig1 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=False)
    plt.plot(t1, com_x1, label = 'Decoupling Model')
    plt.plot(t2, com_x2, label = 'Coupling Model')
    plt.plot(t3, com_x3, label = 'No arm swing')
    plt.xlabel("time (s)")
    plt.ylabel("Y COM Position (m)")
    plt.legend()

    fig2 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=False)
    plt.plot(t1, com_y1, label = 'Decoupling Model')
    plt.plot(t2, com_y2, label = 'Coupling Model')
    plt.plot(t3, com_y3, label = 'Without arm swing')
    plt.xlabel("Time (s)")
    plt.ylabel("COM Y Position (m)")
    plt.legend()

    fig3 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=False)
    plt.plot(t1, P_k1, label = 'Knee Power of Model res')
    plt.plot(t2, P_k2, label = 'Knee Power of traj-opt res')
    plt.plot(t3, P_k3, label = 'Knee Power of noarm res')
    plt.xlabel("time (s)")
    plt.ylabel("Power")
    plt.legend()

    fig4 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=False)
    plt.plot(t1, P_w1, label = 'Waist Power of Model res')
    plt.plot(t2, P_w2, label = 'Waist Power of traj-opt res')
    plt.plot(t3, P_w3, label = 'Waist Power of noarm res')
    plt.xlabel("time (s)")
    plt.ylabel("Power")
    plt.legend()

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    savename1 =  save_dir + "com-y.jpg"
    fig2.savefig(savename1, dpi=500)


    # species = ("Knee", "Waist", "shoulder", "elbow")
    # penguin_means = {
    #     'Work of Model': (I_k1, I_w1, I_s1, I_e1),
    #     'Work of traj-opt': (I_k2, I_w2, I_s2, I_e2),
    #     'Work of noarm': (I_k3, I_w3, I_s3, I_e3),
    # }
    # x = np.array([0, 0.5, 1.0, 1.5])  # the label locations
    # width = 0.05  # the width of the bars
    # multiplier = 0

    # fig5, ax = plt.subplots()

    # for attribute, measurement in penguin_means.items():
    #     offset = width * multiplier
    #     rects = ax.bar(x + offset, measurement, width, label=attribute)
    #     ax.bar_label(rects, padding=-20, fontsize = 15)
    #     multiplier += 1

    # ax.set_ylabel('Work (J)', fontsize = 15)
    # ax.set_title('Work', fontsize = 20)
    # ax.set_xticks(x + width, species , fontsize = 15)
    # ax.legend(loc='upper left')
    # # ax.invert_yaxis()
    # # ax.set_ylim(0, 250)

    # species = ("Knee", "Waist")
    # # penguin_means = {
    # #     'Work of Model': (Ws_k[0], Ws_w[0]),
    # #     'Work of traj-opt': (Ws_k[1], Ws_w[1]),
    # #     'Work of noarm': (Ws_k[2], Ws_w[2]),
    # # }

    # penguin_means = {
    #     'Work of Model': (W_k1, W_w1),
    #     'Work of traj-opt': (W_k2, W_w2),
    #     'Work of noarm': (W_k3, W_w3),
    # }
    # x = np.array([0, 0.2])  # the label locations
    # width = 0.05  # the width of the bars
    # multiplier = 0

    # fig6, ax1 = plt.subplots(layout='constrained')

    # for attribute, measurement in penguin_means.items():
    #     offset = width * multiplier
    #     rects = ax1.bar(x + offset, measurement, width, label=attribute)
    #     ax1.bar_label(rects, padding=-20, fontsize = 15)
    #     multiplier += 1

    # ax1.set_ylabel('Work (J)', fontsize = 15)
    # ax1.set_title('Work', fontsize = 20)
    # ax1.set_xticks(x + width, species , fontsize = 15)
    # ax1.legend(loc='upper left')

    # species = ("Knee", "Waist")
    # penguin_means = {
    #     'Force of Model': (u11, u12),
    #     'Force of traj-opt': (u21, u22),
    #     'Force of noarm': (u31, u32),
    # }
    # x = np.array([0, 0.2])  # the label locations
    # width = 0.05  # the width of the bars
    # multiplier = 0

    # fig6, ax2 = plt.subplots(layout='constrained')

    # for attribute, measurement in penguin_means.items():
    #     offset = width * multiplier
    #     rects = ax2.bar(x + offset, measurement, width, label=attribute)
    #     ax2.bar_label(rects, padding=-20, fontsize = 15)
    #     multiplier += 1

    # ax2.set_ylabel('Force (N)', fontsize = 15)
    # ax2.set_title('Force', fontsize = 20)
    # ax2.set_xticks(x + width, species , fontsize = 15)
    # ax2.legend(loc='upper left')
    # ax2.invert_yaxis()

    plt.show()
    pass

def Momentum():
    
    # region: datafile import
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" +"2023-03-07" + "/"
    # save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name1 = "model_test/model.pkl"
    name2 = "traj_test/traj.pkl"
    name3 = "noarm/noarm.pkl"

    f1 = open(save_dir+name1,'rb')
    data1 = pickle.load(f1)

    f2 = open(save_dir+name2,'rb')
    data2 = pickle.load(f2)

    f3 = open(save_dir+name3,'rb')
    data3 = pickle.load(f3)

    q1 = data1['q']
    dq1 = data1['dq']
    com_x1 = data1['com_x']
    com_y1 = data1['com_y']
    q2 = data2['q']
    dq2 = data2['dq']
    com_x2 = data2['com_x']
    com_y2 = data2['com_y']
    q3 = data3['q']
    dq3 = data3['dq']
    com_x3 = data3['com_x']
    com_y3 = data3['com_y']
    t = data3['t']
    # endregion

    # region: Momentum calculate
    Momt_body_x = np.array([[0.0, 0.0, 0.0]])
    Momt_body_y = np.array([[0.0, 0.0, 0.0]])
    Momt_arm_x = np.array([[0.0, 0.0, 0.0]])
    Momt_arm_y = np.array([[0.0, 0.0, 0.0]])
    H_body = np.array([[0.0, 0.0, 0.0]])
    H_arm = np.array([[0.0, 0.0, 0.0]])
    L = [0.9, 0.5, 0.4, 0.4]
    m1 = [10, 15, 3.5, 4.4]
    m2 = [10, 15, 3.0, 3.75]
    m3 = [10, 15, 3.0, 3.75]
    for i in range(500):
        pos1 = Bipedal_hybrid.get_compos2(q1[i, :])
        pos2 = Bipedal_hybrid.get_compos2(q2[i, :])
        pos3 = Bipedal_hybrid.get_compos2(q3[i, :])

        vel1 = Bipedal_hybrid.get_comvel(q1[i, :], dq1[i, :])
        vel2 = Bipedal_hybrid.get_comvel(q2[i, :], dq2[i, :])
        vel3 = Bipedal_hybrid.get_comvel(q3[i, :], dq3[i, :])

        # region: 质心速度计算
        vel1_cx = (m1[0]*vel1[0]+m1[1]*vel1[2]+m1[2]*vel1[4]+m1[3]*vel1[6])/(m1[0]+m1[1]+m1[2]+m1[3])
        vel1_cy = (m1[0]*vel1[1]+m1[1]*vel1[3]+m1[2]*vel1[5]+m1[3]*vel1[7])/(m1[0]+m1[1]+m1[2]+m1[3])
        vel2_cx = (m2[0]*vel2[0]+m2[1]*vel2[2]+m2[2]*vel2[4]+m2[3]*vel2[6])/(m2[0]+m2[1]+m2[2]+m2[3])
        vel2_cy = (m2[0]*vel2[1]+m2[1]*vel2[3]+m2[2]*vel2[5]+m2[3]*vel2[7])/(m2[0]+m2[1]+m2[2]+m2[3])
        vel3_cx = (m3[0]*vel3[0]+m3[1]*vel3[2]+m3[2]*vel3[4]+m3[3]*vel3[6])/(m3[0]+m3[1]+m3[2]+m3[3])
        vel3_cy = (m3[0]*vel3[1]+m3[1]*vel3[3]+m3[2]*vel3[5]+m3[3]*vel3[7])/(m3[0]+m3[1]+m3[2]+m3[3])
        # endregion

        #region: 线动量计算
        # 身体动量：腿+躯干
        M_body1_x = m1[0]*vel1[0] + m1[1]*vel1[2]
        M_body2_x = m2[0]*vel2[0] + m2[1]*vel2[2]
        M_body3_x = m3[0]*vel3[0] + m3[1]*vel3[2]

        M_body1_y = m1[0]*vel1[1] + m1[1]*vel1[3]
        M_body2_y = m2[0]*vel2[1] + m2[1]*vel2[3]
        M_body3_y = m3[0]*vel3[1] + m3[1]*vel3[3]

        M_body_x = [M_body1_x, M_body2_x, M_body3_x]
        M_body_y = [M_body1_y, M_body2_y, M_body3_y]
        # 手臂动量：大臂+前臂
        M_arm1_x = m1[2]*vel1[4] + m1[3]*vel1[6]
        M_arm2_x = m2[2]*vel2[4] + m2[3]*vel2[6]
        M_arm3_x = m3[2]*vel3[4] + m3[3]*vel3[6]

        M_arm1_y = m1[2]*vel1[5] + m1[3]*vel1[7]
        M_arm2_y = m2[2]*vel2[5] + m2[3]*vel2[7]
        M_arm3_y = m3[2]*vel3[5] + m3[3]*vel3[7]

        M_arm_x = [M_arm1_x, M_arm2_x, M_arm3_x]
        M_arm_y = [M_arm1_y, M_arm2_y, M_arm3_y]

        Momt_body_x = np.concatenate((Momt_body_x, [M_body_x]), axis = 0)
        Momt_body_y = np.concatenate((Momt_body_y, [M_body_y]), axis = 0)
        Momt_arm_x = np.concatenate((Momt_arm_x, [M_arm_x]), axis = 0)
        Momt_arm_y = np.concatenate((Momt_arm_y, [M_arm_y]), axis = 0)
        # endregion

        # region: 相对于原点
        # 身体角动量：腿+躯干
        # H_body1 = m1[0]*(pos1[0]*vel1[1]-pos1[1]*vel1[0]) + m1[0]*L[0]**2*(dq1[i,0])/12 +\
        #           m1[1]*(pos1[2]*vel1[3]-pos1[3]*vel1[2]) + m1[1]*L[1]**2*(dq1[i,0]+dq1[i,1])/12
        # H_body2 = m2[0]*(pos2[0]*vel2[1]-pos2[1]*vel2[0]) + m2[0]*L[0]**2*(dq2[i,0])/12 +\
        #           m2[1]*(pos2[2]*vel2[3]-pos2[3]*vel2[2]) + m2[1]*L[1]**2*(dq2[i,0]+dq2[i,1])/12
        # H_body3 = m3[0]*(pos3[0]*vel3[1]-pos3[1]*vel3[0]) + m3[0]*L[0]**2*(dq3[i,0])/12 +\
        #           m3[1]*(pos3[2]*vel3[3]-pos3[3]*vel3[2]) + m3[1]*L[1]**2*(dq3[i,0]+dq3[i,1])/12
        
        # # 手臂角动量：大臂+前臂
        # H_arm1 = m1[2]*(pos1[4]*vel1[5]-pos1[5]*vel1[4]) + m1[2]*L[2]**2*(dq1[i,0]+dq1[i,1]+dq1[i,2])/12 +\
        #          m1[3]*(pos1[6]*vel1[7]-pos1[7]*vel1[6]) + m1[3]*L[3]**2*(dq1[i,0]+dq1[i,1]+dq1[i,2]+dq1[i,3])/12
        # H_arm2 = m2[2]*(pos2[4]*vel2[5]-pos2[5]*vel2[4]) + m2[2]*L[2]**2*(dq2[i,0]+dq2[i,1]+dq2[i,2])/12 +\
        #          m2[3]*(pos2[6]*vel2[7]-pos2[7]*vel2[6]) + m2[3]*L[3]**2*(dq2[i,0]+dq2[i,1]+dq2[i,2]+dq2[i,3])/12
        # H_arm3 = m3[2]*(pos3[4]*vel3[5]-pos3[5]*vel3[4]) + m3[2]*L[2]**2*(dq3[i,0]+dq3[i,1]+dq3[i,2])/12 +\
        #          m3[3]*(pos3[6]*vel3[7]-pos3[7]*vel3[6]) + m3[3]*L[3]**2*(dq3[i,0]+dq3[i,1]+dq3[i,2]+dq3[i,3])/12
        
        # H_body_tmp = [H_body1, H_body2, H_body3]
        # H_arm_tmp = [H_arm1, H_arm2, H_arm3]
        
        # H_body = np.concatenate((H_body, [H_body_tmp]), axis = 0)
        # H_arm = np.concatenate((H_arm, [H_arm_tmp]), axis= 0)
        # endregion
        
        # region: 相对于肩部参考点
        shoud_x1 = L[0]*np.sin(q1[i,0]) + L[1]*np.sin(q1[i,0]+q1[i,1])
        shoud_y1 = L[0]*np.cos(q1[i,0]) + L[1]*np.cos(q1[i,0]+q1[i,1])
        shoud_x2 = L[0]*np.sin(q2[i,0]) + L[1]*np.sin(q2[i,0]+q2[i,1])
        shoud_y2 = L[0]*np.cos(q2[i,0]) + L[1]*np.cos(q2[i,0]+q2[i,1])
        shoud_x3 = L[0]*np.sin(q3[i,0]) + L[1]*np.sin(q3[i,0]+q3[i,1])
        shoud_y3 = L[0]*np.cos(q3[i,0]) + L[1]*np.cos(q3[i,0]+q3[i,1])

        shoud_vx1 = L[0]*np.cos(q1[i,0])*dq1[i,0] + L[1]*np.cos(q1[i,0]+q1[i,1])*(dq1[i,0]+dq1[i,1])
        shoud_vy1 = -L[0]*np.sin(q1[i,0])*dq1[i,0] - L[1]*np.sin(q1[i,0]+q1[i,1])*(dq1[i,0]+dq1[i,1])
        shoud_vx2 = L[0]*np.cos(q2[i,0])*dq2[i,0] + L[1]*np.cos(q2[i,0]+q2[i,1])*(dq2[i,0]+dq2[i,1])
        shoud_vy2 = -L[0]*np.sin(q2[i,0])*dq2[i,0] - L[1]*np.sin(q2[i,0]+q2[i,1])*(dq2[i,0]+dq2[i,1])
        shoud_vx3 = L[0]*np.cos(q3[i,0])*dq3[i,0] + L[1]*np.cos(q3[i,0]+q3[i,1])*(dq3[i,0]+dq3[i,1])
        shoud_vy3 = -L[0]*np.sin(q3[i,0])*dq3[i,0] - L[1]*np.sin(q3[i,0]+q3[i,1])*(dq3[i,0]+dq3[i,1])
        
        # 身体角动量：腿+躯干
        rxm1_l = (pos1[0]-shoud_x1)*(vel1[1]-shoud_vy1) - (pos1[1]-shoud_y1)*(vel1[0]-shoud_vx1)
        rxm1_t = (pos1[2]-shoud_x1)*(vel1[3]-shoud_vy1) - (pos1[3]-shoud_y1)*(vel1[2]-shoud_vx1)
        H_body1 = m1[0]*rxm1_l + m1[0]*L[0]**2*(dq1[i,0])/12 +\
                  m1[1]*rxm1_t + m1[1]*L[1]**2*(dq1[i,0]+dq1[i,1])/12
        rxm2_l = (pos2[0]-shoud_x2)*(vel2[1]-shoud_vy2) - (pos2[1]-shoud_y2)*(vel2[0]-shoud_vx2)
        rxm2_t = (pos2[2]-shoud_x2)*(vel2[3]-shoud_vy2) - (pos2[3]-shoud_y2)*(vel2[2]-shoud_vx2)
        H_body2 = m2[0]*rxm2_l + m2[0]*L[0]**2*(dq2[i,0])/12 +\
                  m2[1]*rxm2_t + m2[1]*L[1]**2*(dq2[i,0]+dq2[i,1])/12
        rxm3_l = (pos3[0]-shoud_x3)*(vel3[1]-shoud_vy3) - (pos3[1]-shoud_y3)*(vel3[0]-shoud_vx3)
        rxm3_t = (pos3[2]-shoud_x3)*(vel3[3]-shoud_vy3) - (pos3[3]-shoud_y3)*(vel3[2]-shoud_vx3)
        H_body3 = m3[0]*rxm3_l + m3[0]*L[0]**2*(dq3[i,0])/12 +\
                  m3[1]*rxm3_t + m3[1]*L[1]**2*(dq3[i,0]+dq3[i,1])/12
        
        # 手臂角动量：大臂+前臂
        rxm1_a = (shoud_x1-pos1[4])*(shoud_vy1-vel1[5]) - (shoud_y1-pos1[5])*(shoud_vx1-vel1[4])
        rxm1_af = (shoud_x1-pos1[6])*(shoud_vy1-vel1[7]) - (shoud_y1-pos1[7])*(shoud_vx1-vel1[6])
        # print((shoud_x1-pos1[4])*(shoud_vy1-vel1[5]))
        # print((shoud_y1-pos1[5])*(shoud_vx1-vel1[4]))
        if i < 200:
            print(rxm1_a, rxm1_af)
            print(m1[2]*L[2]**2*(dq1[i,0]+dq1[i,1]+dq1[i,2]+dq1[i,3])/12)
        H_arm1 = m1[2]*rxm1_a + m1[2]*L[2]**2*(dq1[i,0]+dq1[i,1]+dq1[i,2])/12 +\
                 m1[3]*rxm1_af + m1[3]*L[3]**2*(dq1[i,0]+dq1[i,1]+dq1[i,2]+dq1[i,3])/12
        rxm2_a = (shoud_x2-pos2[4])*(shoud_vy2-vel2[5]) - (shoud_y2-pos2[5])*(shoud_vx2-vel2[4])
        rxm2_af = (shoud_x2-pos2[6])*(shoud_vy2-vel2[7]) - (shoud_y2-pos2[7])*(shoud_vx2-vel2[6])
        H_arm2 = m2[2]*rxm2_a + m2[2]*L[2]**2*(dq2[i,0]+dq2[i,1]+dq2[i,2])/12 +\
                 m2[3]*rxm2_af + m2[3]*L[3]**2*(dq2[i,0]+dq2[i,1]+dq2[i,2]+dq2[i,3])/12
        rxm3_a = (shoud_x3-pos3[4])*(shoud_vy3-vel3[5]) - (shoud_y3-pos3[5])*(shoud_vx3-vel3[4])
        rxm3_af = (shoud_x3-pos3[6])*(shoud_vy3-vel3[7]) - (shoud_y3-pos3[7])*(shoud_vx3-vel3[6])
        H_arm3 = m3[2]*rxm3_a + m3[2]*L[2]**2*(dq3[i,0]+dq3[i,1]+dq3[i,2])/12 +\
                 m3[3]*rxm3_af + m3[3]*L[3]**2*(dq3[i,0]+dq3[i,1]+dq3[i,2]+dq3[i,3])/12
        
        H_body_tmp = [H_body1, H_body2, H_body3]
        H_arm_tmp = [H_arm1, H_arm2, H_arm3]
        
        H_body = np.concatenate((H_body, [H_body_tmp]), axis = 0)
        H_arm = np.concatenate((H_arm, [H_arm_tmp]), axis= 0)
        # endregion

        # region: 相对于质心参考点
        # 身体角动量：腿+躯干
        # rxm1_l = (pos1[0]-com_x1[i])*(vel1[1]-vel1_cy) - (pos1[1]-com_y1[i])*(vel1[0]-vel1_cx)
        # rxm1_t = (pos1[2]-com_x1[i])*(vel1[3]-vel1_cy) - (pos1[3]-com_y1[i])*(vel1[2]-vel1_cx)
        # # rxm1_l = np.cross([pos1[0]-com_x1[i],])
        # rxm1_t = (pos1[2]-com_x1[i])*(vel1[3]-vel1_cy) - (pos1[3]-com_y1[i])*(vel1[2]-vel1_cx)
        # H_body1 = m1[0]*rxm1_l + m1[0]*L[0]**2*(dq1[i,0])/12 +\
        #           m1[1]*rxm1_t + m1[1]*L[1]**2*(dq1[i,0]+dq1[i,1])/12
        # rxm2_l = (pos2[0]-com_x2[i])*(vel2[1]-vel2_cy) - (pos2[1]-com_y2[i])*(vel2[0]-vel2_cx)
        # rxm2_t = (pos2[2]-com_x2[i])*(vel2[3]-vel2_cy) - (pos2[3]-com_y2[i])*(vel2[2]-vel2_cx)
        # H_body2 = m2[0]*rxm2_l + m2[0]*L[0]**2*(dq2[i,0])/12 +\
        #           m2[1]*rxm2_t + m2[1]*L[1]**2*(dq2[i,0]+dq2[i,1])/12
        # rxm3_l = (pos3[0]-com_x3[i])*(vel3[1]-vel3_cy) - (pos3[1]-com_y3[i])*(vel3[0]-vel3_cx)
        # rxm3_t = (pos3[2]-com_x3[i])*(vel3[3]-vel3_cy) - (pos3[3]-com_y3[i])*(vel3[2]-vel3_cx)
        # H_body3 = m3[0]*rxm3_l + m3[0]*L[0]**2*(dq3[i,0])/12 +\
        #           m3[1]*rxm3_t + m3[1]*L[1]**2*(dq3[i,0]+dq3[i,1])/12
        
        # # 手臂角动量：大臂+前臂
        # rxm1_a = (pos1[4]-com_x1[i])*(vel1[5]-vel1_cy) - (pos1[5]-com_y1[i])*(vel1[4]-vel1_cx)
        # rxm1_af = (pos1[6]-com_x1[i])*(vel1[7]-vel1_cy) - (pos1[7]-com_y1[i])*(vel1[6]-vel1_cx)
        # H_arm1 = m1[2]*rxm1_a + m1[2]*L[2]**2*(dq1[i,0]+dq1[i,1]+dq1[i,2])/12 +\
        #          m1[3]*rxm1_af + m1[3]*L[3]**2*(dq1[i,0]+dq1[i,1]+dq1[i,2]+dq1[i,3])/12
        # rxm2_a = (pos2[4]-com_x2[i])*(vel2[5]-vel2_cy) - (pos2[5]-com_y2[i])*(vel2[4]-vel2_cx)
        # rxm2_af = (pos2[6]-com_x2[i])*(vel2[7]-vel2_cy) - (pos2[7]-com_y2[i])*(vel2[6]-vel2_cx)
        # H_arm2 = m2[2]*rxm2_a + m2[2]*L[2]**2*(dq2[i,0]+dq2[i,1]+dq2[i,2])/12 +\
        #          m2[3]*rxm2_af + m2[3]*L[3]**2*(dq2[i,0]+dq2[i,1]+dq2[i,2]+dq2[i,3])/12
        # rxm3_a = (pos3[4]-com_x3[i])*(vel3[5]-vel3_cy) - (pos3[5]-com_y3[i])*(vel3[4]-vel3_cx)
        # rxm3_af = (pos3[6]-com_x3[i])*(vel3[7]-vel3_cy) - (pos3[7]-com_y3[i])*(vel3[6]-vel3_cx)
        # H_arm3 = m3[2]*rxm3_a + m3[2]*L[2]**2*(dq3[i,0]+dq3[i,1]+dq3[i,2])/12 +\
        #          m3[3]*rxm3_af + m3[3]*L[3]**2*(dq3[i,0]+dq2[i,1]+dq3[i,2]+dq3[i,3])/12
        
        # H_body_tmp = [H_body1, H_body2, H_body3]
        # H_arm_tmp = [H_arm1, H_arm2, H_arm3]
        
        # H_body = np.concatenate((H_body, [H_body_tmp]), axis = 0)
        # H_arm = np.concatenate((H_arm, [H_arm_tmp]), axis= 0)
        # endregion

        pass

    # endregion
    
    # region: data process
    Momt_body_x = Momt_body_x[1:,]
    Momt_body_y = Momt_body_y[1:,]
    Momt_arm_x = Momt_arm_x[1:,]
    Momt_arm_y = Momt_arm_y[1:,]
    H_body = H_body[1:,]
    H_arm = H_arm[1:,]
    # endregion

    # region: dataplot visualization
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

    # region: angular momentum plot
    fig1, ax1 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    ax1.plot(t, H_body[:,0], label = 'Body Ang-Momt of Model res')
    ax1.plot(t, H_arm[:,0], label = 'Arm Ang-Momt of Model res')
    plt.xlabel("time (s)")
    plt.ylabel("Position (m)")
    plt.legend()

    fig2, ax2 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    ax2.plot(t, H_body[:,1], label = 'Body Ang-Momt of traj-opt res')
    ax2.plot(t, H_arm[:,1], label = 'Arm Ang-Momt of traj-opt res')
    plt.xlabel("time (s)")
    plt.ylabel("Position (m)")
    plt.legend()

    fig3, ax3 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    ax3.plot(t, H_body[:,2], label = 'Body Ang-Momt of noarm res')
    ax3.plot(t, H_arm[:,2], label = 'Arm Ang-Momt of noarm res')
    plt.xlabel("time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    # endregion


    # region: linear momentum plot
    fig4, ax4 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    ax4.plot(t, Momt_body_x[:,0], label = 'Body linear Momt of Model res')
    ax4.plot(t, Momt_arm_x[:,0], label = 'Arm linear Momt of Model res')
    plt.xlabel("time (s)")
    plt.ylabel("Position (m)")
    plt.legend()

    fig5, ax5 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    ax5.plot(t, Momt_body_x[:,1], label = 'Body linear Momt of traj-opt res')
    ax5.plot(t, Momt_arm_x[:,1], label = 'Arm  linear Momt of traj-opt res')
    plt.xlabel("time (s)")
    plt.ylabel("Position (m)")
    plt.legend()

    fig6, ax6 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    ax6.plot(t, Momt_body_x[:,2], label = 'Body linear Momt of noarm res')
    ax6.plot(t, Momt_arm_x[:,2], label = 'Arm linear Momt of noarm res')
    plt.xlabel("time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    # endregion

    # plt.show()

    # endregion

    pass

# Momentum()代码简化
def Momentum2():
    
    # region: datafile import
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" +"2023-03-07" + "/"
    # save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name = ["model_test/model.pkl", "traj_test/traj.pkl", "noarm/noarm.pkl"]
 
    q = []
    dq = []
    P_com = []
    for i in range(3):
        f = open(save_dir+name[i],'rb')
        datatmp = pickle.load(f)
        qtmp = datatmp['q']
        dqtmp = datatmp['dq']
        comtmpx = datatmp['com_x']
        comtmpy = datatmp['com_y']
        t = datatmp['t']

        comtmp = np.concatenate(([comtmpx], [comtmpy]),axis=0)
        comtmp = comtmp.T

        q.append(qtmp)
        dq.append(dqtmp)
        P_com.append(comtmp)

    q = np.asarray(q)
    dq = np.asarray(dq)
    P_com = np.asarray(P_com)
    print(q.shape, dq.shape, P_com.shape)
    # endregion

    # region: Momentum calculate: H = (ri-rc)x(vi-vc) + Iw
    Momt_body_x = np.array([[0.0, 0.0, 0.0]])
    Momt_body_y = np.array([[0.0, 0.0, 0.0]])
    Momt_arm_x = np.array([[0.0, 0.0, 0.0]])
    Momt_arm_y = np.array([[0.0, 0.0, 0.0]])
    H_body = []
    H_arm = []
    L = [0.9, 0.5, 0.4, 0.4]
    m1 = [10, 15, 3.5, 4.4]
    m2 = [10, 15, 3.0, 3.75]
    m3 = [10, 15, 3.0, 3.75]
    m = np.asarray([m1, m2, m3])
    print(m)
    for i in range(500):

        # region: 位置矢量和速度矢量计算
        r = []
        v = []
        rc = P_com
        vc = []
        for j in range(3):
            r_tmp = Bipedal_hybrid.get_compos2(q[j, i, :])
            r_tmp = np.asarray(r_tmp).reshape(4,2)
            r.append(r_tmp)

            v_tmp = Bipedal_hybrid.get_comvel(q[j, i, :], dq[j, i, :])
            v_tmp = np.asarray(v_tmp).reshape(4,2)
            v.append(v_tmp)

            # 质心速度计算
            vcx = (m[j][0]*v_tmp[0][0]+m[j][1]*v_tmp[1][0]+m[j][2]*v_tmp[2][0]+m[j][3]*v_tmp[3][0])/ \
                  (m[j][0] + m[j][1] + m[j][2] + m[j][3])
            vcy = (m[j][0]*v_tmp[0][1]+m[j][1]*v_tmp[1][1]+m[j][2]*v_tmp[2][1]+m[j][3]*v_tmp[3][1])/ \
                  (m[j][0] + m[j][1] + m[j][2] + m[j][3])
            vc.append([vcx, vcy])
        
        r = np.asarray(r)
        v = np.asarray(v)
        vc = np.asarray(vc)
        # print(r.shape, v.shape, vc.shape)
        # endregion

        #region: 线动量计算
        # 身体动量：腿+躯干
        # M_body1_x = m1[0]*vel1[0] + m1[1]*vel1[2]
        # M_body2_x = m2[0]*vel2[0] + m2[1]*vel2[2]
        # M_body3_x = m3[0]*vel3[0] + m3[1]*vel3[2]

        # M_body1_y = m1[0]*vel1[1] + m1[1]*vel1[3]
        # M_body2_y = m2[0]*vel2[1] + m2[1]*vel2[3]
        # M_body3_y = m3[0]*vel3[1] + m3[1]*vel3[3]

        # M_body_x = [M_body1_x, M_body2_x, M_body3_x]
        # M_body_y = [M_body1_y, M_body2_y, M_body3_y]
        # # 手臂动量：大臂+前臂
        # M_arm1_x = m1[2]*vel1[4] + m1[3]*vel1[6]
        # M_arm2_x = m2[2]*vel2[4] + m2[3]*vel2[6]
        # M_arm3_x = m3[2]*vel3[4] + m3[3]*vel3[6]

        # M_arm1_y = m1[2]*vel1[5] + m1[3]*vel1[7]
        # M_arm2_y = m2[2]*vel2[5] + m2[3]*vel2[7]
        # M_arm3_y = m3[2]*vel3[5] + m3[3]*vel3[7]

        # M_arm_x = [M_arm1_x, M_arm2_x, M_arm3_x]
        # M_arm_y = [M_arm1_y, M_arm2_y, M_arm3_y]

        # Momt_body_x = np.concatenate((Momt_body_x, [M_body_x]), axis = 0)
        # Momt_body_y = np.concatenate((Momt_body_y, [M_body_y]), axis = 0)
        # Momt_arm_x = np.concatenate((Momt_arm_x, [M_arm_x]), axis = 0)
        # Momt_arm_y = np.concatenate((Momt_arm_y, [M_arm_y]), axis = 0)
        # endregion

        # region: 相对于原点
        # print(rs.shape, vs.shape)
        
        # H_body_i = []
        # H_arm_i = []
        # for j in range(3):
        #     r_rs = []
        #     v_vs = []
        #     for k in range(4):
        #         ri_rs = r[j][k]
        #         vi_vs = v[j][k]
        #         r_rs.append(ri_rs)
        #         v_vs.append(vi_vs)
        #         # print(r_rs, v_vs)

        #     # 身体角动量：腿+躯干
        #     H_body_tmp = m[j,0]*np.cross(r_rs[0], v_vs[0]) + m[j,0]*L[0]**2*(dq[j,i,0])/12 +\
        #                  m[j,1]*np.cross(r_rs[1], v_vs[1]) + m[j,1]*L[1]**2*(dq[j,i,0]+dq[j,i,1])/12
            
        #     # 手臂角动量：大臂+前臂
        #     H_arm_tmp =  m[j,2]*np.cross(r_rs[2], v_vs[2]) + m[j,2]*L[2]**2*(dq[j,i,0]+dq[j,i,1]+dq[j,i,2])/12 +\
        #                  m[j,3]*np.cross(r_rs[3], v_vs[3]) + m[j,3]*L[3]**2*(dq[j,i,0]+dq[j,i,1]+dq[j,i,2]+dq[j,i,3])/12
            
        #     H_body_i.append(H_body_tmp)
        #     H_arm_i.append(H_arm_tmp)
        # H_body.append(H_body_i)
        # H_arm.append(H_arm_i)
        # endregion
        
        # region: 相对于肩部参考点
        # rs = []
        # vs = []
        # for j in range(3):
        #     # 肩部参考点计算
        #     shoud_x = L[0]*np.sin(q[j,i,0]) + L[1]*np.sin(q[j,i,0]+q[j,i,1])
        #     shoud_y = L[0]*np.cos(q[j,i,0]) + L[1]*np.cos(q[j,i,0]+q[j,i,1])
        #     rs.append([shoud_x, shoud_y])
            
        #     shoud_vx = L[0]*np.cos(q[j,i,0])*dq[j,i,0] + L[1]*np.cos(q[j,i,0]+q[j,i,1])*(dq[j,i,0]+dq[j,i,1])
        #     shoud_vy = -L[0]*np.sin(q[j,i,0])*dq[j,i,0] - L[1]*np.sin(q[j,i,0]+q[j,i,1])*(dq[j,i,0]+dq[j,i,1])
        #     vs.append([shoud_vx, shoud_vy])
        # rs = np.asarray(rs)
        # vs = np.asarray(vs)
        # # print(rs.shape, vs.shape)
        
        # H_body_i = []
        # H_arm_i = []
        # for j in range(3):
        #     r_rs = []
        #     v_vs = []
        #     for k in range(4):
        #         ri_rs = r[j][k] - rs[j]
        #         vi_vs = v[j][k] - vs[j]
        #         r_rs.append(ri_rs)
        #         v_vs.append(vi_vs)
        #         # print(r_rs, v_vs)

        #     # 身体角动量：腿+躯干
        #     H_body_tmp = m[j,0]*np.cross(r_rs[0], v_vs[0]) + m[j,0]*L[0]**2*(dq[j,i,0])/12 +\
        #                  m[j,1]*np.cross(r_rs[1], v_vs[1]) + m[j,1]*L[1]**2*(dq[j,i,0]+dq[j,i,1])/12
            
        #     # 手臂角动量：大臂+前臂
        #     H_arm_tmp =  m[j,2]*np.cross(r_rs[2], v_vs[2]) + m[j,2]*L[2]**2*(dq[j,i,0]+dq[j,i,1]+dq[j,i,2])/12 +\
        #                  m[j,3]*np.cross(r_rs[3], v_vs[3]) + m[j,3]*L[3]**2*(dq[j,i,0]+dq[j,i,1]+dq[j,i,2]+dq[j,i,3])/12
            
        #     H_body_i.append(H_body_tmp)
        #     H_arm_i.append(H_arm_tmp)
        # H_body.append(H_body_i)
        # H_arm.append(H_arm_i)
        # endregion

        # region: 相对于质心参考点
        H_body_i = []
        H_arm_i = []
        for j in range(3):
            r_rc = []
            v_vc = []
            for k in range(4):
                ri_rc = r[j][k] - rc[j][i]
                vi_vc = v[j][k] - vc[j]
                r_rc.append(ri_rc)
                v_vc.append(vi_vc)
                # print(r_rc, v_vc)
            
            # 身体角动量：腿+躯干
            H_body_tmp = m[j,0]*np.cross(r_rc[0], v_vc[0]) + m[j,0]*L[0]**2*(dq[j,i,0])/12 +\
                         m[j,1]*np.cross(r_rc[1], v_vc[1]) + m[j,1]*L[1]**2*(dq[j,i,0]+dq[j,i,1])/12
             # 手臂角动量：大臂+前臂
            H_arm_tmp =  m[j,2]*np.cross(r_rc[2], v_vc[2]) + m[j,2]*L[2]**2*(dq[j,i,0]+dq[j,i,1]+dq[j,i,2])/12 +\
                         m[j,3]*np.cross(r_rc[3], v_vc[3]) + m[j,3]*L[3]**2*(dq[j,i,0]+dq[j,i,1]+dq[j,i,2]+dq[j,i,3])/12
            
            H_body_i.append(H_body_tmp)
            H_arm_i.append(H_arm_tmp)
        H_body.append(H_body_i)
        H_arm.append(H_arm_i)
        # endregion

        pass

    # endregion
    
    # region: data process
    H_body = np.asarray(H_body)
    H_arm = np.asarray(H_arm)
    print(H_body.shape)
    # Momt_body_x = Momt_body_x[1:,]
    # Momt_body_y = Momt_body_y[1:,]
    # Momt_arm_x = Momt_arm_x[1:,]
    # Momt_arm_y = Momt_arm_y[1:,]
    # endregion

    # region: dataplot visualization
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib as mpl
    from matplotlib.patches import ConnectionPatch
    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 15,
        'legend.fontsize': 15,
        'axes.labelsize': 20,
        'lines.linewidth': 2,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 8,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }
    plt.rcParams.update(params)

    # region: angular momentum plot
    fig1, ax1 = plt.subplots(1,1,figsize=(6, 6), dpi=180)
    ax1.plot(t, H_body[:,0], label = 'Body')
    ax1.plot(t, H_arm[:,0], label = 'Arm')
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Momentum")
    plt.legend()

    fig2, ax2 = plt.subplots(1,1,figsize=(6, 6), dpi=180)
    ax2.plot(t, H_body[:,1], label = 'Body')
    ax2.plot(t, H_arm[:,1], label = 'Arm')
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Momentum")
    plt.legend()

    fig3, ax3 = plt.subplots(1,1,figsize=(6, 6), dpi=180)
    ax3.plot(t, H_body[:,2], label = 'Body')
    ax3.plot(t, H_arm[:,2], label = 'Arm')
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Momentum")
    plt.legend()

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    savename1 =  save_dir + "Momt-time.jpg"
    savename2 =  save_dir + "Momt-time2.jpg"
    # fig1.savefig(savename1, dpi=500)
    # fig3.savefig(savename2, dpi=500)
    # endregion


    # region: linear momentum plot
    # fig4, ax4 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    # ax4.plot(t, Momt_body_x[:,0], label = 'Body linear Momt of Model res')
    # ax4.plot(t, Momt_arm_x[:,0], label = 'Arm linear Momt of Model res')
    # plt.xlabel("time (s)")
    # plt.ylabel("Position (m)")
    # plt.legend()

    # fig5, ax5 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    # ax5.plot(t, Momt_body_x[:,1], label = 'Body linear Momt of traj-opt res')
    # ax5.plot(t, Momt_arm_x[:,1], label = 'Arm  linear Momt of traj-opt res')
    # plt.xlabel("time (s)")
    # plt.ylabel("Position (m)")
    # plt.legend()

    # fig6, ax6 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    # ax6.plot(t, Momt_body_x[:,2], label = 'Body linear Momt of noarm res')
    # ax6.plot(t, Momt_arm_x[:,2], label = 'Arm linear Momt of noarm res')
    # plt.xlabel("time (s)")
    # plt.ylabel("Position (m)")
    plt.legend()
    # endregion

    plt.show()

    # endregion

    pass

# 相空间和力矩可视化
def PhasePlot():
    # region: datafile import
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" +"2023-03-07" + "/"
    # save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name = ["model_test/model.pkl", "traj_test/traj.pkl", "noarm/noarm.pkl"]
 
    q = []
    dq = []
    u = []
    P_com = []
    for i in range(3):
        f = open(save_dir+name[i],'rb')
        datatmp = pickle.load(f)
        qtmp = datatmp['q']
        dqtmp = datatmp['dq']
        utmp = datatmp['u']
        comtmpx = datatmp['com_x']
        comtmpy = datatmp['com_y']
        t = datatmp['t']

        comtmp = np.concatenate(([comtmpx], [comtmpy]),axis=0)
        comtmp = comtmp.T

        q.append(qtmp)
        dq.append(dqtmp)
        P_com.append(comtmp)
        u.append(utmp)

    u = np.asarray(u)
    q = np.asarray(q)
    dq = np.asarray(dq)
    P_com = np.asarray(P_com)
    print(u.shape, q.shape, dq.shape, P_com.shape)
    # endregion

    # region: 质心速度求解
    L = [0.9, 0.5, 0.4, 0.4]
    m1 = [10, 15, 3.5, 4.4]
    m2 = [10, 15, 3.0, 3.75]
    m3 = [10, 15, 3.0, 3.75]
    m = np.asarray([m1, m2, m3])
    Vc = []
    qsum1 = 0
    qsum2 = 0
    qsum3 = 0
    for i in range(500):
        qsum1 += q[0,i,0]**2 + q[0,i,1]**2
        qsum2 += q[1,i,0]**2 + q[1,i,1]**2
        qsum3 += q[2,i,0]**2 + q[2,i,1]**2

        vctmp = []
        v = []
        for j in range(3):
            v_tmp = Bipedal_hybrid.get_comvel(q[j, i, :], dq[j, i, :])
            v_tmp = np.asarray(v_tmp).reshape(4,2)
            v.append(v_tmp)

            # 质心速度计算
            vcx = (m[j][0]*v_tmp[0][0]+m[j][1]*v_tmp[1][0]+m[j][2]*v_tmp[2][0]+m[j][3]*v_tmp[3][0])/ \
                  (m[j][0] + m[j][1] + m[j][2] + m[j][3])
            vcy = (m[j][0]*v_tmp[0][1]+m[j][1]*v_tmp[1][1]+m[j][2]*v_tmp[2][1]+m[j][3]*v_tmp[3][1])/ \
                  (m[j][0] + m[j][1] + m[j][2] + m[j][3])
            vctmp.append([vcx, vcy])

        # vctmp = np.asarray(vctmp)
        Vc.append(vctmp)
    Vc = np.asarray(Vc)
    print(Vc.shape)
    print("Com x RMS deviation")
    print(np.sqrt(qsum1/500), np.sqrt(qsum2/500), np.sqrt(qsum3/500))
    # endregion

    # region: dataplot visualization
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib as mpl
    from matplotlib.patches import ConnectionPatch
    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 15,
        'legend.fontsize': 15,
        'axes.labelsize': 20,
        'lines.linewidth': 2,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 8,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }
    plt.rcParams.update(params)

    fig1, ax1 = plt.subplots(1,1,figsize=(6, 6), dpi=180)
    line11, = ax1.plot(P_com[0, :, 0], Vc[:, 0, 0], label = 'Decoupling Model')
    line12, = ax1.plot(P_com[1, :, 0], Vc[:, 1, 0], label = 'Coupling Model')
    line13, = ax1.plot(P_com[2, :, 0], Vc[:, 2, 0], label = 'Without arm swing')
    add_arrow_to_line2D(ax1, line11, arrow_locs=[0.2, 0.4, 0.6, 0.8], arrowstyle='-|>')
    add_arrow_to_line2D(ax1, line12, arrow_locs=[0.1, 0.3, 0.7, 0.9], arrowstyle='-|>')
    add_arrow_to_line2D(ax1, line13, arrow_locs=[0.2, 0.4, 0.6, 0.8], arrowstyle='-|>')
    plt.xlabel("Com x Position")
    plt.ylabel("Com x Velocity")
    plt.legend()

    fig2, ax2 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    line21, = ax2.plot(P_com[0, :, 1], Vc[:, 0, 1], label = 'Model res')
    line22, = ax2.plot(P_com[1, :, 1], Vc[:, 1, 1], label = 'Traj-opt res')
    line23, = ax2.plot(P_com[2, :, 1], Vc[:, 2, 1], label = 'Noarm res')
    add_arrow_to_line2D(ax2, line21, arrow_locs=[0.2, 0.4, 0.6, 0.8], arrowstyle='-|>')
    add_arrow_to_line2D(ax2, line22, arrow_locs=[0.1, 0.3, 0.7, 0.9], arrowstyle='-|>')
    add_arrow_to_line2D(ax2, line23, arrow_locs=[0.2, 0.4, 0.6, 0.8], arrowstyle='-|>')
    plt.xlabel("Com y Position")
    plt.ylabel("Com y Velocity")
    plt.legend()

    fig3, ax3 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    ax3.plot(q[0, :, 0], dq[0, :, 0], label = 'Model res')
    ax3.plot(q[1, :, 0], dq[1, :, 0], label = 'Traj-opt res')
    ax3.plot(q[2, :, 0], dq[2, :, 0], label = 'Noarm res')
    plt.xlabel("Waist Angle")
    plt.ylabel("Waist Angular Velocity")
    plt.legend()

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    savename1 =  save_dir + "phase.jpg"
    fig1.savefig(savename1, dpi=500)

    # fig3, ax3 = plt.subplots(1,1,figsize=(10, 6), dpi=180)
    # ax3.plot(t, u[0, :, 1], label = 'Model res')
    # ax3.plot(t, u[1, :, 1], label = 'Traj-opt res')
    # ax3.plot(t, u[2, :, 1], label = 'Noarm res')
    # plt.xlabel("Time(s)")
    # plt.ylabel("Torque(N.m)")
    # plt.legend()

    plt.show()
    # endregion
    pass

def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle='-|>', arrowsize=2, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows

if __name__ == "__main__":
    # main()
    # COMPos()
    Momentum2()
    # PhasePlot()
    pass
