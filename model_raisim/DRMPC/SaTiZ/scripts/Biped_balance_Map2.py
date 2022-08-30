'''
1. 双足机器人轨迹优化
2. 将接触的序列预先制定
3. 格式化配置参数输入和结果输出
4. 混合动力学系统，系统在机器人足底和地面接触的时候存在切换
5. 加入双臂的运动
6. Biped_walk_half2: x0,z0 in hip and hO_hip = O_b+O_hip
7. 2022.08.15: 
       Biped_walk_half3: 从上到下的多杆倒立摆动力学方程建模方法(无脚踝关节力矩)
'''

from ast import walk
import os
import yaml
import datetime
import casadi as ca
from casadi import sin as s
from casadi import cos as c
import numpy as np
from numpy.random import normal
import time
from ruamel.yaml import YAML
from math import acos, atan2, sqrt, sin, cos


class Bipedal_hybrid():
    def __init__(self, Mas, inert, cfg):
        self.opti = ca.Opti()
        # load config parameter
        # self.CollectionNum = cfg['Controller']['CollectionNum']

        # time and collection defination related parameter
        self.T = cfg['Controller']['Period']
        self.dt = cfg['Controller']['dt']
        self.N = int(self.T / self.dt)
        # self.N = cfg['Controller']['CollectionNum']
        # self.dt = self.T / self.N
        
        # mass and geometry related parameter
        # self.m = cfg['Robot']['Mass']['mass']
        # self.I = cfg['Robot']['Mass']['inertia']
        self.m = Mas
        self.I = inert
        self.l = cfg['Robot']['Mass']['massCenter']
        self.I_ = [self.m[i]*self.l[i]**2+self.I[i] for i in range(5)]

        self.L = [cfg['Robot']['Geometry']['L_shank'],
                  cfg['Robot']['Geometry']['L_thigh'],
                  cfg['Robot']['Geometry']['L_body'],             
                  cfg['Robot']['Geometry']['L_arm'],
                  cfg['Robot']['Geometry']['L_forearm']]

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

        self.u_LB = [-12] + [-self.motor_mt] * 4
        self.u_UB = [12] + [self.motor_mt] * 4

        # FF = cfg["Controller"]["Forward"]
        FF = cfg["Controller"]["FrontForward"]
        HF = cfg["Controller"]["HindForward"]

        # shank, thigh, body, arm, forearm
        self.q_LB = [-np.pi/10, -np.pi, -np.pi/30, 0, -np.pi*0.9] 
        self.q_UB = [np.pi/2, 0, np.pi*0.9, 4*np.pi/3, 0]   

        self.dq_LB = [-self.motor_ms,
                      -self.motor_ms, -self.motor_ms,
                      -self.motor_ms, -self.motor_ms]   # arm 

        self.dq_UB = [self.motor_ms,
                      self.motor_ms, self.motor_ms,
                      self.motor_ms, self.motor_ms] # arm 

        # * define variable
        self.q = [self.opti.variable(5) for _ in range(self.N)]
        self.dq = [self.opti.variable(5) for _ in range(self.N)]
        self.ddq = [(self.dq[i+1]-self.dq[i]) /
                        self.dt for i in range(self.N-1)]

        # ! set the last u to be zero at constraint
        self.u = [self.opti.variable(5) for _ in range(self.N)]

        # ! Note, the last force represents the plused force at the contact moment
        # self.F = [self.opti.variable(2) for _ in range(self.N)]

        pass

    def mass_matrix(self, q):
        # region create mass matrix
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
        m3 = self.m[3]
        m4 = self.m[4]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        lc3 = self.l[3]
        lc4 = self.l[4]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]

        m00 = self.I_[0]+self.I_[1]+self.I_[2]+self.I_[3]+self.I_[4] + \
              L0**2*(m1+m2+m3+m4)+L1**2*(m2+m3+m4)+L2**2*(m3+m4)+L3**2*(m4) + \
              2*L0*L1*c(q[1])*(m2+m3+m4)+2*L0*L2*c(q[1]+q[2])*(m3+m4)+2*L0*L3*m4*c(q[1]+q[2]+q[3])+\
              2*L0*lc1*m1*c(q[1])+2*L0*lc2*m2*c(q[1]+q[2])+2*L0*lc3*m3*c(q[1]+q[2]+q[3])+\
              2*L0*lc4*m4*c(q[1]+q[2]+q[3]+q[4])+\
              2*L1*L2*c(q[2])*(m3+m4)+2*L1*L3*c(q[2]+q[3])*(m4)+\
              2*L1*lc2*m2*c(q[2])+2*L1*lc3*m3*c(q[2]+q[3])+2*L1*lc4*m4*c(q[2]+q[3]+q[4])+\
              2*L2*L3*m4*c(q[3])+2*L2*lc3*m3*c(q[3])+2*L2*lc4*m4*c(q[3]+q[4])+\
              2*L3*lc4*m4*c(q[4])

        m01 = self.I_[1]+self.I_[2]+self.I_[3]+self.I_[4] + \
              L1**2*(m2+m3+m4)+L2**2*(m3+m4)+L3**2*(m4) + \
              1*(L0*L1*c(q[1])*(m2+m3+m4)+L0*L2*c(q[1]+q[2])*(m3+m4)+L0*L3*m4*c(q[1]+q[2]+q[3]))+\
              1*(L0*lc1*m1*c(q[1])+L0*lc2*m2*c(q[1]+q[2])+L0*lc3*m3*c(q[1]+q[2]+q[3]))+\
              1*L0*lc4*m4*c(q[1]+q[2]+q[3]+q[4])+\
              2*(L1*L2*c(q[2])*(m3+m4)+L1*L3*c(q[2]+q[3])*(m4))+\
              2*(L1*lc2*m2*c(q[2])+L1*lc3*m3*c(q[2]+q[3])+L1*lc4*m4*c(q[2]+q[3]+q[4]))+\
              2*(L2*L3*m4*c(q[3])+L2*lc3*m3*c(q[3])+L2*lc4*m4*c(q[3]+q[4]))+\
              2*L3*lc4*m4*c(q[4])

        m02 = self.I_[2]+self.I_[3]+self.I_[4] + \
              L2**2*(m3+m4)+L3**2*(m4) + \
              1*(L0*L2*c(q[1]+q[2])*(m3+m4)+L0*L3*m4*c(q[1]+q[2]+q[3]))+\
              1*(L0*lc2*m2*c(q[1]+q[2])+L0*lc3*m3*c(q[1]+q[2]+q[3]))+\
              1*L0*lc4*m4*c(q[1]+q[2]+q[3]+q[4])+\
              1*(L1*L2*c(q[2])*(m3+m4)+L1*L3*c(q[2]+q[3])*(m4))+\
              1*(L1*lc2*m2*c(q[2])+L1*lc3*m3*c(q[2]+q[3])+L1*lc4*m4*c(q[2]+q[3]+q[4]))+\
              2*(L2*L3*m4*c(q[3])+L2*lc3*m3*c(q[3])+L2*lc4*m4*c(q[3]+q[4]))+\
              2*L3*lc4*m4*c(q[4])

        m03 = self.I_[3]+self.I_[4] + \
              L3**2*(m4) + \
              1*(L0*L3*m4*c(q[1]+q[2]+q[3]))+\
              1*(L0*lc3*m3*c(q[1]+q[2]+q[3])+L0*lc4*m4*c(q[1]+q[2]+q[3]+q[4]))+\
              1*(L1*L3*c(q[2]+q[3])*(m4))+\
              1*(L1*lc3*m3*c(q[2]+q[3])+L1*lc4*m4*c(q[2]+q[3]+q[4]))+\
              1*(L2*L3*m4*c(q[3])+L2*lc3*m3*c(q[3])+L2*lc4*m4*c(q[3]+q[4]))+\
              2*L3*lc4*m4*c(q[4])

        m04 = self.I_[4] + \
              1*(L0*lc4*m4*c(q[1]+q[2]+q[3]+q[4]))+\
              1*(L1*lc4*m4*c(q[2]+q[3]+q[4]))+\
              1*(L2*lc4*m4*c(q[3]+q[4]))+\
              1*L3*lc4*m4*c(q[4])

        m10 = m01
        m11 = self.I_[1]+self.I_[2]+self.I_[3]+self.I_[4] + \
              L1**2*(m2+m3+m4)+L2**2*(m3+m4)+L3**2*(m4) + \
              2*(L1*L2*c(q[2])*(m3+m4)+L1*L3*c(q[2]+q[3])*(m4))+\
              2*(L1*lc2*m2*c(q[2])+L1*lc3*m3*c(q[2]+q[3])+L1*lc4*m4*c(q[2]+q[3]+q[4]))+\
              2*(L2*L3*m4*c(q[3])+L2*lc3*m3*c(q[3])+L2*lc4*m4*c(q[3]+q[4]))+\
              2*L3*lc4*m4*c(q[4])

        m12 = self.I_[2]+self.I_[3]+self.I_[4] + \
              L2**2*(m3+m4)+L3**2*(m4) + \
              1*(L1*L2*c(q[2])*(m3+m4)+L1*L3*c(q[2]+q[3])*(m4))+\
              1*(L1*lc2*m2*c(q[2])+L1*lc3*m3*c(q[2]+q[3])+L1*lc4*m4*c(q[2]+q[3]+q[4]))+\
              2*(L2*L3*m4*c(q[3])+L2*lc3*m3*c(q[3])+L2*lc4*m4*c(q[3]+q[4]))+\
              2*L3*lc4*m4*c(q[4])

        m13 = self.I_[3]+self.I_[4] + \
              L3**2*(m4) + \
              1*(L1*L3*c(q[2]+q[3])*(m4))+\
              1*(L1*lc3*m3*c(q[2]+q[3])+L1*lc4*m4*c(q[2]+q[3]+q[4]))+\
              1*(L2*L3*m4*c(q[3])+L2*lc3*m3*c(q[3])+L2*lc4*m4*c(q[3]+q[4]))+\
              2*L3*lc4*m4*c(q[4])

        m14 = self.I_[4] + \
              1*(L1*lc4*m4*c(q[2]+q[3]+q[4]))+\
              1*(L2*lc4*m4*c(q[3]+q[4]))+\
              1*L3*lc4*m4*c(q[4])

        m20 = m02
        m21 = m12
        m22 = self.I_[2]+self.I_[3]+self.I_[4] + \
              L2**2*(m3+m4)+L3**2*(m4) + \
              2*(L2*L3*m4*c(q[3])+L2*lc3*m3*c(q[3])+L2*lc4*m4*c(q[3]+q[4]))+\
              2*L3*lc4*m4*c(q[4])
        m23 = self.I_[3]+self.I_[4] + \
              L3**2*(m4) + \
              1*(L2*L3*m4*c(q[3])+L2*lc3*m3*c(q[3])+L2*lc4*m4*c(q[3]+q[4]))+\
              2*L3*lc4*m4*c(q[4])
        m24 = self.I_[4] + \
              1*(L2*lc4*m4*c(q[3]+q[4]))+\
              1*L3*lc4*m4*c(q[4])

        m30 = m03
        m31 = m13
        m32 = m23
        m33 = self.I_[3]+self.I_[4] + \
              L3**2*(m4) + \
              2*L3*lc4*m4*c(q[4])
        m34 = self.I_[4] + \
              1*L3*lc4*m4*c(q[4])

        m40 = m04
        m41 = m14
        m42 = m24
        m43 = m34
        m44 = self.I_[4]


        return [[m00, m01, m02, m03, m04],
                [m10, m11, m12, m13, m14],
                [m20, m21, m22, m23, m24],
                [m30, m31, m32, m33, m34],
                [m40, m41, m42, m43, m44]]
        # endregion

    def coriolis(self, q, dq):
        # region calculate the coriolis force
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
        m3 = self.m[3]
        m4 = self.m[4]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        lc3 = self.l[3]
        lc4 = self.l[4]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]

        m11 = (L0*L1*s(q[1])*(m2+m3+m4)+L0*L2*s(q[1]+q[2])*(m3+m4)+L0*L3*s(q[1]+q[2]+q[3])*(m4)+\
            L0*lc1*m1*s(q[1])+L0*lc2*m2*s(q[1]+q[2])+L0*lc3*m3*s(q[1]+q[2]+q[3])+L0*lc4*m4*s(q[1]+q[2]+q[3]+q[4]))
        m12 = (L0*L2*s(q[1]+q[2])*(m3+m4)+L0*L3*s(q[1]+q[2]+q[3])*(m4)+\
            L0*lc2*m2*s(q[1]+q[2])+L0*lc3*m3*s(q[1]+q[2]+q[3])+L0*lc4*m4*s(q[1]+q[2]+q[3]+q[4])+\
            L1*L2*s(q[2])*(m3+m4)+L1*L3*s(q[2]+q[3])*(m4)+\
            L1*lc2*m2*s(q[2])+L1*lc3*m3*s(q[2]+q[3])+L1*lc4*m4*s(q[2]+q[3]+q[4]))
        m13 = (L0*L3*s(q[1]+q[2]+q[3])*(m4)+L0*lc3*m3*s(q[1]+q[2]+q[3])+L0*lc4*m4*s(q[1]+q[2]+q[3]+q[4])+\
            L1*L3*s(q[2]+q[3])*(m4)+L1*lc3*m3*s(q[2]+q[3])+L1*lc4*m4*s(q[2]+q[3]+q[4])+\
            L2*L3*m4*s(q[3])+L2*lc3*m3*s(q[3])+L2*lc4*m4*s(q[3]+q[4]))
        m14 = (L0*lc4*m4*s(q[1]+q[2]+q[3]+q[4])+L1*lc4*m4*s(q[2]+q[3]+q[4])+\
            L2*lc4*m4*s(q[3]+q[4])+L3*lc4*m4*s(q[4]))

        m22 = (L1*L2*s(q[2])*(m3+m4)+L1*L3*s(q[2]+q[3])*(m4)+\
            L1*lc2*m2*s(q[2])+L1*lc3*m3*s(q[2]+q[3])+L1*lc4*m4*s(q[2]+q[3]+q[4]))
        m23 = (L1*L3*s(q[2]+q[3])*(m4)+L1*lc3*m3*s(q[2]+q[3])+L1*lc4*m4*s(q[2]+q[3]+q[4])+\
            L2*L3*m4*s(q[3])+L2*lc3*m3*s(q[3])+L2*lc4*m4*s(q[3]+q[4]))
        m24 = L1*lc4*m4*s(q[2]+q[3]+q[4])+L2*lc4*m4*s(q[3]+q[4])+L3*lc4*m4*s(q[4])

        m33 = L2*L3*m4*s(q[3])+L2*lc3*m3*s(q[3])+L2*lc4*m4*s(q[3]+q[4])
        m34 = L2*lc4*m4*s(q[3]+q[4])+L3*lc4*m4*s(q[4])

        m44 = L3*lc4*m4*s(q[4])

        c0 = -2*m11*dq[0]*dq[1]-2*m12*dq[0]*dq[2]-2*m13*dq[0]*dq[3]-2*m14*dq[0]*dq[4]
        c0 += -1*m11*dq[1]*dq[1]-2*m12*dq[1]*dq[2]-2*m13*dq[1]*dq[3]-2*m14*dq[1]*dq[4]
        c0 += -1*m12*dq[2]*dq[2]-2*m13*dq[2]*dq[3]-2*m14*dq[2]*dq[4]
        c0 += -1*m13*dq[3]*dq[3]-2*m14*dq[3]*dq[4]
        c0 += -1*m14*dq[4]*dq[4]

        c1 = 1*m11*dq[0]*dq[0]-2*m22*dq[0]*dq[2]-2*m23*dq[0]*dq[3]-2*m24*dq[0]*dq[4]
        c1 += -2*m22*dq[1]*dq[2]-2*m23*dq[1]*dq[3]-2*m24*dq[1]*dq[4]
        c1 += -1*m22*dq[2]*dq[2]-2*m23*dq[2]*dq[3]-2*m24*dq[2]*dq[4]
        c1 += -1*m23*dq[3]*dq[3]-2*m24*dq[3]*dq[4]
        c1 += -1*m24*dq[4]*dq[4]

        c2 = 1*m12*dq[0]*dq[0]+2*m22*dq[0]*dq[1]-2*m33*dq[0]*dq[3]-2*m34*dq[0]*dq[4]
        c2 += 1*m22*dq[1]*dq[1]-2*m33*dq[1]*dq[3]-2*m34*dq[1]*dq[4]
        c2 += -2*m33*dq[2]*dq[3]-2*m34*dq[2]*dq[4]
        c2 += -1*m33*dq[3]*dq[3]-2*m34*dq[3]*dq[4]
        c2 += -1*m34*dq[4]*dq[4]

        c3 = 1*m13*dq[0]*dq[0]+2*m23*dq[0]*dq[1]+2*m33*dq[0]*dq[2]-2*m44*dq[0]*dq[4]
        c3 += 1*m23*dq[1]*dq[1]+2*m33*dq[1]*dq[2]-2*m44*dq[1]*dq[4]
        c3 += 1*m33*dq[2]*dq[2]-2*m44*dq[2]*dq[4]
        c3 += -2*m44*dq[3]*dq[4]
        c3 += -1*m44*dq[4]*dq[4]

        c4 = 1*m14*dq[0]*dq[0]+2*m24*dq[0]*dq[1]+2*m34*dq[0]*dq[2]+2*m44*dq[0]*dq[3]
        c4 += 1*m24*dq[1]*dq[1]+2*m34*dq[1]*dq[2]+2*m44*dq[1]*dq[3]
        c4 += 1*m34*dq[2]*dq[2]+2*m44*dq[2]*dq[3]
        c4 += 1*m44*dq[3]*dq[3]

        return [c0, c1, c2, c3, c4]
        # endregion

    def gravity(self, q):
        # region calculate the gravity
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
        m3 = self.m[3]
        m4 = self.m[4]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        lc3 = self.l[3]
        lc4 = self.l[4]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]

        g0 = -(L0*s(q[0])*(m1+m2+m3+m4)+L1*s(q[0]+q[1])*(m2+m3+m4)+\
               L2*s(q[0]+q[1]+q[2])*(m3+m4)+L3*s(q[0]+q[1]+q[2]+q[3])*(m4)+\
               lc0*m0*s(q[0])+lc1*m1*s(q[0]+q[1])+lc2*m2*s(q[0]+q[1]+q[2])+\
               lc3*m3*s(q[0]+q[1]+q[2]+q[3])+lc4*m4*s(q[0]+q[1]+q[2]+q[3]+q[4]))
        g1 = -(L1*s(q[0]+q[1])*(m2+m3+m4)+\
               L2*s(q[0]+q[1]+q[2])*(m3+m4)+L3*s(q[0]+q[1]+q[2]+q[3])*(m4)+\
               lc1*m1*s(q[0]+q[1])+lc2*m2*s(q[0]+q[1]+q[2])+\
               lc3*m3*s(q[0]+q[1]+q[2]+q[3])+lc4*m4*s(q[0]+q[1]+q[2]+q[3]+q[4]))
        g2 = -(L2*s(q[0]+q[1]+q[2])*(m3+m4)+L3*s(q[0]+q[1]+q[2]+q[3])*(m4)+\
               lc2*m2*s(q[0]+q[1]+q[2])+\
               lc3*m3*s(q[0]+q[1]+q[2]+q[3])+lc4*m4*s(q[0]+q[1]+q[2]+q[3]+q[4]))
        g3 = -(L3*s(q[0]+q[1]+q[2]+q[3])*(m4)+\
               lc3*m3*s(q[0]+q[1]+q[2]+q[3])+lc4*m4*s(q[0]+q[1]+q[2]+q[3]+q[4]))
        g4 = -(lc4*m4*s(q[0]+q[1]+q[2]+q[3]+q[4]))

        return [g0*self.g, g1*self.g, g2*self.g, g3*self.g, g4*self.g]
        # endregion

    def SupportForce(self, q, dq, ddq):
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
        m3 = self.m[3]
        m4 = self.m[4]
        l0 = self.l[0]
        l1 = self.l[1]
        l2 = self.l[2]
        l3 = self.l[3]
        l4 = self.l[4]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]

        hq0 = q[0]
        hq1 = q[0]+q[1]
        hq2 = q[0]+q[1]+q[2]
        hq3 = q[0]+q[1]+q[2]+q[3]
        hq4 = q[0]+q[1]+q[2]+q[3]+q[4]

        dhq0 = dq[0]
        dhq1 = dq[0]+dq[1]
        dhq2 = dq[0]+dq[1]+dq[2]
        dhq3 = dq[0]+dq[1]+dq[2]+dq[3]
        dhq4 = dq[0]+dq[1]+dq[2]+dq[3]+dq[4]

        ddhq0 = ddq[0]
        ddhq1 = ddq[0]+ddq[1]
        ddhq2 = ddq[0]+ddq[1]+ddq[2]
        ddhq3 = ddq[0]+ddq[1]+ddq[2]+ddq[3]
        ddhq4 = ddq[0]+ddq[1]+ddq[2]+ddq[3]+ddq[4]

        # acceleration cal
        ddx0 = -l0*s(q[0])*dq[0]**2 + l0*c(q[0])*ddq[0]
        ddy0 = -l0*c(q[0])*dq[0]**2 - l0*s(q[0])*ddq[0]
        ddx1 = -L0*s(q[0])*dq[0]**2 - l1*s(hq1)*(dhq1)**2 + \
                L0*c(q[0])*ddq[0] + l1*c(hq1)*(ddhq1)
        ddy1 = -L0*c(q[0])*dq[0]**2 - l1*c(hq1)*(dhq1)**2 - \
                L0*s(q[0])*ddq[0] - l1*s(hq1)*(ddhq1)
        ddx2 = -L0*s(q[0])*dq[0]**2 - L1*s(hq1)*(dhq1)**2 - l2*s(hq2)*(dhq2)**2 + \
                L0*c(q[0])*ddq[0] + L1*c(hq1)*(ddhq1) + l2*c(hq2)*(ddhq2)
        ddy2 = -L0*c(q[0])*dq[0]**2 - L1*c(hq1)*(dhq1)**2 - l2*c(hq2)*(dhq2)**2 - \
                L0*s(q[0])*ddq[0] - L1*s(hq1)*(ddhq1) - l2*s(hq2)*(ddhq2)
        ddx3 = -L0*s(q[0])*dq[0]**2 - L1*s(hq1)*(dhq1)**2 - L2*s(hq2)*(dhq2)**2 - l3*s(hq3)*(dhq3)**2 + \
                L0*c(q[0])*ddq[0] + L1*c(hq1)*(ddhq1) + L2*c(hq2)*(ddhq2) + l3*c(hq3)*(ddhq3)
        ddy3 = -L0*c(q[0])*dq[0]**2 - L1*c(hq1)*(dhq1)**2 - L2*c(hq2)*(dhq2)**2 - l3*c(hq3)*(dhq3)**2 -\
                L0*s(q[0])*ddq[0] - L1*s(hq1)*(ddhq1) - L2*s(hq2)*(ddhq2) - l3*s(hq3)*(ddhq3)
        ddx4 = -L0*s(q[0])*dq[0]**2 - L1*s(hq1)*(dhq1)**2 - L2*s(hq2)*(dhq2)**2 - L3*s(hq3)*(dhq3)**2 - l4*s(hq4)*(dhq4)**2 + \
                L0*c(q[0])*ddq[0] + L1*c(hq1)*(ddhq1) + L2*c(hq2)*(ddhq2) + L3*c(hq3)*(ddhq3)+ l4*c(hq4)*(ddhq4)
        ddy4 = -L0*c(q[0])*dq[0]**2 - L1*c(hq1)*(dhq1)**2 - L2*c(hq2)*(dhq2)**2 - L3*c(hq3)*(dhq3)**2 - l4*c(hq4)*(dhq4)**2-\
                L0*s(q[0])*ddq[0] - L1*s(hq1)*(ddhq1) - L2*s(hq2)*(ddhq2) - L3*s(hq3)*(ddhq3)- l4*s(hq4)*(ddhq4)
        
        AccFx = -(m0*ddx0 + m1*ddx1 + m2*ddx2 + m3*ddx3 + m4*ddx4)
        AccFy = -(m0*ddy0 + m1*ddy1 + m2*ddy2 + m3*ddy3 + m4*ddy4) - \
                 (m0*self.g + m1*self.g + m2*self.g + m3*self.g + m4*self.g)

        AccF = [AccFx, AccFy]

        return AccF

    def contact_force(self, q, F):
        # region calculate the contact force
        # F = [Fxl, Fyl]
        cont0 = F[0]
        cont1 = F[1]
        cont2 = (self.L[1]*c(q[0]+q[1])+self.L[2]*c(q[0]+q[1]+q[2]))*F[0] + \
                (self.L[1]*s(q[0]+q[1])+self.L[2]*s(q[0]+q[1]+q[2]))*F[1]
        cont3 = (self.L[1]*c(q[0]+q[1])+self.L[2]*c(q[0]+q[1]+q[2]))*F[0] + \
                (self.L[1]*s(q[0]+q[1])+self.L[2]*s(q[0]+q[1]+q[2]))*F[1]
        cont4 = (self.L[2]*c(q[0]+q[1]+q[2]))*F[0] + (self.L[2]*s(q[0]+q[1]+q[2]))*F[1]
        cont5 = 0
        cont6 = 0
        return [cont0, cont1, cont2, cont3, cont4, cont5, cont6]
        # endregion

    def inertia_force(self, q, acc):
        # region calculate inertia force
        mm = self.mass_matrix(q)
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2] +
                         mm[i][3]*acc[3]+mm[i][4]*acc[4] for i in range(5)]
        return inertia_force
        # endregion

    def inertia_force2(self, q, acc):
        # region calculate inertia force, split into two parts
        mm = self.mass_matrix(q)
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2] +
                         mm[i][3]*acc[3]+mm[i][4]*acc[4] for i in range(5)]
        inertia_main = [mm[i][i]*acc[i] for i in range(5)]
        inertia_coupling = [inertia_force[i]-inertia_main[i] for i in range(5)]
        return inertia_main, inertia_coupling

    @staticmethod
    def get_posture(q):
        L = [0.5, 0.42, 0.5, 0.3, 0.37]
        lsx = np.zeros(3)
        lsy = np.zeros(3)
        ltx = np.zeros(2)
        lty = np.zeros(2)
        lax = np.zeros(3)
        lay = np.zeros(3)
        lsx[0] = 0
        lsx[1] = lsx[0] + L[0]*np.sin(q[0])
        lsx[2] = lsx[1] + L[1]*np.sin(q[0]+q[1])
        lsy[0] = 0
        lsy[1] = lsy[0] + L[0]*np.cos(q[0])
        lsy[2] = lsy[1] + L[1]*np.cos(q[0]+q[1])

        ltx[0] = 0 + L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1])
        ltx[1] = ltx[0] + L[2]*np.sin(q[0]+q[1]+q[2])
        lty[0] = 0 + L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1])
        lty[1] = lty[0] + L[2]*np.cos(q[0]+q[1]+q[2])

        lax[0] = 0 + L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1]) + L[2]*np.sin(q[0]+q[1]+q[2])
        lax[1] = lax[0] + L[3]*np.sin(q[0]+q[1]+q[2]+q[3])
        lax[2] = lax[1] + L[4]*np.sin(q[0]+q[1]+q[2]+q[3]+q[4])
        lay[0] = 0 + L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1]) + L[2]*np.cos(q[0]+q[1]+q[2])
        lay[1] = lay[0] + L[3]*np.cos(q[0]+q[1]+q[2]+q[3])
        lay[2] = lay[1] + L[4]*np.cos(q[0]+q[1]+q[2]+q[3]+q[4])
        return [lsx, lsy, ltx, lty, lax, lay]

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
            for j in range(5):
                if j == 3:
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
        Pf = [50, 30, 15, 5, 1]
        Vf = [40, 20, 40, 5, 5]
        # Pf = [20, 20, 10, 5, 5]
        # Vf = [5, 5, 2, 2, 1]
        Ptar = [0, 0, 0, 3.14, 0]
        # Ff = [2, 1, 0.5, 0.5]
        Ff = [30, 20, 10, 5, 5]
        for i in range(walker.N):
            for k in range(5):
                if k < 4:
                    power += (walker.dq[i][k+1] * walker.u[i][k])**2 * walker.dt
                    force += (walker.u[i][k] / walker.motor_mt)**2 * walker.dt * Ff[k]  
                    # force += (walker.u[i][k] / walker.motor_mt)**2 * walker.dt

                VelTar += (walker.dq[i][k])**2 * walker.dt * Vf[k]
                PosTar += (walker.q[i][k] - Ptar[k])**2 * walker.dt * Pf[k] 
                # VelTar += (walker.dq[i][k])**2 * walker.dt             
                # PosTar += (walker.q[i][k] - Ptar[k])**2 * walker.dt             
                pass
            pass
        
        for j in range(5):
            VelTar += (walker.dq[-1][j])**2 * Vf[j] * 100
            PosTar += (walker.q[-1][j] - Ptar[j])**2 * Pf[j] * 500
            # VelTar += (walker.dq[-1][j])**2
            # PosTar += (walker.q[-1][j] - Ptar[j])**2

        u = walker.u
        smooth = 0
        AM = [100, 100, 400, 100, 400]
        for i in range(walker.N-1):
            for k in range(4):
                smooth += ((u[i+1][k]-u[i][k])/10)**2
                pass
            pass

        res = 0
        res = (res + power*self.powerCoeff) if (self.powerCoeff > 1e-6) else res
        res = (res + VelTar*self.velCoeff) if (self.velCoeff > 1e-6) else res
        res = (res + PosTar*self.trackingCoeff) if (self.trackingCoeff > 1e-6) else res
        res = (res + force*self.forceCoeff) if (self.forceCoeff > 1e-6) else res
        res = (res + smooth*self.smoothCoeff) if (self.smoothCoeff > 1e-6) else res

        return res

    def getConstraints(self, walker):
        ceq = []
        # region dynamics constraints
        # continuous dynamics
        #! 约束的数量为 (6+6）*（NN1-1+NN2-1）
        for j in range(walker.N):
            if j < (walker.N-1):
                ceq.extend([walker.q[j+1][k]-walker.q[j][k]-walker.dt/2 *
                            (walker.dq[j+1][k]+walker.dq[j][k]) == 0 for k in range(5)])
                inertia = walker.inertia_force(
                    walker.q[j], walker.ddq[j])
                coriolis = walker.coriolis(
                    walker.q[j], walker.dq[j])
                gravity = walker.gravity(walker.q[j])
                # ceq.extend([inertia[0]+gravity[0]+coriolis[0] == 0])
                # ceq.extend([inertia[k+1]+gravity[k+1]+coriolis[k+1] -
                #             walker.u[j][k] == 0 for k in range(4)])
                ceq.extend([inertia[k]+gravity[k]+coriolis[k] -
                            walker.u[j][k] == 0 for k in range(5)])

            if not self.armflag:
                ceq.extend([walker.q[j][3] == np.pi])
                ceq.extend([walker.q[j][4] == 0])
                ceq.extend([walker.dq[j][k+3] == 0 for k in range(2)])
            
            pass


        # region leg locomotion constraint
        for i in range(walker.N-1):
            AccF = walker.SupportForce(walker.q[i], walker.dq[i], walker.ddq[i])
            Fx = -AccF[0]
            Fy = -AccF[1]
            ceq.extend([Fy >= 0])
            ceq.extend([Fy <= 4000])
            ceq.extend([Fx <= 4000])
            ceq.extend([Fx >= -4000])
            ceq.extend([Fy*walker.mu - Fx >= 0])  # 摩擦域条件
            ceq.extend([Fy*walker.mu + Fx >= 0])  # 摩擦域条件
        # endregion

        # region boundary constraint
        for temp_q in walker.q:
            ceq.extend([walker.opti.bounded(walker.q_LB[j],
                        temp_q[j], walker.q_UB[j]) for j in range(5)])
            pass
        for temp_dq in walker.dq:
            ceq.extend([walker.opti.bounded(walker.dq_LB[j],
                        temp_dq[j], walker.dq_UB[j]) for j in range(5)])
            pass
        for temp_u in walker.u:
            ceq.extend([walker.opti.bounded(walker.u_LB[j],
                        temp_u[j], walker.u_UB[j]) for j in range(5)])
            pass
        # endregion

        # region motor external characteristic curve
        cs = walker.motor_cs
        ms = walker.motor_ms
        mt = walker.motor_mt
        for j in range(len(walker.u)):
            ceq.extend([walker.u[j][k]-ca.fmax(mt - (walker.dq[j][k] -
                                                        cs)/(ms-cs)*mt, 0) <= 0 for k in range(5)])
            ceq.extend([walker.u[j][k]-ca.fmin(-mt + (walker.dq[j][k] +
                                                            cs)/(-ms+cs)*mt, 0) >= 0 for k in range(4)])
            pass
        # endregion

        theta = np.pi/50
        ceq.extend([walker.q[0][0]==theta])
        ceq.extend([walker.q[0][1]==-theta*0.2])
        ceq.extend([walker.q[0][2]==theta*0.2])
        # ceq.extend([walker.q[0][3]==np.pi*0.95])
        # ceq.extend([walker.q[0][4]==-np.pi*0.1])

        ceq.extend([walker.q[0][3]==np.pi])
        ceq.extend([walker.q[0][4]==0])

        ceq.extend([walker.dq[0][0]==0])
        ceq.extend([walker.dq[0][1]==0])
        ceq.extend([walker.dq[0][2]==0])
        ceq.extend([walker.dq[0][3]==0])
        ceq.extend([walker.dq[0][4]==0])

        # region smooth constraint
        for j in range(len(walker.u)-1):
            ceq.extend([(walker.u[j][k]-walker.u
                        [j+1][k])**2 <= 100 for k in range(5)])
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
                q.append([sol1.value(robot.q[j][k]) for k in range(5)])
                dq.append([sol1.value(robot.dq[j][k])
                            for k in range(5)])
                if j < (robot.N-1):
                    ddq.append([sol1.value(robot.ddq[j][k])
                                for k in range(5)])
                    u.append([sol1.value(robot.u[j][k])
                                for k in range(5)])
                else:
                    ddq.append([sol1.value(robot.ddq[j-1][k])
                                for k in range(5)])
                    u.append([sol1.value(robot.u[j-1][k])
                                for k in range(5)])
                pass
            pass
        except:
            value = robot.opti.debug.value
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([value(robot.q[j][k])
                            for k in range(5)])
                dq.append([value(robot.dq[j][k])
                            for k in range(5)])
                if j < (robot.N-1):
                    ddq.append([value(robot.ddq[j][k])
                                for k in range(5)])
                    u.append([value(robot.u[j][k])
                                for k in range(5)])
                else:
                    ddq.append([value(robot.ddq[j-1][k])
                                for k in range(5)])
                    u.append([value(robot.u[j-1][k])
                                for k in range(5)])
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


def main(Mass, inertia, armflag):
    # region optimization trajectory for bipedal hybrid robot system
    vis_flag = False
    save_flag = False
    # armflag = False
    # armflag = True

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
    ParamFilePath = FilePath + "/config/Biped_balance2.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # endregion

    # region create robot and NLP problem
    robot = Bipedal_hybrid(Mass, inertia, cfg)
    # nonlinearOptimization = nlp(robot, cfg, seed=seed)
    # nonlinearOptimization = nlp(robot, cfg)
    nonlinearOptimization = nlp(robot, cfg, armflag)
    # endregion
    q, dq, ddq, u, t = nonlinearOptimization.solve_and_output(
        robot, flag_save=save_flag, StorePath=StorePath)
   
    Fx = np.array([0.0])
    Fy = np.array([0.0])
    for i in range(robot.N-1):
        AccF = robot.SupportForce(q[i], dq[i], ddq[i])
        tempx = -AccF[0]
        tempy = -AccF[1]
        Fx = np.concatenate((Fx, [tempx]))
        Fy = np.concatenate((Fy, [tempy]))
        if i == robot.N-2:
            Fx = np.concatenate((Fx, [tempx]))
            Fy = np.concatenate((Fy, [tempy]))
    Fx = Fx[1:]
    Fy = Fy[1:]
    F = np.concatenate(([Fx], [Fy]), axis=1)
    # endregion

    if vis_flag:
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
        g_data = gs[0].subgridspec(3, 6, wspace=0.3, hspace=0.33)
        ax_m = fig2.add_subplot(gm[0])

        # gs = fig.add_gridspec(2, 1, height_ratios=[2,1],
        #                       wspace=0.3, hspace=0.33)
        # g_data = gs[1].subgridspec(3, 6, wspace=0.3, hspace=0.33)

        # ax_m = fig.add_subplot(gs[0])
        ax_p = [fig.add_subplot(g_data[0, i]) for i in range(5)]
        ax_v = [fig.add_subplot(g_data[1, i]) for i in range(5)]
        ax_u = [fig.add_subplot(g_data[2, i]) for i in range(5)]

        # vel = [robot.foot_vel(q[i, :], dq[i, :]) for i in range(len(q[:, 0]))]

        # plot robot trajectory here
        ax_m.axhline(y=0, color='k')
        num_frame = 5
        for tt in np.linspace(0, robot.T, num_frame):
            idx = np.argmin(np.abs(t-tt))
            pos = Bipedal_hybrid.get_posture(q[idx, :])
            ax_m.plot(pos[0], pos[1], 'o-', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1)
            ax_m.plot(pos[2], pos[3], 'o:', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1)
            ax_m.plot(pos[4], pos[5], 'o-', ms=1,
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

        [ax_v[i].plot(t, dq[:, i]) for i in range(5)]
        ax_v[0].set_ylabel('Velocity(m/s)')
        [ax_p[i].plot(t, q[:, i]) for i in range(5)]
        ax_p[0].set_ylabel('Position(m)')

        # ax_u[0].plot(t[1:], Fx)
        # ax_u[0].plot(t[1:], Fy)
        ax_u[0].plot(t, u[:, 0])
        ax_u[0].set_xlabel('ankle')
        ax_u[0].set_ylabel('Torque (N.m)')
        ax_u[1].plot(t, u[:, 1])
        ax_u[1].set_xlabel('knee')
        ax_u[2].plot(t, u[:, 2])
        ax_u[2].set_xlabel('waist')
        ax_u[3].plot(t, u[:, 3])
        ax_u[3].set_xlabel('shoulder')
        ax_u[4].plot(t, u[:, 4])
        ax_u[4].set_xlabel('elbow')
        # [ax_u[j].set_title(title_u[j]) for j in range(4)]

        fig3 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=False)
        plt.plot(t, Fx, label = 'Fx')
        plt.plot(t, Fy, label = 'Fy')
        plt.xlabel("time (s)")
        plt.ylabel("Force (N)")
        plt.legend()

        plt.show()


        pass
    F = [Fx, Fy]
    return u, Fy, t

## use mean value instead of peak value to analysis force map
def ForceMapMV():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pickle
    import random
    from matplotlib.pyplot import MultipleLocator

    saveflag = True

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name = "ForceMap.pkl"
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # M_arm = [1.8, 2.4, 3.2, 4.5, 5.4, 6.2, 7.2, 8.0, 9.0, 10.0, 11.0]
    # M_arm = [1.8, 3.3, 4.5, 6.5, 8.0, 10.0]
    M_arm = [2.0, 4.0, 6.0, 8.0, 10.0]
    M_label = list(map(str, M_arm))
    # I_arm = [0.026, 0.045, 0.082, 0.13, 0.16, 0.20]
    # I_arm = [0.015, 0.032, 0.045, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    I_arm = [0.01, 0.03, 0.05, 0.07, 0.09]
    I_label = list(map(str, I_arm))

    Mass = [6.8, 20, 30]
    inertia = [0.14, 0.3, 0.625]

    Fy = np.array([[0.0]*len(M_arm)])
    u_h = np.array([[0.0]*len(M_arm)])
    u_a = np.array([[0.0]*len(M_arm)])
    u_k = np.array([[0.0]*len(M_arm)])
    u_s = np.array([[0.0]*len(M_arm)])
    u_e = np.array([[0.0]*len(M_arm)])
    CollectNum = 1000
    if saveflag:
        for i in range(len(I_arm)):
            temp_i = []
            temp_i.extend(inertia)
            temp_i.append(I_arm[i])
            temp_i.append(0.03)
            Fy_max = []
            u_h_max = []
            u_a_max = []
            u_k_max = []
            u_s_max = []
            u_e_max = []
            for j in range(len(M_arm)):
                temp_m = []
                temp_m.extend(Mass)
                temp_m.append(M_arm[j])
                temp_m.append(2.6)

                print("="*50)
                print("Mass: ", temp_m)
                print("="*50)
                print("Inertia: ", temp_i)

                u, F, t = main(temp_m, temp_i, True)

                F_1 = 0
                F_3 = 0
                num1 = 0
                num2 = 0
                for k in range(len(F)):
                    if F[k] > 1:
                        F_1 += F[k]
                        num1 += 1

                F_1 = F_1 / num1
                # print(F_1)

                temp_fy_max = F_1

                u4 = u[:, 0]
                temp_ua0_max = np.sum(np.sqrt(u4**2)) / CollectNum
                temp_ua_max = temp_ua0_max

                u0 = u[:, 1]
                temp_uk0_max = np.sum(np.sqrt(u0**2)) / CollectNum
                temp_uk_max = temp_uk0_max

                u1 = u[:, 2]
                temp_uh1_max = np.sum(np.sqrt(u1**2)) / CollectNum
                temp_uh_max = temp_uh1_max

                u2 = u[:, 3]
                temp_us1_max = np.sum(np.sqrt(u2**2)) / CollectNum
                temp_us_max = temp_us1_max

                u3 = u[:, 4]
                temp_ue1_max = np.sum(np.sqrt(u3**2)) / CollectNum
                temp_ue_max = temp_ue1_max

                Fy_max.append(temp_fy_max)
                u_h_max.append(temp_uh_max)
                u_k_max.append(temp_uk_max)
                u_s_max.append(temp_us_max)
                u_e_max.append(temp_ue_max)
                u_a_max.append(temp_ua_max)

                pass
            print(u0.shape,u_k_max)
            Fy = np.concatenate((Fy, [Fy_max]), axis = 0)
            u_h = np.concatenate((u_h, [u_h_max]), axis = 0)
            u_k = np.concatenate((u_k, [u_k_max]), axis = 0)
            u_s = np.concatenate((u_s, [u_s_max]), axis = 0)
            u_e = np.concatenate((u_e, [u_e_max]), axis = 0)
            u_a = np.concatenate((u_a, [u_a_max]), axis = 0)

            pass
        Fy = Fy[1:]
        u_h = u_h[1:]
        u_k = u_k[1:]
        u_s = u_s[1:]
        u_e = u_e[1:]
        u_a = u_a[1:]

        Data = {'Fy': Fy, 'u_h': u_h, "u_k": u_k, "u_s": u_s, "u_e": u_e, "u_a": u_a}
        if os.path.exists(os.path.join(save_dir, name)):
            RandNum = random.randint(0,100)
            name = "ForceMap" + str(RandNum)+ ".pkl"
        with open(os.path.join(save_dir, name), 'wb') as f:
            pickle.dump(Data, f)
    else:
        f = open(save_dir+name,'rb')
        data = pickle.load(f)

        Fy = data['Fy']
        u_h = data['u_h']
        u_k = data['u_k']
        u_s = data['u_s']
        u_e = data['u_e']
        u_a = data['u_a']

    
    plt.style.use("science")
    params = {
        'text.usetex': True,
        'image.cmap': 'inferno',
        'font.size': 20,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.3,
    }

    plt.rcParams.update(params)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    ax1 = axs[0][0]
    ax2 = axs[0][1]
    ax3 = axs[1][0]
    ax4 = axs[1][1]

    pcm1 = ax1.imshow(u_a, vmin = 300, vmax = 1200)
    cb1 = fig.colorbar(pcm1, ax=ax1)

    pcm2 = ax2.imshow(u_h, vmin = 0, vmax = 200)
    cb2 = fig.colorbar(pcm2, ax=ax2)

    pcm3 = ax3.imshow(u_k, vmin = 0, vmax = 100)
    cb3 = fig.colorbar(pcm3, ax=ax3)

    pcm4 = ax4.imshow(u_s, vmin = 0, vmax = 50)
    cb4 = fig.colorbar(pcm4, ax=ax4)
    
    ax = [[ax1, ax2], [ax3, ax4]]
    cb = [[cb1, cb2], [cb3, cb4]]
    title = [["Fy", "Torque-Hip"], ["Torque-Knee", "Torque-shoulder"]]
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xticks(np.arange(len(M_label)))
            ax[i][j].set_xticklabels(M_label)
            ax[i][j].set_yticks(np.arange(len(I_label)))
            ax[i][j].set_ylim(-0.5, len(I_label)-0.5)
            ax[i][j].set_yticklabels(I_label)
            # ax[i][j].xaxis.set_tick_params(top=True, bottom=False,
            #        labeltop=True, labelbottom=False)

            ax[i][j].set_ylabel("Inertia")
            ax[i][j].set_xlabel("Mass")
            ax[i][j].set_title(title[i][j])

            if i==0 and j==0:
                cb[i][j].set_label("Force(N)")
            else:
                cb[i][j].set_label("Torque(N/m)")
    # fig.tight_layout()
    plt.show()
    pass


if __name__ == "__main__":
    # main()
    ForceMapMV()
    # ForceVisualization()
    # power_analysis()
    # Impact_inertia()
    # Impact_process()
    # Power_metrics_analysis()
    pass
