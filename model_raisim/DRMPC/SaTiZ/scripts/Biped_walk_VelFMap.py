'''
1. 双足机器人轨迹优化
2. 将接触的序列预先制定
3. 格式化配置参数输入和结果输出
4. 混合动力学系统，系统在机器人足底和地面接触的时候存在切换
5. 加入双臂的运动
'''

from cProfile import label
import os
from scipy import False_
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
from DataProcess import DataProcess


class Bipedal_hybrid():
    def __init__(self, Period, Stance, Vt, cfg):
        self.opti = ca.Opti()
        # load config parameter
        self.CollectionNum = cfg['Controller']['CollectionNum']
        self.N = cfg['Controller']['CollectionNum']+1

        # time and collection defination related parameter
        # self.T = cfg['Controller']['Period']
        self.T = Period
        self.Ts = Stance
        self.dt = self.T / self.CollectionNum
        self.tc1 = cfg['Controller']['Phase'][0] * \
            self.T           # left leg touch down
        # self.to1 = cfg['Controller']['Stance'] * \
        #     self.T + self.tc1  # left leg lift off
        self.to1 = self.Ts * self.T + self.tc1  # left leg lift off
        self.tc2 = cfg['Controller']['Phase'][1] * \
            self.T           # right leg touch down
        # self.to2 = cfg['Controller']['Stance'] * \
        #     self.T + self.tc2  # right leg lift off
        self.to2 = self.Ts * self.T + self.tc2  # right leg lift off
        self.phase = cfg['Controller']['Phase']
        
        # print((self.tc2-self.tc1) % self.dt)
        # print(self.dt)
        assert (self.tc2-self.tc1) % self.dt <= 1e-10, "Non integer interval"
        assert (self.tc1+self.T-self.tc1) % self.dt <= 1e-10, "Non integer interval"

        self.NN = [int((self.tc2-self.tc1)/self.dt + 1),
                   int((self.tc1+self.T-self.tc2)/self.dt + 1)]
        # self.N_stance = int((cfg['Controller']['Stance'] * self.T) // self.dt)
        self.N_stance = int((Stance * self.T) // self.dt)

        # mass and geometry related parameter
        self.m = cfg['Robot']['Mass']['mass']
        self.I = cfg['Robot']['Mass']['inertia']
        # self.m = Mas
        # self.I = inert
        self.l = cfg['Robot']['Mass']['massCenter']
        self.I_ = [self.m[i]*self.l[i]**2+self.I[i] for i in range(4)]

        self.L = [cfg['Robot']['Geometry']['L_body'],
                  cfg['Robot']['Geometry']['L_thigh'],
                  cfg['Robot']['Geometry']['L_shank'],
                  cfg['Robot']['Geometry']['L_arm']]

        # motor parameter
        self.motor_cs = cfg['Robot']['Motor']['CriticalSpeed']
        self.motor_ms = cfg['Robot']['Motor']['MaxSpeed']
        self.motor_mt = cfg['Robot']['Motor']['MaxTorque']

        # evironemnt parameter
        self.mu = cfg['Environment']['Friction_Coeff']
        self.g = cfg['Environment']['Gravity']
        self.damping = cfg['Robot']['damping']

        # self.vel_aim = cfg['Controller']['Target']
        self.vel_aim = Vt

        print("="*50)
        print("target vel: ", self.vel_aim)
        print("m:, ", self.m)
        print("I:, ", self.I)

        # boundary parameter
        self.bound_fy = cfg['Controller']['Boundary']['Fy']
        self.bound_fx = cfg['Controller']['Boundary']['Fx']
        self.F_LB = [self.bound_fx[0], self.bound_fy[0]] * 2
        self.F_UB = [self.bound_fx[1], self.bound_fy[1]] * 2

        self.u_LB = [-self.motor_mt] * 6
        self.u_UB = [self.motor_mt] * 6

        # FF = cfg["Controller"]["Forward"]
        FF = cfg["Controller"]["FrontForward"]
        HF = cfg["Controller"]["HindForward"]

        self.q_LB = [cfg['Controller']['Boundary']['x'][0],
                     cfg['Controller']['Boundary']['y'][0],
                     -np.pi/2, -np.pi*5/6 if FF else 0, -np.pi/2, -np.pi*5/6 if HF else 0,
                     -np.pi/3, -np.pi/3]    # arm 
        self.q_UB = [cfg['Controller']['Boundary']['x'][1],
                     cfg['Controller']['Boundary']['y'][1],
                     np.pi/2, 0 if FF else np.pi, np.pi/2, 0 if HF else np.pi,
                     np.pi/3, np.pi/3]  # arm 

        self.dq_LB = [cfg['Controller']['Boundary']['dx'][0],
                      cfg['Controller']['Boundary']['dy'][0],
                      -self.motor_ms, -self.motor_ms,
                      -self.motor_ms, -self.motor_ms,
                      -self.motor_ms, -self.motor_ms]   # arm 

        self.dq_UB = [cfg['Controller']['Boundary']['dx'][1],
                      cfg['Controller']['Boundary']['dy'][1],
                      self.motor_ms, self.motor_ms,
                      self.motor_ms, self.motor_ms,
                      self.motor_ms, self.motor_ms] # arm 

        # * define variable
        self.q = []
        self.q.append([self.opti.variable(8) for _ in range(self.NN[0])])
        self.q.append([self.opti.variable(8) for _ in range(self.NN[1])])
        self.dq = []
        self.dq.append([self.opti.variable(8) for _ in range(self.NN[0])])
        self.dq.append([self.opti.variable(8) for _ in range(self.NN[1])])
        self.ddq = []
        self.ddq.append([(self.dq[0][i+1]-self.dq[0][i]) /
                        self.dt for i in range(self.NN[0]-1)])
        self.ddq[0].extend([self.dq[1][0]-self.dq[0][-1]])
        self.ddq.append([(self.dq[1][i+1]-self.dq[1][i]) /
                        self.dt for i in range(self.NN[1]-1)])
        self.ddq[1].extend([self.dq[0][0]-self.dq[1][-1]])

        self.u = []
        # # ! set the last u to be zero at constraint
        self.u.append([self.opti.variable(6) for _ in range(self.NN[0])])
        # ! set the last u to be zero at constraint
        self.u.append([self.opti.variable(6) for _ in range(self.NN[1])])

        self.F = []
        # ! Note, the last force represents the plused force at the contact moment
        self.F.append([self.opti.variable(4) for _ in range(self.NN[0])])
        self.F.append([self.opti.variable(4) for _ in range(self.NN[1])])

        pass

    def mass_matrix(self, q):
        # region create mass matrix
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
        m3 = self.m[3]
        lc1 = self.l[1]
        lc2 = self.l[2]
        lc3 = self.l[3]
        L1 = self.L[1]

        m00 = m0+2*m1+2*m2+2*m3
        m01 = 0
        m02 = (m1*lc1+m2*L1)*c(q[0])+m2*lc2*c(q[0]+q[1])
        m03 = m2*lc2*c(q[0]+q[1])
        m04 = (m1*lc1+m2*L1)*c(q[2])+m2*lc2*c(q[2]+q[3])
        m05 = m2*lc2*c(q[2]+q[3])
        m06 = (m3*lc3)*c(q[4])
        m07 = m3*lc3*c(q[5])

        m10 = m01
        m11 = m0+2*m1+2*m2+2*m3
        m12 = (m1*lc1+m2*L1)*s(q[0])+m2*lc2*s(q[0]+q[1])
        m13 = m2*lc2*s(q[0]+q[1])
        m14 = (m1*lc1+m2*L1)*s(q[2])+m2*lc2*s(q[2]+q[3])
        m15 = m2*lc2*s(q[2]+q[3])
        m16 = (m3*lc3)*s(q[4])
        m17 = (m3*lc3)*s(q[5])

        m20 = m02
        m21 = m12
        m22 = self.I_[1] + self.I_[2] + m2*L1**2 + 2*m2*lc2*L1*c(q[1])
        m23 = self.I_[2] + m2*lc2*L1*c(q[1])
        m24 = 0
        m25 = 0
        m26 = 0
        m27 = 0

        m30 = m03
        m31 = m13
        m32 = m23
        m33 = self.I_[2]
        m34 = 0
        m35 = 0
        m36 = 0
        m37 = 0

        m40 = m04
        m41 = m14
        m42 = m24
        m43 = m34
        m44 = self.I_[1] + self.I_[2] + m2*L1**2 + 2*m2*lc2*L1*c(q[3])
        m45 = self.I_[2] + m2*lc2*L1*c(q[3])
        m46 = 0
        m47 = 0

        m50 = m05
        m51 = m15
        m52 = m25
        m53 = m35
        m54 = m45
        m55 = self.I_[2]
        m56 = 0
        m57 = 0

        m60 = m06
        m61 = m16
        m62 = m26
        m63 = m36
        m64 = m46
        m65 = m56
        m66 = self.I_[3]
        m67 = 0

        m70 = m07
        m71 = m17
        m72 = m27
        m73 = m37
        m74 = m47
        m75 = m57
        m76 = m67
        m77 = self.I_[3]


        return [[m00, m01, m02, m03, m04, m05, m06, m07],
                [m10, m11, m12, m13, m14, m15, m16, m17],
                [m20, m21, m22, m23, m24, m25, m26, m27],
                [m30, m31, m32, m33, m34, m35, m36, m37],
                [m40, m41, m42, m43, m44, m45, m46, m47],
                [m50, m51, m52, m53, m54, m55, m56, m57],
                [m60, m61, m62, m63, m64, m65, m66, m67],
                [m70, m71, m72, m73, m74, m75, m76, m77]]
        # endregion

    def coriolis(self, q, dq):
        # region calculate the coriolis force
        m1 = self.m[1]
        m2 = self.m[2]
        m3 = self.m[3]
        lc1 = self.l[1]
        lc2 = self.l[2]
        lc3 = self.l[3]
        L1 = self.L[1]

        c0 = -(m1*lc1*s(q[0])+m2*L1*s(q[0]))*dq[0]*dq[0] - \
            (m2*lc2*s(q[0]+q[1]))*(dq[0]+dq[1])*(dq[0]+dq[1])
        c0 += (-(m1*lc1*s(q[2])+m2*L1*s(q[2]))*dq[2]*dq[2] -
               (m2*lc2*s(q[2]+q[3]))*(dq[2]+dq[3])*(dq[2]+dq[3]))
        c0 += -m3*lc3*s(q[4])*dq[4]**2 - m3*lc3*s(q[5])*dq[5]**2

        c1 = (m1*lc1*c(q[0])+m2*L1*c(q[0]))*dq[0]*dq[0] + \
            (m2*lc2*c(q[0]+q[1]))*(dq[0]+dq[1])*(dq[0]+dq[1])
        c1 += ((m1*lc1*c(q[2])+m2*L1*c(q[2]))*dq[2]*dq[2] +
               (m2*lc2*c(q[2]+q[3]))*(dq[2]+dq[3])*(dq[2]+dq[3]))
        c1 += m3*lc3*c(q[4])*dq[4]**2 + m3*lc3*c(q[5])*dq[5]**2

        c2 = -m2*L1*lc2*s(q[1])*(2*dq[0]+dq[1])*dq[1]

        c3 = m2*lc2*L1*s(q[1])*dq[0]*dq[0]

        c4 = -m2*L1*lc2*s(q[3])*(2*dq[2]+dq[3])*dq[3]

        c5 = m2*lc2*L1*s(q[3])*dq[2]*dq[2]

        c6 = 0
        c7 = 0
        return [c0, c1, c2, c3, c4, c5, c6, c7]
        # endregion

    def gravity(self, q):
        # region calculate the gravity
        g0 = 0
        g1 = self.m[0]+2*self.m[1]+2*self.m[2]+2*self.m[3]
        g2 = (self.m[1]*self.l[1]+self.m[2]*self.L[1]) * \
            s(q[0]) + self.m[2]*self.l[2]*s(q[0]+q[1])
        g3 = self.m[2]*self.l[2]*s(q[0]+q[1])
        g4 = (self.m[1]*self.l[1]+self.m[2]*self.L[1]) * \
            s(q[2]) + self.m[2]*self.l[2]*s(q[2]+q[3])
        g5 = self.m[2]*self.l[2]*s(q[2]+q[3])
        g6 = self.m[3]*self.l[3]*s(q[4])
        g7 = self.m[3]*self.l[3]*s(q[5])
        return [g0*self.g, g1*self.g, g2*self.g, g3*self.g, g4*self.g, g5*self.g, g6*self.g, g7*self.g]
        # endregion

    def contact_force(self, q, F):
        # region calculate the contact force
        # F = [Fxl, Fyl, Fxr, Fyr]
        cont0 = F[0] + F[2]
        cont1 = F[1] + F[3]
        cont2 = (self.L[1]*c(q[0])+self.L[2]*c(q[0]+q[1]))*F[0] + \
                (self.L[1]*s(q[0])+self.L[2]*s(q[0]+q[1]))*F[1]
        cont3 = (self.L[2]*c(q[0]+q[1]))*F[0] + (self.L[2]*s(q[0]+q[1]))*F[1]
        cont4 = (self.L[1]*c(q[2])+self.L[2]*c(q[2]+q[3]))*F[2] + \
                (self.L[1]*s(q[2])+self.L[2]*s(q[2]+q[3]))*F[3]
        cont5 = (self.L[2]*c(q[2]+q[3]))*F[2] + (self.L[2]*s(q[2]+q[3]))*F[3]
        cont6 = 0
        cont7 = 0
        return [cont0, cont1, cont2, cont3, cont4, cont5, cont6, cont7]
        # endregion

    def inertia_force(self, q, acc):
        # region calculate inertia force
        mm = self.mass_matrix(q[2:])
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2] +
                         mm[i][3]*acc[3]+mm[i][4]*acc[4]+mm[i][5]*acc[5] + 
                         mm[i][6]*acc[6]+mm[i][7]*acc[7] for i in range(8)]
        return inertia_force
        # endregion

    def inertia_force2(self, q, acc):
        # region calculate inertia force, split into two parts
        mm = self.mass_matrix(q[2:])
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2] +
                         mm[i][3]*acc[3]+mm[i][4]*acc[4]+mm[i][5]*acc[5] + 
                         mm[i][6]*acc[6]+mm[i][7]*acc[7] for i in range(8)]
        inertia_main = [mm[i][i]*acc[i] for i in range(8)]
        inertia_coupling = [inertia_force[i]-inertia_main[i] for i in range(8)]
        return inertia_main, inertia_coupling
    
    def inertia_coupling_force(self, q, acc):
        mm = self.mass_matrix(q[2:])
        inertia_coupling_x = [mm[0][1]*acc[1], mm[0][2]*acc[2]+mm[0][3]*acc[3], mm[0][4]*acc[4]+mm[0][5]*acc[5],
                              mm[0][6]*acc[6], mm[0][7]*acc[7]]
        inertia_coupling_y = [mm[1][0]*acc[0], mm[1][2]*acc[2]+mm[1][3]*acc[3], mm[1][4]*acc[4]+mm[1][5]*acc[5],
                              mm[1][6]*acc[6], mm[1][7]*acc[7]]
        return inertia_coupling_x, inertia_coupling_y

    def foot_pos(self, q):
        # region calculate foot coordinate in world frame
        # q = [x, y, q0, q1, q2, q3]
        foot_lx = q[0] + self.L[1]*s(q[2]) + self.L[2]*s(q[2]+q[3])
        foot_ly = q[1] - self.L[1]*c(q[2]) - self.L[2]*c(q[2]+q[3])
        foot_rx = q[0] + self.L[1]*s(q[4]) + self.L[2]*s(q[4]+q[5])
        foot_ry = q[1] - self.L[1]*c(q[4]) - self.L[2]*c(q[4]+q[5])
        return foot_lx, foot_ly, foot_rx, foot_ry
        # endregion

    def foot_vel(self, q, dq):
        # region calculate foot coordinate in world frame
        # q = [x, y, q0, q1, q2, q3]
        # dq = [dx, dy, dq0, dq1, dq2, dq3]
        df_lx = dq[0] + self.L[1]*c(q[2])*dq[2] + \
            self.L[2]*c(q[2]+q[3])*(dq[2]+dq[3])
        df_ly = dq[1] + self.L[1]*s(q[2])*dq[2] + \
            self.L[2]*s(q[2]+q[3])*(dq[2]+dq[3])
        df_rx = dq[0] + self.L[1]*c(q[4])*dq[4] + \
            self.L[2]*c(q[4]+q[5])*(dq[4]+dq[5])
        df_ry = dq[1] + self.L[1]*s(q[4])*dq[4] + \
            self.L[2]*s(q[4]+q[5])*(dq[4]+dq[5])
        return df_lx, df_ly, df_rx, df_ry
        # endregion

    def get_jacobian(self, q):
        J = [[0 for i in range(4)] for j in range(8)]
        J[0][0] = 1
        J[0][2] = 1
        J[1][1] = 1
        J[1][3] = 1
        J[2][0] = (self.L[1]*c(q[0])+self.L[2]*c(q[0]+q[1]))
        J[2][1] = (self.L[1]*s(q[0])+self.L[2]*s(q[0]+q[1]))
        J[3][0] = (self.L[2]*c(q[0]+q[1]))
        J[3][1] = (self.L[2]*s(q[0]+q[1]))
        J[4][2] = (self.L[1]*c(q[2])+self.L[2]*c(q[2]+q[3]))
        J[4][3] = (self.L[1]*s(q[2])+self.L[2]*s(q[2]+q[3]))
        J[5][2] = (self.L[2]*c(q[2]+q[3]))
        J[5][3] = (self.L[2]*s(q[2]+q[3]))
        return J

    @staticmethod
    def get_posture(q):
        L = [0, 0.42, 0.5, 0.4]
        # L = [0, 0.2, 0.3, 0.3]
        lx = np.zeros(3)
        ly = np.zeros(3)
        rx = np.zeros(3)
        ry = np.zeros(3)
        lax = np.zeros(2)
        lay = np.zeros(2)
        rax = np.zeros(2)
        ray = np.zeros(2)
        lx[0] = q[0]
        lx[1] = lx[0] + L[1]*np.sin(q[2])
        lx[2] = lx[1] + L[2]*np.sin(q[2]+q[3])
        ly[0] = q[1]
        ly[1] = ly[0] - L[1]*np.cos(q[2])
        ly[2] = ly[1] - L[2]*np.cos(q[2]+q[3])

        rx[0] = q[0]
        rx[1] = rx[0] + L[1]*np.sin(q[4])
        rx[2] = rx[1] + L[2]*np.sin(q[4]+q[5])
        ry[0] = q[1]
        ry[1] = ry[0] - L[1]*np.cos(q[4])
        ry[2] = ry[1] - L[2]*np.cos(q[4]+q[5])

        lax[0] = q[0]
        lax[1] = lax[0] + L[3]*np.sin(q[6])
        # lay[0] = 0.2+q[1]
        lay[0] = 0.5+q[1]
        lay[1] = lay[0] - L[3]*np.cos(q[6])

        rax[0] = q[0]
        rax[1] = rax[0] + L[3]*np.sin(q[7])
        # ray[0] = 0.2+q[1]
        ray[0] = 0.5+q[1]
        ray[1] = ray[0] - L[3]*np.cos(q[7])
        return (lx, ly, rx, ry, lax, lay, rax, ray)

    @staticmethod
    def get_motor_boundary(speed, MaxTorque=36, CriticalSpeed=27, MaxSpeed=53):
        upper = MaxTorque - (speed-CriticalSpeed) / \
            (MaxSpeed-CriticalSpeed)*MaxTorque
        upper = np.clip(upper, 0, MaxTorque)
        lower = -MaxTorque + (speed+CriticalSpeed) / \
            (-MaxSpeed+CriticalSpeed)*MaxTorque
        lower = np.clip(lower, -MaxTorque, 0)
        return upper, lower

    @staticmethod
    def inverse_kinematics(foot, L, is_forward=True):
        l1 = L[0]
        l2 = L[1]

        l = sqrt(foot[0]**2+foot[1]**2)

        if l > l1+l2:
            foot[0] = foot[0] * (l1+l2-1e-5) / l
            foot[1] = foot[1] * (l1+l2-1e-5) / l
            pass
        l = sqrt(foot[0]**2+foot[1]**2)
        foot[1] = abs(foot[1])

        theta2 = np.pi - acos((l1**2+l2**2-foot[0]**2-foot[1]**2)/2/l1/l2)
        theta2 = theta2 if (not is_forward) else -theta2

        theta1 = atan2(foot[0], foot[1]) - acos((l1**2+l**2-l2**2)/2/l1/l) if (
            not is_forward) else atan2(foot[0], foot[1]) + acos((l1**2+l**2-l2**2)/2/l1/l)

        return np.asarray([theta1, theta2])

    pass


class nlp():
    def __init__(self, Period, Stance, Vt, legged_robot, cfg, armflag = True, seed=None, is_ref=False):
        # load parameter
        self.cfg = cfg
        self.armflag = armflag
        self.trackingCoeff = cfg["Optimization"]["CostCoeff"]["trackingCoeff"]
        self.powerCoeff = cfg["Optimization"]["CostCoeff"]["powerCoeff"]
        self.forceCoeff = cfg["Optimization"]["CostCoeff"]["forceCoeff"]
        self.smoothCoeff = cfg["Optimization"]["CostCoeff"]["smoothCoeff"]
        self.impactCoeff = cfg["Optimization"]["CostCoeff"]["ImpactCoeff"]
        self.forceRatio = cfg["Environment"]["ForceRatio"]
        # self.FF_flag = cfg["Controller"]["Forward"]
        self.FF_flag = cfg["Controller"]["FrontForward"]
        self.HF_flag = cfg["Controller"]["HindForward"]
        max_iter = cfg["Optimization"]["MaxLoop"]
        self.random_seed = cfg["Optimization"]["RandomSeed"]

        self.cost = self.Cost(legged_robot)
        legged_robot.opti.minimize(self.cost)

        self.ceq = self.getConstraints(legged_robot)
        legged_robot.opti.subject_to(self.ceq)

        p_opts = {"expand": True, "error_on_fail": False}
        s_opts = {"max_iter": max_iter}
        legged_robot.opti.solver("ipopt", p_opts, s_opts)
        self.initialGuess(legged_robot, seed=seed, is_ref=is_ref,
                          lam=Stance,
                          delta=cfg['Ref']['delta'],
                          target=Vt,
                          offset=cfg['Ref']['offset'],
                          alpha=cfg['Ref']['alpha'],
                          beta=cfg['Ref']['beta'],
                          geometry=legged_robot.L[1:])
        pass

    def initialGuess(self, walker, seed=None, is_ref=False, **param):
        np.random.seed(self.random_seed)
        if seed is None:
            init = walker.opti.set_initial
            q = walker.q
            dq = walker.dq
            if not is_ref:
                for i in range(2):
                    for j in range(walker.NN[i]):
                        init(dq[i][j][0], walker.vel_aim)
                        init(q[i][j][0], walker.dt *
                             (j+i*(walker.NN[0]-1))*walker.vel_aim)
                        init(q[i][j][1], 0.45)
                        init(q[i][j][2], 0.5 if self.FF_flag else -0.5)
                        init(q[i][j][4], 0.5 if self.HF_flag else -0.5)
                        init(q[i][j][3], -1 if self.FF_flag else 1)
                        init(q[i][j][5], -1 if self.HF_flag else 1)
                        # if self.armflag:
                        #     init(q[i][j][6], -0.5)
                        #     init(q[i][j][7], 0.5)
                        # else:
                        #     init(q[i][j][6], 0)
                        #     init(q[i][j][7], 0)

                        if self.armflag:
                            init(q[i][j][6], -0.5*cos(j/walker.NN[i]*np.pi)*(1 if i==0 else -1))
                            init(q[i][j][7], 0.5*cos(j/walker.NN[i]*np.pi)*(1 if i==0 else -1))
                        else:
                            init(q[i][j][6], 0)
                            init(q[i][j][7], 0)
                        pass
                    pass
                pass
            else:
                t1 = (np.linspace(0, 1, walker.N) + walker.phase[0]) % 1.0
                t2 = (np.linspace(0, 1, walker.N) + walker.phase[1]) % 1.0
                theta_l_ref = nlp.refTraj(
                    t1, walker.T, is_forward=self.cfg['Controller']['FrontForward'], **param)
                theta_r_ref = nlp.refTraj(
                    t2, walker.T, is_forward=self.cfg['Controller']['HindForward'], **param)
                vel_l_ref = [(theta_l_ref[i+1]-theta_l_ref[i]) /
                             walker.dt for i in range(len(theta_l_ref)-1)]
                vel_l_ref.append(vel_l_ref[0])

                vel_r_ref = [(theta_r_ref[i+1]-theta_r_ref[i]) /
                             walker.dt for i in range(len(theta_r_ref)-1)]
                vel_r_ref.append(vel_r_ref[0])

                # initialize the solution
                for i in range(2):
                    for j in range(walker.NN[i]):
                        # int x dof
                        init(q[i][j][0], walker.dt *
                             (j+i*(walker.NN[0]-1))*walker.vel_aim)
                        init(dq[i][j][0], walker.vel_aim)
                        # init z dof
                        init(q[i][j][1], param['offset'])
                        init(dq[i][j][1], 0)
                        # int left leg
                        init(q[i][j][2], theta_l_ref[j+i*(walker.NN[0]-1)][0])
                        init(dq[i][j][2], vel_l_ref[j+i*(walker.NN[0]-1)][0])
                        init(q[i][j][3], theta_l_ref[j+i*(walker.NN[0]-1)][1])
                        init(dq[i][j][3], vel_l_ref[j+i*(walker.NN[0]-1)][1])
                        # int right leg
                        init(q[i][j][4], theta_r_ref[j+i*(walker.NN[0]-1)][0])
                        init(dq[i][j][4], vel_r_ref[j+i*(walker.NN[0]-1)][0])
                        init(q[i][j][5], theta_r_ref[j+i*(walker.NN[0]-1)][1])
                        init(dq[i][j][5], vel_r_ref[j+i*(walker.NN[0]-1)][1])
                        if self.armflag:
                            init(q[i][j][6], -0.5*cos(j/walker.NN[i]*np.pi)*(1 if i==0 else -1))
                            init(q[i][j][7], 0.5*cos(j/walker.NN[i]*np.pi)*(1 if i==0 else -1))
                        else:
                            init(q[i][j][6], 0)
                            init(q[i][j][7], 0)
                        # # int arm
                        # init(q[i][j][6], theta_r_ref[j+i*(walker.NN[0]-1)][0])
                        # init(dq[i][j][6], vel_r_ref[j+i*(walker.NN[0]-1)][0])
                        # init(q[i][j][7], theta_r_ref[j+i*(walker.NN[0]-1)][1])
                        # init(dq[i][j][7], vel_r_ref[j+i*(walker.NN[0]-1)][1])
                        pass
                    pass

                pass
            pass
        else:
            old_solution = np.load(seed)
            assert old_solution.shape[0] == (
                walker.CollectionNum+2), "The collection number must be the same, please check the config file"
            assert abs(old_solution[1, -1]-old_solution[0, -1] -
                       walker.dt) <= 1e-6, "Time step and Period must be the same"
            for i in range(2):
                for j in range(walker.NN[i]):
                    for k in range(8):
                        # set position
                        walker.opti.set_initial(
                            walker.q[i][j][k], old_solution[i*walker.NN[0]+j, k])
                        # set velocity
                        walker.opti.set_initial(
                            walker.dq[i][j][k], old_solution[i*walker.NN[0]+j, 6+k])
                        if(k < 4):
                            # set external force
                            walker.opti.set_initial(
                                walker.F[i][j][k], old_solution[i*walker.NN[0]+j, 18+4+k])
                        if(k < 6):
                            if j < walker.NN[i]-1:
                                # set actuator
                                walker.opti.set_initial(
                                    walker.u[i][j][k], old_solution[i*(walker.NN[0]-1)+j, 18+k])
                                pass
                            pass
                        pass
                    pass
                pass
            pass
        pass

    def Cost(self, walker):
        # region aim function of optimal control
        FM = [walker.bound_fx[1], walker.bound_fy[1]]*2
        power = 0
        force = 0
        VelTrack = 0

        for j in range(2):
            for i in range(walker.NN[j]-1):
                for k in range(6):
                    power += (walker.dq[j][i][k+2]
                              * walker.u[j][i][k])**2 * walker.dt
                    force += (walker.u[j][i][k]/walker.motor_mt)**2
                    pass
                pass
                for k in range(4):
                    force += (walker.F[j][i][k]/FM[k])**2
                    pass
                pass
            pass
        # VelTrack = ca.fabs(walker.q[1][-1][0]-walker.q[0]                                                                                                                                                                                                                                                                                                                     
        #                    [0][0] - walker.vel_aim*walker.T)*walker.N
        VelTrack = (walker.q[1][-1][0]-walker.q[0]
                    [0][0] - walker.vel_aim*walker.T)**2*walker.N
        # endregion

        u = walker.u
        F = walker.F
        smooth = 0
        AM = [100, 400, 100, 400, 100, 100]
        for j in range(2):
            for i in range(walker.NN[j]-2):
                for k in range(4):
                    smooth += ((F[j][i+1][k]-F[j][i][k])/AM[k])**2
                    pass
                pass 
                for k in range(6):
                    smooth += ((walker.u[j][i+1][k]-walker.u[j][i][k])/20)**2
                    pass
                pass
            pass

        # minimize the impact loss
        vel_l_x, vel_l_y, _, _ = walker.foot_vel(
            walker.q[1][-1], walker.dq[1][-1])
        _, _, vel_r_x, vel_r_y = walker.foot_vel(
            walker.q[0][-1], walker.dq[0][-1])
        impact = vel_l_x**2 + vel_l_y**2 + vel_r_x**2 + vel_r_y**2
        impact = impact / walker.vel_aim**2 * walker.N

        res = 0
        res = (res + VelTrack *
               self.trackingCoeff) if (self.trackingCoeff > 1e-6) else res
        res = (res + power*self.powerCoeff) if (self.powerCoeff > 1e-6) else res
        res = (res + force*self.forceCoeff) if (self.forceCoeff > 1e-6) else  res
        res = (res + smooth*self.smoothCoeff) if (self.smoothCoeff > 1e-6) else res
        res = (res + impact*self.impactCoeff) if (self.impactCoeff > 1e-6) else res

        return res

    def getConstraints(self, walker):
        ceq = []
        # region dynamics constraints
        # continuous dynamics
        #! 约束的数量为 (6+6）*（NN1-1+NN2-1）
        for i in range(2):
            for j in range(walker.NN[i]-1):
                ceq.extend([walker.q[i][j+1][k]-walker.q[i][j][k]-walker.dt/2 *
                           (walker.dq[i][j+1][k]+walker.dq[i][j][k]) == 0 for k in range(8)])
                inertia = walker.inertia_force(
                    walker.q[i][j], walker.ddq[i][j])
                coriolis = walker.coriolis(
                    walker.q[i][j][2:], walker.dq[i][j][2:])
                gravity = walker.gravity(walker.q[i][j][2:])
                contact = walker.contact_force(
                    walker.q[i][j][2:], walker.F[i][j])
                ceq.extend([inertia[k]+gravity[k]+coriolis[k] -
                            contact[k] == 0 for k in range(2)])
                ceq.extend([inertia[k+2]+gravity[k+2]+coriolis[k+2] -
                            contact[k+2] - walker.u[i][j][k] == 0 for k in range(6)])
                # arm bound constraints
                if not self.armflag:
                    ceq.extend([walker.q[i][j][k+6] == 0 for k in range(2)])
                    ceq.extend([walker.dq[i][j][k+6] == 0 for k in range(2)])
                pass
            pass

        # discrete dynamics
        #! 约束的数量为 5*2+1+4
        for i in range(2):
            ceq.extend([walker.q[i][-1][j]-walker.q[1-i]
                        [0][j] == 0 for j in [1, 2, 3, 4, 5, 6, 7]])  # x dof is not periodic
            ceq.extend([walker.dq[i][-1][j]-walker.dq[1-i]
                        [0][j] == 0 for j in [6, 7]])
            # acc = (walker.dq[1-i][0] - walker.dq[i][-1])/walker.dt
            # ! to avoid to big impluse which lead to big numerical error
            acc = walker.ddq[i][-1]
            inertia_force = walker.inertia_force(walker.q[i][-1], acc)
            contact_force = walker.contact_force(
                walker.q[i][-1][2:], walker.F[i][-1])
            ceq.extend([inertia_force[j]-contact_force[j]
                       == 0 for j in range(6)])
            pass
        #! 将第一段轨迹与第二段轨迹的x自由度相连接，但是第二段的末尾不与第一段连接, 否则机器人无法向前奔跑
        ceq.extend([walker.q[0][-1][0]-walker.q[1][0][0] == 0])

        #! 左脚轨迹的末端，是右脚接触的开始，右脚有脉冲力，左脚没有
        ceq.extend([walker.F[0][-1][0] == 0])
        ceq.extend([walker.F[0][-1][1] == 0])
        #! 右脚轨迹的末端，是左脚接触的开始，左脚有脉冲力，右脚没有
        ceq.extend([walker.F[1][-1][2] == 0])
        ceq.extend([walker.F[1][-1][3] == 0])

        ceq.extend([walker.u[0][-1][i] == 0 for i in range(4)])
        ceq.extend([walker.u[1][-1][i] == 0 for i in range(4)])

        # endregion

        # region periodicity constraints
        # ! this version no periodicity constraints
        # ! this constraint is implicit in dynamics constraints
        # endregion

        # region leg locomotion constraint
        # TODO: 尝试将足底和地面接触的条件放松，允许足底有微量的位移，is_ref即假设地面或足底具有一定的变形能力（怀疑过于刚性的约束会导致求解的时候力出现抖动）
        for i in range(walker.CollectionNum):

            pl = i // (walker.NN[0]-1)
            jl = i % (walker.NN[0]-1)

            pr = 1 - i // (walker.NN[1]-1)
            jr = i % (walker.NN[1]-1)

            _, foot_y_l, _, _ = walker.foot_pos(walker.q[pl][jl])
            dfoot_x_l, dfoot_y_l, _, _ = walker.foot_vel(
                walker.q[pl][jl], walker.dq[pl][jl])

            _, _, _, foot_y_r = walker.foot_pos(walker.q[pr][jr])
            _, _, dfoot_x_r, dfoot_y_r = walker.foot_vel(
                walker.q[pr][jr], walker.dq[pr][jr])

            if i < walker.N_stance:
                # stance phase
                ceq.extend([foot_y_l == 0])  # 足底与地面接触
                ceq.extend([foot_y_r == 0])
                ceq.extend([walker.F[pl][jl][1]*walker.mu -
                           ca.fabs(walker.F[pl][jl][0]) >= 0])  # 摩擦域条件
                ceq.extend([walker.F[pr][jr][3]*walker.mu -
                           ca.fabs(walker.F[pr][jr][2]) >= 0])
                # 无滑移条件
                ceq.extend([dfoot_x_l == 0])
                ceq.extend([dfoot_x_r == 0])
                # ceq.extend([dfoot_y_l == 0])
                # ceq.extend([dfoot_y_r == 0])
                pass
            else:
                # swing phase
                ceq.extend([foot_y_l*10 > 0])  # 足底在空中
                ceq.extend([foot_y_r*10 > 0])
                ceq.extend([walker.F[pl][jl][0] == 0])  # 足底无力的作用
                ceq.extend([walker.F[pl][jl][1] == 0])
                ceq.extend([walker.F[pr][jr][2] == 0])
                ceq.extend([walker.F[pr][jr][3] == 0])
                pass
            pass

        # endregion

        # region target velocity constraint
        average_vel = 0
        for i in range(2):
            for temp_dq in walker.dq[i]:
                average_vel += temp_dq[0]
                pass
            pass
        average_vel = average_vel / (len(walker.dq[0])+len(walker.dq[1]))
        ceq.extend([average_vel == walker.vel_aim])
        # endregion

        # region boundary constraint
        for i in range(2):
            for temp_q in walker.q[i]:
                ceq.extend([walker.opti.bounded(walker.q_LB[j],
                           temp_q[j], walker.q_UB[j]) for j in range(8)])
                pass
            for temp_dq in walker.dq[i]:
                ceq.extend([walker.opti.bounded(walker.dq_LB[j],
                           temp_dq[j], walker.dq_UB[j]) for j in range(8)])
                pass
            for temp_u in walker.u[i]:
                ceq.extend([walker.opti.bounded(walker.u_LB[j],
                           temp_u[j], walker.u_UB[j]) for j in range(6)])
                pass
            for temp_f in walker.F[i]:
                ceq.extend([walker.opti.bounded(walker.F_LB[j],
                           temp_f[j], walker.F_UB[j]) for j in range(4)])
                pass
            pass
        # endregion

        # region motor external characteristic curve
        cs = walker.motor_cs
        ms = walker.motor_ms
        mt = walker.motor_mt
        for i in range(2):
            for j in range(len(walker.u[i])):
                ceq.extend([walker.u[i][j][k]-ca.fmax(mt - (walker.dq[i][j][k+2] -
                                                            cs)/(ms-cs)*mt, 0) <= 0 for k in range(6)])
                ceq.extend([walker.u[i][j][k]-ca.fmin(-mt + (walker.dq[i][j][k+2] +
                                                             cs)/(-ms+cs)*mt, 0) >= 0 for k in range(6)])
                pass
            pass
        # endregion

        # region smooth constraint
        for i in range(2):
            for j in range(len(walker.u[i])-1):
                ceq.extend([ca.fabs(walker.F[i][j][k]-walker.F[i]
                           [j+1][k]) <= 100 for k in range(4)])
                ceq.extend([ca.fabs(walker.u[i][j][k]-walker.u[i]
                           [j+1][k]) <= 10 for k in range(6)])
                pass
            
            pass
        # endregion

        return ceq

    def solve_and_output(self, robot, flag_save=True, StorePath="./"):
        # solve the nlp and stroge the solution
        q = []
        dq = []
        ddq = []
        u = []
        f = []
        t = []
        try:
            sol1 = robot.opti.solve()
            # stat = robot.opti.solver.stats()
            for i in range(2):
                for j in range(robot.NN[i]):
                    # print("=============================")
                    # print(stat)
                    # print(sol1.value(f))
                    t.append(j*robot.dt+i*(robot.NN[0]-1)*robot.dt)
                    q.append([sol1.value(robot.q[i][j][k]) for k in range(8)])
                    dq.append([sol1.value(robot.dq[i][j][k])
                              for k in range(8)])
                    ddq.append([sol1.value(robot.ddq[i][j][k])
                               for k in range(8)] if j < (robot.NN[i]-1) else [0]*8)
                    u.append([sol1.value(robot.u[i][j][k])
                             for k in range(6)] if j < (robot.NN[i]-1) else [0]*6)
                    f.append([sol1.value(robot.F[i][j][k]) for k in range(4)])
                    pass
                pass
            pass
        except:
            value = robot.opti.debug.value
            # stat = robot.opti.solver.stats()
            for i in range(2):
                for j in range(robot.NN[i]):
                    # print("=============================")
                    # print(stat)
                    # print(value(f))
                    t.append(j*robot.dt+i*(robot.NN[0]-1)*robot.dt)
                    q.append([value(robot.q[i][j][k])
                             for k in range(8)])
                    dq.append([value(robot.dq[i][j][k])
                              for k in range(8)])
                    ddq.append([value(robot.ddq[i][j][k])
                               for k in range(8)])
                    u.append([value(robot.u[i][j][k])
                             for k in range(6)])
                    f.append([value(robot.F[i][j][k])
                             for k in range(4)])
                    pass
                pass
            pass
        finally:
            q = np.asarray(q)
            dq = np.asarray(dq)
            ddq = np.asarray(ddq)
            u = np.asarray(u)
            f = np.asarray(f)
            t = np.asarray(t).reshape([-1, 1])

            # if(flag_save):
            #     import time
            #     date = time.strftime("%d_%m_%Y_%H_%M_%S")
            #     # output solution of NLP
            #     np.save(StorePath+date+"_sol.npy",
            #             np.hstack((q, dq, ddq, u, f, t)))
            #     # output the config yaml file
            #     with open(StorePath+date+"config.yaml", mode='w') as file:
            #         YAML().dump(self.cfg, file)
            #     pass

            return q, dq, ddq, u, f, t

    @staticmethod
    def refTraj(t, period, **param):
        # get the reference trajectory of joints
        ref_x = []
        ref_y = []
        lam = param['lam']
        delta = param['delta']
        target_velocity = param['target']
        offset = -param['offset']
        alpha = param['alpha']  # default 0.15
        beta = param['beta'] * period    # default -200*period
        geometry = param['geometry']  # link length
        is_forward = param['is_forward']

        v1 = (lam+delta)/(1-lam-3*delta)*target_velocity*period
        v2 = -target_velocity*period

        # reference of x
        for tt in t:
            if tt < lam:
                ref_x.append(v2*tt - v2 * lam / 2)
            elif lam <= tt < lam + delta:
                ref_x.append((tt-lam)*(v2+(lam+delta-tt)/(delta)*v2) /
                             2+v2*lam - v2 * lam / 2)
            elif lam+delta <= tt < lam + delta * 2:
                ref_x.append(v2/2*(delta+lam)+v1/2/delta*(tt-lam-delta)**2)
            elif lam + delta * 2 <= tt < 1-2*delta:
                ref_x.append(-v1*(1-lam-4*delta)/2+v1*(tt-lam-delta*2))
            elif 1-2*delta <= tt < 1-delta:
                ttt = tt - (1-2*delta)
                ref_x.append(v1*(1-lam-4*delta)/2 +
                             ((delta-ttt)/delta*v1+v1)/2*ttt)
            else:
                ttt = tt - (1-delta)
                ref_x.append(0.5*v1*(1-lam-3*delta)+ttt/delta*v2*ttt/2)
            pass

        # reference of y
        for tt in t:
            if tt < lam:
                ref_y.append(0+offset)
            else:
                ttt = tt - (1+lam)/2
                T = (1-lam)/2
                ref_y.append((0.25*ttt**4-0.5*alpha**2*ttt **
                              2-0.25*T**4+0.5*alpha**2*T**2)*beta+offset)
            pass

        # calculate the inverse kinematics
        ik = Bipedal_hybrid.inverse_kinematics
        theta = [ik([ref_x[i], ref_y[i]], geometry, is_forward=is_forward)
                 for i in range(len(ref_x))]
        return theta

    pass


class SolutionData():
    def __init__(self, old_solution=None):
        if old_solution is None:
            self.q = None
            self.dq = None
            self.ddq = None
            self.u = None
            self.f = None
            self.t = None
            self.N = 0
            self.dt = None
            self.sudden_id = None
            pass
        else:
            self.q = old_solution[:, 0:8]
            self.dq = old_solution[:, 8:16]
            self.ddq = old_solution[:, 16:24]
            self.u = old_solution[:, 24:30]
            self.f = old_solution[:, 30:34]
            self.t = old_solution[:, 34]

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


def main(Period, Stance, Vt, armflag):
    # region optimization trajectory for bipedal hybrid robot system
    vis_flag = False
    save_flag = False
    # armflag = True

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    ani_path = StorePath + "/data/animation/"
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # seed = None
    # seed = StorePath + str(todaytime) + "_sol.npy"
    # region load config file
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Biped_walk.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # endregion

    # region create robot and NLP problem
    robot = Bipedal_hybrid(Period, Stance, Vt, cfg)
    # nonlinearOptimization = nlp(robot, cfg, seed=seed)
    # nonlinearOptimization = nlp(robot, cfg)
    nonlinearOptimization = nlp(Period, Stance, Vt, robot, cfg, armflag, is_ref=True)
    # endregion
    q, dq, ddq, u, F, t = nonlinearOptimization.solve_and_output(
        robot, flag_save=save_flag, StorePath=StorePath)
    # endregion
    if save_flag:
        visual = DataProcess(cfg, robot, q, dq, ddq, u, F, t, save_dir)
        SaveDir = visual.DataSave(save_flag)

    if vis_flag:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib as mpl
        from matplotlib.patches import ConnectionPatch
        plt.style.use("science")
        # cmap = mpl.cm.get_cmap('Paired')
        params = {
            'text.usetex': True,
            'font.size': 8,
            'pgf.preamble': [r'\usepackage{color}'],
        }
        mpl.rcParams.update(params)

        labelfont = 12
        labelfonty = 10

        cmap = mpl.cm.get_cmap('viridis')
        fig = plt.figure(figsize=(10, 7), dpi=180, constrained_layout=False)
        fig2 = plt.figure(figsize=(10, 7), dpi=180, constrained_layout=False)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.8],
                              wspace=0.3, hspace=0.33)
        gs = fig.add_gridspec(1, 1)
        gm = fig2.add_gridspec(1, 1)
        g_data = gs[0].subgridspec(3, 5, wspace=0.3, hspace=0.33)

        ax_m = fig2.add_subplot(gm[0])
        ax_p = [fig.add_subplot(g_data[0, i]) for i in range(5)]
        ax_v = [fig.add_subplot(g_data[1, i]) for i in range(5)]
        ax_u = [fig.add_subplot(g_data[2, i]) for i in range(5)]

        vel = [robot.foot_vel(q[i, :], dq[i, :]) for i in range(len(q[:, 0]))]

        # plot robot trajectory here
        ax_m.axhline(y=0, color='k')
        num_frame = 10
        for tt in np.linspace(0, robot.T, num_frame):
            idx = np.argmin(np.abs(t-tt))
            pos = Bipedal_hybrid.get_posture(q[idx, :])
            bodyy = [pos[1][0], pos[1][0]+0.5]     # Biped params len
            # bodyy = [pos[1][0], pos[1][0]+0.2]     # quadruped params len
            bodyx = [pos[0][0], pos[0][0]]
            ax_m.plot(pos[0], pos[1], 'o-', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1)
            ax_m.plot(bodyx, bodyy, 'o-', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1)
            ax_m.plot(pos[2], pos[3], 'o:', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1, ls='--')
            ax_m.plot(pos[4], pos[5], 'o-', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=2)
            ax_m.plot(pos[6], pos[7], 'o:', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=2, ls='--')
            patch = patches.Rectangle(
                (pos[0][0]-0.02, pos[1][0]-0.05), 0.04, 0.1, alpha=tt/robot.T*0.8+0.2, lw=0, color=cmap(tt/robot.T))
            ax_m.add_patch(patch)
            ax_m.axis('equal')

            # plot velocity
            vel_st = (pos[0][-1], pos[1][-1])
            vel_ed = (pos[0][-1]+vel[idx][0]*0.02, pos[1][-1]+vel[idx][1]*0.02)
            # vel_con = ConnectionPatch(vel_st, vel_ed, "data", "data",
            #                           arrowstyle="->", shrinkA=0, shrinkB=0,
            #                           mutation_scale=5, fc="w", ec='C5', zorder=10)
            # ax_m.add_patch(vel_con)
            pass
        # ax_m.axis('equal')
        ax_m.set_ylabel('z(m)', fontsize = 15)
        ax_m.set_xlabel('x(m)', fontsize = 15)
        ax_m.xaxis.set_tick_params(labelsize = 12)
        ax_m.yaxis.set_tick_params(labelsize = 12)
        # title_v = [r'$\dot{x}$', r'$\dot{y}$',
        #            r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']
        # title_u = [r'$F_x$', r'$F_y$', r'$\tau_1$', r'$\tau_2$']

        p_idx1 = [0, 1, 2, 2, 3, 3, 4, 4]
        p_idx2 = [0, 1, 2, 4, 3, 5, 6, 7]
        [ax_p[p_idx1[i]].plot(t, q[:, p_idx2[i]]) for i in range(8)]
        ax_p[0].set_ylabel('Position(m)', fontsize = labelfonty)
        
        v_idx1 = [0, 1, 2, 2, 3, 3, 4, 4]
        v_idx2 = [0, 1, 2, 4, 3, 5, 6, 7]
        [ax_v[v_idx1[i]].plot(t, dq[:, v_idx2[i]]) for i in range(8)]
        ax_v[0].set_ylabel('Velocity(m/s)', fontsize = labelfonty)
        # [ax_v[i].set_title(title_v[i]) for i in range(4)]

        ax_u[0].plot(t, F[:, 0])
        ax_u[0].plot(t, F[:, 2])
        ax_u[0].set_ylabel('Force(N)', fontsize = labelfonty)
        ax_u[0].set_xlabel('x', fontsize = labelfont)
        ax_u[1].plot(t, F[:, 1])
        ax_u[1].plot(t, F[:, 3])
        ax_u[1].set_xlabel('y', fontsize = labelfont)
        ax_u[2].plot(t, u[:, 0])
        ax_u[2].plot(t, u[:, 2])
        ax_u[2].set_xlabel('hip', fontsize = labelfont)
        ax_u[3].plot(t, u[:, 1])
        ax_u[3].plot(t, u[:, 3])
        ax_u[3].set_xlabel('knee', fontsize = labelfont)
        ax_u[4].plot(t, u[:, 4])
        ax_u[4].plot(t, u[:, 5])
        ax_u[4].set_xlabel('shoulder', fontsize = labelfont)

        # [ax_u[j].set_title(title_u[j]) for j in range(4)]
        

        if save_flag:
            savename1 =  SaveDir + "Traj"
            savename2 =  SaveDir + "Pos-Vel-uF"
            fig.savefig(savename2)
            fig2.savefig(savename1)
        plt.show()

        pass
    
    return u, F, t

    pass

def ForceMap():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pickle

    saveflag = True

    M_arm = [1.4, 2.2, 3.3, 4.2, 5.2, 6.2, 7.2, 8.0, 9.0, 10.0, 11.0]
    # M_arm = [3.3, 4.2]
    M_label = list(map(str, M_arm))
    I_arm = [0.015, 0.032, 0.045, 0.062, 0.082, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    # I_arm = [0.032, 0.045, ]
    I_label = list(map(str, I_arm))

    Tp = [0.53, 0.46, 0.46, 0.4, 0.4, 0.4, 0.35, 0.35, 0.35]
    # Tp = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
    Ts = [0.45, 0.41, 0.37, 0.34, 0.35, 0.32, 0.25, 0.24, 0.23]
    Vt = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    Mass = [30, 10, 3.2]
    inertia = [0, 0.15, 0.06]

    Fy = np.array([[0.0]*len(M_arm)])
    u_h = np.array([[0.0]*len(M_arm)])
    u_k = np.array([[0.0]*len(M_arm)])
    u_s = np.array([[0.0]*len(M_arm)])

    for i in range(len(I_arm)):
        temp_i = []
        temp_i.extend(inertia)
        temp_i.append(I_arm[i])
        Fy_max = []
        u_h_max = []
        u_k_max = []
        u_s_max = []
        for j in range(len(M_arm)):
            temp_m = []
            temp_m.extend(Mass)
            temp_m.append(M_arm[j])

            print("="*50)
            print("Mass: ", temp_m)
            print("="*50)
            print("Inertia: ", temp_i)

            u, F, t = main(temp_m, temp_i, True)

            temp_fy1_max = max(F[:, 1])
            temp_fy2_max = max(F[:, 3])
            temp_fy_max = max(temp_fy1_max, temp_fy2_max)

            temp_uh1_max = max(u[:, 0])
            temp_uh2_max = max(u[:, 2])
            temp_uh_max = max(temp_uh1_max, temp_uh2_max)

            temp_uk1_max = max(u[:, 1])
            temp_uk2_max = max(u[:, 3])
            temp_uk_max = max(temp_uk1_max, temp_uk2_max)

            temp_us1_max = max(u[:, 4])
            temp_us2_max = max(u[:, 5])
            temp_us_max = max(temp_us1_max, temp_us2_max)

            Fy_max.append(temp_fy_max)
            u_h_max.append(temp_uh_max)
            u_k_max.append(temp_uk_max)
            u_s_max.append(temp_us_max)

            pass
        
        Fy = np.concatenate((Fy, [Fy_max]), axis = 0)
        u_h = np.concatenate((u_h, [u_h_max]), axis = 0)
        u_k = np.concatenate((u_k, [u_k_max]), axis = 0)
        u_s = np.concatenate((u_s, [u_s_max]), axis = 0)

        pass
    Fy = Fy[1:]
    u_h = u_h[1:]
    u_k = u_k[1:]
    u_s = u_s[1:]

    Data = {'Fy': Fy, 'u_h': u_h, "u_k": u_k, "u_s": u_s}
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    if saveflag:
        name = "ForceMap.pkl"
        with open(os.path.join(save_dir, name), 'wb') as f:
            pickle.dump(Data, f)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    ax1 = axs[0][0]
    ax2 = axs[0][1]
    ax3 = axs[1][0]
    ax4 = axs[1][1]
    print(Fy, u_h, u_k, u_s)
    print(M_label, I_label)

    plt.style.use("science")
    # cmap = mpl.cm.get_cmap('Paired')
    params = {
        'text.usetex': True,
        'font.size': 8,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    }

    mpl.rcParams.update(params)
    # pcm1 = ax1.pcolormesh(Fy, cmap='inferno', vmin = 800, vmax = 2000)
    pcm1 = ax1.imshow(Fy, cmap='inferno', vmin = 600, vmax = 2000)
    cb1 = fig.colorbar(pcm1, ax=ax1)
    # ax1.set_xticks(np.arange(len(M_label)))
    # ax1.set_xticklabels(M_label)
    # ax1.set_yticks(np.arange(len(I_label)))
    # ax1.set_yticklabels(I_label)
    # ax1.xaxis.set_tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # pcm2 = ax2.pcolormesh(u_h, cmap='inferno', vmin = 300, vmax = 800)
    pcm2 = ax2.imshow(u_h, cmap='inferno', vmin = 300, vmax = 800)
    cb2 = fig.colorbar(pcm2, ax=ax2)
    # ax2.set_xticks(np.arange(len(M_label)))
    # ax2.set_xticklabels(M_label)
    # ax2.set_yticks(np.arange(len(I_label)))
    # ax2.set_yticklabels(I_label)
    # ax2.xaxis.set_tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # pcm3 = ax3.pcolormesh(u_k, cmap='inferno', vmin = 200, vmax = 500)
    pcm3 = ax3.imshow(u_k, cmap='inferno', vmin = 100, vmax = 500)
    cb3 = fig.colorbar(pcm3, ax=ax3)
    # ax3.set_xticks(np.arange(len(M_label)))
    # ax3.set_xticklabels(M_label)
    # ax3.set_yticks(np.arange(len(I_label)))
    # ax3.set_yticklabels(I_label)
    # ax3.xaxis.set_tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # pcm4 = ax4.pcolormesh(u_s, cmap='inferno', vmin = 50, vmax = 200)
    pcm4 = ax4.imshow(u_s, cmap='inferno', vmin = 50, vmax = 300)
    cb4 = fig.colorbar(pcm4, ax=ax4)
    # ax4.set_xticks(np.arange(len(M_label)))
    # ax4.set_xticklabels(M_label)
    # ax4.set_yticks(np.arange(len(I_label)))
    # ax4.set_yticklabels(I_label)
    # ax4.xaxis.set_tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)
    
    ax = [[ax1, ax2], [ax3, ax4]]
    cb = [[cb1, cb2], [cb3, cb4]]
    title = [["Fy", "Torque_Hip"], ["Torque_Knee", "Torque_shoulder"]]
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xticks(np.arange(len(M_label)))
            ax[i][j].set_xticklabels(M_label)
            ax[i][j].set_xticks(np.arange(len(I_label)))
            ax[i][j].set_xticklabels(I_label)
            ax[i][j].xaxis.set_tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

            ax[i][j].set_ylabel("Inertia")
            ax[i][j].set_xlabel("Mass")
            ax[i][j].set_tilte(title[i][j])

            if i==0 and j==0:
                cb[i][j].set_label("Force(N)")
            else:
                cb[i][j].set_label("Torque(N/m)")
    plt.show()
    pass

def VelForceMap():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pickle

    saveflag = False

    Period = [0.53, 0.46, 0.46, 0.4, 0.4, 0.4, 0.35, 0.35, 0.35]
    # Period = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
    Stance = [0.45, 0.41, 0.37, 0.34, 0.35, 0.32, 0.25, 0.24, 0.23]

    Vt = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    Vt_label = list(map(str, Vt))

    Fy = np.array([0.0])
    Fx = np.array([0.0])
    u_h = np.array([0.0])
    u_k = np.array([0.0])
    u_s = np.array([0.0])

    # for i in range(len(Period)):
    #     temp_p = Period[i]
    #     temp_s = Stance[i]
    #     temp_v = Vt[i]

    #     print("="*50)
    #     print("Period: ", temp_p)
    #     print("="*50)
    #     print("Stance: ", temp_s)
    #     print("="*50)
    #     print("Vt: ", temp_v)

    #     u, F, t = main(temp_p, temp_s, temp_v, True)
    #     # u, F, t = main(temp_p, temp_s, 10.0, True)

    #     temp_fx1_max = max(F[:, 0])
    #     temp_fx2_max = max(F[:, 2])
    #     temp_fx_max = max(temp_fx1_max, temp_fx2_max)

    #     temp_fy1_max = max(F[:, 1])
    #     temp_fy2_max = max(F[:, 3])
    #     temp_fy_max = max(temp_fy1_max, temp_fy2_max)

    #     temp_uh1_max = max(u[:, 0])
    #     temp_uh2_max = max(u[:, 2])
    #     temp_uh_max = max(temp_uh1_max, temp_uh2_max)

    #     temp_uk1_max = max(u[:, 1])
    #     temp_uk2_max = max(u[:, 3])
    #     temp_uk_max = max(temp_uk1_max, temp_uk2_max)

    #     temp_us1_max = max(u[:, 4])
    #     temp_us2_max = max(u[:, 5])
    #     temp_us_max = max(temp_us1_max, temp_us2_max)
        
    #     Fx = np.concatenate((Fx, [temp_fx_max]))
    #     Fy = np.concatenate((Fy, [temp_fy_max]))
    #     u_h = np.concatenate((u_h, [temp_uh_max]))
    #     u_k = np.concatenate((u_k, [temp_uk_max]))
    #     u_s = np.concatenate((u_s, [temp_us_max]))

    #     pass

    # Fx = Fx[1:]
    # Fy = Fy[1:]
    # u_h = u_h[1:]
    # u_k = u_k[1:]
    # u_s = u_s[1:]
    # print(Fx)
    Data = {'Fx': Fx, 'Fy': Fy, 'u_h': u_h, "u_k": u_k, "u_s": u_s}

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name = "Vel-Force.pkl"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    if saveflag:
        with open(os.path.join(save_dir, name), 'wb') as f:
            pickle.dump(Data, f)

    f = open(save_dir+name,'rb')
    data = pickle.load(f)

    Fx = data['Fx']
    Fy = data['Fy']
    u_h = data['u_h']
    u_k = data['u_k']
    u_s = data['u_s']

    fig, axs = plt.subplots(1, 2, figsize=(10, 7))
    ax1 = axs[0]
    ax2 = axs[1]
    # print(Fy, u_h, u_k, u_s)

    plt.style.use("science")
    # cmap = mpl.cm.get_cmap('Paired')
    params = {
        'text.usetex': True,
        'axes.labelsize': 12,
        'lines.linewidth': 3,
        'legend.fontsize': 15,
    }
    mpl.rcParams.update(params)
    # plt.style.use('fivethirtyeight')
    title = [["Fy", "Torque_Hip"], ["Torque_Knee", "Torque_shoulder"]]
    ax1.plot(Vt, Fx, label="Fx")
    ax1.plot(Vt, Fy, label="Fy")

    ax1.set_ylabel('Force(N)', fontsize = 15)
    ax1.set_xlabel('Velocity(m/s)', fontsize = 15)
    ax1.xaxis.set_tick_params(labelsize = 15)
    ax1.yaxis.set_tick_params(labelsize = 15)
    # ax1.legend(loc='upper right', fontsize = 12)
    ax1.grid()
    ax1.legend()

    ax2.plot(Vt, u_h, label="Hip Torque")
    ax2.plot(Vt, u_k, label="Knee Torque")
    ax2.plot(Vt, u_s, label="Shoulder Torque")
    ax2.set_ylabel('Torque(N/m) ', fontsize = 15)
    ax2.set_xlabel('Velocity(m/s)', fontsize = 15)
    ax2.xaxis.set_tick_params(labelsize = 15)
    ax2.yaxis.set_tick_params(labelsize = 15)
    # ax2.legend(loc='upper right', fontsize = 12)
    ax2.grid()
    ax2.legend()
    ax = [ax1, ax2]
    for i in range(2):
        ax[i].set_title(title[i][1])

    plt.show()

def VelAndAcc():
    # 这部分的脚本是用于对每个时刻的受力状态可视化
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import transforms

    # ------------------------------------------------
    # load data and preprocess
    armflag = False
    # armflag = True
    store_path = "/home/stylite-y/Documents/Master/Manipulator/Simulation/model_raisim/DRMPC/SaTiZ/data/"
    today = "2022-07-06/"
    if armflag:
        date = "2022-07-06-09-30-18-Traj-Tcf_0.0-Pcf_0.0-Fcf_0.6-Scf_0.2-Icf_0.2-Vt_5-Tp_0.26-Tst_0.3"
    else:
        date = "2022-07-06-09-29-45-Traj-Tcf_0.0-Pcf_0.0-Fcf_0.6-Scf_0.2-Icf_0.2-Vt_5-Tp_0.26-Tst_0.3"
    
    solution_file = store_path + today + date + "/" + date + "-sol.npy"
    config_file = store_path + today + date + "/" + date + "-config.yaml"
    solution = np.load(solution_file)
    cfg = YAML().load(open(config_file, 'r'))

    robot = Bipedal_hybrid(cfg)    # create robot
    data = SolutionData(old_solution=solution)  # format solution data

    foot_lx = []
    foot_ly = []
    foot_rx = []
    foot_ry = []
    for i in range(data.N):
        df_lx, df_ly, df_rx, df_ry = robot.foot_vel(data.q[i, :], data.dq[i, :])
        # df_lx, df_ly, df_rx, df_ry = robot.foot_pos(data.q[i, :])
        foot_lx.append(df_lx)
        foot_ly.append(df_ly)
        foot_rx.append(df_rx)
        foot_ry.append(df_ry)
        pass

    fig, axes = plt.subplots(2,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]

    ax1.plot(data.t, foot_lx, label="left foot x", lw=3)
    ax1.plot(data.t, foot_rx, label="right foot x", lw=3)
    ax1.set_ylabel('x-vel(m/s)', fontsize = 18)
    ax1.set_xlabel('Time(s)', fontsize = 18)
    ax1.xaxis.set_tick_params(labelsize = 15)
    ax1.yaxis.set_tick_params(labelsize = 15)
    ax1.legend(loc='upper right',fontsize = 15)

    ax2.plot(data.t, foot_ly, label="left foot y", lw=3)
    ax2.plot(data.t, foot_ry, label="right foot y", lw=3)
    ax2.set_ylabel('z-vel(m/s)', fontsize = 18)
    ax2.set_xlabel('Time(s)', fontsize = 18)
    ax2.xaxis.set_tick_params(labelsize = 15)
    ax2.yaxis.set_tick_params(labelsize = 15)
    ax2.legend(loc='upper right',fontsize = 15)

    plt.show()
    pass


def ForceAnalysis():
    # 这部分的脚本是用于对每个时刻的受力状态可视化
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import transforms

    # ------------------------------------------------
    # load data and preprocess
    # armflag = False
    armflag = True
    store_path = "/home/stylite-y/Documents/Master/Manipulator/Simulation/model_raisim/DRMPC/SaTiZ/data/"
    today = "2022-07-06/"
    if armflag:
        date = "2022-07-06-09-30-18-Traj-Tcf_0.0-Pcf_0.0-Fcf_0.6-Scf_0.2-Icf_0.2-Vt_5-Tp_0.26-Tst_0.3"
    else:
        date = "2022-07-06-09-29-45-Traj-Tcf_0.0-Pcf_0.0-Fcf_0.6-Scf_0.2-Icf_0.2-Vt_5-Tp_0.26-Tst_0.3"
    
    solution_file = store_path + today + date + "/" + date + "-sol.npy"
    config_file = store_path + today + date + "/" + date + "-config.yaml"
    solution = np.load(solution_file)
    cfg = YAML().load(open(config_file, 'r'))

    robot = Bipedal_hybrid(cfg)    # create robot
    data = SolutionData(old_solution=solution)  # format solution data
    # print(data.sudden_id)

    # ------------------------------------------------
    # calculate force
    Inertia_main = []
    Inertia_coupling = []
    Inertia_coupling_x = []
    Inertia_coupling_y = []
    Corialis = []
    Gravity = []
    Contact = []
    Control = np.hstack((np.zeros([data.N, 2]), data.u))
    for i in range(data.N):
        temp1, temp2 = robot.inertia_force2(data.q[i, :], data.ddq[i, :])
        temp3, temp4 = robot.inertia_coupling_force(data.q[i, :], data.ddq[i, :])
        Inertia_coupling_x.append(temp3)
        Inertia_coupling_y.append(temp4)
        Inertia_main.append(temp1)
        Inertia_coupling.append(temp2)
        Corialis.append(robot.coriolis(
            data.q[i, 2:8], data.dq[i, 2:8]) if i != data.sudden_id else np.zeros(8))
        Gravity.append(robot.gravity(
            data.q[i, 2:8]) if i != data.sudden_id else np.zeros(8))
        Contact.append(robot.contact_force(data.q[i, 2:8], data.f[i, :]))
        pass

    Force = [np.asarray(temp) for temp in [Inertia_main, Corialis,
                                           Inertia_coupling, Gravity, Contact, Control]]
    Force[4] = -Force[4]
    Force[5] = -Force[5]
    print(np.array(Inertia_coupling_x).shape)
    Inertia_coupling_x = np.asarray(Inertia_coupling_x)
    Inertia_coupling_y = np.asarray(Inertia_coupling_y)
    # temp1 = Inertia_coupling_x[:,0]
    # print(len(temp1))
    linewd = 3
    fig, axes = plt.subplots(1,1, dpi=100,figsize=(12,10))
    # ax1 = axes[0]
    # ax2 = axes[1]
    # ax3 = axes[2]
    ax4 = axes
    
    labeltex = ["Inertia_main","Corialis", "Inertia_coupling", "Gravity", "Contact", "Control"]
    labeltex2 = ["xy","lg","rg", "la", "ra"]
    for i in range(6):
        if i < 5:
            temp1 = Inertia_coupling_x[:, i]
        #     ax1.plot(data.t, temp1, label=labeltex2[i], lw=linewd)
        #     ax2.plot(data.t, Inertia_coupling_y[:, i], label=labeltex2[i], lw=linewd)
        # ax3.plot(data.t, Force[i][:,0], label=labeltex[i], lw=linewd)
        ax4.plot(data.t, Force[i][:,0], label=labeltex[i], lw=linewd)
    # ax1.plot(data.t, Force[0][:,0], label = "Inertia_main", lw=linewd)
    # ax2.plot(data.t, Force[0][:,1], label = "Inertia_main", lw=linewd)
    
    # ax1.legend(loc='upper right')
    # ax2.legend(loc='upper right')
    # ax3.legend(loc='upper right')
    ax4.set_xlabel('Time(s)', fontsize = 18)
    ax4.set_ylabel('Torque(N/m)', fontsize = 18)
    ax4.xaxis.set_tick_params(labelsize = 15)
    ax4.yaxis.set_tick_params(labelsize = 15)
    ax4.legend(loc='upper right')

    plt.show()


def ForceVisualization():
    # 这部分的脚本是用于对每个时刻的受力状态可视化
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import transforms

    # ------------------------------------------------
    # load data and preprocess
    armflag = False
    saveflag = False
    store_path = "/home/stylite-y/Documents/Master/Manipulator/Simulation/model_raisim/DRMPC/SaTiZ/data/"
    today = "2022-07-13/"
    if armflag:
        date = "2022-07-13-11-21-51-Traj-Tcf_0.0-Pcf_0.0-Fcf_0.6-Scf_0.2-Icf_0.2-Vt_5-Tp_0.26-Tst_0.3"
    else:
        date = "2022-07-13-11-57-05-Traj-Tcf_0.0-Pcf_0.0-Fcf_0.6-Scf_0.2-Icf_0.2-Vt_5-Tp_0.26-Tst_0.3"

    pic_store_path = store_path + today + date + "/" 
    solution_file = store_path + today + date + "/" + date + "-sol.npy"
    config_file = store_path + today + date + "/" + date + "-config.yaml"
    solution = np.load(solution_file)
    cfg = YAML().load(open(config_file, 'r'))

    robot = Bipedal_hybrid(cfg)    # create robot
    data = SolutionData(old_solution=solution)  # format solution data

    # ------------------------------------------------
    # calculate force
    Inertia_main = []
    Inertia_coupling = []
    Corialis = []
    Gravity = []
    Contact = []
    Control = np.hstack((np.zeros([data.N, 2]), data.u))
    for i in range(data.N):
        temp1, temp2 = robot.inertia_force2(data.q[i, :], data.ddq[i, :])
        Inertia_main.append(temp1)
        Inertia_coupling.append(temp2)
        Corialis.append(robot.coriolis(
            data.q[i, 2:8], data.dq[i, 2:8]) if i != data.sudden_id else np.zeros(8))
        Gravity.append(robot.gravity(
            data.q[i, 2:8]) if i != data.sudden_id else np.zeros(8))
        Contact.append(robot.contact_force(data.q[i, 2:8], data.f[i, :]))
        pass
    Force = [np.asarray(temp) for temp in [Inertia_main,
                                           Inertia_coupling, Corialis, Gravity, Contact, Control]]
    Force[4] = -Force[4]
    Force[5] = -Force[5]
    print(np.array(Force).shape)

    # ------------------------------------------------
    plt.style.use("science")
    cmap = mpl.cm.get_cmap('Paired')
    params = {
        'text.usetex': True,
        'font.size': 8,
        'pgf.preamble': [r'\usepackage{color}'],
    }
    mpl.rcParams.update(params)

    # axis and labels
    labelfont = 8
    labelfonty = 10
    ## arm
    up = [1000, 2400, 800, 300, 800, 300, 300, 300]
    textlim = [1100, 2700, 900, 340, 900, 340, 340, 340]
    ## noarm
    up = [2000, 2400, 800, 300, 800, 300, 30, 30]
    textlim = [2200, 2700, 900, 340, 900, 340, 35, 35]
    labels = ['X', 'Y', 'Hip-L', 'Knee-L', 'Hip-R', 'Knee-R', 'Shoulder-L', 'Shoulder-R']


    # fig = plt.figure(figsize=(7, 3), dpi=300, constrained_layout=False)
    # ax = fig.subplots(2, 3)

    # ------------------------------------------------
    # self-define function
    def rainbow_text(fig, ax, x, y, ls, lc, **kw):

        # t = plt.gca().transData
        t = ax.transData

        # horizontal version
        for s, c in zip(ls, lc):
            text = fig.text(x, y, s, color=c, transform=t, **kw)
            text.draw(fig.canvas.get_renderer())
            ex = text.get_window_extent()
            t = transforms.offset_copy(
                text._transform, x=ex.width, units='dots')
        pass

    def pos_or_neg_flag(value_list):
        res = 0
        s = np.sign(np.asarray(value_list))

        pass

    # ------------------------------------------------
    # fig plot setting
    fig_zero_holder = plt.figure(
        figsize=(7, 4), dpi=300, constrained_layout=False)
    ax_zero_holder = fig_zero_holder.subplots(2, 4, sharex=True)

    ColorCandidate = ['C'+str(i) for i in range(6)]
    rainbow_text(fig_zero_holder, ax_zero_holder[0, 0], 0.01, textlim[0]+300, [r'$M_{ii}\ddot{q}_i$', r'$+$',
                                                                             r'$\sum M_{ij}\ddot{q}_j$', r'+',
                                                                             r'$C(q,\dot{q})$', r'+',
                                                                             r'$G({q})$', r'+',
                                                                             r'$-J^T\lambda$', r'+',
                                                                             r'$-u$', r'$=0$'],
                     ['C0', 'k', 'C1', 'k', 'C2', 'k', 'C3', 'k', 'C4', 'k', 'C5', 'k'], size=10)

    for i in range(2):
        for j in range(4):
            for k in range(data.N):
                dof_id = i + j * 2
                pos = 0
                neg = 0
                for kk in range(6):
                    # print(Force[kk][k][dof_id])
                    ax_zero_holder[i, j].bar(data.t[k], Force[kk][k][dof_id], width=data.dt[k],
                                             bottom=pos if Force[kk][k][dof_id] >= 0 else neg, align='edge', color=ColorCandidate[kk], linewidth=0, ecolor=ColorCandidate[kk])
                    if Force[kk][k][dof_id] >= 0:
                        pos += Force[kk][k][dof_id]
                    else:
                        neg += Force[kk][k][dof_id]
                    pass
                pass
            ax_zero_holder[i, j].set_ylim([-up[dof_id], up[dof_id]])
            ax_zero_holder[i, j].set_xlabel(labels[dof_id], fontsize = labelfont)
            pass
        pass
    [a.set_xlim([0, 0.2]) for a in ax_zero_holder.reshape(-1)]
    
    if saveflag:
        savename = pic_store_path + 'dynamics-eq'
        fig_zero_holder.savefig(savename)

    # -------------------------------------------------
    # plot the force of each dof separately
    fig = [plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False)
           for _ in range(len(textlim))]
    gs = [f.add_gridspec(2, 1, height_ratios=[1, 2.2],
                         wspace=0.3, hspace=0.33) for f in fig]
    ax_posture = [fig[i].add_subplot(gs[i][0]) for i in range(len(fig))]
    ax_force = [fig[i].add_subplot(gs[i][1]) for i in range(len(fig))]

    # plot posture
    # ------------------------------------------
    for tt in [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]:
        idx = np.argmin(np.abs(data.t-tt))
        pos = Bipedal_hybrid.get_posture(data.q[idx, :])
        for a in ax_posture:
            a.axhline(y=0, color='k')
            a.plot(pos[0], pos[1], color=cmap(tt/data.t[-1]))
            a.plot(pos[2], pos[3], color=cmap(tt/data.t[-1]), ls=':')
            a.scatter(pos[0][0], pos[1][0], marker='s',
                      color=cmap(tt/data.t[-1]), s=30)
            a.set_xlim([-1.5, 1.5])
            a.axis('equal')
            a.set_xticklabels([])
            pass
        pass

    # plot force
    # -----------------------------------------
    for i in range(len(textlim)):
        for k in range(data.N):
            dof_id = i
            pos = 0
            neg = 0
            for kk in range(6):
                ax_force[i].bar(data.t[k], Force[kk][k][dof_id], width=data.dt[k],
                                bottom=pos if Force[kk][k][dof_id] >= 0 else neg, align='edge', color=ColorCandidate[kk], linewidth=0, ecolor=ColorCandidate[kk])
                if Force[kk][k][dof_id] >= 0:
                    pos += Force[kk][k][dof_id]
                else:
                    neg += Force[kk][k][dof_id]
                pass
            pass
        ax_force[i].set_xlim([0, 0.2])
        ax_force[i].set_ylim([-up[i], up[i]])
        rainbow_text(fig[i], ax_force[i], 0.01, textlim[i], [r'$M_{ii}\ddot{q}_i$', r'$+$',
                                                                             r'$\sum M_{ij}\ddot{q}_j$', r'+',
                                                                             r'$C(q,\dot{q})$', r'+',
                                                                             r'$G({q})$', r'+',
                                                                             r'$-J^T\lambda$', r'+',
                                                                             r'$-u$', r'$=0$'],
                     ['C0', 'k', 'C1', 'k', 'C2', 'k', 'C3', 'k', 'C4', 'k', 'C5', 'k'], size=10)
        pass

    plt.show()

    pass


def power_analysis():
    # analysis the power and motor active work
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # ------------------------------------------------
    # load data and preprocess
    store_path = "/home/wooden/Desktop/spring-leg/data/TrajOpt/"
    # date = "11_03_2022_22_25_22"
    # date = "13_03_2022_16_08_24"   # FF
    # date = "13_03_2022_21_19_23"   # BB   #! 这个BB奔跑基本不耗能， 如果不考铜的损耗的话
    date = []
    date.append("25_03_2022_08_48_47")  # FF, 8m/s
    date.append("25_03_2022_08_56_29")  # BB, 8m/s
    date.append("25_03_2022_09_06_40")  # BF, 8m/s

    label = [r'$FF$', r'$BB$', r'$BF$']

    solution_file = [store_path + d + "_sol.npy" for d in date]
    config_file = [store_path + d + "config.yaml" for d in date]
    solution = [np.load(s) for s in solution_file]
    cfg = [YAML().load(open(c, 'r')) for c in config_file]

    robot = [Bipedal_hybrid(c) for c in cfg]    # create robot
    data = [SolutionData(old_solution=s)
            for s in solution]  # format solution data

    plt.style.use("science")
    # plt.style.use("nature")
    params = {
        'text.usetex': True,
        'font.size': 8,
        'pgf.preamble': [r'\usepackage{color}'],
    }

    mpl.rcParams.update(params)

    fig = plt.figure(figsize=(4, 3), dpi=300, constrained_layout=False)
    ax = [fig.subplots(1, 1, sharex=True)]

    tt = [np.tile(d.t.reshape(-1, 1), 4) for d in data]
    power = [np.asarray([d.u[:, i]*d.dq[:, 2+i]
                         for i in range(4)]).transpose() for d in data]
    # work = [np.trapz(power[i], data.t) for i in range(4)]
    # print(work[0], work[1], work[2], work[3])
    work = [np.asarray([np.trapz(power[j][0:i+1, :], x=tt[j][0:i+1, :], axis=0)
                       for i in range(data[j].N)]) for j in range(len(date))]

    [ax[0].plot(data[i].t, np.sum(work[i], axis=1), label=label[i])
     for i in range(len(date))]
    [a.set_xlim([0, 0.2]) for a in ax]
    [a.set_xlabel(r"$Time\ (s)$") for a in ax]
    [a.set_ylabel(r"$Work\ (J)$") for a in ax]
    ax[0].legend(loc=2)
    ax[0].grid(axis='y')

    # label = [r"$Hip_l$", r"$Knee_l$", r"$Hip_r$", r"$Knee_r$"]
    # ls = ['-', '-', '--', '--']
    # alpha = [1.0, 1.0, 0.6, 0.6]
    # [ax[0].plot(data.t, power[:, i], label=label[i], ls=ls[i], alpha=alpha[i])
    #  for i in range(2)]
    # [ax[1].plot(data.t, work[:, i], label=label[i], ls=ls[i], alpha=alpha[i])
    #  for i in range(2)]

    # [a.set_xlim([0, 0.2]) for a in ax]
    # [a.set_xlabel(r"$Time\ (s)$") for a in ax]
    # ax[0].set_ylabel(r"$Power\ (W)$")
    # ax[1].set_ylabel(r"$Work\ (J)$")
    # ax[0].set_ylim([-600, 600])
    # ax[1].set_ylim([-10, 40])
    # [a.legend(loc=1) for a in ax]

    plt.show()

    pass


def Impact_inertia():
    # compare the impact imertia
    # load thrid part package
    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import transforms
    import matplotlib.patches as patches
    from math import atan2

    # ------------------------------------------------
    # load data and preprocess
    store_path = "/home/wooden/Desktop/spring-leg/data/TrajOpt/"
    date = "25_03_2022_08_48_47"  # FF, 8m/s
    # date = "25_03_2022_08_56_29"  # BB, 8m/s
    # date = "25_03_2022_09_06_40"  # BF, 8m/s
    solution_file = store_path + date + "_sol.npy"
    config_file = store_path + date + "config.yaml"
    solution = np.load(solution_file)
    cfg = YAML().load(open(config_file, 'r'))

    robot = Bipedal_hybrid(cfg)    # create robot
    data = SolutionData(old_solution=solution)  # format solution data

    num_frame = 8

    # ---------------------------
    # calculate impact inertia matrix
    def get_ellipse_param(cov):
        a = cov[0, 0]
        b = cov[0, 1]
        c = cov[1, 1]
        lam1 = (a+c)/2+np.sqrt(((a-c)/2)**2+b**2)
        lam2 = (a+c)/2-np.sqrt(((a-c)/2)**2+b**2)
        if abs(b) < 1e-6:
            theta = 0 if a >= c else np.pi/2
            pass
        else:
            theta = atan2(lam1-a, b)
            pass
        return [lam1, lam2, theta]

    impact_matrix = []
    ellipse_param = []
    for tt in np.linspace(0, robot.T, num_frame):
        idx = np.argmin(np.abs(data.t-tt))
        MassMatrix = robot.mass_matrix(data.q[idx, 2:])
        J = robot.get_jacobian(data.q[idx, 2:])
        J = np.asarray(J)
        impact_matrix.append(np.linalg.inv(
            J.T.dot(np.linalg.inv(MassMatrix).dot(J)))[0:2, 0:2])
        ellipse_param.append(get_ellipse_param(impact_matrix[-1]))
        pass

    # ---------------------------
    # visualize the trajectory and impact inertia matrix
    class HandlerEllipse(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = patches.Ellipse(xy=center, width=width + xdescent,
                                height=height + ydescent)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 6,
    }
    mpl.rcParams.update(params)
    cmap = mpl.cm.get_cmap('Paired')
    fig = plt.figure(figsize=(5, 2.0), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 1, height_ratios=[1], wspace=0.3)
    ax_traj = fig.add_subplot(gs[0])
    ax_traj.axhline(y=0, color='k', lw=1, zorder=0, ls=':')

    i = 0
    for tt in np.linspace(0, robot.T, num_frame):
        idx = np.argmin(np.abs(data.t-tt))
        pos = Bipedal_hybrid.get_posture(
            data.q[idx, :])
        if i == 0:
            offset = pos[0][0]
            pass
        patch = patches.Rectangle(
            (pos[0][0]-offset-0.1, pos[1][0]-0.025), 0.2, 0.05, lw=0, color=cmap(tt/robot.T))

        ellipse = patches.Ellipse(
            (pos[0][-1]-offset, pos[1][-1]), ellipse_param[i][0]*0.5, ellipse_param[i][1]*0.5, angle=ellipse_param[i][2]/np.pi*180, color=cmap(tt/robot.T), alpha=0.8, lw=0)

        if i == 1:
            ell = ax_traj.add_artist(ellipse)
            body = ax_traj.add_patch(patch)
            right_leg, = ax_traj.plot(pos[0]-offset, pos[1], 'o-',
                                      color=cmap(tt/robot.T), markersize=2, label="Right Leg")
            left_leg, = ax_traj.plot(pos[2]-offset, pos[3], 'o:', color=cmap(
                tt/robot.T), alpha=0.6, markersize=2, label="Left Leg")
        else:
            ax_traj.add_artist(ellipse)
            ax_traj.add_patch(patch)
            ax_traj.plot(pos[0]-offset, pos[1], 'o-',
                         color=cmap(tt/robot.T), markersize=2)
            ax_traj.plot(pos[2]-offset, pos[3], 'o:', color=cmap(
                tt/robot.T), alpha=0.6, markersize=2)
        i += 1
        pass
    # ax_traj.axis('equal')
    ax_traj.legend([ell, body, right_leg, left_leg], ["Impact Inertia Matrix", "Body", "Right Leg", "Left Leg"],
                   handler_map={patches.Ellipse: HandlerEllipse()}, frameon=True, fancybox=True, loc=1)
    ax_traj.set_ylim([-0.2, 0.8])
    ax_traj.set_xlim([-0.5, 2.5])
    ax_traj.set_xlabel(r"$x\ (m)$")
    ax_traj.set_ylabel(r"$z\ (m)$")
    plt.show()

    pass


def Impact_process():
    # Delicately characterize the moment of contact
    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import transforms
    import matplotlib.patches as patches
    from matplotlib.patches import ConnectionPatch
    from math import atan2

    # ---------------------------
    # load data
    store_path = "/home/wooden/Desktop/spring-leg/data/TrajOpt/"
    # date = "25_03_2022_08_48_47"  # FF, 8m/s
    date = "25_03_2022_08_56_29"  # BB, 8m/s
    # date = "25_03_2022_09_06_40"  # BF, 8m/s
    solution_file = store_path + date + "_sol.npy"
    config_file = store_path + date + "config.yaml"
    solution = np.load(solution_file)
    cfg = YAML().load(open(config_file, 'r'))
    robot = Bipedal_hybrid(cfg)  # create robot
    data = SolutionData(old_solution=solution)

    # ---------------------------
    # calculate impact inertia matrix

    def get_ellipse_param(cov):
        a = cov[0, 0]
        b = cov[0, 1]
        c = cov[1, 1]
        lam1 = (a+c)/2+np.sqrt(((a-c)/2)**2+b**2)
        lam2 = (a+c)/2-np.sqrt(((a-c)/2)**2+b**2)
        if abs(b) < 1e-6:
            theta = 0 if a >= c else np.pi/2
            pass
        else:
            theta = atan2(lam1-a, b)
            pass
        return [lam1, lam2, theta]

    impact_matrix = None
    ellipse_param = None
    MassMatrix = robot.mass_matrix(data.q[0, 2:])
    J = robot.get_jacobian(data.q[0, 2:])
    J = np.asarray(J)
    impact_matrix = np.linalg.inv(
        J.T.dot(np.linalg.inv(MassMatrix).dot(J)))[0:2, 0:2]
    ellipse_param = get_ellipse_param(impact_matrix)

    pos = Bipedal_hybrid.get_posture(data.q[0, :])

    impluse = data.f[-1][0:2]*data.dt[-1]

    vel = []
    for i in range(len(data.q[:, 0])):
        vel.append(robot.foot_vel(data.q[i, :], data.dq[i, :]))
        pass

    # visualization data
    # visualize the trajectory and impact inertia matrix
    class HandlerEllipse(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = patches.Ellipse(xy=center, width=width + xdescent,
                                height=height + ydescent)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    # class HandlerCon(HandlerPatch):
    #     def create_artists(self, legend, orig_handle,
    #                        xdescent, ydescent, width, height, fontsize, trans):
    #         center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
    #         p = ConnectionPatch(center, (center[0], center[1]+width), "data", "data",
    #                             arrowstyle="->")
    #         self.update_prop(p, orig_handle, legend)
    #         p.set_transform(trans)
    #         return [p]

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 6,
    }
    mpl.rcParams.update(params)
    cmap = mpl.cm.get_cmap('Paired')
    fig = plt.figure(figsize=(5, 5), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 1, height_ratios=[1], wspace=0.3)
    ax_traj = fig.add_subplot(gs[0])
    ax_traj.axhline(y=0, color='k', lw=1, zorder=0, ls=':')

    patch = patches.Rectangle(
        (pos[0][0]-0.1, pos[1][0]-0.025), 0.2, 0.05, lw=0, color=cmap(0))

    ellipse = patches.Ellipse(
        (pos[0][-1], pos[1][-1]), ellipse_param[0], ellipse_param[1], angle=ellipse_param[2]/np.pi*180, color=cmap(0.4), alpha=0.8, lw=0)

    ell = ax_traj.add_artist(ellipse)
    body = ax_traj.add_patch(patch)
    right_leg, = ax_traj.plot(pos[0], pos[1], 'o-',
                              color=cmap(0.3), markersize=2, label="Right Leg")
    left_leg, = ax_traj.plot(pos[2], pos[3], 'o:', color=cmap(
        0.3), alpha=1.0, markersize=2, label="Left Leg")

    vel_st = (pos[0][-1], pos[1][-1])
    vel_ed = (pos[0][-1]+vel[-1][0]*0.02, pos[1][-1]+vel[-1][1]*0.02)
    vel_con = ConnectionPatch(vel_st, vel_ed, "data", "data",
                              arrowstyle="->", shrinkA=0, shrinkB=0,
                              mutation_scale=10, fc="w", ec=cmap(0.6), zorder=10)
    ax_traj.add_patch(vel_con)
    ax_traj.text(pos[0][-1]-0.05, 0.12,
                 r"$v^-=[{:.1f},{:.1f}]m/s$".format(vel[-1][0], vel[-1][1]),
                 color=cmap(0.6), zorder=100)

    force_st = (pos[0][-1], pos[1][-1])
    force_ed = (pos[0][-1]+impluse[0]*0.2, pos[1][-1]+impluse[1]*0.2)
    force_con = ConnectionPatch(force_st, force_ed, "data", "data",
                                arrowstyle="->", shrinkA=0, shrinkB=0,
                                mutation_scale=10, fc="w", ec=cmap(0.8), zorder=10)
    ax_traj.add_patch(force_con)
    ax_traj.text(pos[0][-1]-0.05, 0.16,
                 r"$\Lambda=[{:.1f},{:.1f}]N\cdot s$".format(
                     impluse[0], impluse[1]),
                 color=cmap(0.8), zorder=100)

    ax_traj.axis('equal')
    ax_traj.set_ylim([-0.2, 0.6])
    ax_traj.set_xlim([-1.4, -0.1])
    ax_traj.legend([ell, body, right_leg, left_leg], ["Impact Inertia Matrix", "Body", "Right Leg", "Left Leg"],
                   handler_map={patches.Ellipse: HandlerEllipse()}, frameon=True, fancybox=True, loc=1)

    mass_text = r"$\mathrm{M}_I=\begin{bmatrix} {m1} &{m2}\\ {m3} &{m4} \end{bmatrix}kg\cdot m/s^2$".format(
        bmatrix="{bmatrix}", m1="{:.2f}".format(impact_matrix[0, 0]), m2="{:.2f}".format(impact_matrix[0, 1]), m3="{:.2f}".format(impact_matrix[1, 0]), m4="{:.2f}".format(impact_matrix[1, 1]), M="{M}")

    ax_traj.text(pos[0][-1]-0.05, 0.2,
                 mass_text,
                 color=cmap(0.4), zorder=100)

    ax_traj.set_xlabel(r"$x\ (m)$")
    ax_traj.set_ylabel(r"$z\ (m)$")
    plt.show()

    pass


def Power_metrics_analysis():
    # analysis the power metrics of system
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import transforms
    import matplotlib.patches as patches

    # ---------------------------
    # load data
    store_path = "/home/wooden/Desktop/spring-leg/data/TrajOpt/"
    # date = "25_03_2022_08_48_47"  # FF, 8m/s
    date = "25_03_2022_08_56_29"  # BB, 8m/s
    # date = "25_03_2022_09_06_40"  # BF, 8m/s
    solution_file = store_path + date + "_sol.npy"
    config_file = store_path + date + "config.yaml"
    solution = np.load(solution_file)
    cfg = YAML().load(open(config_file, 'r'))
    robot = Bipedal_hybrid(cfg)  # create robot
    data = SolutionData(old_solution=solution)

    # --------------------------
    # power metrics analysis
    p = []
    p.append(data.dq[:, 2]*data.u[:, 0])
    p.append(data.dq[:, 3]*data.u[:, 1])
    p.append(data.dq[:, 4]*data.u[:, 2])
    p.append(data.dq[:, 5]*data.u[:, 3])

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 6,
    }
    mpl.rcParams.update(params)
    cmap = mpl.cm.get_cmap('Paired')
    fig = plt.figure(figsize=(3.5, 5), dpi=300, constrained_layout=True)

    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], wspace=0.3)
    ax_p = fig.add_subplot(gs[0])
    ax_q = fig.add_subplot(gs[1])

    label = [r"$P_{l,h}$", r"$P_{l,k}$",
             r"$P_{r,h}$", r"$P_{r,k}$"]
    color = ['C0', 'C1', 'C0', 'C1']
    ls = ['-', '-', '--', '--']
    alpha = [1, 1, 0.6, 0.6]
    [ax_p.plot(data.t, p[i], label=label[i], color=color[i],
               ls=ls[i], alpha=alpha[i]) for i in range(4)]
    ax_p.legend(loc=1, frameon=True, fancybox=True)

    p = np.asarray(p).T
    q = np.sum(p, axis=1)**2-np.sum(p**2, axis=1)
    ax_q.plot(data.t, q, zorder=3)
    ax_q.axhline(y=0, color='C3', lw=1, zorder=2)

    ax_p.set_xticklabels([])
    ax_q.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_p.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_q.set_xlabel(r"$Time\ (\mathsf{s})$")
    ax_p.set_ylabel(r"$Power\ (\mathsf{w})$")
    ax_q.set_ylabel(r"$Power\ Quality\ (\mathsf{w}^2)$")
    ax_q.set_xlim([0, 0.2])
    ax_p.set_xlim([0, 0.2])
    ax_q.set_ylim([-2e6, 4e6])
    ax_p.set_ylim([-1.5e3, 1.5e3])
    ax_p.grid(axis='y')
    ax_q.grid(axis='y')

    plt.show()

    pass


if __name__ == "__main__":
    # main(True)
    # main(False)
    # ForceMap()
    VelForceMap()
    # ForceVisualization()
    # ForceAnalysis()
    # VelAndAcc()
    # power_analysis()
    # Impact_inertia()
    # Impact_process()
    # Power_metrics_analysis()
    pass
