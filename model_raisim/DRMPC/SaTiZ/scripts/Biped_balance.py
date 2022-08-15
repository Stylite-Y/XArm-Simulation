'''
1. 双足机器人轨迹优化
2. 将接触的序列预先制定
3. 格式化配置参数输入和结果输出
4. 混合动力学系统，系统在机器人足底和地面接触的时候存在切换
5. 加入双臂的运动
6. Biped_walk_half: x0,z0 in hip and hO_hip = O_hip
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
    def __init__(self, cfg):
        self.opti = ca.Opti()
        # load config parameter
        self.CollectionNum = cfg['Controller']['CollectionNum']
        self.N = cfg['Controller']['CollectionNum']

        # time and collection defination related parameter
        self.T = cfg['Controller']['Period']
        self.dt = self.T / self.CollectionNum

        # mass and geometry related parameter
        self.m = cfg['Robot']['Mass']['mass']
        self.I = cfg['Robot']['Mass']['inertia']
        self.l = cfg['Robot']['Mass']['massCenter']
        self.I_ = [self.m[i]*self.l[i]**2+self.I[i] for i in range(5)]

        self.L = [cfg['Robot']['Geometry']['L_body'],
                  cfg['Robot']['Geometry']['L_thigh'],
                  cfg['Robot']['Geometry']['L_shank'],
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

        # self.vel_aim = cfg['Controller']['Target']

        # boundary parameter
        self.bound_fy = cfg['Controller']['Boundary']['Fy']
        self.bound_fx = cfg['Controller']['Boundary']['Fx']
        self.F_LB = [self.bound_fx[0], self.bound_fy[0]]
        self.F_UB = [self.bound_fx[1], self.bound_fy[1]]

        self.u_LB = [-self.motor_mt] * 5
        self.u_UB = [self.motor_mt] * 5

        # FF = cfg["Controller"]["Forward"]
        FF = cfg["Controller"]["FrontForward"]
        HF = cfg["Controller"]["HindForward"]

        self.q_LB = [cfg['Controller']['Boundary']['x'][0],
                     cfg['Controller']['Boundary']['y'][0],
                     -np.pi,  # body x,y,theta
                     -np.pi/2, -np.pi,
                     -np.pi, 0]    # arm 
        self.q_UB = [cfg['Controller']['Boundary']['x'][1],
                     cfg['Controller']['Boundary']['y'][1],
                     np.pi/4,   # body x,y,theta
                     np.pi/2, 0,
                     np.pi/2, np.pi]  # arm 

        self.dq_LB = [cfg['Controller']['Boundary']['dx'][0],
                      cfg['Controller']['Boundary']['dy'][0],
                      -self.motor_ms,
                      -self.motor_ms, -self.motor_ms,
                      -self.motor_ms, -self.motor_ms]   # arm 

        self.dq_UB = [cfg['Controller']['Boundary']['dx'][1],
                      cfg['Controller']['Boundary']['dy'][1],
                      self.motor_ms,
                      self.motor_ms, self.motor_ms,
                      self.motor_ms, self.motor_ms] # arm 

        # * define variable
        self.q = [self.opti.variable(7) for _ in range(self.N)]
        self.dq = [self.opti.variable(7) for _ in range(self.N)]
        self.ddq = [(self.dq[i+1]-self.dq[i]) /
                        self.dt for i in range(self.N-1)]

        # ! set the last u to be zero at constraint
        self.u = [self.opti.variable(5) for _ in range(self.N)]

        # ! Note, the last force represents the plused force at the contact moment
        self.F = [self.opti.variable(2) for _ in range(self.N)]

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
        L3 = self.L[3]

        m00 = m0+m1+m2+m3+m4
        m01 = 0
        m02 = -L0*(m3+m4)*c(q[0])-lc0*m0*c(q[0])+L3*m4*c(q[0]+q[3])+\
              lc3*m3*c(q[0]+q[3])+lc4*m4*c(q[0]+q[3]+q[4])
        m03 = L1*m2*c(q[1])+lc1*m1*c(q[1])+lc2*m2*c(q[1]+q[2])
        m04 = lc2*m2*c(q[1]+q[2])
        m05 = L3*m4*c(q[0]+q[3])+lc3*m3*c(q[0]+q[3])+lc4*m4*c(q[0]+q[3]+q[4])
        m06 = lc4*m4*c(q[0]+q[3]+q[4])

        m10 = m01
        m11 = m0+m1+m2+m3+m4
        m12 = -L0*(m3+m4)*s(q[0])-lc0*m0*s(q[0])+L3*m4*s(q[0]+q[3])+\
              lc3*m3*s(q[0]+q[3])+lc4*m4*s(q[0]+q[3]+q[4])
        m13 = L1*m2*s(q[1])+lc1*m1*s(q[1])+lc2*m2*s(q[1]+q[2])
        m14 = lc2*m2*s(q[1]+q[2])
        m15 = L3*m4*s(q[0]+q[3])+lc3*m3*s(q[0]+q[3])+lc4*m4*s(q[0]+q[3]+q[4])
        m16 = lc4*m4*s(q[0]+q[3]+q[4])

        m20 = m02
        m21 = m12
        m22 = self.I_[0]+self.I_[3]+self.I_[4]+L0**2*(m3+m4)+L3**2*m4-\
              2*L0*L3*m4*c(q[3])-2*L0*lc3*m3*c(q[3])-2*L0*lc4*m4*c(q[3]+q[4])+\
              2*L3*lc4*m4*(q[4])
        m23 = 0
        m24 = 0
        m25 = self.I_[3]+self.I_[4]+L3**2*m4+2*L3*lc4*m4*(q[4])-\
              L0*L3*m4*c(q[3])-L0*lc3*m3*c(q[3])-L0*lc4*m4*c(q[3]+q[4])
        m26 = self.I_[4]+L3*lc4*m4*c(q[4])-L0*lc4*m4*c(q[3]+q[4])

        m30 = m03
        m31 = m13
        m32 = m23
        m33 = self.I_[1] + self.I_[2] + m2*L1**2+2*L1*lc2*m2*c(q[2])
        m34 = self.I_[2]+L1*lc2*m2*c(q[2])
        m35 = 0
        m36 = 0

        m40 = m04
        m41 = m14
        m42 = m24
        m43 = m34
        m44 = self.I_[2]
        m45 = 0
        m46 = 0

        m50 = m05
        m51 = m15
        m52 = m25
        m53 = m35
        m54 = m45
        m55 = self.I_[3]+self.I_[4]+L3**2*m4+2*L3*lc4*m4*c(q[4])
        m56 = self.I_[4]+L3*lc4*m4*c(q[4])

        m60 = m06
        m61 = m16
        m62 = m26
        m63 = m36
        m64 = m46
        m65 = m56
        m66 = self.I_[4]


        return [[m00, m01, m02, m03, m04, m05, m06],
                [m10, m11, m12, m13, m14, m15, m16],
                [m20, m21, m22, m23, m24, m25, m26],
                [m30, m31, m32, m33, m34, m35, m36],
                [m40, m41, m42, m43, m44, m45, m46],
                [m50, m51, m52, m53, m54, m55, m56],
                [m60, m61, m62, m63, m64, m65, m66]]
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
        L3 = self.L[3]

        mbs = L0*(m3+m4)*s(q[0])-L3*m4*s(q[0]+q[3])+lc0*m0*s(q[0])-\
              lc3*m3*s(q[0]+q[3])-lc4*m4*s(q[0]+q[3]+q[4])
        m11 = L1*m2*s(q[1])+lc1*m1*s(q[1])+lc2*m2*s(q[1]+q[2])
        m12 = lc2*m2*s(q[1]+q[2])
        m31 = L3*m4*s(q[0]+q[3])+lc3*m3*s(q[0]+q[3])+lc4*m4*s(q[0]+q[3]+q[4])
        m32 = lc4*m4*s(q[0]+q[3]+q[4])

        m41 = L0*(L3*m4*s(q[3])+lc3*m3*s(q[3])+lc4*m4*s(q[3]+q[4]))
        m42 = lc4*m4*(L0*s(q[3]+q[4])-L3*s(q[4]))
        m43 = L3*lc4*m4*s(q[4])
        m44 = L1*lc2*m2*s(q[2])

        mbc = L0*(m3+m4)*c(q[0])-L3*m4*c(q[0]+q[3])+lc0*m0*c(q[0])-\
              lc3*m3*c(q[0]+q[3])-lc4*m4*c(q[0]+q[3]+q[4])
        m11c = L1*m2*c(q[1])+lc1*m1*c(q[1])+lc2*m2*c(q[1]+q[2])
        m12c = lc2*m2*c(q[1]+q[2])
        m31c = L3*m4*c(q[0]+q[3])+lc3*m3*c(q[0]+q[3])+lc4*m4*c(q[0]+q[3]+q[4])
        m32c = lc4*m4*c(q[0]+q[3]+q[4])

        c0 = mbs*dq[0]**2-2*m31*dq[0]*dq[3]-2*m32*dq[0]*dq[4]
        c0 += -m11*dq[1]*dq[1]-2*m12*dq[1]*dq[2]-m12*dq[2]*dq[2]
        c0 += -m31*dq[3]*dq[3]-2*m32*dq[3]*dq[4]-m32*dq[4]*dq[4]

        c1 = -mbc*dq[0]*dq[0]+2*m31c*dq[0]*dq[3]+2*m32c*dq[0]*dq[4]
        c1 += m11c*dq[1]*dq[1]+2*m12c*dq[1]*dq[2]+m12c*dq[2]*dq[2]
        c1 += m31c*dq[3]**2 + 2*m32c*dq[3]*dq[4]+m32c*dq[4]*dq[4]

        c2 = 2*m41*dq[0]*dq[3]+2*m42*dq[0]*dq[4]
        c2+= m41*dq[3]*dq[3]+2*m42*dq[3]*dq[4]+m42*dq[4]*dq[4]

        c3 = -2*m44*dq[1]*dq[2]-m44*dq[2]*dq[2]

        c4 = m44*dq[1]*dq[1]

        c5 = -m41*dq[0]*dq[0]-2*m43*dq[0]*dq[4]-2*m43*dq[3]*dq[4]-m43*dq[4]*dq[4]

        c6 = -m42*dq[0]*dq[0]+2*m43*dq[0]*dq[3]+m43*dq[3]*dq[3]
        return [c0, c1, c2, c3, c4, c5, c6]
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
        L3 = self.L[3]

        g0 = 0
        g1 = m0+m1+m2+m3+m4
        g2 = -L0*(m3+m4)*s(q[0])-lc0*m0*s(q[0])+L3*m4*s(q[0]+q[3])+\
              lc3*m3*s(q[0]+q[3])+lc4*m4*s(q[0]+q[3]+q[4])
        g3 = L1*m2*s(q[1])+lc1*m1*s(q[1])+lc2*m2*s(q[1]+q[2])
        g4 = lc2*m2*s(q[1]+q[2])
        g5 = L3*m4*s(q[0]+q[3])+lc3*m3*s(q[0]+q[3])+lc4*m4*s(q[0]+q[3]+q[4])
        g6 = lc4*m4*s(q[0]+q[3]+q[4])
        return [g0*self.g, g1*self.g, g2*self.g, g3*self.g, g4*self.g, g5*self.g, g6*self.g]
        # endregion

    def contact_force(self, q, F):
        # region calculate the contact force
        # F = [Fxl, Fyl]
        cont0 = F[0]
        cont1 = F[1]
        cont2 = 0
        cont3 = (self.L[1]*c(q[1])+self.L[2]*c(q[1]+q[2]))*F[0] + \
                (self.L[1]*s(q[1])+self.L[2]*s(q[1]+q[2]))*F[1]
        cont4 = (self.L[2]*c(q[1]+q[2]))*F[0] + (self.L[2]*s(q[1]+q[2]))*F[1]
        cont5 = 0
        cont6 = 0
        return [cont0, cont1, cont2, cont3, cont4, cont5, cont6]
        # endregion

    def inertia_force(self, q, acc):
        # region calculate inertia force
        mm = self.mass_matrix(q[2:])
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2] +
                         mm[i][3]*acc[3]+mm[i][4]*acc[4]+mm[i][5]*acc[5] + 
                         mm[i][6]*acc[6] for i in range(7)]
        return inertia_force
        # endregion

    def inertia_force2(self, q, acc):
        # region calculate inertia force, split into two parts
        mm = self.mass_matrix(q[2:])
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2] +
                         mm[i][3]*acc[3]+mm[i][4]*acc[4]+mm[i][5]*acc[5] + 
                         mm[i][6]*acc[6] for i in range(7)]
        inertia_main = [mm[i][i]*acc[i] for i in range(7)]
        inertia_coupling = [inertia_force[i]-inertia_main[i] for i in range(7)]
        return inertia_main, inertia_coupling

    def foot_pos(self, q):
        # region calculate foot coordinate in world frame
        # q = [x, y, q0, q1, q2, q3, q4]
        foot_lx = q[0]  + self.L[1]*s(q[3]) + self.L[2]*s(q[3]+q[4])
        foot_ly = q[1]  - self.L[1]*c(q[3]) - self.L[2]*c(q[3]+q[4])
        return foot_lx, foot_ly
        # endregion

    def foot_vel(self, q, dq):
        # region calculate foot coordinate in world frame
        # q = [x, y, q0, q1, q2, q3, q4]
        # dq = [dx, dy, dq0, dq1, dq2, dq3, dq4]
        df_lx = dq[0] + self.L[1]*c(q[3])*(dq[3]) + \
            self.L[2]*c(q[3]+q[4])*(dq[3]+dq[4])
        df_ly = dq[1] + self.L[1]*s(q[3])*(dq[3]) + \
            self.L[2]*s(q[3]+q[4])*(dq[3]+dq[4])
        return df_lx, df_ly
        # endregion

    def get_jacobian(self, q):
        J = [[0 for i in range(2)] for j in range(8)]
        J[0][0] = 1
        J[1][1] = 1
        J[2][0] = (self.L[1]*c(q[0]+q[1])+self.L[2]*c(q[0]+q[1]+q[2]))
        J[2][1] = (self.L[1]*s(q[0]+q[1])+self.L[2]*s(q[0]+q[1]+q[2]))
        J[3][0] = (self.L[2]*c(q[0]+q[1]+q[2]))
        J[3][1] = (self.L[2]*s(q[0]+q[1]+q[2]))
        return J

    @staticmethod
    def get_posture(q):
        L = [0.5, 0.42, 0.5, 0.3, 0.37]
        lx = np.zeros(3)
        ly = np.zeros(3)
        lbx = np.zeros(2)
        lby = np.zeros(2)
        lax = np.zeros(3)
        lay = np.zeros(3)
        lx[0] = q[0]
        lx[1] = lx[0] + L[1]*np.sin(q[3])
        lx[2] = lx[1] + L[2]*np.sin(q[3]+q[4])
        ly[0] = q[1]
        ly[1] = ly[0] - L[1]*np.cos(q[3])
        ly[2] = ly[1] - L[2]*np.cos(q[3]+q[4])

        lbx[0] = q[0]
        lbx[1] = lbx[0] - L[0]*np.sin(q[2])
        lby[0] = q[1]
        lby[1] = lby[0] + L[0]*np.cos(q[2])

        lax[0] = q[0]- L[0]*np.sin(q[2])
        lax[1] = lax[0] + L[3]*np.sin(q[2]+q[5])
        lax[2] = lax[1] + L[4]*np.sin(q[2]+q[5]+q[6])
        lay[0] = q[1]+ L[0]*np.cos(q[2])
        lay[1] = lay[0] - L[3]*np.cos(q[2]+q[5])
        lay[2] = lay[1] - L[4]*np.cos(q[2]+q[5]+q[6])
        return (lx, ly, lbx, lby, lax, lay)

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
    def __init__(self, legged_robot, cfg, seed=None, is_ref=False):
        # load parameter
        self.cfg = cfg
        self.trackingCoeff = cfg["Optimization"]["CostCoeff"]["trackingCoeff"]
        self.velCoeff = cfg["Optimization"]["CostCoeff"]["VelCoeff"]
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
                          lam=cfg['Controller']['Stance'],
                          delta=cfg['Ref']['delta'],
                          target=cfg['Controller']['Target'],
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
            for i in range(walker.N):
                init(q[i][0], 0)
                init(q[i][1], walker.L[1]+walker.L[2])
                init(dq[i][0], 0)
                init(dq[i][1], 0)
                for j in range(5):
                    init(q[i][j+2], 0)
                    init(dq[i][j+2], 0)
                pass
            pass
        else:
            old_solution = np.load(seed)
            assert old_solution.shape[0] == (
                walker.CollectionNum+2), "The collection number must be the same, please check the config file"
            assert abs(old_solution[1, -1]-old_solution[0, -1] -
                       walker.dt) <= 1e-6, "Time step and Period must be the same"
            for j in range(walker.N):
                for k in range(8):
                    # set position
                    walker.opti.set_initial(
                        walker.q[j][k], old_solution[j, k])
                    # set velocity
                    walker.opti.set_initial(
                        walker.dq[j][k], old_solution[+j, 6+k])
                    if(k < 2):
                        # set external force
                        walker.opti.set_initial(
                            walker.F[j][k], old_solution[+j, 18+4+k])
                    if(k < 5):
                        if j < walker.N-1:
                            # set actuator
                            walker.opti.set_initial(
                                walker.u[j][k], old_solution[j, 18+k])
                            pass
                        pass
                    pass
                pass
            pass
        pass

    def Cost(self, walker):
        # region aim function of optimal control
        FM = [walker.bound_fx[1], walker.bound_fy[1]]
        power = 0
        force = 0
        Veltar = 0
        PosTar = 0
        for i in range(walker.N):
            for k in range(5):
                power += (walker.dq[i][k+2]
                            * walker.u[i][k])**2 * walker.dt
                force += (walker.u[i][k]/walker.motor_mt)**2
                
                pass
            for k in range(2):
                force += (walker.F[i][k]/FM[k])**2
                pass
            pass

        for i in range(walker.N):
            PosTar += (walker.q[i][1]-(walker.L[1]+walker.L[2]))**2 * walker.dt
            PosTar += (walker.q[i][0])**2 * walker.dt
            VelTar += (walker.dq[i][1])**2 * walker.dt
            VelTar += (walker.dq[i][0])**2 * walker.dt

            for k in range(5):
                PosTar += (walker.q[i][k+2])**2 * walker.dt
                VelTar += (walker.dq[i][k+2])**2 * walker.dt
        
        for j in range(7):
            if j == 1:
                PosTar += (walker.q[-1][j]-(walker.L[1]+walker.L[2]))**2 * walker.dt
                VelTar += (walker.dq[-1][j])**2 * walker.dt
            else:
                PosTar += (walker.q[-1][j])**2 * walker.dt
                VelTar += (walker.dq[-1][j])**2 * walker.dt


        u = walker.u
        F = walker.F
        smooth = 0
        AM = [100, 100, 400, 100, 400]
        for i in range(walker.N-1):
            for k in range(2):
                smooth += ((F[i+1][k]-F[i][k])/AM[k])**2
                pass
            pass 
            for k in range(5):
                smooth += ((u[i+1][k]-u[i][k])/20)**2
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
        for j in range(walker.N-1):
            ceq.extend([walker.q[j+1][k]-walker.q[j][k]-walker.dt/2 *
                        (walker.dq[j+1][k]+walker.dq[j][k]) == 0 for k in range(7)])
            inertia = walker.inertia_force(
                walker.q[j], walker.ddq[j])
            coriolis = walker.coriolis(
                walker.q[j][2:], walker.dq[j][2:])
            gravity = walker.gravity(walker.q[j][2:])
            contact = walker.contact_force(
                walker.q[j][2:], walker.F[j])
            ceq.extend([inertia[k]+gravity[k]+coriolis[k] -
                        contact[k] == 0 for k in range(2)])
            ceq.extend([inertia[k+2]+gravity[k+2]+coriolis[k+2] -
                        contact[k+2] - walker.u[j][k] == 0 for k in range(5)])
            pass


        # region periodicity constraints
        # ! this version no periodicity constraints
        # ! this constraint is implicit in dynamics constraints
        # endregion

        # region leg locomotion constraint
        for i in range(walker.N):
            pos_l_x, pos_l_y = walker.foot_pos(walker.q[i])
            vel_l_x, vel_l_y = walker.foot_vel(walker.q[i],walker.dq[i])
            ceq.extend([pos_l_x==0])
            ceq.extend([pos_l_y==0])
            ceq.extend([vel_l_x==0])
            ceq.extend([vel_l_y==0])
        # endregion

        # region boundary constraint
        for temp_q in walker.q:
            ceq.extend([walker.opti.bounded(walker.q_LB[j],
                        temp_q[j], walker.q_UB[j]) for j in range(7)])
            pass
        for temp_dq in walker.dq:
            ceq.extend([walker.opti.bounded(walker.dq_LB[j],
                        temp_dq[j], walker.dq_UB[j]) for j in range(7)])
            pass
        for temp_u in walker.u:
            ceq.extend([walker.opti.bounded(walker.u_LB[j],
                        temp_u[j], walker.u_UB[j]) for j in range(5)])
            pass
        for temp_f in walker.F:
            ceq.extend([walker.opti.bounded(walker.F_LB[j],
                        temp_f[j], walker.F_UB[j]) for j in range(2)])
            pass
        # endregion

        # region motor external characteristic curve
        cs = walker.motor_cs
        ms = walker.motor_ms
        mt = walker.motor_mt
        for j in range(len(walker.u)):
            ceq.extend([walker.u[j][k]-ca.fmax(mt - (walker.dq[j][k+2] -
                                                        cs)/(ms-cs)*mt, 0) <= 0 for k in range(5)])
            ceq.extend([walker.u[j][k]-ca.fmin(-mt + (walker.dq[j][k+2] +
                                                            cs)/(-ms+cs)*mt, 0) >= 0 for k in range(5)])
            pass
        # endregion

        theta = -np.pi/40
        x_ref = -(walker.L[1]+walker.L[2])*np.sin(theta)
        y_ref = (walker.L[1]+walker.L[2])*np.cos(theta)
        ceq.extend([walker.q[0][0]==x_ref])
        ceq.extend([walker.q[0][1]==y_ref])
        ceq.extend([walker.q[0][2]==theta])
        ceq.extend([walker.q[0][3]==theta])
        ceq.extend([walker.q[0][4]==0])
        ceq.extend([walker.q[0][5]==0])
        ceq.extend([walker.q[0][6]==0])

        ceq.extend([walker.dq[0][0]==0])
        ceq.extend([walker.dq[0][1]==0])
        ceq.extend([walker.dq[0][2]==0])
        ceq.extend([walker.dq[0][3]==0])
        ceq.extend([walker.dq[0][4]==0])
        ceq.extend([walker.dq[0][5]==0])
        ceq.extend([walker.dq[0][6]==0])

        # region smooth constraint
        for j in range(len(walker.u)-1):
            ceq.extend([ca.fabs(walker.F[j][k]-walker.F
                        [j+1][k]) <= 100 for k in range(2)])
            ceq.extend([ca.fabs(walker.u[j][k]-walker.u
                        [j+1][k]) <= 10 for k in range(5)])
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
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([sol1.value(robot.q[j][k]) for k in range(7)])
                dq.append([sol1.value(robot.dq[j][k])
                            for k in range(7)])
                ddq.append([sol1.value(robot.ddq[j][k])
                            for k in range(7)] if j < (robot.N-1) else [0]*7)
                u.append([sol1.value(robot.u[j][k])
                            for k in range(5)] if j < (robot.N-1) else [0]*5)
                f.append([sol1.value(robot.F[j][k]) for k in range(2)])
                pass
            pass
        except:
            value = robot.opti.debug.value
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([value(robot.q[j][k])
                            for k in range(7)])
                dq.append([value(robot.dq[j][k])
                            for k in range(7)])
                ddq.append([value(robot.ddq[j][k])
                            for k in range(7)])
                u.append([value(robot.u[j][k])
                            for k in range(5)])
                f.append([value(robot.F[j][k])
                            for k in range(2)])
                pass
            pass
        finally:
            q = np.asarray(q)
            dq = np.asarray(dq)
            ddq = np.asarray(ddq)
            u = np.asarray(u)
            f = np.asarray(f)
            t = np.asarray(t).reshape([-1, 1])

            if(flag_save):
                import time
                date = time.strftime("%d_%m_%Y_%H_%M_%S")
                # output solution of NLP
                np.save(StorePath+date+"_sol.npy",
                        np.hstack((q, dq, ddq, u, f, t)))
                # output the config yaml file
                with open(StorePath+date+"config.yaml", mode='w') as file:
                    YAML().dump(self.cfg, file)
                pass

            return q, dq, ddq, u, f, t


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
            self.q = old_solution[:, 0:6]
            self.dq = old_solution[:, 6:12]
            self.ddq = old_solution[:, 12:18]
            self.u = old_solution[:, 18:22]
            self.f = old_solution[:, 22:26]
            self.t = old_solution[:, 26]

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
    save_flag = False
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
    ParamFilePath = FilePath + "/config/Biped_balance.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # endregion

    # region create robot and NLP problem
    robot = Bipedal_hybrid(cfg)
    # nonlinearOptimization = nlp(robot, cfg, seed=seed)
    # nonlinearOptimization = nlp(robot, cfg)
    nonlinearOptimization = nlp(robot, cfg, is_ref=True)
    # endregion
    q, dq, ddq, u, F, t = nonlinearOptimization.solve_and_output(
        robot, flag_save=save_flag, StorePath=StorePath)
    # endregion

    if vis_flag:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib as mpl
        from matplotlib.patches import ConnectionPatch
        # plt.style.use("science")
        # params = {
        #     'text.usetex': True,
        #     'font.size': 8,
        # }
        # mpl.rcParams.update(params)
        cmap = mpl.cm.get_cmap('viridis')
        fig = plt.figure(figsize=(8, 5), dpi=180, constrained_layout=False)
        gs = fig.add_gridspec(2, 1, height_ratios=[2,1],
                              wspace=0.3, hspace=0.33)
        g_data = gs[1].subgridspec(2, 6, wspace=0.3, hspace=0.33)

        ax_m = fig.add_subplot(gs[0])
        ax_v = [fig.add_subplot(g_data[0, i]) for i in range(6)]
        ax_u = [fig.add_subplot(g_data[1, i]) for i in range(6)]

        vel = [robot.foot_vel(q[i, :], dq[i, :]) for i in range(len(q[:, 0]))]

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
                (pos[0][0]-0.02, pos[1][0]-0.05), 0.04, 0.1, alpha=tt/robot.T*0.8+0.2, lw=0, color=cmap(tt/robot.T))
            ax_m.add_patch(patch)

            # plot velocity
            vel_st = (pos[0][-1], pos[1][-1])
            vel_ed = (pos[0][-1]+vel[idx][0]*0.02, pos[1][-1]+vel[idx][1]*0.02)
            vel_con = ConnectionPatch(vel_st, vel_ed, "data", "data",
                                      arrowstyle="->", shrinkA=0, shrinkB=0,
                                      mutation_scale=5, fc="w", ec='C5', zorder=10)
            ax_m.add_patch(vel_con)
            ax_m.axis('equal')
            pass
        # ax_m.axis('equal')
        # title_v = [r'$\dot{x}$', r'$\dot{y}$',
        #            r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']
        # title_u = [r'$F_x$', r'$F_y$', r'$\tau_1$', r'$\tau_2$']

        v_idx1 = [0, 0, 1, 2, 3, 4, 5]
        v_idx2 = [0, 1, 2, 3, 4, 5, 6]
        [ax_v[v_idx1[i]].plot(t, q[:, v_idx2[i]]) for i in range(7)]
        # [ax_v[i].set_title(title_v[i]) for i in range(4)]

        ax_u[0].plot(t[1:], F[:, 0])
        ax_u[0].plot(t[1:], F[:, 1], color='C' + str(3))
        ax_u[1].plot(t[1:], u[:, 0])
        ax_u[2].plot(t[1:], u[:, 1])
        ax_u[3].plot(t[1:], u[:, 2])
        ax_u[4].plot(t[1:], u[:, 3])
        ax_u[5].plot(t[1:], u[:, 4])
        # [ax_u[j].set_title(title_u[j]) for j in range(4)]
        plt.show()

        pass

    pass


def ForceVisualization():
    # 这部分的脚本是用于对每个时刻的受力状态可视化
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import transforms

    # ------------------------------------------------
    # load data and preprocess
    store_path = "/home/wooden/Desktop/spring-leg/data/TrajOpt/"
    # date = "11_03_2022_22_25_22"
    # date = "13_03_2022_16_08_24"
    # date = "24_03_2022_19_48_57"
    # date = "25_03_2022_08_42_47"
    # date = "25_03_2022_08_48_47"  # FF, 8m/s
    # date = "25_03_2022_08_56_29"  # BB, 8m/s
    # date = "25_03_2022_09_06_40"  # BF, 8m/s

    # date = "04_04_2022_15_54_42"  # FF, 8m/s, right dynamics
    # date = "17_05_2022_15_15_31"  # FF, 8m/s, right dynamics
    # date = "17_05_2022_15_21_43"  # FF, 8m/s, right dynamics   low mass of leg
    # FF, 8m/s, right dynamics     low mass of leg and low inertia
    date = "17_05_2022_15_29_05"

    solution_file = store_path + date + "_sol.npy"
    config_file = store_path + date + "config.yaml"
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
            data.q[i, 2:6], data.dq[i, 2:6]) if i != data.sudden_id else np.zeros(6))
        Gravity.append(robot.gravity(
            data.q[i, 2:6]) if i != data.sudden_id else np.zeros(6))
        Contact.append(robot.contact_force(data.q[i, 2:6], data.f[i, :]))
        pass
    Force = [np.asarray(temp) for temp in [Inertia_main,
                                           Inertia_coupling, Corialis, Gravity, Contact, Control]]
    Force[4] = -Force[4]
    Force[5] = -Force[5]
    # Force = [Force[4], Force[0], Force[1], Force[2], Force[3]]
    # Force = [Force[5], Force[4], Force[0], Force[1], Force[2], Force[3]]
    # ------------------------------------------------
    plt.style.use("science")
    cmap = mpl.cm.get_cmap('Paired')
    params = {
        'text.usetex': True,
        'font.size': 8,
        'pgf.preamble': [r'\usepackage{color}'],
    }
    mpl.rcParams.update(params)

    # fig = plt.figure(figsize=(7, 3), dpi=300, constrained_layout=False)
    # ax = fig.subplots(2, 3)

    fig_zero_holder = plt.figure(
        figsize=(7, 3), dpi=300, constrained_layout=False)
    ax_zero_holder = fig_zero_holder.subplots(2, 3, sharex=True)

    def pos_or_neg_flag(value_list):
        res = 0
        s = np.sign(np.asarray(value_list))

        pass

    # ColorCandidate = [cmap(i/6) for i in range(6)]
    ColorCandidate = ['C'+str(i) for i in range(6)]

    for i in range(2):
        for j in range(3):
            for k in range(data.N):
                dof_id = i + j * 2
                pos = 0
                neg = 0
                for kk in range(6):
                    ax_zero_holder[i, j].bar(data.t[k], Force[kk][k, dof_id], width=data.dt[k],
                                             bottom=pos if Force[kk][k, dof_id] >= 0 else neg, align='edge', color=ColorCandidate[kk], linewidth=0, ecolor=ColorCandidate[kk])
                    if Force[kk][k, dof_id] >= 0:
                        pos += Force[kk][k, dof_id]
                    else:
                        neg += Force[kk][k, dof_id]
                    pass
                pass
            pass
        pass
    [a.set_xlim([0, 0.2]) for a in ax_zero_holder.reshape(-1)]
    ax_zero_holder[0, 0].set_ylim([-600, 600])
    ax_zero_holder[1, 0].set_ylim([-600, 600])
    ax_zero_holder[0, 1].set_ylim([-100, 100])
    ax_zero_holder[1, 1].set_ylim([-100, 100])
    ax_zero_holder[0, 2].set_ylim([-100, 100])
    ax_zero_holder[1, 2].set_ylim([-100, 100])

    # -------------------------------------------------
    # plot the force of each dof separately
    fig = [plt.figure(figsize=(4, 7), dpi=300, constrained_layout=False)
           for _ in range(6)]
    gs = [f.add_gridspec(2, 1, height_ratios=[1, 2.2],
                         wspace=0.3, hspace=0.33) for f in fig]
    ax_posture = [fig[i].add_subplot(gs[i][0]) for i in range(len(fig))]
    ax_force = [fig[i].add_subplot(gs[i][1]) for i in range(len(fig))]

    # plot posture
    # ------------------------------------------
    for tt in [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]:
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

    # plot force
    # -----------------------------------------
    up = [600, 600, 100, 100, 100, 100]
    low = [-600, -600, -100, -100, -100, -100]
    for i in range(6):
        for k in range(data.N):
            dof_id = i
            pos = 0
            neg = 0
            for kk in range(6):
                ax_force[i].bar(data.t[k], Force[kk][k, dof_id], width=data.dt[k],
                                bottom=pos if Force[kk][k, dof_id] >= 0 else neg, align='edge', color=ColorCandidate[kk], linewidth=0, ecolor=ColorCandidate[kk])
                if Force[kk][k, dof_id] >= 0:
                    pos += Force[kk][k, dof_id]
                else:
                    neg += Force[kk][k, dof_id]
                pass
            pass
        ax_force[i].set_xlim([0, 0.2])
        ax_force[i].set_ylim([low[i], up[i]])
        rainbow_text(fig[i], ax_force[i], 0.01, 650 if dof_id < 2 else 108, [r'$M_{ii}\ddot{q}_i$', r'$+$',
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
    main()
    # ForceVisualization()
    # power_analysis()
    # Impact_inertia()
    # Impact_process()
    # Power_metrics_analysis()
    pass
