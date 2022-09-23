'''
1. 双足机器人轨迹优化
2. 将接触的序列预先制定
3. 格式化配置参数输入和结果输出
4. 混合动力学系统，系统在机器人足底和地面接触的时候存在切换
5. 加入双臂的运动
6. Biped_walk_half2: x0,z0 in hip and hO_hip = O_b+O_hip
7. 2022.09.12: 
       Biped_walk_half3: 从上到下的三连杆倒立摆动力学方程建模方法(有脚踝关节力矩)
8. 2022.09.23：
        - 对质量和惯量参数遍历，计算力矩、功率、支撑力、平衡时间相对于惯性参数的计算结果
        - 对比参数遍历后有臂和无臂对四个指标的影响
        - 观察有臂和无臂目标函数变化情况
        - 计算有臂和无臂各部分的角动量变化情况
        - 将目标函数归一化（暂时失败）
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
        self.I_ = [self.m[i]*self.l[i]**2+self.I[i] for i in range(3)]

        self.L = [cfg['Robot']['Geometry']['L_leg'],
                  cfg['Robot']['Geometry']['L_body'],             
                  cfg['Robot']['Geometry']['L_arm']]

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

        self.u_LB = [-13] + [-self.motor_mt] * 2
        self.u_UB = [13] + [self.motor_mt] * 2

        # shank, thigh, body, arm, forearm
        self.q_LB = [-np.pi/10, -np.pi/30, 0,] 
        self.q_UB = [np.pi/2, np.pi*0.9, 4*np.pi/3]   

        self.dq_LB = [-self.motor_ms,
                      -self.motor_ms, -self.motor_ms]   # arm 

        self.dq_UB = [self.motor_ms,
                      self.motor_ms, self.motor_ms] # arm 

        # * define variable
        self.q = [self.opti.variable(3) for _ in range(self.N)]
        self.dq = [self.opti.variable(3) for _ in range(self.N)]
        self.ddq = [(self.dq[i+1]-self.dq[i]) /
                        self.dt for i in range(self.N-1)]

        # ! set the last u to be zero at constraint
        self.u = [self.opti.variable(3) for _ in range(self.N)]

        # ! Note, the last force represents the plused force at the contact moment
        # self.F = [self.opti.variable(2) for _ in range(self.N)]

        pass

    def MassMatrix(self, q):
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        I0 = self.I[0]
        I1 = self.I[1]
        I2 = self.I[2]

        M11 = I0 + I1 + I2 + (L0**2 + L1**2 + lc2**2)*m2+\
            (L0**2 + lc1**2) * m1 + lc0**2*m0 + \
            2*L0*m2*(L1*c(q[1]) + lc2*c(q[1]+q[2])) + \
            2*L0*lc1*m1*c(q[1]) + 2*L1*lc2*m2*c(q[2])

        M12 = I1 + I2 + (L1**2 + lc2**2)*m2 + lc1**2*m1 + \
            L0*L1*m2*c(q[1]) + L0*m2*lc2*c(q[1]+q[2]) + \
            L0*lc1*m1*c(q[1]) + 2*L1*lc2*m2*c(q[2])

        M13 = I2 + lc2**2*m2 + L0*lc2*m2*c(q[1]+q[2]) + L1*lc2*m2*c(q[2])
        
        M21 = M12
        M22 = I1 + I2 + (L1**2 + lc2**2)*m2 + lc2**1*m1 + \
            2*L1*lc2*m2*c(q[2])
        M23 = I2 + lc2**2*m2 + L1*lc2*m2*c(q[2])

        M31 = M13
        M32 = M23
        M33 = I2 + lc2**2*m2

        return [[M11, M12, M13],
                [M21, M22, M23],
                [M31, M32, M33]]

    def coriolis(self, q, dq):
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]

        C1 = -2*L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[0]*dq[1] \
            - 2*lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[0]*dq[2] \
            - L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[1]*dq[1] \
            - 2*lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[1]*dq[2] \
            - lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[2]*dq[2]
        # C1 = C1 - 0.2 * dq[0] 
        
        C2 = L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[0]*dq[0] \
            - 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[2] \
            - 2*L1*lc2*m2*s(q[2]) * dq[1]*dq[2] \
            - L1*lc2*m2*s(q[2]) * dq[2]*dq[2]
        # C2 = C2 - 0.2 * dq[1] 

        C3 = lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[0]*dq[0] \
            + 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[1] \
            + L1*lc2*m2*s(q[2]) * dq[1]*dq[1]
        # C3 = C3 - 0.2 * dq[2] 

        return [C1, C2, C3]

    def gravity(self, q):
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
        lc0 = self.l[0]
        lc1 = self.l[1]
        lc2 = self.l[2]
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]

        G1 = -(L0*m1*s(q[0]) + L0*m2*s(q[0]) + L1*m2*s(q[0]+q[1]) + \
            lc0*m0*s(q[0]) + lc1*m1*s(q[0]+q[1]) + lc2*m2*s(q[0]+q[1]+q[2]))
        
        G2 = -(L1*m2*s(q[0]+q[1]) + lc1*m1*s(q[0]+q[1]) + lc2*m2*s(q[0]+q[1]+q[2]))

        G3 = -lc2*m2*s(q[0]+q[1]+q[2])

        return [G1*self.g, G2*self.g, G3*self.g]

        pass

    def inertia_force(self, q, acc):
        mm = self.MassMatrix(q)
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2] for i in range(3)]
        
        return inertia_force

    def inertia_force2(self, q, acc):
        # region calculate inertia force, split into two parts
        mm = self.MassMatrix(q)
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1]+mm[i][2]*acc[2] for i in range(3)]
        inertia_main = [mm[i][i]*acc[i] for i in range(3)]
        inertia_coupling = [inertia_force[i]-inertia_main[i] for i in range(3)]
        return inertia_main, inertia_coupling

    def SupportForce(self, q, dq, ddq):
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        l0 = self.l[0]
        l1 = self.l[1]
        l2 = self.l[2]
        m0 = self.m[0]
        m1 = self.m[1]
        m2 = self.m[2]
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
        
        AccFx = -(m0*ddx0 + m1*ddx1 + m2*ddx2)
        AccFy = -(m0*ddy0 + m1*ddy1 + m2*ddy2) - (m0*self.g + m1*self.g + m2*self.g)

        AccF = [AccFx, AccFy]

        return AccF
        pass


    @staticmethod
    def get_posture(q):
        L = [0.9, 0.5, 0.4]
        lsx = np.zeros(2)
        lsy = np.zeros(2)
        ltx = np.zeros(2)
        lty = np.zeros(2)
        lax = np.zeros(2)
        lay = np.zeros(2)
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
            for j in range(3):
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
        Pf = [80, 20, 5]
        Vf = [40, 20, 5]
        Ff = [30, 10, 5]

        # Pf = [0.1, 0.1, 0.1]
        # Vf = [0.8, 0.8, 0.8]
        # Ff = [0.0, 0.2, 0.2]
        # Pwf = [0.05, 0.2, 0.2]
        Ptar = [0, 0, np.pi]

        qmax = 1.5*np.pi
        dqmax = 20
        umax = 600
        
        # for i in range(walker.N):
        #     for k in range(3):
        #         power += ((walker.dq[i][k]*walker.u[i][k]) / (dqmax*umax))**2 * walker.dt*Pwf[k]
        #         force += (walker.u[i][k] / umax)**2 * walker.dt * Ff[k]  

        #         VelTar += (walker.dq[i][k]/dqmax)**2 * walker.dt * Vf[k]
        #         PosTar += ((walker.q[i][k] - Ptar[k])/qmax)**2 * walker.dt * Pf[k]              
        #         pass
        #     pass
        
        # for j in range(3):
        #     VelTar += (walker.dq[-1][j]/dqmax)**2 * Vf[j] * 100
        #     PosTar += ((walker.q[-1][j] - Ptar[j])/qmax)**2 * Pf[j] * 500

        for i in range(walker.N):
            for k in range(3):
                power += ((walker.dq[i][k]*walker.u[i][k]))**2 * walker.dt
                force += (walker.u[i][k] / walker.motor_mt)**2 * walker.dt * Ff[k]  

                VelTar += (walker.dq[i][k])**2 * walker.dt * Vf[k]
                PosTar += ((walker.q[i][k] - Ptar[k]))**2 * walker.dt * Pf[k]
                pass
            pass
        
        for j in range(3):
            VelTar += (walker.dq[-1][j])**2 * Vf[j]*99
            PosTar += ((walker.q[-1][j] - Ptar[j]))**2 * Pf[j]*500
       
        u = walker.u

        smooth = 0
        AM = [100, 400, 100]
        for i in range(walker.N-1):
            for k in range(3):
                smooth += ((u[i+1][k]-u[i][k])/10)**2
                pass
            pass

        res = 0
        res = (res + power*self.powerCoeff) if (self.powerCoeff > 1e-6) else res
        res = (res + VelTar*self.velCoeff) if (self.velCoeff > 1e-6) else res
        res = (res + PosTar*self.trackingCoeff) if (self.trackingCoeff > 1e-6) else res
        res = (res + force*self.forceCoeff) if (self.forceCoeff > 1e-6) else res
        res = (res + smooth*self.smoothCoeff) if (self.smoothCoeff > 1e-6) else res
        # res = (res + self.powerCoeff)
        # res = (res + self.velCoeff)
        # res = (res + self.trackingCoeff)
        # res = (res + self.forceCoeff)

        return res

    def getConstraints(self, walker):
        ceq = []
        # region dynamics constraints
        # continuous dynamics
        #! 约束的数量为 (6+6）*（NN1-1+NN2-1）
        for j in range(walker.N):
            if j < (walker.N-1):
                ceq.extend([walker.q[j+1][k]-walker.q[j][k]-walker.dt/2 *
                            (walker.dq[j+1][k]+walker.dq[j][k]) == 0 for k in range(3)])
                inertia = walker.inertia_force(
                    walker.q[j], walker.ddq[j])
                coriolis = walker.coriolis(
                    walker.q[j], walker.dq[j])
                gravity = walker.gravity(walker.q[j])
                # ceq.extend([inertia[0]+gravity[0]+coriolis[0] == 0])
                # ceq.extend([inertia[k+1]+gravity[k+1]+coriolis[k+1] -
                #             walker.u[j][k] == 0 for k in range(4)])
                ceq.extend([inertia[k]+gravity[k]+coriolis[k] -
                            walker.u[j][k] == 0 for k in range(3)])

            if not self.armflag:
                ceq.extend([walker.q[j][2] == np.pi])
                ceq.extend([walker.dq[j][2] == 0])
            
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
                        temp_q[j], walker.q_UB[j]) for j in range(3)])
            pass
        for temp_dq in walker.dq:
            ceq.extend([walker.opti.bounded(walker.dq_LB[j],
                        temp_dq[j], walker.dq_UB[j]) for j in range(3)])
            pass
        for temp_u in walker.u:
            ceq.extend([walker.opti.bounded(walker.u_LB[j],
                        temp_u[j], walker.u_UB[j]) for j in range(3)])
            pass
        # endregion

        # region motor external characteristic curve
        cs = walker.motor_cs
        ms = walker.motor_ms
        mt = walker.motor_mt
        for j in range(len(walker.u)):
            ceq.extend([walker.u[j][k]-ca.fmax(mt - (walker.dq[j][k] -
                                                        cs)/(ms-cs)*mt, 0) <= 0 for k in range(3)])
            ceq.extend([walker.u[j][k]-ca.fmin(-mt + (walker.dq[j][k] +
                                                            cs)/(-ms+cs)*mt, 0) >= 0 for k in range(3)])
            pass
        # endregion

        theta = np.pi/40
        ceq.extend([walker.q[0][0]==theta])
        ceq.extend([walker.q[0][1]==theta*0.2])
        ceq.extend([walker.q[0][2]==np.pi])

        ceq.extend([walker.dq[0][0]==0])
        ceq.extend([walker.dq[0][1]==0])
        ceq.extend([walker.dq[0][2]==0])

        # region smooth constraint
        for j in range(len(walker.u)-1):
            ceq.extend([(walker.u[j][k]-walker.u
                        [j+1][k])**2 <= 80 for k in range(3)])
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
                q.append([sol1.value(robot.q[j][k]) for k in range(3)])
                dq.append([sol1.value(robot.dq[j][k])
                            for k in range(3)])
                if j < (robot.N-1):
                    ddq.append([sol1.value(robot.ddq[j][k])
                                for k in range(3)])
                    u.append([sol1.value(robot.u[j][k])
                                for k in range(3)])
                else:
                    ddq.append([sol1.value(robot.ddq[j-1][k])
                                for k in range(3)])
                    u.append([sol1.value(robot.u[j-1][k])
                                for k in range(3)])
                pass
            pass
        except:
            value = robot.opti.debug.value
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([value(robot.q[j][k])
                            for k in range(3)])
                dq.append([value(robot.dq[j][k])
                            for k in range(3)])
                if j < (robot.N-1):
                    ddq.append([value(robot.ddq[j][k])
                                for k in range(3)])
                    u.append([value(robot.u[j][k])
                                for k in range(3)])
                else:
                    ddq.append([value(robot.ddq[j-1][k])
                                for k in range(3)])
                    u.append([value(robot.u[j-1][k])
                                for k in range(3)])
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


    vis_flag = True

def main(Mass, inertia, armflag, vis_flag):
    # region optimization trajectory for bipedal hybrid robot system
    # vis_flag = True
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
    ParamFilePath = FilePath + "/config/Biped_balance3.yaml"
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
    b,a = signal.butter(3, 0.12, 'lowpass')
    Fy2 = signal.filtfilt(b, a, Fy)

    #region: costfun cal
    # Pf = [0.1, 0.1, 0.1]
    # Vf = [0.8, 0.8, 0.8]
    # Ff = [0.0, 0.2, 0.2]
    # Pwf = [0.05, 0.2, 0.2]
    Ptar = [0, 0, np.pi]

    # qmax = 1.5*np.pi
    # dqmax = 20
    # umax = 600

    Pf = [80, 20, 5]
    Vf = [40, 20, 5]
    Ff = [30, 10, 5]

    Pcostfun = 0.0
    Vcostfun = 0.0
    Fcostfun = 0.0
    Power = 0.0
    # for i in range(robot.N):
    #     for k in range(3):
    #         if i < robot.N-1:
    #             Pcostfun += ((q[i][k] - Ptar[k])/qmax)**2 * robot.dt * Pf[k]
    #             Fcostfun += (u[i][k] / umax)**2 * robot.dt * Ff[k]  
    #             Vcostfun += ((dq[i][k])/dqmax)**2 * robot.dt * Vf[k]
    #             Power += ((dq[i][k] * u[i][k])/(qmax*umax))
    #         else:
    #             Pcostfun += ((q[i][k] - Ptar[k])/qmax)**2 * robot.dt * Pf[k]
    #             Vcostfun += ((dq[i][k])/dqmax)**2 * robot.dt * Vf[k]
    for i in range(robot.N):
        for k in range(3):
            if i < robot.N-1:
                Pcostfun += ((q[i][k] - Ptar[k]))**2 * robot.dt * Pf[k]
                Fcostfun += (u[i][k] / robot.motor_mt)**2 * robot.dt * Ff[k]  
                Vcostfun += ((dq[i][k]))**2 * robot.dt * Vf[k]
                Power += np.abs(dq[i][k] * u[i][k])
            else:
                Pcostfun += ((q[i][k] - Ptar[k]))**2 * robot.dt * Pf[k]
                Vcostfun += ((dq[i][k]))**2 * robot.dt * Vf[k]
    print(Pcostfun, Vcostfun, Fcostfun)
    # endregion
    theta = np.pi/40
    visual = DataProcess(cfg, robot, Mass[2], inertia[2], theta, q, dq, ddq, u, F, t, save_dir, save_flag)
    if save_flag:
        SaveDir = visual.DataSave(save_flag)

    if vis_flag:
        visual.animation(0, save_flag)

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
        g_data = gs[0].subgridspec(3, 3, wspace=0.3, hspace=0.33)
        ax_m = fig2.add_subplot(gm[0])

        # gs = fig.add_gridspec(2, 1, height_ratios=[2,1],
        #                       wspace=0.3, hspace=0.33)
        # g_data = gs[1].subgridspec(3, 6, wspace=0.3, hspace=0.33)

        # ax_m = fig.add_subplot(gs[0])
        ax_p = [fig.add_subplot(g_data[0, i]) for i in range(3)]
        ax_v = [fig.add_subplot(g_data[1, i]) for i in range(3)]
        ax_u = [fig.add_subplot(g_data[2, i]) for i in range(3)]

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
            patch = patches.Rectangle(
                (pos[2][0]-0.02, pos[3][0]-0.05), 0.04, 0.1, alpha=tt/robot.T*0.8+0.2, lw=0, color=cmap(tt/robot.T))
            ax_m.add_patch(patch)
            ax_m.axis('equal')
            pass

        ax_m.set_ylabel('z(m)')
        ax_m.set_xlabel('x(m)')
        ax_m.xaxis.set_tick_params()
        ax_m.yaxis.set_tick_params()

        [ax_v[i].plot(t, dq[:, i]) for i in range(3)]
        ax_v[0].set_ylabel('Velocity(m/s)')
        [ax_p[i].plot(t, q[:, i]) for i in range(3)]
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
        # [ax_u[j].set_title(title_u[j]) for j in range(4)]

        fig3 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=False)
        plt.plot(t, Fx, label = 'Fx')
        plt.plot(t, Fy, label = 'Fy')
        # plt.plot(t, Fy2, label = 'Fy')
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
    F = [Fx, Fy]
    return u, Fy2, t, Pcostfun, Vcostfun, Fcostfun, Power

## use mean value instead of peak value to analysis force map
def ForceMapMV():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pickle
    import random
    from matplotlib.pyplot import MultipleLocator
    from mpl_toolkits.mplot3d import Axes3D

    saveflag = True
    armflag = False
    vis_flag = False
    ani_flag = False

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name = "ForceMap3-7-noarm-cfun.pkl"
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    M_arm = [3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5]
    # M_arm = [4.0, 6.0, 7.0]
    # M_arm = [4.0]
    M_label = list(map(str, M_arm))
    I_arm = [0.012, 0.015, 0.03, 0.04, 0.06, 0.07, 0.09]
    # I_arm = [0.012, 0.04, 0.09]
    # I_arm = [0.03]
    I_label = list(map(str, I_arm))

    Mass = [15, 20]
    inertia = [1.0125, 0.417]
    # Mass = [25, 30]
    # inertia = [1.675, 0.625]

    Fy = np.array([[0.0]*len(M_arm)])
    u_h = np.array([[0.0]*len(M_arm)])
    u_a = np.array([[0.0]*len(M_arm)])
    u_k = np.array([[0.0]*len(M_arm)])
    u_s = np.array([[0.0]*len(M_arm)])
    u_e = np.array([[0.0]*len(M_arm)])
    t_b = np.array([[0.0]*len(M_arm)])

    Pcostfun = np.array([[0.0]*len(M_arm)])
    Vcostfun = np.array([[0.0]*len(M_arm)])
    Fcostfun = np.array([[0.0]*len(M_arm)])
    Power = np.array([[0.0]*len(M_arm)])

    CollectNum = 1000
    dt = 0.002
    index = int(1.3 / dt)
    if saveflag:
        for i in range(len(I_arm)):
            temp_i = []
            temp_i.extend(inertia)
            temp_i.append(I_arm[i])
            Fy_max = []
            u_h_max = []
            u_a_max = []
            u_s_max = []
            t_p = []

            P_J = []
            V_J = []
            F_J = []
            Pw_J=[]
            for j in range(len(M_arm)):
                temp_m = []
                temp_m.extend(Mass)
                temp_m.append(M_arm[j])

                print("="*50)
                print("Mass: ", temp_m)
                print("="*50)
                print("Inertia: ", temp_i)
                print("="*50)
                print("armflag: ", armflag)

                u, F, t, Ptmp, Vtmp, Ftmp, Pwtmp = main(temp_m, temp_i, armflag, vis_flag)

                F_1 = 0
                num1 = 0
                for k in range(len(F)):
                    if F[k] > 1:
                        F_1 += F[k]
                        num1 += 1
                    
                    if F[k] > F[index]*1.04 and k*dt < 1.2:
                        tk = k * dt

                F_1 = F_1 / num1
                # print(F_1)

                temp_fy_max = F_1

                u4 = u[:, 0]
                temp_ua0_max = np.sum(np.sqrt(u4**2)) / CollectNum
                temp_ua_max = temp_ua0_max

                u0 = u[:, 1]
                temp_uh0_max = np.sum(np.sqrt(u0**2)) / CollectNum
                temp_uh_max = temp_uh0_max

                u1 = u[:, 2]
                temp_us1_max = np.sum(np.sqrt(u1**2)) / CollectNum
                temp_us_max = temp_us1_max

                Fy_max.append(temp_fy_max)
                u_h_max.append(temp_uh_max)
                u_s_max.append(temp_us_max)
                u_a_max.append(temp_ua_max)
                t_p.append(tk)
                P_J.append(Ptmp)
                V_J.append(Vtmp)
                F_J.append(Ftmp)
                Pw_J.append(Pwtmp)

                pass
            # print(u0.shape,u_k_max)
            print(Fy_max)
            Fy = np.concatenate((Fy, [Fy_max]), axis = 0)
            u_h = np.concatenate((u_h, [u_h_max]), axis = 0)
            u_s = np.concatenate((u_s, [u_s_max]), axis = 0)
            u_a = np.concatenate((u_a, [u_a_max]), axis = 0)
            t_b = np.concatenate((t_b, [t_p]), axis = 0)
            Pcostfun = np.concatenate((Pcostfun, [P_J]), axis = 0)
            Vcostfun = np.concatenate((Vcostfun, [V_J]), axis = 0)
            Fcostfun = np.concatenate((Fcostfun, [F_J]), axis = 0)
            Power = np.concatenate((Power, [Pw_J]), axis = 0)

            pass
        Fy = Fy[1:]
        u_h = u_h[1:]
        u_s = u_s[1:]
        u_a = u_a[1:]
        t_b = t_b[1:]
        Pcostfun = Pcostfun[1:]
        Vcostfun = Vcostfun[1:]
        Fcostfun = Fcostfun[1:]
        Power = Power[1:]

        Data = {'Fy': Fy, 'u_h': u_h, "u_s": u_s,"u_a": u_a, "t_b": t_b,
                'P_J': Pcostfun, 'V_J': Vcostfun, 'F_J': Fcostfun, 'Pw_J': Power}
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
        u_s = data['u_s']
        u_a = data['u_a']
        t_b = data['t_b']
        Pcostfun = data['P_J']
        Vcostfun = data['V_J']
        Fcostfun = data['F_J']

    Sumcostfun = Pcostfun + Vcostfun + Fcostfun
    print(Sumcostfun)

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

    # region: imshow
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    ax1 = axs[0][0]
    ax2 = axs[0][1]
    ax3 = axs[1][0]
    ax4 = axs[1][1]

    print(t_b)
    if armflag:
        pcm1 = ax1.imshow(Fy, vmin = 345, vmax = 420)
        pcm2 = ax2.imshow(u_h, vmin = 60, vmax = 80)
        pcm3 = ax3.imshow(u_s, vmin = 3, vmax = 12)
        pcm4 = ax4.imshow(t_b, vmin = 0.4, vmax = 0.6)
    else:
        pcm1 = ax1.imshow(Fy, vmin = 340, vmax = 430)
        pcm2 = ax2.imshow(u_h, vmin = 70, vmax = 90)
        pcm3 = ax3.imshow(u_s, vmin = 2, vmax = 15)
        pcm4 = ax4.imshow(t_b, vmin = 0.7, vmax = 1.2)

    cb1 = fig.colorbar(pcm1, ax=ax1)
    cb2 = fig.colorbar(pcm2, ax=ax2)
    cb3 = fig.colorbar(pcm3, ax=ax3)
    cb4 = fig.colorbar(pcm4, ax=ax4)
    
    ax = [[ax1, ax2], [ax3, ax4]]
    cb = [[cb1, cb2], [cb3, cb4]]
    title = [["Fy", "Torque-Hip"], ["Torque-shoulder", "balance time"]]
    Dataset = {"Fy":Fy, "Torque-shoulder":u_s, "Torque-Hip":u_h, "balance time":t_b}
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
            elif i==1 and j==1:
                cb[i][j].set_label("Time(s)")

            else:
                cb[i][j].set_label("Torque(N/m)")
            
            for k in range(len(M_arm)):
                for m in range(len(I_arm)):
                    ids = i*2+1
                    data = Dataset[title[i][j]]
                    data = np.round(data, ids)
                    ax[i][j].text(m,k,data[k][m], ha="center", va="center",color="w",fontsize=10)
    fig.tight_layout()
    # endregion
    
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": "3d"})
    axes1 = axs2[0][0]
    axes2 = axs2[0][1]
    axes3 = axs2[1][0]
    axes4 = axs2[1][1]
    M_arm, I_arm = np.meshgrid(M_arm, I_arm)


    surf1 = axes1.plot_surface(M_arm, I_arm, Pcostfun)
    surf2 = axes2.plot_surface(M_arm, I_arm, Vcostfun)
    surf3 = axes3.plot_surface(M_arm, I_arm, Fcostfun)
    surf4 = axes4.plot_surface(M_arm, I_arm, Sumcostfun)
    for i in range(2):
        for j in range(2):
            axs2[i][j].set_ylabel("Inertia")
            axs2[i][j].set_xlabel("Mass")
            axs2[i][j].set_zlabel(title[i][j])
            axs2[i][j].set_title(title[i][j])

    # def rotate(angle):
    #     axes1.view_init(30, angle)
    #     axes2.view_init(30, angle)
    #     axes3.view_init(30, angle)
    #     axes4.view_init(30, angle)
    # rot_animation = animation.FuncAnimation(fig2, rotate, frames=np.arange(0,362,2), interval=0.1, blit=False)
    
    # if ani_flag:
    #     ani_name = save_dir + "rotation.gif"
    #     rot_animation.save(ani_name, fps=30, writer='pillow')
    plt.show()
    
    pass

def ResCmp():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pickle
    import random
    from matplotlib.pyplot import MultipleLocator
    from mpl_toolkits.mplot3d import Axes3D
    from scipy import interpolate

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name1 = "ForceMap3-7-arm-cfun.pkl"
    name2 = "ForceMap3-7-noarm-cfun.pkl"
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    M_arm = [3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5]
    I_arm = [0.012, 0.015, 0.03, 0.04, 0.06, 0.07, 0.09]
    M_label = list(map(str, M_arm))
    I_label = list(map(str, I_arm))

    f1 = open(save_dir+name1,'rb')
    data1 = pickle.load(f1)
    f2 = open(save_dir+name2,'rb')
    data2 = pickle.load(f2)

    Fy = data1['Fy']
    u_h = data1['u_h']
    u_s = data1['u_s']
    u_a = data1['u_a']
    t_b = data1['t_b']
    Pwcostfun1 = data1['P_J']


    Fy2 = data2['Fy']
    u_h2 = data2['u_h']
    u_s2 = data2['u_s']
    u_a2 = data2['u_a']
    t_b2 = data2['t_b']
    Pwcostfun2 = data2['P_J']
    t_b2[0][6] = 0.9

    # 数据插值光滑
    Mnew = np.linspace(3, 7.5, 30)
    Inew = np.linspace(0.012, 0.09, 30)
    ffy = interpolate.interp2d(M_arm, I_arm, Fy, kind='cubic')
    fuh = interpolate.interp2d(M_arm, I_arm, u_h, kind='cubic')
    fus = interpolate.interp2d(M_arm, I_arm, u_s, kind='cubic')
    ftb = interpolate.interp2d(M_arm, I_arm, t_b, kind='linear')
    fw1 = interpolate.interp2d(M_arm, I_arm, Pwcostfun1, kind='cubic')
    Fynew = ffy(Mnew, Inew)
    uhnew = fuh(Mnew, Inew)
    usnew = fus(Mnew, Inew)
    tbnew = ftb(Mnew, Inew)
    Pwnew1 = fw1(Mnew, Inew)

    ffy2 = interpolate.interp2d(M_arm, I_arm, Fy2, kind='cubic')
    fuh2 = interpolate.interp2d(M_arm, I_arm, u_h2, kind='cubic')
    fus2 = interpolate.interp2d(M_arm, I_arm, u_s2, kind='cubic')
    ftb2 = interpolate.interp2d(M_arm, I_arm, t_b2, kind='linear')
    fw2 = interpolate.interp2d(M_arm, I_arm, Pwcostfun2, kind='cubic')
    Fynew2 = ffy2(Mnew, Inew)
    uhnew2 = fuh2(Mnew, Inew)
    usnew2 = fus2(Mnew, Inew)
    tbnew2 = ftb2(Mnew, Inew)
    Pwnew2 = fw2(Mnew, Inew)

    
    plt.style.use("science")
    params = {
        'text.usetex': True,
        'image.cmap': 'inferno',
        'font.size': 18,
        'axes.labelsize': 15,
        'axes.titlesize': 20,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'axes.titlepad': 15.0,
        'axes.labelpad': 12.0,
        'figure.subplot.wspace': 0.2,
        'figure.subplot.hspace': 0.3,
    }

    plt.rcParams.update(params)
    title = [["Fy", "Torque-Hip"], ["Power", "balance time"]]
    Dataset = {"Fy":Fy, "Torque-shoulder":u_s, "Torque-Hip":u_h, "balance time":t_b}

    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    ax1 = axs
    print(t_b)
    pcm1 = ax1.imshow(Pwcostfun1, vmin = 40, vmax = 50)

    cb1 = fig.colorbar(pcm1, ax=ax1)
    ax1.set_xticks(np.arange(len(M_label)))
    ax1.set_xticklabels(M_label)
    ax1.set_yticks(np.arange(len(I_label)))
    ax1.set_ylim(-0.5, len(I_label)-0.5)
    ax1.set_yticklabels(I_label)
    # ax[i][j].xaxis.set_tick_params(top=True, bottom=False,
    #        labeltop=True, labelbottom=False)

    ax1.set_ylabel("Inertia")
    ax1.set_xlabel("Mass")
    ax1.set_title("Power")
    cb1.set_label("Power (N)")
    for k in range(len(M_arm)):
        for m in range(len(I_arm)):
            data = Pwcostfun1
            data = np.round(data, 2)
            ax1.text(m,k,data[k][m], ha="center", va="center",color="w",fontsize=10)
    fig.tight_layout()

    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": "3d"})
    axes1 = axs2[0][0]
    axes2 = axs2[0][1]
    axes3 = axs2[1][0]
    axes4 = axs2[1][1]
    M_arm, I_arm = np.meshgrid(M_arm, I_arm)
    Mnew, Inew = np.meshgrid(Mnew, Inew)


    # surf1 = axes1.plot_surface(M_arm, I_arm, Fy)
    # surf12 = axes1.plot_surface(M_arm, I_arm, Fy2)
    # surf2 = axes2.plot_surface(M_arm, I_arm, u_h)
    # surf22 = axes2.plot_surface(M_arm, I_arm, u_h2)
    # surf3 = axes3.plot_surface(M_arm, I_arm, u_s)
    # surf32 = axes3.plot_surface(M_arm, I_arm, u_s2)
    # surf4 = axes4.plot_surface(M_arm, I_arm, t_b, cmap="inferno")
    # surf42 = axes4.plot_surface(M_arm, I_arm, t_b2, cmap="inferno")
    surf1 = axes1.plot_surface(Mnew, Inew, Fynew, label="arm free")
    surf12 = axes1.plot_surface(Mnew, Inew, Fynew2, label="arm bound")
    surf2 = axes2.plot_surface(Mnew, Inew, uhnew, label="arm free")
    surf22 = axes2.plot_surface(Mnew, Inew, uhnew2, label="arm bound")
    surf3 = axes3.plot_surface(Mnew, Inew, Pwnew1, label="arm free")
    surf32 = axes3.plot_surface(Mnew, Inew, Pwnew2, label="arm bound")
    surf4 = axes4.plot_surface(Mnew, Inew, tbnew, label="arm free")
    surf42 = axes4.plot_surface(Mnew, Inew, tbnew2, label="arm bound")
    surf = [surf1,surf12,surf2,surf22,surf3,surf32,surf4,surf42]
   
    for k in range(8):
        surf[k]._facecolors2d=surf[k]._facecolors3d
        surf[k]._edgecolors2d=surf[k]._edgecolors3d
    for i in range(2):
        for j in range(2):

            axs2[i][j].set_ylabel("Inertia")
            axs2[i][j].set_xlabel("Mass")
            axs2[i][j].set_zlabel(title[i][j])
            axs2[i][j].set_title(title[i][j])
            axs2[i][j].legend(loc="upper left")
    axes4.set_zlim(0.0, 1.2)
    axes3.set_zlim(0.0, 150)

    # elev =20
    # def rotate(angle):
    #     axes1.view_init(elev, angle)
    #     axes2.view_init(elev, angle)
    #     axes3.view_init(elev, angle)
    #     axes4.view_init(elev, angle)
    # rot_animation = animation.FuncAnimation(fig2, rotate, frames=np.arange(0,150,0.8), interval=0.3, blit=False)
    # ani_flag = True
    # if ani_flag:
    #     ani_name = save_dir + "rotation ResCmp.gif"
    #     rot_animation.save(ani_name, fps=30, writer='pillow')

    plt.show()
    pass

def CostFunAnalysis():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pickle
    import random
    from matplotlib.pyplot import MultipleLocator
    from mpl_toolkits.mplot3d import Axes3D
    from scipy import interpolate

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name1 = "ForceMap3-7-arm-pw2.pkl"
    name2 = "ForceMap3-7-noarm-pw.pkl"
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    M_arm = [3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5]
    I_arm = [0.012, 0.015, 0.03, 0.04, 0.06, 0.07, 0.09]
    M_label = list(map(str, M_arm))
    I_label = list(map(str, I_arm))

    f1 = open(save_dir+name1,'rb')
    data1 = pickle.load(f1)
    f2 = open(save_dir+name2,'rb')
    data2 = pickle.load(f2)

    Pcostfun1 = data1['P_J']
    Vcostfun1 = data1['V_J']
    Pwcostfun1 = data1['Pw_J']
    Pwcostfun1 = Pwcostfun1/1000
    Fcostfun1 = data1['F_J']
    Pcostfun2 = data2['P_J']
    Vcostfun2 = data2['V_J']
    Fcostfun2 = data2['F_J']
    Pwcostfun2 = data2['Pw_J']
    Pwcostfun2 = Pwcostfun2/1000

    Sumcostfun1 = Pcostfun1 + Vcostfun1 + Fcostfun1
    Sumcostfun2 = Pcostfun2 + Vcostfun2 + Fcostfun2

    # 数据插值光滑
    Mnew = np.linspace(3, 7.5, 30)
    Inew = np.linspace(0.012, 0.09, 30)
    fp1 = interpolate.interp2d(M_arm, I_arm, Pcostfun1, kind='cubic')
    fv1 = interpolate.interp2d(M_arm, I_arm, Vcostfun1, kind='cubic')
    ff1 = interpolate.interp2d(M_arm, I_arm, Fcostfun1, kind='cubic')
    fs1 = interpolate.interp2d(M_arm, I_arm, Sumcostfun1, kind='cubic')
    fw1 = interpolate.interp2d(M_arm, I_arm, Pwcostfun1, kind='cubic')
    Pnew = fp1(Mnew, Inew)
    Vnew = fv1(Mnew, Inew)
    Fnew = ff1(Mnew, Inew)
    Snew = fs1(Mnew, Inew)
    Pwnew1 = fw1(Mnew, Inew)

    fp2 = interpolate.interp2d(M_arm, I_arm, Pcostfun2, kind='cubic')
    fv2 = interpolate.interp2d(M_arm, I_arm, Vcostfun2, kind='cubic')
    ff2 = interpolate.interp2d(M_arm, I_arm, Fcostfun2, kind='cubic')
    fs2 = interpolate.interp2d(M_arm, I_arm, Sumcostfun2, kind='cubic')
    fw2 = interpolate.interp2d(M_arm, I_arm, Pwcostfun2, kind='cubic')
    Pnew2 = fp2(Mnew, Inew)
    Vnew2 = fv2(Mnew, Inew)
    Fnew2 = ff2(Mnew, Inew)
    Snew2 = fs2(Mnew, Inew)
    Pwnew2 = fw2(Mnew, Inew)

    
    plt.style.use("science")
    params = {
        'text.usetex': True,
        'image.cmap': 'inferno',
        'font.size': 18,
        'axes.labelsize': 15,
        'axes.titlesize': 20,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'axes.titlepad': 15.0,
        'axes.labelpad': 12.0,
        'figure.subplot.wspace': 0.2,
        'figure.subplot.hspace': 0.3,
    }

    plt.rcParams.update(params)
    title = [[r'$\sum (\theta_i-\theta_d)^2*c_p$', r'$\sum (\dot{\theta}_i-\dot{\theta}_d)^2*c_v$'], 
            [r'$\sum (u_i/u_{max})^2*c_f$', r'$\sum (u_i*\theta_i)^2$']]

    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": "3d"})
    axes1 = axs2[0][0]
    axes2 = axs2[0][1]
    axes3 = axs2[1][0]
    axes4 = axs2[1][1]
    M_arm, I_arm = np.meshgrid(M_arm, I_arm)
    Mnew, Inew = np.meshgrid(Mnew, Inew)

    surf1 = axes1.plot_surface(Mnew, Inew, Pnew, label="arm free")
    surf12 = axes1.plot_surface(Mnew, Inew, Pnew2, label="arm bound")
    surf2 = axes2.plot_surface(Mnew, Inew, Vnew, label="arm free")
    surf22 = axes2.plot_surface(Mnew, Inew, Vnew2, label="arm bound")
    surf3 = axes3.plot_surface(Mnew, Inew, Fnew, label="arm free")
    surf32 = axes3.plot_surface(Mnew, Inew, Fnew2, label="arm bound")
    surf4 = axes4.plot_surface(Mnew, Inew, Pwnew1, label="arm free")
    surf42 = axes4.plot_surface(Mnew, Inew, Pwnew2, label="arm bound")
    surf = [surf1,surf12,surf2,surf22,surf3,surf32,surf4,surf42]
   
    for k in range(8):
        surf[k]._facecolors2d=surf[k]._facecolors3d
        surf[k]._edgecolors2d=surf[k]._edgecolors3d
    for i in range(2):
        for j in range(2):

            axs2[i][j].set_ylabel("Inertia")
            axs2[i][j].set_xlabel("Mass")
            axs2[i][j].set_zlabel("CostFun")
            axs2[i][j].set_title(title[i][j])
            axs2[i][j].legend(loc="upper left")

    ani_flag = False
    if ani_flag:
        elev = 20
        def rotate(angle):
            axes1.view_init(elev, angle)
            axes2.view_init(elev, angle)
            axes3.view_init(elev, angle)
            axes4.view_init(elev, angle)
        rot_animation = animation.FuncAnimation(fig2, rotate, frames=np.arange(0,120,1), interval=0.2, blit=False)
    
        ani_name = save_dir + "rotation Cfun.gif"
        rot_animation.save(ani_name, fps=25, writer='pillow')

    print(Pwcostfun1)
    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    ax1 = axs
    pcm1 = ax1.imshow(Pwcostfun1, vmin = -4, vmax = 0)
    cb1 = fig.colorbar(pcm1, ax=ax1)
    ax1.set_xticks(np.arange(len(M_label)))
    ax1.set_xticklabels(M_label)
    ax1.set_yticks(np.arange(len(I_label)))
    ax1.set_ylim(-0.5, len(I_label)-0.5)
    ax1.set_yticklabels(I_label)
    # ax[i][j].xaxis.set_tick_params(top=True, bottom=False,
    #        labeltop=True, labelbottom=False)

    ax1.set_ylabel("Inertia")
    ax1.set_xlabel("Mass")
    ax1.set_title("Power")
    cb1.set_label("Power(N.s)")
    
    for k in range(len(M_arm)):
        for m in range(len(I_arm)):
            ids = i*2+1
            data = Pwcostfun1
            data = np.round(data, ids)
            ax1.text(m,k,data[k][m], ha="center", va="center",color="w",fontsize=10)
    fig.tight_layout()
    plt.show()
    pass

def CharactTime():
    M_arm = [3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5]
    M_label = list(map(str, M_arm))
    I_arm = [0.012, 0.015, 0.03, 0.04, 0.06, 0.07, 0.09]
    I_label = list(map(str, I_arm))

    lc = [0.5, 0.25, 0.2]
    L = [0.9, 0.5, 0.4]
    g = 9.8

    Mass = [15, 20]
    inertia = [1.0125, 0.417]
    T=[]
    for j in range(len(M_arm)):
        m = Mass[0]+Mass[1]+M_arm[j]
        l = (Mass[0]*lc[0]+Mass[1]*(L[0]+lc[1])+M_arm[j]*(L[0]+L[1]-lc[2])) / m
        I = m*l**2/12
        Tp = 2*np.pi*np.sqrt((m*l**2+I)/(m*g*l))
        T.append(Tp)
    print(T)
    pass

def VelAndPos(q, dq):
    L = [0.9, 0.5, 0.4]
    lc = [0.5, 0.25, 0.2]

    x0 = lc[0]*sin(q[0])
    y0 = lc[0]*cos(q[0])
    x1 = L[0]*sin(q[0])+lc[1]*sin(q[0]+q[1])
    y1 = L[0]*cos(q[0])+lc[1]*cos(q[0]+q[1])
    x2 = L[0]*sin(q[0])+L[1]*sin(q[0]+q[1])+lc[2]*sin(q[0]+q[1]+q[2])
    y2 = L[0]*cos(q[0])+L[1]*cos(q[0]+q[1])+lc[2]*cos(q[0]+q[1]+q[2])

    dx0 = lc[0]*cos(q[0])*dq[0]
    dy0 = -lc[0]*sin(q[0])*dq[0]
    dx1 = L[0]*cos(q[0])*(dq[0])+lc[1]*cos(q[0]+q[1])*(dq[0]+dq[1])
    dy1 = -L[0]*sin(q[0])*dq[0]-lc[1]*sin(q[0]+q[1])*(dq[0]+dq[1])
    dx2 = L[0]*cos(q[0])*dq[0]+L[1]*cos(q[0]+q[1])*(dq[0]+dq[1])+lc[2]*cos(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])
    dy2 = -L[0]*sin(q[0])*dq[0]-L[1]*sin(q[0]+q[1])*(dq[0]+dq[1])-lc[2]*sin(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])

    r = np.asarray([[x0, y0],
        [x1, y1],
        [x2, y2]])
    v = np.asarray([[dx0, dy0],
        [dx1, dy1],
        [dx2, dy2]])
    return r, v
    pass

def MOmentCal():
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name1 = "arm.pkl"
    name2 = "noarm.pkl"
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    f = open(save_dir+name1,'rb')
    data = pickle.load(f)
    f2 = open(save_dir+name2,'rb')
    data2 = pickle.load(f2)

    I = [15, 20, 4.0]
    m = [1.0125, 0.417, 0.03]
    q = data['q']
    dq = data['dq']
    t = data['t']
    q2 = data2['q']
    dq2 = data2['dq']
    t2 = data2['t']

    LegMomt = np.array([0.0])
    BodyMomt = np.array([0.0])
    ArmMomt = np.array([0.0])
    LegMomt2 = np.array([0.0])
    BodyMomt2 = np.array([0.0])
    ArmMomt2 = np.array([0.0])

    for i in range(len(t)):
        r, v = VelAndPos(q[i], dq[i])
        r2, v2 = VelAndPos(q2[i], dq2[i])
        legtmp = I[0]*dq[i][0] + m[0]*np.cross(r[0,:], v[0,:])
        bodytmp = I[1]*dq[i][1] + m[1]*np.cross(r[1,:], v[1,:])
        armtmp = I[2]*dq[i][2] + m[2]*np.cross(r[2,:], v[2,:])

        LegMomt = np.concatenate((LegMomt, [legtmp]))
        BodyMomt = np.concatenate((BodyMomt, [bodytmp]))
        ArmMomt = np.concatenate((ArmMomt, [armtmp]))

        legtmp2 = I[0]*dq2[i][0] + m[0]*np.cross(r2[0,:], v2[0,:])
        bodytmp2 = I[1]*dq2[i][1] + m[1]*np.cross(r2[1,:], v2[1,:])
        armtmp2 = I[2]*dq2[i][2] + m[2]*np.cross(r2[2,:], v2[2,:])

        LegMomt2 = np.concatenate((LegMomt2, [legtmp2]))
        BodyMomt2 = np.concatenate((BodyMomt2, [bodytmp2]))
        ArmMomt2 = np.concatenate((ArmMomt2, [armtmp2]))
        pass

    LegMomt = LegMomt[1:]
    BodyMomt = BodyMomt[1:]
    ArmMomt = ArmMomt[1:]
    LegMomt2 = LegMomt2[1:]
    BodyMomt2 = BodyMomt2[1:]
    ArmMomt2 = ArmMomt2[1:]

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'image.cmap': 'inferno',
        'font.size': 20,
        "lines.linewidth": 3,
        'axes.labelsize': 15,
        'axes.titlesize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.3,
    }

    plt.rcParams.update(params)

    # region: imshow
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    ax1 = axs[0]
    ax2 = axs[1]

    ax1.plot(t, LegMomt, label="Leg Momt")
    ax1.plot(t, BodyMomt, label="Body Momt")
    ax1.plot(t, ArmMomt, label="Arm Momt")
    ax1.set_ylabel("Angular Momentum (Kg.m2/s)")
    ax1.legend()
    ax1.grid()

    ax2.plot(t, LegMomt2, label="Leg Momt")
    ax2.plot(t, BodyMomt2, label="Body Momt")
    ax2.plot(t, ArmMomt2, label="Arm Momt")
    ax2.set_ylabel("Angular Momentum (Kg.m2/s)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid()


    plt.show()
    pass

if __name__ == "__main__":
    # main()
    # ForceMapMV()
    # MOmentCal()
    ResCmp()
    # CostFunAnalysis()
    # CharactTime()
    pass
