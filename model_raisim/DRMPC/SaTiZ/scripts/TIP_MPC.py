"""
三级摆的平衡控制
"""

from re import A
import casadi as ca
from casadi import sin as s
from casadi import cos as c
from matplotlib.pyplot import savefig
import numpy as np
from scipy import signal
from numpy.random import normal
import time
import os
import yaml
from ruamel.yaml import YAML
import raisimpy as raisim
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint

from Dynamics_MPC import RobotProperty
from RobotInterface import RobotInterface
from DataProcess import DataProcess

class TriplePendulum():
    def __init__(self, cfg):
        self.opti = ca.Opti()
        self.dt = cfg['Controller']['dt']       # sample time
        self.Tp = cfg['Controller']['Tp']       # predictive time
        self.NS = int(self.Tp / self.dt)        # samples number
        self.Frict_coef = cfg['Environment']['Friction_Coeff']
        # print("number of sample: ", self.NS)
        
        # mass and geometry related parameter
        self.m = cfg['Robot']['Mass']['mass']
        self.I = cfg['Robot']['Mass']['inertia']
        self.l = cfg['Robot']['Mass']['massCenter']
        self.I_ = [self.m[i]*self.l[i]**2+self.I[i] for i in range(3)]

        self.L = [cfg['Robot']['Geometry']['L_body'],
                  cfg['Robot']['Geometry']['L_thigh'],
                  cfg['Robot']['Geometry']['L_shank']]

        # motor parameter
        self.motor_cs = cfg['Robot']['Motor']['CriticalSpeed']
        self.motor_ms = cfg['Robot']['Motor']['MaxSpeed']
        self.motor_mt = cfg['Robot']['Motor']['MaxTorque']

        # evironemnt parameter
        self.mu = cfg['Environment']['Friction_Coeff']
        self.g = cfg['Environment']['Gravity']
        self.damping = cfg['Robot']['damping']

        # control parameter
        self.postar = cfg['Controller']['PosTar']
        self.veltar = cfg['Controller']['VelTar']

        # boundary parameter
        self.bound_fy = cfg['Controller']['Boundary']['Fy']
        self.bound_fx = cfg['Controller']['Boundary']['Fx']
        self.u_LB = [-self.motor_mt] * 2
        self.u_UB = [self.motor_mt] * 2

        self.q_LB = [cfg['Controller']['Boundary']['theta1'][0],
                     cfg['Controller']['Boundary']['theta2'][0],
                     cfg['Controller']['Boundary']['theta3'][0]]
        self.q_UB = [cfg['Controller']['Boundary']['theta1'][1],
                     cfg['Controller']['Boundary']['theta2'][1],
                     cfg['Controller']['Boundary']['theta3'][1]]

        self.dq_LB = [-self.motor_ms, -self.motor_ms, -self.motor_ms]

        self.dq_UB = [self.motor_ms, self.motor_ms, self.motor_ms]

        self.F_LB = [self.bound_fx[0], self.bound_fy[0]]
        self.F_UB = [self.bound_fx[1], self.bound_fy[1]]

        ## define variable
         # * define variable
        # self.q = []
        self.q = [self.opti.variable(3) for _ in range(self.NS)]
        # self.q.append([self.opti.variable(3) for _ in range(self.NS)])
        # self.dq = []
        self.dq = [self.opti.variable(3) for _ in range(self.NS)]
        # self.dq.append([self.opti.variable(3) for _ in range(self.NS)])
        # self.ddq = []
        self.ddq = [(self.dq[i+1]-self.dq[i]) /
                        self.dt for i in range(self.NS-1)]
        # self.ddq.append(self.ddq[0]) 
        # self.ddq.append([(self.dq[i+1]-self.dq[i]) /
        #                 self.dt for i in range(self.NS-1)])
        # self.u = []
        self.u = [self.opti.variable(2) for _ in range(self.NS)]
        # self.u.append([self.opti.variable(2) for _ in range(self.NS-1)])

        # support force
        self.F = []
        self.F = [self.opti.variable(2) for _ in range(self.NS-1)]


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

    def Coriolis(self, q, dq):
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
        C1 = C1 - 0.2 * dq[0] 
        
        C2 = L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[0]*dq[0] \
            - 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[2] \
            - 2*L1*lc2*m2*s(q[2]) * dq[1]*dq[2] \
            - L1*lc2*m2*s(q[2]) * dq[2]*dq[2]
        C2 = C2 - 0.2 * dq[1] 

        C3 = lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[0]*dq[0] \
            + 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[1] \
            + L1*lc2*m2*s(q[2]) * dq[1]*dq[1]
        C3 = C3 - 0.2 * dq[2] 

        return [C1, C2, C3]

    def Gravity(self, q):
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

    def InertiaForce(self, q, acc):
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

    def MassCenter(self, q):
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        l0 = self.l[0]
        l1 = self.l[1]
        l2 = self.l[2]

        z0 = l0*c(q[0])
        z1 = L0 * c(q[0]) + l1*c(q[0]+q[1])
        z2 = L0 * c(q[0]) + L1*c(q[0]+q[1]) + l2*c(q[0]+q[1]+q[2])
        return z0,z1,z2
        pass

class NLP():
    def __init__(self, robot, cfg, x0, dq0, seed=None):
        self.x0 = x0
        self.dq0 = dq0
        self.cfg = cfg
        self.TorqueCoef = cfg["Optimization"]["CostCoef"]["torqueCoef"]
        self.PostarCoef = cfg["Optimization"]["CostCoef"]["postarCoef"]
        self.VeltarCoef = cfg["Optimization"]["CostCoef"]["VeltarCoef"]
        self.DtorqueCoef = cfg["Optimization"]["CostCoef"]["DtorqueCoef"]
        self.SupportFCoef = cfg["Optimization"]["CostCoef"]["SupportFCoef"]
        max_iter = cfg["Optimization"]["MaxLoop"]
        self.random_seed = cfg["Optimization"]["RandomSeed"]
        self.dt = cfg['Controller']['dt']       # sample time
        self.Nc = cfg['Controller']['Nc']       # control time

        # self.cost = self.CostFun(robot)
        self.cost = self.CostFunMPC(robot)
        # self.cost = self.CostFunMPCHuman(robot)
        robot.opti.minimize(self.cost)

        self.ceq = self.getConstraints(robot)
        # self.ceq = self.getConstraintsMPC(robot)
        robot.opti.subject_to(self.ceq)

        p_opts = {"expand": True, "error_on_fail": False}
        s_opts = {"max_iter": max_iter}
        robot.opti.solver("ipopt", p_opts, s_opts)
        self.InitialGuess(robot, seed=seed)
        pass

    def InitialGuess(self, Arm, seed=None):
        np.random.seed(self.random_seed)
        for i in range(Arm.NS):
            for j in range(3):
                Arm.opti.set_initial(Arm.dq[i][j], 0.0)
                pass
 
            Arm.opti.set_initial(Arm.q[i][0], 0.0)
            Arm.opti.set_initial(Arm.q[i][1], np.pi)
            Arm.opti.set_initial(Arm.q[i][2], 0.0)
            pass    

    def CostFunMPC(self, Arm):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        Torque = 0
        PosTar = 0
        VelTar = 0
        Dtorque = 0
        FM = [Arm.bound_fx[1], Arm.bound_fy[1]]
        for i in range(Arm.NS):
            Torque += (Arm.u[i][0]/Arm.motor_mt)**2 * Arm.dt * self.TorqueCoef[0]
            Torque += (Arm.u[i][1]/Arm.motor_mt)**2 * Arm.dt * self.TorqueCoef[1]
            if i > 0:
                Dtorque += (Arm.u[i][0]- Arm.u[i-1][0])**2 * Arm.dt * self.DtorqueCoef[0]
                Dtorque += (Arm.u[i][1]- Arm.u[i-1][1])**2 * Arm.dt * self.DtorqueCoef[1]
            if i< Arm.NS-1:
                Torque += (Arm.F[i][0]/FM[0])**2 * Arm.dt * self.SupportFCoef[0]
                Torque += (Arm.F[i][1]/FM[1])**2 * Arm.dt * self.SupportFCoef[1]

            PosTar += (Arm.q[i][0] - Arm.postar)**2 * Arm.dt * self.PostarCoef[0]
            PosTar += (Arm.q[i][1] - np.pi)**2 * Arm.dt * self.PostarCoef[1]
            PosTar += (Arm.q[i][2] - Arm.postar)**2 * Arm.dt * self.PostarCoef[2]

            VelTar += (Arm.dq[i][0] - Arm.veltar)**2 * Arm.dt * self.VeltarCoef[0]
            VelTar += (Arm.dq[i][1] - Arm.veltar)**2 * Arm.dt * self.VeltarCoef[1]
            VelTar += (Arm.dq[i][2] - Arm.veltar)**2 * Arm.dt * self.VeltarCoef[2]
            pass

        PosTar += (Arm.q[-1][0] - Arm.postar)**2 * self.PostarCoef[0]
        PosTar += (Arm.q[-1][1] - np.pi)**2 * self.PostarCoef[1]
        PosTar += (Arm.q[-1][2] - Arm.postar)**2 * self.PostarCoef[2]
        VelTar += (Arm.dq[-1][0] - Arm.veltar)**2 * self.VeltarCoef[0]
        VelTar += (Arm.dq[-1][1] - Arm.veltar)**2 * self.VeltarCoef[1]
        VelTar += (Arm.dq[-1][2] - Arm.veltar)**2 * self.VeltarCoef[2]

        return PosTar + Torque + VelTar + Dtorque

    def CostFunMPCHuman(self, Arm):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        Torque = 0
        AngMom = 0
        MassCent = 0
        Dtorque = 0
        PosTar = 0
        VelTar = 0
        FM = [Arm.bound_fx[1], Arm.bound_fy[1]]
        for i in range(Arm.NS):
            Torque += (Arm.u[i][0]/Arm.motor_mt)**2 * Arm.dt * self.TorqueCoef[0]
            Torque += (Arm.u[i][1]/Arm.motor_mt)**2 * Arm.dt * self.TorqueCoef[1]
            if i > 0:
                Dtorque += (Arm.u[i][0]- Arm.u[i-1][0])**2 * Arm.dt * self.DtorqueCoef[0]
                Dtorque += (Arm.u[i][1]- Arm.u[i-1][1])**2 * Arm.dt * self.DtorqueCoef[1]
            if i< Arm.NS-1:
                Torque += (Arm.F[i][0]/FM[0])**2 * Arm.dt * self.SupportFCoef[0]
                Torque += (Arm.F[i][1]/FM[1])**2 * Arm.dt * self.SupportFCoef[1]

            PosTar += (Arm.q[i][1] - np.pi)**2 * Arm.dt * self.PostarCoef[1]
            PosTar += (Arm.q[i][2] - Arm.postar)**2 * Arm.dt * self.PostarCoef[2]

            # VelTar += (Arm.dq[i][1] - Arm.veltar)**2 * Arm.dt * self.VeltarCoef[1]
            # VelTar += (Arm.dq[i][2] - Arm.veltar)**2 * Arm.dt * self.VeltarCoef[2]

            MI = Arm.I[0] * Arm.dq[i][0]
            AngMom += MI ** 2 * Arm.dt * self.PostarCoef[0]

            Mc0, Mc1, Mc2 = Arm.MassCenter(Arm.q[i])
            Mc = (Arm.m[0]*Mc0 + Arm.m[1]*Mc1 + Arm.m[2]*Mc2) / (Arm.m[0] + Arm.m[1] + Arm.m[2])
            MassCent += (1/Mc)**2 * Arm.dt * self.VeltarCoef[0]
            pass

        PosTar += (Arm.q[-1][1] - np.pi)**2 * self.PostarCoef[1]
        PosTar += (Arm.q[-1][2] - Arm.postar)**2 * self.PostarCoef[2]
        # VelTar += (Arm.dq[-1][0] - Arm.veltar)**2 * self.VeltarCoef[0]
        VelTar += (Arm.dq[-1][1] - Arm.veltar)**2 * self.VeltarCoef[1]
        VelTar += (Arm.dq[-1][2] - Arm.veltar)**2 * self.VeltarCoef[2]

        return Torque + Dtorque + AngMom + MassCent + PosTar + VelTar

    def getConstraints(self, Arm):                     
        ceq = []

        ## continuous dynamics constraints
        for i in range(Arm.NS - 1):
            ceq.extend([Arm.q[i+1][j] - Arm.q[i][j] - Arm.dt/2 * \
                        (Arm.dq[i+1][j] + Arm.dq[i][j]) == 0 for j in range(3)])

            Inertia = Arm.InertiaForce(Arm.q[i], Arm.ddq[i])
            Coriolis = Arm.Coriolis(Arm.q[i], Arm.dq[i])
            Gravity = Arm.Gravity(Arm.q[i])
            AccF = Arm.SupportForce(Arm.q[i], Arm.dq[i], Arm.ddq[i])

            ceq.extend([Inertia[0] + Gravity[0] + Coriolis[0] == 0])
            ceq.extend([Inertia[j+1] + Gravity[j+1] + Coriolis[j+1] - Arm.u[i][j] == 0 for j in range(2)])
            # ceq.extend([Arm.F[i][j] + AccF[j] == 0 for j in range(2)])
            pass
    
        ## Boundary constraints
        for temp_q in Arm.q:
            ceq.extend([Arm.opti.bounded(Arm.q_LB[j], temp_q[j], Arm.q_UB[j]) for j in range(3)])
            pass

        for temp_dq in Arm.dq:
            ceq.extend([Arm.opti.bounded(Arm.dq_LB[j], temp_dq[j], Arm.dq_UB[j]) for j in range(3)])
            pass

        for temp_u in Arm.u:
            ceq.extend([Arm.opti.bounded(Arm.u_LB[j], temp_u[j], Arm.u_UB[j]) for j in range(2)])
            pass

        # for temp_f in Arm.F:
        #     ceq.extend([Arm.opti.bounded(Arm.F_LB[j], temp_f[j], Arm.F_UB[j]) for j in range(2)])
        #     pass
        
        ## support force and friction
        for i in range(Arm.NS-1):
            # AccF = Arm.SupportForce(Arm.q[i], Arm.dq[i], Arm.ddq[i])
            # Fx = -AccF[0]
            # Fy = -AccF[1]
            # ceq.extend([Fy*Arm.Frict_coef - Fx >= 0.0])
            # ceq.extend([-Fy*Arm.Frict_coef - Fx <= 0.0])
            # ceq.extend([Fy >= 0])

            # ceq.extend([Arm.F[i][1]*Arm.Frict_coef - Arm.F[i][0] >= 0.0])
            # ceq.extend([-Arm.F[i][1]*Arm.Frict_coef - Arm.F[i][0] <= 0.0])
            pass

        ## motion smooth constraint
        for i in range(len(Arm.u) -1):
            ceq.extend([(Arm.u[i][j] - Arm.u[i+1][j])**2 <= 100 for j in range(2)])
            pass

        ceq.extend([Arm.q[0][0]==self.x0[0]])
        ceq.extend([Arm.q[0][1]==self.x0[1]])
        ceq.extend([Arm.q[0][2]==self.x0[2]])

        ceq.extend([Arm.dq[0][0]==self.dq0[0]])
        ceq.extend([Arm.dq[0][1]==self.dq0[1]])
        ceq.extend([Arm.dq[0][2]==self.dq0[2]])

        return ceq

    def getConstraintsMPC(self, Arm):
        ceq = []

        ## continuous dynamics constraints
        for i in range(Arm.NS - 1):
            ceq.extend([Arm.q[i+1][j] - Arm.q[i][j] - Arm.dt/2 * \
                        (Arm.dq[i+1][j] + Arm.dq[i][j]) == 0 for j in range(3)])

            Inertia = Arm.InertiaForce(Arm.q[i], Arm.ddq[i])
            Coriolis = Arm.Coriolis(Arm.q[i], Arm.dq[i])
            Gravity = Arm.Gravity(Arm.q[i])

            ceq.extend([Inertia[0] + Gravity[0] + Coriolis[0] == 0])
            ceq.extend([Inertia[j+1] + Gravity[j+1] + Coriolis[j+1] - Arm.u[i][j] == 0 for j in range(2)])
            pass
    
        ## Boundary constraints
        for temp_q in Arm.q:
            ceq.extend([Arm.opti.bounded(Arm.q_LB[j], temp_q[j], Arm.q_UB[j]) for j in range(3)])
            pass

        for temp_dq in Arm.dq:
            ceq.extend([Arm.opti.bounded(Arm.dq_LB[j], temp_dq[j], Arm.dq_UB[j]) for j in range(3)])
            pass

        for temp_u in Arm.u:
            ceq.extend([Arm.opti.bounded(Arm.u_LB[j], temp_u[j], Arm.u_UB[j]) for j in range(2)])
            pass

        ## motion smooth constraint
        for i in range(len(Arm.u) -1):
            ceq.extend([(Arm.u[i][j] - Arm.u[i+1][j])**2 <= 4 for j in range(2)])
            pass
        
        ceq.extend([Arm.q[0][0]==self.x0[0]])
        ceq.extend([Arm.q[0][1]==self.x0[1]])
        ceq.extend([Arm.q[0][2]==self.x0[2]])

        # ceq.extend([Arm.dq[0][0]==self.dq0[0]])
        # ceq.extend([Arm.dq[0][1]==self.dq0[1]])
        # ceq.extend([Arm.dq[0][2]==self.dq0[2]])

        return ceq

    def Solve_StateReturn(self, robot):
        u = []
        ddq = []
        try:
            sol = robot.opti.solve()
            u.append([sol.value(robot.u[0][j]) for j in range(2)])
            ddq.append([sol.value(robot.ddq[0][j]) for j in range(3)])
            pass
        except:
            value = robot.opti.debug.value
            u.append([value(robot.u[0][j]) for j in range(2)])
            ddq.append([value(robot.ddq[0][j]) for j in range(3)])
            pass

        return u[0], ddq[0]

    def Solve_StateReturn2(self, robot):
        u = []
        ddq = []
        try:
            sol = robot.opti.solve()
            for i in range(self.Nc):
                u.append([sol.value(robot.u[i][j]) for j in range(2)])
                ddq.append([sol.value(robot.ddq[0][j]) for j in range(3)])
            pass
        except:
            value = robot.opti.debug.value
            for i in range(self.Nc):
                u.append([value(robot.u[i][j]) for j in range(2)])
                ddq.append([value(robot.ddq[0][j]) for j in range(3)])
            pass

        return u, ddq

    def Solve_Output(self, robot, flag_save=True, StorePath="./"):
        # solve the nlp and stroge the solution
        q = []
        dq = []
        ddq = []
        u = []
        t = []

        try:
            sol = robot.opti.solve()
            for i in range(robot.NS):
                t.append(i*robot.dt)
                q.append([sol.value(robot.q[i][j]) for j in range(3)])
                dq.append([sol.value(robot.dq[i][j]) for j in range(3)])
                ddq.append([sol.value(robot.ddq[i][j]) for j in range(3)])
                u.append([sol.value(robot.u[i][j]) for j in range(2)])
                pass
            pass
        except:
            value = robot.opti.debug.value
            for i in range(robot.NS):
                t.append(i*robot.dt)
                q.append([value(robot.q[i][j]) for j in range(3)])
                dq.append([value(robot.dq[i][j]) for j in range(3)])
                ddq.append([value(robot.ddq[i][j]) for j in range(3)])
                u.append([value(robot.u[i][j]) for j in range(2)])
                if i == 0:
                    print([value(robot.u[i][j]) for j in range(2)])
                pass
            pass
        finally:
            q = np.asarray(q)
            dq = np.asarray(dq)
            ddq = np.asarray(ddq)
            u = np.asarray(u)
            t = np.asarray(t).reshape([-1, 1])

            ML = self.cfg["Optimization"]["MaxLoop"] / 1000

            if flag_save:
                date = time.strftime("%Y-%m-%d-%H-%M-%S")
                name = "-Pos_"+str(self.PostarCoef)+"-Tor_"+str(self.TorqueCoef) \
                     + "-dt_"+str(robot.dt)+"-T_"+str(robot.Tp)+"-ML_"+str(ML)+ "k"
                np.save(StorePath+date+name+"-sol.npy",
                        np.hstack((q, dq, ddq, u, t)))
                # output the config yaml file
                # with open(os.path.join(StorePath, date + name+"-config.yaml"), 'wb') as file:
                #     yaml.dump(self.cfg, file)
                with open(StorePath+date+name+"-config.yaml", mode='w') as file:
                    YAML().dump(self.cfg, file)
                pass

            return q, dq, ddq, u, t


class MPC():
    def __init__(self, cfg, q0, dq0, u1):
        self.cfg = cfg
        self.q0 = q0
        self.u1 = u1
        self.dq0 = dq0
        self.t = cfg['Controller']['dt']
        pass

    def updateState(self, robot):
        mass_matrix = robot.MassMatrix(self.q0)
        mass_matrix = np.asarray(mass_matrix)
        mass_inv = np.linalg.inv(mass_matrix)
        m0 = robot.m[0]
        m1 = robot.m[1]
        m2 = robot.m[2]
        lc0 = robot.l[0]
        lc1 = robot.l[1]
        lc2 = robot.l[2]
        L0 = robot.L[0]
        L1 = robot.L[1]
        L2 = robot.L[2]

        def odefun(y, t):
            q1, q2, q3, dq1, dq2, dq3 = y
            corilios = robot.Coriolis([q1, q2, q3], [dq1, dq2, dq3])
            gravity = robot.Gravity([q1, q2, q3])
            dydt = [dq1, dq2, dq3,
                    mass_inv[0][1]*self.u1[0] + mass_inv[0][2]*self.u1[1] - mass_inv[0][0]*(corilios[0]+gravity[0]) - mass_inv[0][1]*(corilios[1]+gravity[1]) - mass_inv[0][2]*(corilios[2]+gravity[2]),
                    mass_inv[1][1]*self.u1[0] + mass_inv[1][2]*self.u1[1] - mass_inv[1][0]*(corilios[0]+gravity[0]) - mass_inv[1][1]*(corilios[1]+gravity[1]) - mass_inv[1][2]*(corilios[2]+gravity[2]),
                    mass_inv[2][1]*self.u1[0] + mass_inv[2][2]*self.u1[1] - mass_inv[2][0]*(corilios[0]+gravity[0]) - mass_inv[2][1]*(corilios[1]+gravity[1]) - mass_inv[2][2]*(corilios[2]+gravity[2])]
            
            # C1 = -2*L0*(L1*m2*s(q2) + lc2*m2*s(q2+q3) + lc1*m1*s(q2)) * dq1*dq2 \
            # - 2*lc2*m2*(L0*s(q2+q3) + L1*s(q3))* dq1*dq3 \
            # - L0*(L1*m2*s(q2) + lc2*m2*s(q1+q3) + lc1*m1*s(q2)) * dq2*dq2 \
            # - 2*lc2*m2*(L0*s(q2+q3) + L1*s(q3))* dq2*dq3 \
            # - lc2*m2*(L0*s(q2+q3) + L1*s(q3))* dq3*dq3

            # C2 = L0*(L1*m2*s(q2) + lc2*m2*s(q2+q3) + lc1*m1*s(q2)) * dq1*dq1 \
            #     - 2*L1*lc2*m2*s(q3) * dq1*dq3 \
            #     - 2*L1*lc2*m2*s(q3) * dq2*dq3 \
            #     - L1*lc2*m2*s(q3) * dq3*dq3

            # C3 = lc2*m2*(L0*s(q2+q3) + L1*s(q3))* dq1*dq1 \
            #     + 2*L1*lc2*m2*s(q3) * dq1*dq2 \
            #     + L1*lc2*m2*s(q3) * dq2*dq2 

            # G1 = -(L0*m1*s(q1) + L0*m2*s(q1) + L1*m2*s(q1+q2) + \
            #     lc0*m0*s(q1) + lc1*m1*s(q1+q2) + lc2*m2*s(q1+q2+q3)) *robot.g
            
            # G2 = -(L1*m2*s(q1+q2) + lc1*m1*s(q1+q2) + lc2*m2*s(q1+q2+q3)) *robot.g

            # G3 = -lc2*m2*s(q1+q2+q3) *robot.g
            # dydt1 = [dq1, dq2, dq3,
            #         q1]
            return dydt

        q_init = []
        q_init.extend(self.q0)
        q_init.extend(self.dq0)

        t_ode = [0.0, 0.0025, 0.005, 0.0075, 0.01]

        sol = odeint(odefun, q_init, t_ode)

        # solres = sol.tolist()

        return sol

class DynamicsAnalysis():
    def __init__(self, cfg):
        self.cfg = cfg
        pass

    def StepResponse(self):
        Robot = RobotProperty(self.cfg)
        MassMatrix = Robot.getMassMatrix(0, np.pi, 0)
        Gravity = Robot.getGravityLinearMatrix()
        pass

## trajectory optimizaition main function
def main():
    # region optimization trajectory for bipedal hybrid robot system
    vis_flag = True
    ani_flag = True
    save_flag = False
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    ani_path = StorePath + "/data/animation/"
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # region load config file
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    ## initial state setting
    q0 = [0.1, 3.3, -0.5]
    dq0 = [0.0, 0.0, 0.0]

    #create robot and NLP problem
    robot = TriplePendulum(cfg)
    # nonlinearOptimization = nlp(robot, cfg, seed=seed)
    nonlinearOptimization = NLP(robot, cfg, q0, dq0)
    q, dq, ddq, u, t = nonlinearOptimization.Solve_Output(
        robot, flag_save=save_flag, StorePath=save_dir)

    DataProcess.DataPlot(q, dq, u, t, vis_flag)

    if ani_flag:
        DataProcess.animation(q, dq, u, t, robot, ani_path, cfg, 0, 0)

    pass

## mpc control main function
def MPC_main():
    ## file path 
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    ani_path = StorePath + "/data/animation/"
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    ## region load config file
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    ## controller paramters
    dt = cfg['Controller']['dt']        # sample time
    T = cfg['Controller']['T']          # predictive time
    N = int(T / dt)                     # samples number

    ## initial state setting
    q0 = [0.1, 3.3, -0.5]
    # q0 = [0.2, 3.2, -0.5]
    dq0 = [0.0, 0.0, 0.0]

    ## create robot and NLP problem
    # robot = TriplePendulum(cfg)
    q = []
    dq = []
    tor = []
    t = []
    time_start = time.time() 
    for i in range(N):
        ## create NLP problem
        robot = TriplePendulum(cfg)
        nonlinearOptimization = NLP(robot, cfg, q0, dq0)

        ## get the first input of the optimal sequence
        u1 = nonlinearOptimization.Solve_StateReturn(robot)

        ## state save
        t.append(i*robot.dt)
        q.append(q0)
        dq.append(dq0)
        tor.append(u1)
        # print(q0)
        # print(tor)

        ## state update accoding to current state and input
        mpccontroller = MPC(cfg, q0, dq0, u1)
        sol = mpccontroller.updateState(robot)
        # sol2 = sol.tolist()
        q0 = sol[-1, 0:3]
        dq0 = sol[-1, 3:6]
        q0 = q0.tolist()
        dq0 = dq0.tolist()
        print("=================================")
        print("the number of receding optimia: ", i, "/", N)
        time_end = time.time()
        time_run = (time_end - time_start) / 60
        print("whole running time of mpc is: ", time_run , " min")

        # if i == 1:
        #     print(q)
        #     # print(q[:,1])
        #     print(sol, q0, dq0)
        #     break

    time_run = (time_end - time_start) / 60
    print("="*50)
    print("whole running time of mpc is: ", time_run , " min")

    q = np.asarray(q)
    dq = np.asarray(dq)
    tor = np.asarray(tor)
    t = np.asarray(t).reshape([-1, 1])
    
    print(t.shape, q.shape)

    # region visualization
    fig_flag = True
    ani_flag = True
    save_flag = True
    # fig_flag = False
    # ani_flag = False
    # save_flag = False
    visual = DataProcess(cfg, robot, q, dq, tor, t, save_dir)
    visual.DataPlot(fig_flag)
    visual.animation(0, ani_flag)
    visual.DataSave(save_flag)
    # endregion

def Sim_main():
    # region filepath and configure file load
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    ani_path = save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)
    cfg = ParamData

    raisim.World.setLicenseFile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/activation.raisim")
    # TIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/TIP.urdf"
    # TIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/TIP_low.urdf"
    TIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/TIP_high_inertia.urdf"
    # TIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/TIP_params.urdf"
    world = raisim.World()
    # endregion

    # region raisim ecvsetting
    t_step = ParamData["Environment"]["t_step"] 
    sim_time = ParamData["Environment"]["sim_time"]
    world.setTimeStep(t_step)
    ground = world.addGround(0)

    gravity = world.getGravity()
    print(gravity,t_step)
    UrdfParams = RobotInterface.LoadUrdfParam(cfg)
    # TIP = world.addArticulatedSystem(TIP_urdf, UrdfParams)
    TIP = world.addArticulatedSystem(TIP_urdf)
    TIP.setName("TIP")
    print(TIP.getDOF())

    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)
    # endregion

    # region controller initial params setting
    dt = cfg['Controller']['dt']        # sample time
    T = cfg['Controller']['T']          # predictive time
    Nc = cfg['Controller']['Nc']        # control time
    N = int(T / dt)                     # samples number

    # q0 = [0.1, 3.3, -0.5]
    q0 = cfg["Robot"]["q_init"]["q0"]
    dq0 = cfg["Robot"]["q_init"]["dq0"]
    jointNominalConfig = np.array([q0[0], q0[1], q0[2]])
    jointVelocityTarget = np.array([dq0[0], dq0[1], dq0[2]])
    TIP.setGeneralizedCoordinate(jointNominalConfig)
    TIP.setGeneralizedVelocity(jointVelocityTarget)

    # create robot and NLP problem
    robot = TriplePendulum(cfg)
    # endregion

    # region data setting
    q = []
    dq = []
    ddq = []
    t = []
    u = []
    # endregion
    
    time_start = time.time() 

    # region mpc main loop
    for i in range(N):
        # time.sleep(0.01)
        TIP.updateMassInfo() 
        JointPos, JointVel = TIP.getState()
        JointPos = JointPos.tolist()
        JointVel = JointVel.tolist()

        robot = TriplePendulum(cfg)
        nonlinearOptimization = NLP(robot, cfg, JointPos, JointVel)
        tor, ddq0 = nonlinearOptimization.Solve_StateReturn(robot)

        u1 = tor[0]
        u2 = tor[1]
        u_temp = [u1, u2]
        ddq_temp = [ddq0[0], ddq0[1], ddq0[2]]

        t.append((i)*robot.dt)
        q.append(JointPos)
        dq.append(JointVel)
        ddq.append(ddq_temp)
        u.append(u_temp)

        print("=================================")
        print("the number of receding optimia: ", i, "/", N)
        time_end = time.time()
        time_run = (time_end - time_start) / 60
        print("whole running time of mpc is: ", time_run , " min")

        TIP.setGeneralizedForce([0.0, u1, u2])

        server.integrateWorldThreadSafe()
        pass
    # endregion

    # region mpc multi Tc loop
    # Nc_flag = 0
    # for i in range(N):
    #     # time.sleep(0.01)
    #     TIP.updateMassInfo() 
    #     JointPos, JointVel = TIP.getState()
    #     JointPos = JointPos.tolist()
    #     JointVel = JointVel.tolist()
    #     if Nc_flag == Nc:
    #         Nc_flag = 0
    #     if Nc_flag == 0:
    #         robot = TriplePendulum(cfg)
    #         nonlinearOptimization = NLP(robot, cfg, JointPos, JointVel)
    #         tor, ddq0 = nonlinearOptimization.Solve_StateReturn2(robot)
    #         # print(tor)
    #         u1 = tor[Nc_flag][0]
    #         u2 = tor[Nc_flag][1]
    #         Nc_flag += 1
    #     elif Nc_flag > 0 and Nc_flag < Nc:
    #         u1 = tor[Nc_flag][0]
    #         u2 = tor[Nc_flag][1]
    #         Nc_flag += 1
    #         pass
    #     if Nc_flag <= Nc:
    #         ddq.append([ddq0[Nc_flag-1][0], ddq0[Nc_flag-1][1], ddq0[Nc_flag-1][2]])
    #     u_temp = [u1, u2]

    #     t.append((i)*robot.dt)
    #     q.append(JointPos)
    #     dq.append(JointVel)
    #     u.append(u_temp)

    #     print("=================================")
    #     print("the number of receding optimia: ", i, "/", N)
    #     time_end = time.time()
    #     time_run = (time_end - time_start) / 60
    #     print("whole running time of mpc is: ", time_run , " min")

    #     TIP.setGeneralizedForce([0.0, u1, u2])

    #     server.integrateWorldThreadSafe()
    #     pass
    # endregion
    time_end = time.time()
    
    # region data process
    q = np.asarray(q)
    dq = np.asarray(dq)
    ddq = np.asarray(ddq)
    tor = np.asarray(u)
    t = np.asarray(t).reshape([-1, 1])
    # endregion

    # region params print 
    time_run = (time_end - time_start) / 60
    print("="*50)
    print("whole running time of mpc is: ", time_run , " min")
    # endregion

    # region visualization
    fig_flag = True
    ani_flag = True
    save_flag = True
    # fig_flag = False
    # ani_flag = False
    # save_flag = False
    visual = DataProcess(cfg, robot, q, dq, ddq, tor, t, save_dir)
    visual.DataPlot(fig_flag)
    visual.ForceAnalysis(fig_flag)
    visual.SupportForce(fig_flag)
    visual.MomentumAnalysis(fig_flag)
    visual.PowerAnalysis(fig_flag)
    visual.animation(0, ani_flag)
    visual.DataSave(save_flag)

    # endregion
    pass

def DataVisual():
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(FilePath)
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    DataFile = FilePath + "/data/2022-04-21/X-2022-04-21-18-00-29-MPC-Pos_50-Tor_5-Vel_20-dt_0.02-T_5.0-Tp_0.8-Tc_1-ML_0.1k/2022-04-21-18-00-29-MPC-Pos_50-Tor_5-Vel_20-dt_0.02-T_5.0-Tp_0.8-Tc_1-ML_0.1k-sol.npy"
    Data = np.load(DataFile)
    q = Data[:, 0:3]
    dq = Data[:, 3:6]
    ddq = Data[:, 6:9]
    u = Data[:, 9:11]
    t = Data[:, 11]
    print(max(t))

    todaytime=datetime.date.today()
    save_dir = FilePath + "/data/" + str(todaytime) + "/"
    robot = TriplePendulum(cfg)
    visual = DataProcess(cfg, robot, q, dq, ddq, u, t, save_dir)
    visual.DataPlot()
    visual.SupportForce()
    # visual.ForceAnalysis()
    # visual.PowerAnalysis()
    # visual.MomentumAnalysis()
    pass

def DataResLoad(Data):
    q = Data[:, 0:3]
    dq = Data[:, 3:6]
    ddq = Data[:, 6:9]
    u = Data[:, 9:11]
    t = Data[:, 11]
    return q, dq, ddq, u, t

def ParamsAnalysis():
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(FilePath)
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    MaxTor_m = []
    MaxTime_m = []
    MaxAngle_m = []
    MaxTor_r = []
    MaxTime_r = []
    MaxAngle_r = []
    q_ref = 0.04 * 0.2
    mM = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    Ir = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 8.0, 10.0, 20.0]
    # Ir = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(mM)):
        DataFile_m = FilePath + "/data/2022-04-25/mM_" + str(mM[i]) + "/mM_" + str(mM[i]) + "-sol.npy"
        Data_m = np.load(DataFile_m)
        q_m, dq_m, ddq_m, u_m, t_m = DataResLoad(Data_m)
        for j in range(len(t_m)):
            if j >0:
                ## min recovery time
                if q_m[j][0] <= q_ref and q_m[j-1][0] > q_ref:
                    MaxTime_m.append(t_m[j])
        
        ## max theta 2 angle
        theta2_m = q_m[:, 1]
        theta2_m_max = max(theta2_m)
        MaxAngle_m.append(theta2_m_max)

        ## max tor 1, 2
        tor1_m = u_m[:, 0]
        tor2_m = u_m[:, 1]
        tor1_m_max = max(tor1_m)
        tor2_m_max = max(tor2_m)
        MaxTor_m.append([tor1_m_max, tor2_m_max])
    
    for i in range(len(Ir)):
        DataFile_I = FilePath + "/data/2022-04-25/Ir_" + str(Ir[i]) + "/Ir_" + str(Ir[i]) + "-sol.npy"
        Data_r = np.load(DataFile_I)
        q_r, dq_r, ddq_r, u_r, t_r = DataResLoad(Data_r)
        flag = 0
        for j in range(len(t_r)):
            if j >0:
                ## min recovery time
                if q_r[j][0] <= q_ref and q_r[j-1][0] > q_ref:
                    if flag == 0:
                        MaxTime_r.append(t_r[j])
                    else:
                        MaxTime_r[-1] = t_r[j]
                    print(i, t_r[j])
                    flag = 1

        ## max theta 2 angle
        theta2_r = q_r[:, 1]
        theta2_r_max = max(theta2_r)
        MaxAngle_r.append(theta2_r_max)

        ## max tor 1, 2
        tor1_r = u_r[:, 0]
        tor2_r = u_r[:, 1]
        tor1_r_max = max(tor1_r)
        tor2_r_max = max(tor2_r)
        MaxTor_r.append([tor1_r_max, tor2_r_max])

    MaxTor_m = np.asarray(MaxTor_m)
    MaxTor_r = np.asarray(MaxTor_r)
    
    fig, axes = plt.subplots(2,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]
    x_index = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '2.0', '5.0', '8.0', '10.0','20.0']
    values = range(len(Ir))
    ax1.plot(mM, MaxTime_m, label="Settling time", linewidth = 3)

    ax1.set_xlabel('M_arm / M_body', fontsize = 18)
    ax1.set_ylabel('Settling time(s) ', fontsize = 20)
    ax1.xaxis.set_tick_params(labelsize = 20)
    ax1.yaxis.set_tick_params(labelsize = 20)
    ax1.legend(loc='lower right', fontsize = 20)
    # ax1.grid()

    ax2.plot(values, MaxTime_r, label="Settling time", linewidth = 3)

    ax2.set_xlabel('I_arm / I_body ', fontsize = 18)
    ax2.set_ylabel('Settling time(s) ', fontsize = 20)
    ax2.xaxis.set_tick_params(labelsize = 20)
    ax2.xaxis.set_ticks(values)
    ax2.xaxis.set_ticklabels(x_index)
    ax2.yaxis.set_tick_params(labelsize = 20)
    ax2.legend(loc='upper right', fontsize = 20)
    # ax2.grid()
    plt.show()

    fig, axes = plt.subplots(2,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]
    ax1.plot(mM, MaxAngle_m, label="Max Joint 2 angle", linewidth = 3)

    ax1.set_xlabel('M_arm / M_body', fontsize = 18)
    ax1.set_ylabel('Angle(rad)', fontsize = 20)
    ax1.xaxis.set_tick_params(labelsize = 20)
    ax1.yaxis.set_tick_params(labelsize = 20)
    ax1.legend(loc='lower right', fontsize = 20)
    # ax1.grid()

    ax2.plot(values, MaxAngle_r, label="Max Joint 2 angle", linewidth = 3)

    ax2.set_xlabel('I_arm / I_body ', fontsize = 18)
    ax2.set_ylabel('Angle(rad) ', fontsize = 20)
    ax2.xaxis.set_tick_params(labelsize = 20)
    ax2.xaxis.set_ticks(values)
    ax2.xaxis.set_ticklabels(x_index)
    ax2.yaxis.set_tick_params(labelsize = 20)
    ax2.legend(loc='upper right', fontsize = 20)
    # ax2.grid()
    plt.show()

    fig, axes = plt.subplots(2,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]
    ax1.plot(mM, MaxTor_m[:, 0], label="Max Joint 2 Torque", linewidth = 3)
    ax1.plot(mM, MaxTor_m[:, 1], label="Max Joint 3 Torque", linewidth = 3)

    ax1.set_xlabel('M_arm / M_body', fontsize = 18)
    ax1.set_ylabel('Torque(N.m) ', fontsize = 20)
    ax1.xaxis.set_tick_params(labelsize = 20)
    ax1.yaxis.set_tick_params(labelsize = 20)
    ax1.legend(loc='lower right', fontsize = 20)
    # ax1.grid()

    ax2.plot(values, MaxTor_r[:, 0], label="Max Joint 2 Torque", linewidth = 3)
    ax2.plot(values, MaxTor_r[:, 1], label="Max Joint 3 Torque", linewidth = 3)

    ax2.set_xlabel('I_arm / I_body ', fontsize = 18)
    ax2.set_ylabel('Torque(N.m) ', fontsize = 25)
    ax2.xaxis.set_tick_params(labelsize = 20)
    ax2.xaxis.set_ticks(values)
    ax2.xaxis.set_ticklabels(x_index)
    ax2.yaxis.set_tick_params(labelsize = 20)
    ax2.legend(loc='upper right', fontsize = 20)
    # ax2.grid()
    plt.show()
    
    pass

if __name__ == "__main__":
    # main()
    # visualization()
    # MPC_main()
    Sim_main()
    # DataVisual()
    # DataAnalysis()
    pass