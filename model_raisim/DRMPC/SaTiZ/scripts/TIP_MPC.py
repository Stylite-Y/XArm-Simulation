"""
三级摆的平衡控制
"""

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
from scipy.integrate import odeint
from Dynamics_MPC import RobotProperty

class TriplePendulum():
    def __init__(self, cfg):
        self.opti = ca.Opti()
        self.dt = cfg['Controller']['dt']       # sample time
        self.Tp = cfg['Controller']['Tp']       # predictive time
        self.NS = int(self.Tp / self.dt)        # samples number
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
        self.u_LB = [-self.motor_mt] * 3
        self.u_UB = [self.motor_mt] * 3

        self.q_LB = [cfg['Controller']['Boundary']['theta1'][0],
                     cfg['Controller']['Boundary']['theta2'][0],
                     cfg['Controller']['Boundary']['theta3'][0]]
        self.q_UB = [cfg['Controller']['Boundary']['theta1'][1],
                     cfg['Controller']['Boundary']['theta2'][1],
                     cfg['Controller']['Boundary']['theta3'][1]]

        self.dq_LB = [-self.motor_ms, -self.motor_ms, -self.motor_ms]

        self.dq_UB = [self.motor_ms, self.motor_ms, self.motor_ms]

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
        self.ddq.append(self.ddq[0]) 
        # self.ddq.append([(self.dq[i+1]-self.dq[i]) /
        #                 self.dt for i in range(self.NS-1)])
        # self.u = []
        self.u = [self.opti.variable(2) for _ in range(self.NS)]
        # self.u.append([self.opti.variable(2) for _ in range(self.NS-1)])

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
        # C1 = C1 - 1.0 * dq[0] 
        
        C2 = L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[0]*dq[0] \
            - 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[2] \
            - 2*L1*lc2*m2*s(q[2]) * dq[1]*dq[2] \
            - L1*lc2*m2*s(q[2]) * dq[2]*dq[2]

        C3 = lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[0]*dq[0] \
            + 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[1] \
            + L1*lc2*m2*s(q[2]) * dq[1]*dq[1]

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


class NLP():
    def __init__(self, robot, cfg, x0, dq0, seed=None):
        self.x0 = x0
        self.dq0 = dq0
        self.cfg = cfg
        self.TorqueCoef = cfg["Optimization"]["CostCoef"]["torqueCoef"]
        self.PostarCoef = cfg["Optimization"]["CostCoef"]["postarCoef"]
        self.VeltarCoef = cfg["Optimization"]["CostCoef"]["VeltarCoef"]
        max_iter = cfg["Optimization"]["MaxLoop"]
        self.random_seed = cfg["Optimization"]["RandomSeed"]
        self.dt = cfg['Controller']['dt']       # sample time
        self.Nc = cfg['Controller']['Nc']       # control time

        self.cost = self.CostFun(robot)
        # self.cost = self.CostFunMPC(robot)
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

    def CostFun(self, Arm):
        Torque = 0
        PosTar = 0
        VelTar = 0
        for i in range(Arm.NS):
            for j in range(2):
                Torque += ca.fabs(Arm.u[i][j]/Arm.motor_mt) * Arm.dt
                pass
            PosTar += ca.fabs(Arm.q[i][0] - Arm.postar) * Arm.dt
            PosTar += ca.fabs(Arm.q[i][1] - np.pi) * Arm.dt
            PosTar += ca.fabs(Arm.q[i][2] - Arm.postar) * Arm.dt
            VelTar += ca.fabs(Arm.dq[i][0] - Arm.veltar) * Arm.dt
            VelTar += ca.fabs(Arm.dq[i][1] - Arm.veltar) * Arm.dt
            VelTar += ca.fabs(Arm.dq[i][2] - Arm.veltar) * Arm.dt
            pass

        PosTar += ca.fabs(Arm.q[-1][0] - Arm.postar)
        PosTar += ca.fabs(Arm.q[-1][1] - np.pi)
        PosTar += ca.fabs(Arm.q[-1][2] - Arm.postar)
        VelTar += ca.fabs(Arm.dq[-1][0] - Arm.veltar)
        VelTar += ca.fabs(Arm.dq[-1][ 1] - Arm.veltar)
        VelTar += ca.fabs(Arm.dq[-1][2] - Arm.veltar)

        return PosTar * self.PostarCoef + Torque * self.TorqueCoef + VelTar * self.VeltarCoef
        # return PosTar * self.PostarCoef + Torque * self.TorqueCoef

    def CostFunMPC(self, Arm):
        Torque = 0
        PosTar = 0
        VelTar = 0
        for i in range(Arm.NS):
            for j in range(2):
                Torque += ca.fabs(Arm.u[i][j]/Arm.motor_mt) * Arm.dt
                pass
            PosTar += ca.fabs(Arm.q[i][0] - Arm.postar) * Arm.dt
            PosTar += ca.fabs(Arm.q[i][1] - np.pi) * Arm.dt
            # PosTar += ca.fabs(Arm.q[i][2] - Arm.postar) * Arm.dt
            VelTar += ca.fabs(Arm.dq[i][0] - Arm.veltar) * Arm.dt
            VelTar += ca.fabs(Arm.dq[i][1] - Arm.veltar) * Arm.dt
            VelTar += ca.fabs(Arm.dq[i][2] - Arm.veltar) * Arm.dt
            pass

        # PosTar += ca.fabs(Arm.q[-1][0] - Arm.postar)
        # PosTar += ca.fabs(Arm.q[-1][1] - np.pi)
        # # PosTar += ca.fabs(Arm.q[-1][2] - Arm.postar)
        # VelTar += ca.fabs(Arm.dq[-1][0] - Arm.veltar)
        # VelTar += ca.fabs(Arm.dq[-1][1] - Arm.veltar)
        # VelTar += ca.fabs(Arm.dq[-1][2] - Arm.veltar)

        return PosTar * self.PostarCoef + Torque * self.TorqueCoef + VelTar / 10 * self.PostarCoef
        # return PosTar * self.PostarCoef + Torque * self.TorqueCoef

    def getConstraints(self, Arm):
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
            ceq.extend([ca.fabs(Arm.u[i][j] - Arm.u[i+1][j]) <= 10 for j in range(2)])
            pass
        
        # ceq.extend([Arm.q[0][0]==0.1])
        # ceq.extend([Arm.q[0][1]==3.2])
        # ceq.extend([Arm.q[0][2]==-0.2])

        # ceq.extend([Arm.q[0][0]==0])
        # ceq.extend([Arm.q[0][1]==0])
        # ceq.extend([Arm.q[0][2]==0])

        # ceq.extend([Arm.q[0][0]==0.1])
        # ceq.extend([Arm.q[0][1]==-0.2])
        # ceq.extend([Arm.q[0][2]==0.3])

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
            ceq.extend([ca.fabs(Arm.u[i][j] - Arm.u[i+1][j]) <= 2 for j in range(2)])
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
        try:
            sol = robot.opti.solve()
            u.append([sol.value(robot.u[0][j]) for j in range(2)])
            pass
        except:
            value = robot.opti.debug.value
            u.append([value(robot.u[0][j]) for j in range(2)])
            pass

        return u[0]

    def Solve_StateReturn2(self, robot):
        u = []
        try:
            sol = robot.opti.solve()
            for i in range(self.Nc):
                u.append([sol.value(robot.u[i][j]) for j in range(2)])
            pass
        except:
            value = robot.opti.debug.value
            for i in range(self.Nc):
                u.append([value(robot.u[i][j]) for j in range(2)])
            pass

        return u

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

class DataProcess():
    def __init__(self, cfg, robot, q, dq, u, t, savepath):
        self.cfg = cfg
        self.robot = robot
        self.q = q
        self.dq = dq
        self.u = u
        self.t = t
        self.savepath = savepath

        self.ML = self.cfg["Optimization"]["MaxLoop"] / 1000
        self.Tp = self.cfg['Controller']['Tp']
        self.Nc = self.cfg['Controller']['Nc']
        self.T = self.cfg['Controller']['T']
        self.PostarCoef = self.cfg["Optimization"]["CostCoef"]["postarCoef"]
        self.TorqueCoef = self.cfg["Optimization"]["CostCoef"]["torqueCoef"]
        self.VeltarCoef = cfg["Optimization"]["CostCoef"]["VeltarCoef"]

        self.save_dir, self.name, self.date = self.DirCreate()
        pass

    def visualization(q, dq, u, t, robot, flag):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque

        if flag:

            FilePath = "/home/stylite-y/Documents/Master/Manipulator/Simulation/model_raisim/DRMPC/SaTiZ_3D/data/2022-04-06" 
            config_file = FilePath + "/2022-04-06-17-59-06-Pos_1-Tor_0.1-dt_0.01-T_2-ML_1.0k-config.yaml"
            datafile = FilePath + "/2022-04-06-17-59-06-Pos_1-Tor_0.1-dt_0.01-T_2-ML_1.0k-sol.npy"

            data = np.load(datafile)
            cfg = YAML().load(open(config_file, 'r'))

            print(data.shape)
            q = data[:, 0:3]
            dq = data[:, 3:6]
            ddq = data[:, 6:9]
            u = data[:, 9:11]
            t = data[:, 11:14]

        else:
            pass
        fig, axes = plt.subplots(2,1, dpi=100,figsize=(12,10))
        ax1 = axes[0]
        ax2 = axes[1]

        ax1.plot(t, q[:, 0], label="theta 1")
        ax1.plot(t, q[:, 1], label="theta 2")
        ax1.plot(t, q[:, 2], label="theta 3")

        ax1.set_ylabel('Angular ', fontsize = 15)
        ax1.legend(loc='upper right', fontsize = 12)
        ax1.grid()

        ax2.plot(t, u[:, 0], label="torque 1")
        ax2.plot(t, u[:, 1], label="torque 2")
        ax2.set_ylabel('Torque ', fontsize = 15)
        ax2.legend(loc='upper right', fontsize = 12)
        ax2.grid()

        plt.show()
        
        pass

    def DirCreate(self):
        date = time.strftime("%Y-%m-%d-%H-%M-%S")
        dirname = "-MPC-Pos_"+str(self.PostarCoef)+"-Tor_"+str(self.TorqueCoef) +"-Vel_"+str(self.VeltarCoef)\
                + "-dt_"+str(self.robot.dt)+"-T_"+str(self.T)+"-Tp_"+str(self.Tp)+"-Tc_"+str(self.Nc)+"-ML_"+str(self.ML)+ "k" 

        save_dir = self.savepath + date + dirname+ "/"

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        return save_dir, dirname, date

    def DataPlot(self, saveflag):
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        ax1.plot(self.t, self.q[:, 0], label="theta 1")
        # ax1.plot(t, q[:, 1], label="theta 2")
        ax1.plot(self.t, self.q[:, 2], label="theta 3")

        ax11 = ax1.twinx()
        ax11.plot(self.t, self.q[:, 1], color='forestgreen', label="theta 2")
        # ax11.plot(t, q[:, 1], color='mediumseagreen', label="theta 2")
        ax11.legend(loc='lower right', fontsize = 12)
        ax11.yaxis.set_tick_params(labelsize = 12)

        ax1.set_ylabel('Angle ', fontsize = 15)
        ax1.xaxis.set_tick_params(labelsize = 12)
        ax1.yaxis.set_tick_params(labelsize = 12)
        ax1.legend(loc='upper right', fontsize = 12)
        ax1.grid()

        ax2.plot(self.t, self.dq[:, 0], label="theta 1 Vel")
        ax2.plot(self.t, self.dq[:, 1], label="theta 2 Vel")
        ax2.plot(self.t, self.dq[:, 2], label="theta 3 Vel")

        ax2.set_ylabel('Angular Vel ', fontsize = 15)
        ax2.xaxis.set_tick_params(labelsize = 12)
        ax2.yaxis.set_tick_params(labelsize = 12)
        ax2.legend(loc='upper right', fontsize = 12)
        ax2.grid()

        ax3.plot(self.t, self.u[:, 0], label="torque 1")
        ax3.plot(self.t, self.u[:, 1], label="torque 2")
        ax3.set_ylabel('Torque ', fontsize = 15)
        ax3.xaxis.set_tick_params(labelsize = 12)
        ax3.yaxis.set_tick_params(labelsize = 12)
        ax3.legend(loc='upper right', fontsize = 12)
        ax3.grid()

        date = self.date
        name = self.name + ".png"

        savename = self.save_dir + date + name

        if saveflag:
            plt.savefig(savename)
    
        plt.show()

    def animation(self, fileflag, saveflag):
        from numpy import sin, cos
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque

        # if fileflag:

        #     FilePath = "/home/stylite-y/Documents/Master/Manipulator/Simulation/model_raisim/DRMPC/SaTiZ_3D/data/2022-04-06" 
        #     config_file = FilePath + "/2022-04-06-17-59-06-Pos_1-Tor_0.1-dt_0.01-T_2-ML_1.0k-config.yaml"
        #     datafile = FilePath + "/2022-04-06-17-59-06-Pos_1-Tor_0.1-dt_0.01-T_2-ML_1.0k-sol.npy"

        #     data = np.load(datafile)
        #     cfg = YAML().load(open(config_file, 'r'))

        #     print(data.shape)
        #     q = data[:, 0:3]
        #     dq = data[:, 3:6]
        #     ddq = data[:, 6:9]
        #     u = data[:, 9:11]
        #     t = data[:, 11:14]

        # else:
        #     pass

        ## kinematic equation
        L0 = self.robot.L[0]
        L1 = self.robot.L[1]
        L2 = self.robot.L[2]
        L_max = L0+L1+L2
        x1 = L0*sin(self.q[:, 0])
        y1 = L0*cos(self.q[:, 0])
        x2 = L1*sin(self.q[:, 0] + self.q[:, 1]) + x1
        y2 = L1*cos(self.q[:, 0] + self.q[:, 1]) + y1
        x3 = L2*sin(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + x2
        y3 = L2*cos(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + y2

        history_len = 100
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-L0, L0), ylim=(-L2, L0+L1))
        ax.set_aspect('equal')
        ax.set_xlabel('X axis ', fontsize = 15)
        ax.set_ylabel('Y axis ', fontsize = 15)
        ax.xaxis.set_tick_params(labelsize = 12)
        ax.yaxis.set_tick_params(labelsize = 12)
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=3,markersize=8)
        trace, = ax.plot([], [], '.-', lw=1, ms=1)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=15)
        history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

        def animate(i):
            thisx = [0, x1[i], x2[i], x3[i]]
            thisy = [0, y1[i], y2[i], y3[i]]

            if i == 0:
                history_x.clear()
                history_y.clear()

            history_x.appendleft(thisx[3])
            history_y.appendleft(thisy[3])

            alpha = (i / history_len) ** 2
            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            # trace.set_alpha(alpha)
            time_text.set_text(time_template % (i*self.robot.dt))
            return line, trace, time_text
        
        ani = animation.FuncAnimation(
            fig, animate, len(self.t), interval=self.robot.dt*1000, blit=True)

        ## animation save to gif
        date = self.date
        name = self.name + ".gif"

        savename = self.save_dir + date + name

        if saveflag:
            ani.save(savename, writer='pillow', fps=72)

        plt.show()
        
        pass

    def animation2(q, dq, u, t, robot, savepath, cfg, flag):
        from numpy import sin, cos
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque
        from matplotlib.patches import Circle

        if flag:

            FilePath = "/home/stylite-y/Documents/Master/Manipulator/Simulation/model_raisim/DRMPC/SaTiZ_3D/data/2022-04-06" 
            config_file = FilePath + "/2022-04-06-17-59-06-Pos_1-Tor_0.1-dt_0.01-T_2-ML_1.0k-config.yaml"
            datafile = FilePath + "/2022-04-06-17-59-06-Pos_1-Tor_0.1-dt_0.01-T_2-ML_1.0k-sol.npy"

            data = np.load(datafile)
            cfg = YAML().load(open(config_file, 'r'))

            print(data.shape)
            q = data[:, 0:3]
            dq = data[:, 3:6]
            ddq = data[:, 6:9]
            u = data[:, 9:11]
            t = data[:, 11:14]

        else:
            pass

        ## kinematic equation
        L0 = robot.L[0]
        L1 = robot.L[1]
        L2 = robot.L[2]
        L_max = L0+L1+L2
        x1 = L0*sin(q[:, 0])
        y1 = L0*cos(q[:, 0])
        x2 = L1*sin(q[:, 0] + q[:, 1]) + x1
        y2 = L1*cos(q[:, 0] + q[:, 1]) + y1
        x3 = L2*sin(q[:, 0] + q[:, 1]+q[:, 2]) + x2
        y3 = L2*cos(q[:, 0] + q[:, 1]+q[:, 2]) + y2

        fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(autoscale_on=False, xlim=(-L0, L0), ylim=(-L2, L0+L1))
        # ax.set_aspect('equal')
        # ax.set_xlabel('X axis ', fontsize = 15)
        # ax.set_ylabel('Y axis ', fontsize = 15)
        # ax.grid()

        # Plotted bob circle radius
        r = 0.05
        # This corresponds to max_trail time points.
        max_trail = int(1.0 / robot.dt)

        def make_plot(i):
            # Plot and save an image of the double pendulum configuration for time
            # point i.
            # The pendulum rods.
            ax.plot([0, x1[i], x2[i], x3[i]], [0, y1[i], y2[i], y3[i]], lw=2, c='k')
            # Circles representing the anchor point of rod 1, and bobs 1 and 2.
            c0 = Circle((0, 0), r/2, fc='k', zorder=10)
            c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
            c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
            c3 = Circle((x3[i], y3[i]), r, fc='r', ec='r', zorder=10)
            ax.add_patch(c0)
            ax.add_patch(c1)
            ax.add_patch(c2)
            ax.add_patch(c3)

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
                ax.plot(x3[imin:imax], y3[imin:imax], c='r', solid_capstyle='butt',
                        lw=2, alpha=alpha)

            # Centre the image on the fixed anchor point, and ensure the axes are equal
            ax.set_xlabel('X axis ', fontsize = 15)
            ax.set_ylabel('Y axis ', fontsize = 15)
            ax.set_xlim(-L0, L0)
            ax.set_ylim(-L2, L0+L1)
            ax.set_aspect('equal', adjustable='box')
            plt.axis('off')
            # plt.savefig('frames/_img{:04d}.png'.format(i//di), dpi=72)
            plt.cla()


        # Make an image every di time points, corresponding to a frame rate of fps
        # frames per second.
        # Frame rate, s-1
        fps = 10
        di = int(1/fps/robot.dt)
        fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
        ax = fig.add_subplot(111)

        for i in range(0, t.size, di):
            print(i // di, '/', t.size // di)
            make_plot(i)
        
        pass

    def DataSave(self, saveflag):
        date = self.date
        name = self.name 

        if saveflag:
            np.save(self.save_dir+date+name+"-sol.npy",
                    np.hstack((self.q, self.dq, self.u, self.t)))
            # output the config yaml file
            # with open(os.path.join(StorePath, date + name+"-config.yaml"), 'wb') as file:
            #     yaml.dump(self.cfg, file)
            with open(self.save_dir+date+name+"-config.yaml", mode='w') as file:
                YAML().dump(self.cfg, file)
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
    TIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/TIP copy.urdf"
    world = raisim.World()
    # endregion

    # region raisim ecvsetting
    t_step = ParamData["Environment"]["t_step"] 
    sim_time = ParamData["Environment"]["sim_time"]
    world.setTimeStep(t_step)
    ground = world.addGround(0)

    gravity = world.getGravity()
    print(gravity,t_step)
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

    q0 = [0.1, 3.3, -0.5]
    # q0 = [0.5, 3.3, -0.5]
    dq0 = [0.0, 0.0, 0.0]
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
    t = []
    u = []
    # endregion
    
    time_start = time.time() 

    # region mpc main loop
    # for i in range(N):
    #     # time.sleep(0.01)
    #     TIP.updateMassInfo() 
    #     JointPos, JointVel = TIP.getState()
    #     JointPos = JointPos.tolist()
    #     JointVel = JointVel.tolist()

    #     robot = TriplePendulum(cfg)
    #     nonlinearOptimization = NLP(robot, cfg, JointPos, JointVel)
    #     tor = nonlinearOptimization.Solve_StateReturn(robot)

    #     u1 = tor[0]
    #     u2 = tor[1]
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

    # region mpc multi Tc loop
    Nc_flag = 0
    for i in range(N):
        # time.sleep(0.01)
        TIP.updateMassInfo() 
        JointPos, JointVel = TIP.getState()
        JointPos = JointPos.tolist()
        JointVel = JointVel.tolist()
        if Nc_flag == Nc:
            Nc_flag = 0
        if Nc_flag == 0:
            robot = TriplePendulum(cfg)
            nonlinearOptimization = NLP(robot, cfg, JointPos, JointVel)
            tor = nonlinearOptimization.Solve_StateReturn2(robot)
            # print(tor)
            u1 = tor[Nc_flag][0]
            u2 = tor[Nc_flag][1]
            Nc_flag += 1
        elif Nc_flag > 0 and Nc_flag < Nc:
            u1 = tor[Nc_flag][0]
            u2 = tor[Nc_flag][1]
            Nc_flag += 1
            pass

        u_temp = [u1, u2]

        t.append((i)*robot.dt)
        q.append(JointPos)
        dq.append(JointVel)
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
    time_end = time.time()
    
    # region data process
    q = np.asarray(q)
    dq = np.asarray(dq)
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
    visual = DataProcess(cfg, robot, q, dq, tor, t, save_dir)
    visual.DataPlot(fig_flag)
    visual.animation(0, ani_flag)
    visual.DataSave(save_flag)
    # endregion
    pass

if __name__ == "__main__":
    # main()
    # visualization()
    # MPC_main()
    Sim_main()
    pass