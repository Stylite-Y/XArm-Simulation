"""
三级摆的平衡控制
"""

import casadi as ca
from casadi import sin as s
from casadi import cos as c
import numpy as np
from numpy.random import normal
import time
import os
import yaml
from ruamel.yaml import YAML
import datetime
from scipy.integrate import odeint

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
        self.postar = cfg['Controller']['Target']

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
        pass

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
        
        C2 = L0*(L1*m2*s(q[1]) + lc2*m2*s(q[1]+q[2]) + lc1*m1*s(q[1])) * dq[0]*dq[0] \
            - 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[2] \
            - 2*L1*lc2*m2*s(q[2]) * dq[1]*dq[2] \
            - L1*lc2*m2*s(q[2]) * dq[2]*dq[2]

        C3 = lc2*m2*(L0*s(q[1]+q[2]) + L1*s(q[2]))* dq[0]*dq[0] \
            + 2*L1*lc2*m2*s(q[2]) * dq[0]*dq[1] \
            + L1*lc2*m2*s(q[2]) * dq[1]*dq[1] \

        return [C1, C2, C3]
        pass

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
        self.TorqueCoef = cfg["Optimization"]["CostCoeff"]["torqueCoef"]
        self.PostarCoef = cfg["Optimization"]["CostCoeff"]["postarCoeff"]
        max_iter = cfg["Optimization"]["MaxLoop"]
        self.random_seed = cfg["Optimization"]["RandomSeed"]

        self.cost = self.CostFun(robot)
        robot.opti.minimize(self.cost)

        # self.ceq = self.getConstraints(robot)
        self.ceq = self.getConstraintsMPC(robot)
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
 
            Arm.opti.set_initial(Arm.q[i][0], 0.05)
            Arm.opti.set_initial(Arm.q[i][1], -0.05)
            Arm.opti.set_initial(Arm.q[i][2], 0.05)
            pass    

    def CostFun(self, Arm):
        Torque = 0
        PosTar = 0
        for i in range(Arm.NS):
            for j in range(2):
                Torque += ca.fabs(Arm.u[i][j]/Arm.motor_mt)
                PosTar += ca.fabs(Arm.q[i][0] - Arm.postar)
                PosTar += ca.fabs(Arm.q[i][1] - Arm.postar)
                PosTar += ca.fabs(Arm.q[i][2] - Arm.postar)
                pass
            pass

        return PosTar * self.PostarCoef + Torque * self.TorqueCoef

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
            ceq.extend([ca.fabs(Arm.u[i][j] - Arm.u[i+1][j]) <= 2 for j in range(2)])
            pass
        
        ceq.extend([Arm.q[0][0]==0.1])
        ceq.extend([Arm.q[0][1]==-0.2])
        ceq.extend([Arm.q[0][2]==0.3])

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

        ceq.extend([Arm.dq[0][0]==self.dq0[0]])
        ceq.extend([Arm.dq[0][1]==self.dq0[1]])
        ceq.extend([Arm.dq[0][2]==self.dq0[2]])

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
                     + "-dt_"+str(robot.dt)+"-T_"+str(robot.T)+"-ML_"+str(ML)+ "k"
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

        def odefun(y, t):
            q1, q2, q3, dq1, dq2, dq3 = y
            corilios = robot.Coriolis([q1, q2, q3], [dq1, dq2, dq3])
            gravity = robot.Gravity([q1, q2, q3])
            dydt = [dq1, dq2, dq3,
                    -mass_inv[0][0]*(corilios[0]+gravity[0]) - mass_inv[0][1]*(corilios[1]+gravity[1]) - mass_inv[0][2]*(corilios[2]+gravity[2]),
                    self.u1[0] - mass_inv[1][0]*(corilios[0]+gravity[0]) - mass_inv[1][1]*(corilios[1]+gravity[1]) - mass_inv[1][2]*(corilios[2]+gravity[2]),
                    self.u1[1] - mass_inv[2][0]*(corilios[0]+gravity[0]) - mass_inv[2][1]*(corilios[1]+gravity[1]) - mass_inv[2][2]*(corilios[2]+gravity[2])]
            dydt1 = [dq1, dq2, dq3,
                    q1]
            return dydt

        q_init = []
        q_init.extend(self.q0)
        q_init.extend(self.dq0)

        t_ode = np.linspace(0, self.t, 4)

        sol = odeint(odefun, q_init, t_ode)

        return sol[-1,0:3], sol[-1,3:6]

## trajectory optimizaition main function
def main():
    # region optimization trajectory for bipedal hybrid robot system
    vis_flag = True
    save_flag = True
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # region load config file
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # region create robot and NLP problem
    robot = TriplePendulum(cfg)
    # nonlinearOptimization = nlp(robot, cfg, seed=seed)
    nonlinearOptimization = NLP(robot, cfg)
    # endregion
    q, dq, ddq, u, t = nonlinearOptimization.Solve_Output(
        robot, flag_save=save_flag, StorePath=save_dir)

    if vis_flag:
        import matplotlib.pyplot as plt
        import matplotlib as mpl

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


## mpc control main function
def MPC_main():
    ## file path 
    vis_flag = True
    save_flag = True
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
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
    q0 = [0.1, -0.3, 0.2]    
    dq0 = [0.0, 0.0, 0.0]

    ## create robot and NLP problem
    # robot = TriplePendulum(cfg)
    q = []
    dq = []
    tor = []
    t = []
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

        ## state update accoding to current state and input
        mpccontroller = MPC(cfg, q0, dq0, u1)
        q0, dq0 = mpccontroller.updateState(robot)

        # if i > 2:
        #     break
        pass

    q = np.asarray(q)
    dq = np.asarray(dq)
    tor = np.asarray(tor)
    t = np.asarray(t).reshape([-1, 1])
    
    print(t.shape, q.shape)

    ML = cfg["Optimization"]["MaxLoop"] / 1000
    Tp = cfg['Controller']['Tp']
    PostarCoef = cfg["Optimization"]["CostCoeff"]["postarCoeff"]
    TorqueCoef = cfg["Optimization"]["CostCoeff"]["torqueCoef"]

    if save_flag:
        date = time.strftime("%Y-%m-%d-%H-%M-%S")
        name = "-Pos_"+str(PostarCoef)+"-Tor_"+str(TorqueCoef) \
                + "-dt_"+str(dt)+"-T_"+str(T)+"-Tp_"+str(Tp)+"-ML_"+str(ML)+ "k"
        np.save(save_dir+date+name+"-sol.npy",
                np.hstack((q, dq, tor, t)))
        # output the config yaml file
        # with open(os.path.join(StorePath, date + name+"-config.yaml"), 'wb') as file:
        #     yaml.dump(self.cfg, file)
        with open(save_dir+date+name+"-config.yaml", mode='w') as file:
            YAML().dump(cfg, file)

    if vis_flag:
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        ax1.plot(t, q[:, 0], label="theta 1")
        ax1.plot(t, q[:, 1], label="theta 2")
        ax1.plot(t, q[:, 2], label="theta 3")

        ax1.set_ylabel('Joint Angle ', fontsize = 15)
        ax1.legend(loc='upper right', fontsize = 12)
        ax1.grid()

        ax2.plot(t, dq[:, 0], label="theta 1 Vel")
        ax2.plot(t, dq[:, 1], label="theta 2 Vel")
        ax2.plot(t, dq[:, 2], label="theta 3 Vel")

        ax2.set_ylabel('Angular Vel ', fontsize = 15)
        ax2.legend(loc='upper right', fontsize = 12)
        ax2.grid()

        ax3.plot(t, tor[:, 0], label="torque 1")
        ax3.plot(t, tor[:, 1], label="torque 2")
        ax3.set_ylabel('Torque ', fontsize = 15)
        ax3.legend(loc='upper right', fontsize = 12)
        ax3.grid()
    
        plt.show()



def visualization():
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

    import matplotlib.pyplot as plt
    import matplotlib as mpl

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


if __name__ == "__main__":
    # main()
    # visualization()
    MPC_main()