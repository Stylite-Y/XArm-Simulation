'''
1. 2023.05.30:
        - 羽毛球二连杆模型: 基于角动量最大化优化机械臂参数
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
from PostProcess import DataProcess
from scipy import signal
import matplotlib.animation as animation


class Bipedal_hybrid():
    def __init__(self, Im, cfg):
        self.opti = ca.Opti()

        # time and collection defination related parameter
        self.T = cfg['Controller']['Period']
        self.Im = Im
        self.N = cfg['Controller']['CollectionNum']
        self.dt = self.T / self.N
        
        # mass and geometry related parameter
        # self.mm1 = [0.0, 0.0]
        self.m = self.opti.variable(2)
        self.gam = self.opti.variable(2)

        self.L = [cfg['Robot']['Geometry']['L1'],
                  cfg['Robot']['Geometry']['L1']]
        
        self.I = [self.m[0]*self.L[0]**2/12, self.m[1]*self.L[1]**2/12]
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

        # boundry
        self.m_UB = [10.0, 10.0]
        self.m_LB = [1.0, 1.0]

        self.gam_UB = [10.0, 10.0]
        self.gam_LB = [1.0, 1.0]

        self.u_LB = [-self.motor_mt * self.gam[0], -self.motor_mt * self.gam[1]]
        self.u_UB = [self.motor_mt * self.gam[0], self.motor_mt * self.gam[1]]

        self.q_LB = [-np.pi/10, -np.pi] 
        self.q_UB = [np.pi/2, 0.0]   

        self.dq_LB = [-self.motor_ms/self.gam[0], -self.motor_ms/self.gam[1]]   # arm 

        self.dq_UB = [self.motor_ms/self.gam[0], self.motor_ms/self.gam[1]] # arm 

        # * define variable
        self.q = [self.opti.variable(2) for _ in range(self.N)]
        self.dq = [self.opti.variable(2) for _ in range(self.N)]
        self.ddq = [(self.dq[i+1]-self.dq[i]) /
                        self.dt for i in range(self.N-1)]

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
        gam0 = self.gam[0]
        gam1 = self.gam[1]
        Im = self.Im

        M11 = I0 + I1 + Im*gam0**2 + L0**2*m1+2*L0*lc1*m1*c(q[1]) + lc0**2*m0 + lc1**2*m1
        M12 = I1 + L0*lc1*m1*c(q[1]) + lc1**2*m1

        M21 = M12
        M22 = I1+lc1**2*m1+Im*gam1**2

        return [[M11, M12],
                [M21, M22]]

    def coriolis(self, q, dq):
        m0 = self.m[0]
        m1 = self.m[1]
        lc0 = self.l[0]
        lc1 = self.l[1]
        L0 = self.L[0]
        L1 = self.L[1]

        C0 = -2*L0*lc1*m1*s(q[1])*dq[0]*dq[1] - L0*lc1*m1*s(q[1])*dq[1]*dq[1]

        C1 = L0*lc1*m1*s(q[1])*dq[0]*dq[0]

        return [C0, C1]

    def gravity(self, q):
        m0 = self.m[0]
        m1 = self.m[1]
        lc0 = self.l[0]
        lc1 = self.l[1]
        L0 = self.L[0]
        L1 = self.L[1]

        G1 = -(L0*m1*s(q[0]) + lc0*m0*s(q[0]) + lc1*m1*s(q[0]+q[1]))
        
        G2 = -(lc1*m1*s(q[0]+q[1]))

        return [G1*self.g, G2*self.g]

        pass

    def inertia_force(self, q, acc):
        mm = self.MassMatrix(q)
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1] for i in range(2)]
        
        return inertia_force

    def inertia_force2(self, q, acc):
        # region calculate inertia force, split into two parts
        mm = self.MassMatrix(q)
        inertia_force = [mm[i][0]*acc[0]+mm[i][1]*acc[1] for i in range(2)]
        inertia_main = [mm[i][i]*acc[i] for i in range(2)]
        inertia_coupling = [inertia_force[i]-inertia_main[i] for i in range(2)]
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

    @staticmethod
    def get_posture(q):
        L = [0.35, 0.35]
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
        init(walker.m[0], 5.0)
        init(walker.m[1], 4.0)
        # endregion
        for i in range(walker.N):
            for j in range(2):
                init(walker.q[i][j], np.pi)
                init(walker.dq[i][j], 0)
            pass

    def Cost(self, walker):
        # region aim function of optimal control
        power = 0
        force = 0
        VelTar = 0
        PosTar = 0
        Ptar = [0, 0, np.pi, 0.0]
        # Pf = [0.3]*4
        # Vf = [0.6]*4
        # Ff = [0.1]*4 
        # endregion
        
        for i in range(walker.N):
            for k in range(2):
                power += ((walker.dq[i][k]*walker.u[i][k]) / (walker.motor_ms*walker.motor_mt))**2 * walker.dt
                force += (walker.u[i][k] / walker.u_UB[k])**2 * walker.dt

                VelTar += (walker.dq[i][k]/walker.dq_UB[k])**2 * walker.dt
                PosTar += ((walker.q[i][k] - Ptar[k])/walker.q_UB[k])**2 * walker.dt            
                pass
            pass
        
        for j in range(2):
            VelTar += (walker.dq[-1][j]/walker.dq_UB[k])**2 
            PosTar += ((walker.q[-1][j] - Ptar[j])/walker.q_UB[k])**2
       
        u = walker.u

        smooth = 0
        AM = [100, 400, 100]
        for i in range(walker.N-1):
            for k in range(2):
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
        for j in range(walker.N):
            if j < (walker.N-1):
                ceq.extend([walker.q[j+1][k]-walker.q[j][k]-walker.dt/2 *
                            (walker.dq[j+1][k]+walker.dq[j][k]) == 0 for k in range(2)])
                inertia = walker.inertia_force(
                    walker.q[j], walker.ddq[j])
                coriolis = walker.coriolis(
                    walker.q[j], walker.dq[j])
                gravity = walker.gravity(walker.q[j])
                ceq.extend([inertia[k]+gravity[k]+coriolis[k] -
                            walker.u[j][k] == 0 for k in range(2)])

        # endregion

        # ceq.extend([walker.mm2[0]-0.8*walker.mm2[1] >= 0])

        # region boundary constraint
        for temp_q in walker.q:
            ceq.extend([walker.opti.bounded(walker.q_LB[j],
                        temp_q[j], walker.q_UB[j]) for j in range(2)])
            pass
        for temp_dq in walker.dq:
            ceq.extend([walker.opti.bounded(walker.dq_LB[j],
                        temp_dq[j], walker.dq_UB[j]) for j in range(2)])
            pass
        for temp_u in walker.u:
            ceq.extend([walker.opti.bounded(walker.u_LB[j],
                        temp_u[j], walker.u_UB[j]) for j in range(2)])

        for temp_m in walker.m:
            ceq.extend([walker.opti.bounded(walker.m_LB[j],
                        temp_m[j], walker.m_UB[j]) for j in range(2)])

        for temp_gam in walker.gam:
            ceq.extend([walker.opti.bounded(walker.gam_LB[j],
                        temp_gam[j], walker.gam_UB[j]) for j in range(2)])
            pass
        # endregion

        # region motor external characteristic curve
        cs = []
        ms = []
        mt = []
        for k in range(2):
            cs.append(walker.motor_cs/walker.gam[k])
            ms.append(walker.motor_ms/walker.gam[k])
            mt.append(walker.motor_mt*walker.gam[k])
        for j in range(len(walker.u)):
            ceq.extend([walker.u[j][k]-ca.fmax(mt[k] - (walker.dq[j][k] -
                                                        cs[k])/(ms[k]-cs[k])*mt[k], 0) <= 0 for k in range(2)])
            ceq.extend([walker.u[j][k]-ca.fmin(-mt[k] + (walker.dq[j][k] +
                                                            cs[k])/(-ms[k]+cs[k])*mt[k], 0) >= 0 for k in range(2)])
            pass

        # endregion

        theta = np.pi/30
        ceq.extend([walker.q[0][0]==theta])
        ceq.extend([walker.q[0][1]==theta*0.2])

        ceq.extend([walker.dq[0][0]==0])
        ceq.extend([walker.dq[0][1]==0])

        # region smooth constraint
        for j in range(len(walker.u)-1):
            ceq.extend([(walker.u[j][k]-walker.u
                        [j+1][k])**2 <= 50 for k in range(2)])
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
        m = []
        try:
            sol1 = robot.opti.solve()
            gamma.append(sol1.value(robot.gam))
            m.append(sol1.value(robot.m))
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([sol1.value(robot.q[j][k]) for k in range(2)])
                dq.append([sol1.value(robot.dq[j][k])
                            for k in range(2)])
                if j < (robot.N-1):
                    ddq.append([sol1.value(robot.ddq[j][k])
                                for k in range(2)])
                    u.append([sol1.value(robot.u[j][k])
                                for k in range(2)])
                else:
                    ddq.append([sol1.value(robot.ddq[j-1][k])
                                for k in range(2)])
                    u.append([sol1.value(robot.u[j-1][k])
                                for k in range(2)])
                pass
            pass
        except:
            value = robot.opti.debug.value
            gamma.append(value(robot.gam))
            m.append(value(robot.m))
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([value(robot.q[j][k])
                            for k in range(2)])
                dq.append([value(robot.dq[j][k])
                            for k in range(2)])
                if j < (robot.N-1):
                    ddq.append([value(robot.ddq[j][k])
                                for k in range(2)])
                    u.append([value(robot.u[j][k])
                                for k in range(2)])
                else:
                    ddq.append([value(robot.ddq[j-1][k])
                                for k in range(2)])
                    u.append([value(robot.u[j-1][k])
                                for k in range(2)])
                pass
            pass
        finally:
            q = np.asarray(q)
            dq = np.asarray(dq)
            ddq = np.asarray(ddq)
            u = np.asarray(u)
            t = np.asarray(t).reshape([-1, 1])

            return q, dq, ddq, u, t, gamma, m

def main():
    # region optimization trajectory for bipedal hybrid robot system
    vis_flag = True
    save_flag = True
    # endregion

    # region: filepath
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    ani_path = StorePath + "/data/animation/"
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # endregion

    # region load config file
    ParamFilePath = StorePath + "/config/ManiDesign.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)
    # endregion

    # region create robot and NLP problem
    Im = 5e-4
    robot = Bipedal_hybrid(Im, cfg)
    nonlinearOptimization = nlp(robot, cfg)
    q, dq, ddq, u, t, gamma, m = nonlinearOptimization.solve_and_output(
        robot, flag_save=save_flag, StorePath=StorePath)
    # endregion
    
    print("="*50)
    print("gamma:", gamma)
    print("="*50)
    print("m:", m[0],m[1])
    print("="*50)

    # visulization
    if vis_flag:

        pass

if __name__ == "__main__":
    main()
    pass
