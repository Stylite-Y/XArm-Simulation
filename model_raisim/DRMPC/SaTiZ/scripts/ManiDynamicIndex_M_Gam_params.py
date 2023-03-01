'''
1. 2023.02.19:
        - 二连杆两个关节以功率和为最大优化目标,但是减速比不同,遍历减速比
2. 2023.02.21:
        - 二连杆两个关节以功率和为最大优化目标,但是减速比不同, 并把减速比作为优化变量
3. 2023.02.22:
        - 二连杆两个关节减速比作为优化变量，目标函数加入速度项
4. 2023.02.23:
        - 二连杆两个关节减速比作为优化变量，目标函数不需要速度项也可
5. 2023.02.24:
        - 二连杆两个关节减速比作为优化变量，伸展轨迹优化计算

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
    # def __init__(self, cfg, gam1, gam2, Im):
    def __init__(self, cfg, Im):
        self.opti = ca.Opti()
        # load config parameter
        # self.CollectionNum = cfg['Controller']['CollectionNum']
        # self.gam1 = gam1
        # self.gam2 = gam2
        self.Im = Im

        self.Pf = [0.5, 0.5]

        self.qmax = 0.75*np.pi
        self.dqmax = 64
        self.umax = 27

        # time and collection defination related parameter
        self.T = cfg['Controller']['Period']
        self.dt = cfg['Controller']['dt']
        self.N = int(self.T / self.dt)
        # self.N = cfg['Controller']['CollectionNum']
        # self.dt = self.T / self.N
        
        # mass and geometry related parameter
        self.m = cfg['Robot']['Mass']['mass']
        self.I = cfg['Robot']['Mass']['inertia']
        # self.m = Mas
        # self.I = inert
        self.l = cfg['Robot']['Mass']['massCenter']
        self.I_ = [self.m[i]*self.l[i]**2+self.I[i] for i in range(2)]

        self.L = [cfg['Robot']['Geometry']['L1'],
                  cfg['Robot']['Geometry']['L2']]

        # motor parameter
        self.motor_cs = cfg['Robot']['Motor']['CriticalSpeed']
        self.motor_ms = cfg['Robot']['Motor']['MaxSpeed']
        self.motor_mt = cfg['Robot']['Motor']['MaxTorque']

        # evironemnt parameter
        self.mu = cfg['Environment']['Friction_Coeff']
        self.g = cfg['Environment']['Gravity']
        self.damping = cfg['Robot']['damping']

        self.u_LB = [-self.motor_mt] * 2
        self.u_UB = [self.motor_mt] * 2

        # shank, thigh, body, arm, forearm
        self.q_LB = [-np.pi/2, 0] 
        self.q_UB = [2*np.pi/3, 2*np.pi/3]   

        self.dq_LB = [-self.motor_ms, -self.motor_ms]   # arm 

        self.dq_UB = [self.motor_ms, self.motor_ms] # arm 

        # * define variable
        self.gam = self.opti.variable(2)
        self.q = [self.opti.variable(2) for _ in range(self.N)]
        self.dq = [            self.opti.variable(2) for _ in range(self.N)]
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
        Im = self.Im
        gam1 = self.gam[0]
        gam2 = self.gam[1]

        M11 = Im*gam1**2+I0 + I1 + m0*lc0**2+m1*(L0**2+lc1**2+2*L0*lc1*c(q[1]))

        M12 = I1 + m1*(lc1**2+L0*lc1*c(q[1]))
        M21 = M12
        M22 = I1 + m1*lc1**2+Im*gam2**2

        return [[M11, M12],
                [M21, M22]]

    def coriolis(self, q, dq):
        m1 = self.m[0]
        m2 = self.m[1]
        lc1 = self.l[0]
        lc2 = self.l[1]
        L1 = self.L[0]
        L2 = self.L[1]

        C1 = -2*m2*L1*lc2*s(q[1])*dq[0]*dq[1]-m2*L1*lc2*s(q[1])*dq[1]*dq[1]
        C2 = m2*L1*lc2*s(q[1])*dq[0]*dq[0]

        return [C1, C2]

    def gravity(self, q):
        m1 = self.m[0]
        m2 = self.m[1]
        lc1 = self.l[0]
        lc2 = self.l[1]
        L1 = self.L[0]
        L2 = self.L[1]

        G1 = (m1*lc1+m2*L1)*c(q[0])+m2*lc2*c(q[0]+q[1])
        
        G2 = m2*lc2*c(q[0]+q[1])

        return [G1*self.g, G2*self.g]

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
        return inertia_main, inertia_coupling
        # endregion


    @staticmethod
    def get_posture(q):
        L = [0.4, 0.4]
        lsx = np.zeros(2)
        lsy = np.zeros(2)
        ltx = np.zeros(2)
        lty = np.zeros(2)
        lsx[0] = 0
        lsx[1] = lsx[0] + L[0]*np.cos(q[0])
        lsy[0] = 0
        lsy[1] = lsy[0] + L[0]*np.sin(q[0])

        ltx[0] = 0 + L[0]*np.cos(q[0])
        ltx[1] = ltx[0] + L[1]*np.cos(q[0]+q[1])
        lty[0] = 0 + L[0]*np.sin(q[0])
        lty[1] = lty[0] + L[1]*np.sin(q[0]+q[1])
        return [lsx, lsy, ltx, lty]

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
    def inverse_kinematics(foot, L):
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
        # theta2 = theta2 if (not is_forward) else -theta2

        theta1 = atan2(foot[1], foot[0]) - acos((l1**2+l**2-l2**2)/2/l1/l)

        return np.asarray([theta1, theta2])

    pass

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
        # init(walker.gam[0], 5.0)
        # init(walker.gam[1], 7.0) # 5.2-7.12
        # init(walker.gam[1], 3.3)    # 4.7-5.57
        # init(walker.gam[1], 4.3)    # 5.13-3.47
        init(walker.gam[0], 5.0)
        init(walker.gam[1], 2.6)    # 3.24-3.42
        for i in range(walker.N):
            init(walker.q[i][0], np.pi/6)
            # init(walker.q[i][0], np.pi*0.4)
            # init(walker.q[i][1], 2*np.pi/3)
            # init(walker.dq[i][0], vel_f[i][0])
            # init(walker.dq[i][1], vel_f[i][0])
            pass

    def Cost(self, walker):
        # region aim function of optimal control
        power = 0
        vel = 0
        l1 = walker.L[0]
        l2 = walker.L[1]
        # ms = walker.motor_ms/walker.gam

        for i in range(walker.N):
            q0 = walker.q[i]
            # Jq = [[-l1*np.sin(q0[0])-l2*np.sin(q0[0]+q0[1]), -l2*np.sin(q0[0]+q0[1])],
            #         [l1*np.cos(q0[0])+l2*np.cos(q0[0]+q0[1]), l2*np.cos(q0[0]+q0[1])]]
            # vx_end = Jq[0][0]*walker.dq[i][0]+Jq[0][1]*walker.dq[i][1]
            # vy_end = Jq[1][0]*walker.dq[i][0]+Jq[1][1]*walker.dq[i][1]
            # v_end = (vx_end**2+vy_end**2)**0.5
            # vel -= (v_end/60)**2 * walker.dt * 0.4
            # vel -= (walker.dq[i][0]/walker.dqmax)**2 * walker.dt * 0.3
            # vel -= (walker.dq[i][1]/walker.dqmax)**2 * walker.dt * 0.3
            # for k in range(2):
            #     power -= (walker.dq[i][k] * walker.u[i][k])**2 * walker.dt * walker.Pf[k]              
            #     pass
            # power -= ((walker.dq[i][0] * walker.u[i][0] + walker.dq[i][1] * walker.u[i][1])/150)**2 * walker.dt * 0.7
            power -= ((walker.dq[i][0] * walker.u[i][0] + walker.dq[i][1] * walker.u[i][1])/150)**2 * walker.dt
            pass

        power -= ((walker.dq[-1][0] * walker.u[-1][0] + walker.dq[-1][1] * walker.u[-1][1])/150)**2 * 50
        # vel -= (walker.dq[-1][0]/walker.dqmax)**2 * walker.dt * 10
        # vel -= (walker.dq[-1][1]/walker.dqmax)**2 * walker.dt * 10

        
        # for j in range(2):
        #     power -= (walker.dq[-1][j] * walker.u[-1][j])**2 * walker.dt * walker.Pf[j]  * 50
        #     pass
        
        # endregion

       
        u = walker.u

        smooth = 0
        AM = [100, 100]
        for i in range(walker.N-1):
            for k in range(2):
                smooth += ((u[i+1][k]-u[i][k])/5)**2*0.5
                pass
            pass
        
        res = 0
        res = res + power
        res = res + vel

        return res

    def getConstraints(self, walker):
        ceq = []
        # region dynamics constraints
        # continuous dynamics
        #! 约束的数量为 (6+6）*（NN1-1+NN2-1）
        for j in range(walker.N):
            if j < (walker.N-1):
                ceq.extend([walker.q[j+1][k]-walker.q[j][k]-walker.dt/2 *
                            (walker.dq[j+1][k]+walker.dq[j][k]) == 0 for k in range(2)])
                inertia = walker.inertia_force(
                    walker.q[j], walker.ddq[j])
                coriolis = walker.coriolis(
                    walker.q[j], walker.dq[j])
                ceq.extend([inertia[k]+coriolis[k] -
                            walker.u[j][k] == 0 for k in range(2)])
            
            pass
        # endregion
        gam1 = walker.gam[0]
        gam2 = walker.gam[1]
        # region boundary constraint
        for temp_q in walker.q:
            ceq.extend([walker.opti.bounded(walker.q_LB[j],
                        temp_q[j], walker.q_UB[j]) for j in range(2)])
            pass
        for temp_dq in walker.dq:
            ceq.extend([walker.opti.bounded(walker.dq_LB[0]/gam1,
                        temp_dq[0], walker.dq_UB[0]/gam1)])
            ceq.extend([walker.opti.bounded(walker.dq_LB[1]/gam2,
                        temp_dq[1], walker.dq_UB[1]/gam2)])
            pass
        for temp_u in walker.u:
            ceq.extend([walker.opti.bounded(walker.u_LB[0]*gam1,
                        temp_u[0], walker.u_UB[0]*gam1)])
            ceq.extend([walker.opti.bounded(walker.u_LB[1]*gam2,
                        temp_u[1], walker.u_UB[1]*gam2)])
            pass
        ceq.extend([walker.opti.bounded(1.0,
                        walker.gam[0], 20.0)])
        ceq.extend([walker.opti.bounded(1.0,
                        walker.gam[1], 20.0)])
        # endregion

        # region motor external characteristic curve
        cs = walker.motor_cs/gam1
        ms = walker.motor_ms/gam1
        mt = gam1*walker.motor_mt
        cs2 = walker.motor_cs/gam2
        ms2 = walker.motor_ms/gam2
        mt2 = gam2*walker.motor_mt
        for j in range(len(walker.u)):
            ceq.extend([walker.u[j][0]-ca.fmax(mt - (walker.dq[j][0] -
                                                        cs)/(ms-cs)*mt, 0) <= 0])
            ceq.extend([walker.u[j][0]-ca.fmin(-mt + (walker.dq[j][0] +
                                                            cs)/(-ms+cs)*mt, 0) >= 0])
            ceq.extend([walker.u[j][1]-ca.fmax(mt2 - (walker.dq[j][1] -
                                                        cs2)/(ms2-cs2)*mt2, 0) <= 0])
            ceq.extend([walker.u[j][1]-ca.fmin(-mt2 + (walker.dq[j][1] +
                                                            cs2)/(-ms2+cs2)*mt2, 0) >= 0])
            pass
        # endregion

        ## line traj 
        # ceq.extend([walker.q[0][0]== np.pi/10])
        # ceq.extend([walker.q[0][1]== 8*np.pi/10])

        ## 收缩轨迹 ellipse traj
        # x_eff = 0.64
        # y_eff = 0.78
        # q11 = np.pi - np.arccos(x_eff/(0.8))
        # q22 = 2*np.arccos(x_eff/(0.8))
        # x = walker.L[0]*np.cos(q11) + walker.L[1]*np.cos(q11+q22)
        # y = walker.L[0]*np.sin(q11) + walker.L[1]*np.sin(q11+q22)
        # print(q11, q22)
        # print(x, y)
        # ceq.extend([walker.q[0][0]== q11])
        # ceq.extend([walker.q[0][1]== q22])

        ## 伸展轨迹 ellipse traj 
        ceq.extend([walker.q[0][0]== -np.pi/6])
        ceq.extend([walker.q[0][1]== np.pi/3])

        ceq.extend([walker.dq[0][0]==0.0])
        ceq.extend([walker.dq[0][1]==0.0])

        ## ellipse traj based on same traj and different time
        # ceq.extend([walker.q[-1][0]== np.pi/6])
        # ceq.extend([walker.q[-1][1]== 2*np.pi/3])        

        for k in range(walker.N):
            # region: line traj contraints
            # ceq.extend([2*walker.q[k][0]+walker.q[k][1]-np.pi==0])
            # ceq.extend([2*walker.dq[k][0]+walker.dq[k][1]==0.0])
            # ceq.extend([walker.dq[k][0]>=0.0])
            # endregion

            x_end = walker.L[0]*c(walker.q[k][0]) + walker.L[1]*c(walker.q[k][0]+walker.q[k][1])
            y_end = walker.L[0]*s(walker.q[k][0]) + walker.L[1]*s(walker.q[k][0]+walker.q[k][1])
            # 伸展轨迹 ellipse traj contraints
            ceq.extend([x_end**2/0.48+y_end**2/0.27 - 1==0])
            ceq.extend([walker.dq[k][0]>=0.0])

            # 收缩轨迹 ellipse traj contraints
            # ceq.extend([x_end**2/(x_eff**2)+y_end**2/(y_eff**2) - 1==0])
            # ceq.extend([walker.dq[k][0]<=0.0])

            # slapse traj contraints
            # ceq.extend([-2*x_end+y_end - 0.6==0])
            # ceq.extend([x_end+y_end - 0.7==0])

        # region smooth constraint
        for j in range(len(walker.u)-1):
            ceq.extend([(walker.u[j][k]-walker.u
                        [j+1][k])**2 <= 5 for k in range(2)])
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
        try:
            sol1 = robot.opti.solve()
            gamma.append(sol1.value(robot.gam))
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
            gamma = np.asarray(gamma)

            return q, dq, ddq, u, t, gamma

        

# def main(gamma, gamma2):
def main():
    # region optimization trajectory for bipedal hybrid robot system
    # vis_flag = False
    # save_flag = False
    vis_flag = True
    save_flag = True
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
    ParamFilePath = FilePath + "/config/ManiDynamic.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # endregion

    # region create robot and NLP problem
    Im = 5e-4
    # robot = Bipedal_hybrid(cfg, gamma, gamma2, Im)
    robot = Bipedal_hybrid(cfg, Im)
    nonlinearOptimization = nlp(robot, cfg, armflag)
    # endregion
    # q, dq, ddq, u, t = nonlinearOptimization.solve_and_output(
    #     robot, flag_save=save_flag, StorePath=StorePath)
    q, dq, ddq, u, t, gamma = nonlinearOptimization.solve_and_output(
        robot, flag_save=save_flag, StorePath=StorePath)

    print("="*50)
    print("gamma:", gamma)
    print("="*50)
    print("qmax:", q[-1])
    gam1 = gamma[0][0]
    gam2 = gamma[0][1]
    qm1 = q[:, 0]
    qm2 = q[:, 1]
    qmax1 = max(qm1)
    qmax2 = max(qm2)

    power = []
    Lam = []
    m = cfg['Robot']['Mass']['mass']
    L = [cfg['Robot']['Geometry']['L1'],
        cfg['Robot']['Geometry']['L2']]
    m1 = m[0]
    m2 = m[1]
    l1 = L[0]
    l2 = L[1]
    for i in range(len(t)):
        p1 = q[i][0]*u[i][0]
        p2 = q[i][1]*u[i][1]
        power.append([p1, p2])

        q0 = [q[i][0], q[i][1]]
        dq0 = [dq[i][0], dq[i][1]]
        Jq = np.array([[-l1*np.sin(q0[0])-l2*np.sin(q0[0]+q0[1]), -l2*np.sin(q0[0]+q0[1])],
                            [l1*np.cos(q0[0])+l2*np.cos(q0[0]+q0[1]), l2*np.cos(q0[0]+q0[1])]])
        Mq = np.array([[Im*gam1**2 + m2*l1**2+m1*l1**2/3+m2*l2**2/3+m2*l1*l2*np.cos(q0[1]), m2*l2**2/3+m2*l1*l2*np.cos(q0[1])/2],
                    [m2*l2**2/3+m2*l1*l2*np.cos(q0[1])/2, m2*l2**2/3+Im*gam2**2]])
        M_inv = np.linalg.inv(Mq)
        Mtmp = Jq @ M_inv @ Jq.T
        Mc = np.linalg.inv(Mtmp)
        Ltmp = Mc @ Jq @ dq0
        Lsmp = np.sqrt(Ltmp[0]**2+Ltmp[1]**2)
        # Lambda_p.append(Ltmp)
        Lam.append(Lsmp)
    
    power = np.asarray(power)

    visual = DataProcess(cfg, robot, 1.0, 0.0075, np.pi/6, q, dq, ddq, u, t, save_dir, save_flag)
    if save_flag:
        SaveDir = visual.DataSave(save_flag)

    if vis_flag:
        visual.animationTwoLink(0, save_flag)
        # animation(L, qm1, qm2, t, 0.001, save_dir, 2, 2)

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

        fig = plt.figure(figsize=(8, 8), dpi=180, constrained_layout=False)
        gs = fig.add_gridspec(1, 1)
        g_data = gs[0].subgridspec(4, 2, wspace=0.3, hspace=0.33)
        
        fig2 = plt.figure(figsize=(10, 6), dpi=180, constrained_layout=False)
        gm = fig2.add_gridspec(1, 1)        
        ax_m = fig2.add_subplot(gm[0])

        # gs = fig.add_gridspec(2, 1, height_ratios=[2,1],
        #                       wspace=0.3, hspace=0.33)
        # g_data = gs[1].subgridspec(3, 6, wspace=0.3, hspace=0.33)

        # ax_m = fig.add_subplot(gs[0])
        ax_p = [fig.add_subplot(g_data[0, i]) for i in range(2)]
        ax_v = [fig.add_subplot(g_data[1, i]) for i in range(2)]
        ax_u = [fig.add_subplot(g_data[3, i]) for i in range(2)]
        ax_pow = [fig.add_subplot(g_data[2, i]) for i in range(2)]

        # vel = [robot.foot_vel(q[i, :], dq[i, :]) for i in range(len(q[:, 0]))]

        # plot robot trajectory here
        ax_m.axhline(y=0, color='k')
        num_frame = 5
        for tt in np.linspace(0, robot.T, num_frame):
            idx = np.argmin(np.abs(t-tt))
            # print(idx)
            pos = Bipedal_hybrid.get_posture(q[idx, :])
            ax_m.plot(pos[0], pos[1], 'o-', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1)
            ax_m.plot(pos[2], pos[3], 'o:', ms=1,
                      color=cmap(tt/robot.T), alpha=tt/robot.T*0.8+0.2, lw=1)
            # patch = patches.Rectangle(
            #     (pos[2][0]-0.02, pos[3][0]-0.05), 0.04, 0.1, alpha=tt/robot.T*0.8+0.2, lw=0, color=cmap(tt/robot.T))
            # ax_m.add_patch(patch)
            ax_m.axis('equal')
            pass

        ax_m.set_ylabel('y(m)')
        ax_m.set_xlabel('x(m)')
        ax_m.xaxis.set_tick_params()
        ax_m.yaxis.set_tick_params()

        [ax_v[i].plot(t, dq[:, i]) for i in range(2)]
        ax_v[0].set_ylabel('Velocity(m/s)')
        [ax_p[i].plot(t, q[:, i]) for i in range(2)]
        ax_p[0].set_ylabel('Position(m)')
        [ax_pow[i].plot(t, power[:, i]) for i in range(2)]
        ax_pow[0].set_ylabel('Power(W)')

        # ax_u[0].plot(t[1:], Fx)
        # ax_u[0].plot(t[1:], Fy)
        ax_u[0].plot(t, u[:, 0])
        ax_u[0].set_xlabel('joint 1')
        ax_u[0].set_ylabel('Torque (N.m)')
        ax_u[1].plot(t, u[:, 1])
        ax_u[1].set_xlabel('joint 2')
        # [ax_u[j].set_title(title_u[j]) for j in range(4)]
        plt.legend()

        fig3, axs= plt.subplots(1, 1, figsize=(12, 12))
        ax1 = axs
        # print(dq*gamma)

        # ax1.plot(I_l, Lambda,'o-')
        ax1.plot(t, Lam,'o-')
        ax1.set_xlabel(r't(s)')
        ax1.set_ylabel(r'Impact $\Lambda (kg.m.s^{-1})$')


        if save_flag:
            savename1 =  SaveDir + "Lambda_e.jpg"
            savename2 =  SaveDir + "Pos-Vel-u-P_sl.jpg"
            fig.savefig(savename2)
            fig3.savefig(savename1)
        

        plt.show()

    print(Lam[-1])
    return Lam[-1], qmax1, qmax2

## use mean value instead of peak value to analysis force map
def ForceMapMV():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pickle
    import random
    from matplotlib.pyplot import MultipleLocator
    from mpl_toolkits.mplot3d import Axes3D

    saveflag = False
    # saveflag = True
    armflag = True
    vis_flag = True
    ani_flag = False

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name = "Lam.pkl"
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)    

    gamma = np.linspace(1,20,20)
    gamma2 = np.linspace(1,20,20)
    
    Lambda_p = []
    Lambda_s = np.array([[0.0]*len(gamma)])
    index = []
    dqmax = np.array([[0.0]*len(gamma)])
    dqmax2 = np.array([[0.0]*len(gamma)])
    if saveflag:
        for i in range(len(gamma)):
            dqtmp = []
            dqtmp2 = []
            LamTmp = []
            for j in range(len(gamma2)):
                print("="*50)
                print("Gamma1: ", gamma[i])
                print("Gamma2: ", gamma2[j])
                print("="*50)

                Lam, qmax1, qmax2 = main(gamma[i], gamma2[j])
                print("Lam:", Lam)
                LamTmp.append(Lam)
                dqtmp.append(qmax1)
                dqtmp2.append(qmax2)
            pass
            Lambda_s = np.concatenate((Lambda_s, [LamTmp]), axis = 0)
            dqmax = np.concatenate((dqmax, [dqtmp]), axis = 0)
            dqmax2 = np.concatenate((dqmax2, [dqtmp2]), axis = 0)

        Lambda_s = Lambda_s[1:,]
        dqmax = dqmax[1:,]
        dqmax = np.array(dqmax)
        dqmax = np.around(dqmax,2)
        dqmax2 = dqmax2[1:,]
        dqmax2 = np.array(dqmax2)
        dqmax2 = np.around(dqmax2,2)
        Lambda_s = np.around(Lambda_s,2)

        print(Lambda_s)
        print(dqmax2)
        Data = {'Lambda': Lambda_s, 'dqmax1': dqmax, 'dqmax2': dqmax2}
        with open(os.path.join(save_dir, name), 'wb') as f:
            pickle.dump(Data, f) 

    else:
        with open(os.path.join(save_dir, name), 'rb') as f:
            data=pickle.load(f)

        Lambda_s = data['Lambda']
        dqmax = data['dqmax1']
        dqmax2 = data['dqmax2']
        print(Lambda_s.shape)

    plt.style.use("science")
    params = {  
        'text.usetex': True,
        'image.cmap': 'inferno',
        'font.size': 20,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'lines.linewidth': 1,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.3,
    }

    plt.rcParams.update(params)

    gam = gamma.astype(int) 
    gam2 = gamma2.astype(int)
    gam_label = list(map(str, gam))
    gam2_label = list(map(str, gam2))
    print(gam_label)
    print(dqmax)

    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    ax1 = axs

    pcm1 = ax1.imshow(Lambda_s, cmap='inferno', vmin = -1, vmax = 6)
    cb1 = fig.colorbar(pcm1, ax=ax1)
    ax1.set_xticks(np.arange(len(gam2)))
    ax1.set_xticklabels(gam2_label)
    ax1.set_yticks(np.arange(len(gam)))
    ax1.set_ylim(-0.5, len(gam)-0.5)
    ax1.set_yticklabels(gam_label)
    # ax[i][j].xaxis.set_tick_params(top=True, bottom=False,
    #        labeltop=True, labelbottom=False)

    ax1.set_xlabel(r'Joint 2 Reduction ratio $\gamma_2$')
    ax1.set_ylabel(r'Joint 1 Reduction ratio $\gamma_1$')
    cb1.set_label(r'Impact $\Lambda (kg.m.s^{-1})$')
    for k in range(len(gam)):
        for m in range(len(gam2)):
            ax1.text(m,k,Lambda_s[k][m], ha="center", va="center",color="black",fontsize=10)

    fig2, axs2 = plt.subplots(1, 1, figsize=(12, 12))
    ax2 = axs2

    pcm2 = ax2.imshow(dqmax, cmap='inferno', vmin = -1.0, vmax = 0.6)
    cb2 = fig2.colorbar(pcm2, ax=ax2)
    ax2.set_xticks(np.arange(len(gam2)))
    ax2.set_xticklabels(gam2_label)
    ax2.set_yticks(np.arange(len(gam)))
    ax2.set_ylim(-0.5, len(gam)-0.5)
    ax2.set_yticklabels(gam_label)
    # ax[i][j].xaxis.set_tick_params(top=True, bottom=False,
    #        labeltop=True, labelbottom=False)

    ax2.set_xlabel(r'Joint 2 Reduction ratio $\gamma_2$')
    ax2.set_ylabel(r'Joint 1 Reduction ratio $\gamma_1$')
    cb2.set_label(r'Joint 1 Angle $\theta (rad)$')
    for k in range(len(gam)):
        for m in range(len(gam2)):
            ax2.text(m,k,dqmax[k][m], ha="center", va="center",color="black",fontsize=10)
    
    fig3, axs3 = plt.subplots(1, 1, figsize=(12, 12))
    ax3 = axs3

    pcm3 = ax3.imshow(dqmax2, cmap='inferno', vmin = 0, vmax = np.pi*0.8)
    cb3 = fig3.colorbar(pcm3, ax=ax3)
    ax3.set_xticks(np.arange(len(gam2)))
    ax3.set_xticklabels(gam2_label)
    ax3.set_yticks(np.arange(len(gam)))
    ax3.set_ylim(-0.5, len(gam)-0.5)
    ax3.set_yticklabels(gam_label)
    # ax[i][j].xaxis.set_tick_params(top=True, bottom=False,
    #        labeltop=True, labelbottom=False)

    ax3.set_xlabel(r'Joint 2 Reduction ratio $\gamma_2$')
    ax3.set_ylabel(r'Joint 1 Reduction ratio $\gamma_1$')
    cb3.set_label(r'Joint 2 Angle $\theta (rad)$')
    for k in range(len(gam)):
        for m in range(len(gam2)):
            ax3.text(m,k,dqmax2[k][m], ha="center", va="center",color="black",fontsize=10)
    plt.show()
    
    pass

def animation(L, q1, q2, t, dt, save_dir, gam1, gam2):
    from numpy import sin, cos
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from collections import deque


    ## kinematic equation
    L0 = L[0]
    L1 = L[1]
    L_max = L0+L1
    x1 = L0*cos(q1)
    y1 = L0*sin(q1)
    x2 = L1*cos(q1 + q2) + x1
    y2 = L1*sin(q1 + q2) + y1

    history_len = 100
    
    fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(autoscale_on=False, xlim=(-L_max, L_max), ylim=(-0.5, (L0+L1)*1.0))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L_max, L_max), ylim=(-(L0+L1)*1.2, (L0+L1)*0.8))
    ax.set_aspect('equal')
    ax.set_xlabel('X axis ', fontsize = 20)
    ax.set_ylabel('Y axis ', fontsize = 20)
    ax.xaxis.set_tick_params(labelsize = 18)
    ax.yaxis.set_tick_params(labelsize = 18)
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=3,markersize=8)
    trace, = ax.plot([], [], '.-', lw=1, ms=1)
    time_template = 'time = %.2fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=15)
    history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        if i == 0:
            history_x.clear()
            history_y.clear()

        history_x.appendleft(thisx[2])
        history_y.appendleft(thisy[2])

        alpha = (i / history_len) ** 2
        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        # trace.set_alpha(alpha)
        time_text.set_text(time_template % (i*dt))
        return line, trace, time_text
    
    ani = animation.FuncAnimation(
        fig, animate, len(t), interval=0.1, save_count = 30, blit=True)

    ## animation save to gif
    # date = self.date
    # name = "traj_ani" + ".gif"

    saveflag = True
    # save_dir = "/home/hyyuan/Documents/Master/Manipulator/XArm-Simulation/model_raisim/DRMPC/SaTiZ/data/2023-02-01/"
    # savename = save_dir + "t_"+str(t[-1])+"-pm_"+str(gam1+1)+"-"+str(gam2+1)+".gif"
    savename = save_dir + "g-t_"+str(t[-1])+"-pm_"+str(gam1+1)+"-"+str(gam2+1)+".gif"
    # savename = save_dir +date+ name

    if saveflag:
        ani.save(savename, writer='pillow', fps=30)


if __name__ == "__main__":
    main()
    # ForceMapMV()
    # MOmentCal()
    # ResCmp()
    # CostFunAnalysis()
    # CharactTime()
    pass
