import os
import pickle
import datetime
import numpy as np
import scipy.linalg
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import casadi as ca
from casadi import sin as s
from casadi import cos as c

## use casadi to optimize params
class Bipedal_hybrid():
    def __init__(self):
        self.opti = ca.Opti()
        # load config parameter
        # self.CollectionNum = cfg['Controller']['CollectionNum']

        # time and collection defination related parameter
        # self.T = cfg['Controller']['Period']
        self.dt = 0.001
        self.N = 200

        self.q_LB = [1.0, 0.01] 
        self.q_UB = [9.0, 0.1] 
        self.u_LB = [20, 0] 
        self.u_UB = [120, 15] 
        self.F_LB = [200] 
        self.F_UB = [600] 
        self.t_LB = [0.2] 
        self.t_UB = [1.0]   

        # * define variable
        self.q = [self.opti.variable(2) for _ in range(self.N)]
        self.u = [self.opti.variable(2) for _ in range(self.N)]
        self.F = [self.opti.variable(1) for _ in range(self.N)]
        self.t_b = [self.opti.variable(1) for _ in range(self.N)]

        pass

class nlp():
    def __init__(self, legged_robot):
        # load parameter
        max_iter = 500

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
            init(walker.q[i][0], 6.0)
            init(walker.q[i][1], 0.04)
            init(walker.u[i][0], 72.0)
            init(walker.u[i][1], 7.0)
            init(walker.F[i], 385)
            init(walker.t_b[i], 0.47)
            pass

    def Cost(self, walker):
        CostFun = 0

        for i in range(walker.N):
            CostFun += 0.8*(walker.t_b[i]/1.0)**2 + 0.2*(walker.u[i][0]/120)**2+\
                        0.2*(walker.u[i][1]/15)**2 + 0.2*(walker.F[i]/600)**2
            pass

        return CostFun

    def getConstraints(self, walker):
        ceq = []
        c1 = [326.456, 10.12, -4.089, 0.449, -0.01166, 4.857]
        c2 = [61.225, 1.173, 6.267, -1.455, 0.1, -5.767]
        c3 = [0.583, 1.004, 18.9, 0.1045, -0.00287, -19.05]
        c4 = [0.737, -0.0677, -0.3886, 0.01737, 0.0044, -0.4758]
        for i in range(walker.N):
            ceq.extend([c1[0] + c1[1]*walker.q[i][0] + c1[2]*walker.q[i][1] + c1[3]*walker.q[i][0]*walker.q[i][1] + 
                                c1[4]*walker.q[i][0]**2 + c1[5]*walker.q[i][1]**2 == walker.F[i]])
            ceq.extend([c2[0] + c2[1]*walker.q[i][0] + c2[2]*walker.q[i][1] + c2[3]*walker.q[i][0]*walker.q[i][1] + 
                                c2[4]*walker.q[i][0]**2 + c2[5]*walker.q[i][1]**2 == walker.u[i][0]])
            ceq.extend([c3[0] + c3[1]*walker.q[i][0] + c3[2]*walker.q[i][1] + c3[3]*walker.q[i][0]*walker.q[i][1] + 
                                c3[4]*walker.q[i][0]**2 + c3[5]*walker.q[i][1]**2 == walker.u[i][1]])
            ceq.extend([c4[0] + c4[1]*walker.q[i][0] + c4[2]*walker.q[i][1] + c4[3]*walker.q[i][0]*walker.q[i][1] + 
                                c4[4]*walker.q[i][0]**2 + c4[5]*walker.q[i][1]**2 == walker.t_b[i]])
            
            pass

        # region boundary constraint
        for temp_q in walker.q:
            ceq.extend([walker.opti.bounded(walker.q_LB[j],
                        temp_q[j], walker.q_UB[j]) for j in range(2)])
            pass
        for temp_u in walker.u:
            ceq.extend([walker.opti.bounded(walker.u_LB[j],
                        temp_u[j], walker.u_UB[j]) for j in range(2)])
            pass
        for temp_F in walker.F:
            ceq.extend([walker.opti.bounded(walker.F_LB[j],
                        temp_F[j], walker.F_UB[j]) for j in range(1)])
            pass
        for temp_t in walker.t_b:
            ceq.extend([walker.opti.bounded(walker.t_LB[j],
                        temp_t[j], walker.t_UB[j]) for j in range(1)])
            pass
        # endregion

        return ceq

    def solve_and_output(self, robot, flag_save=True, StorePath="./"):
        # solve the nlp and stroge the solution
        q = []
        F = []
        t_b = []
        u = []
        t = []
        try:
            sol1 = robot.opti.solve()
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([sol1.value(robot.q[j][k]) for k in range(2)])
                u.append([sol1.value(robot.u[j][k]) for k in range(2)])
                F.append([sol1.value(robot.F[j])])
                t_b.append([sol1.value(robot.t_b[j])])
                
                pass
            pass
        except:
            value = robot.opti.debug.value
            for j in range(robot.N):
                t.append(j*robot.dt)
                q.append([value(robot.q[j][k]) for k in range(2)])
                u.append([value(robot.u[j][k]) for k in range(2)])
                F.append([value(robot.F[j])])
                t_b.append([value(robot.t_b[j])])
                pass
            pass
        finally:
            q = np.asarray(q)
            t_b = np.asarray(t_b)
            F = np.asarray(F)
            u = np.asarray(u)
            t = np.asarray(t).reshape([-1, 1])

            return q, F, t_b, u, t


## use scipy to optimize params
def SciOpt():
    c1 = [326.456, 10.12, -4.089, 0.449, -0.01166, 4.857]       # F
    c2 = [61.225, 1.173, 6.267, -1.455, 0.1, -5.767]            # u_h
    c3 = [0.583, 1.004, 18.9, 0.1045, -0.00287, -19.05]         # u_s
    c4 = [0.737, -0.0677, -0.3886, 0.01737, 0.0044, -0.4758]    # t_b

    def objective(x):
        return 0.4*(x[5]/0.8)**2 + 0.2*(x[2]/120)**2+\
                        0.2*(x[3]/15)**2 + 0.2*(x[4]/600)**2
    def objective2(x):
        return (x[3]-7)**2
    def constraint1(x):
        return c1[0] + c1[1]*x[0] + c1[2]*x[1] + c1[3]*x[0]*x[1] + c1[4]*x[0]**2 + c1[5]*x[1]**2 - x[4]
    def constraint2(x):
        return c2[0] + c2[1]*x[0] + c2[2]*x[1] + c2[3]*x[0]*x[1] + c2[4]*x[0]**2 + c2[5]*x[1]**2 - x[2]
    def constraint3(x):
        return c3[0] + c3[1]*x[0] + c3[2]*x[1] + c3[3]*x[0]*x[1] + c3[4]*x[0]**2 + c3[5]*x[1]**2 - x[3]
    def constraint4(x):
        return c4[0] + c4[1]*x[0] + c4[2]*x[1] + c4[3]*x[0]*x[1] + c4[4]*x[0]**2 + c4[5]*x[1]**2 - x[5]
    
    n = 6
    x0 = np.zeros(n)
    x0[0] = 6.0
    x0[1] = 0.04
    x0[2] = 72.0
    x0[3] = 7.0
    x0[4] = 385
    x0[5] = 0.47

    # fun = lambda x: (x[3]-10)**2
    bnds = ((1.0, 9.0),(0.01, 0.1),(20, 120),(0, 15),(200, 600),(0.2, 1.0))
    cons = ({'type': 'eq', 'fun': constraint1},
            {'type': 'eq', 'fun': constraint2},
            {'type': 'eq', 'fun': constraint3},
            {'type': 'eq', 'fun': constraint4})

    solution = minimize(objective,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)
    x = solution.x

    print(x)
    pass

## scipy.linalg.lstsq方法
def Opt(M_arm, I_arm, F):
    Mass0, Inertia0 = np.meshgrid(M_arm, I_arm)
    Mass = Mass0.flatten()
    Inertia = Inertia0.flatten()
    order = 2
    if order ==1:
        A = np.c_[Mass, Inertia, np.ones(Mass.shape[0])]
        fit, residual, rnk, s = scipy.linalg.lstsq(A, F)

        Z = fit[0]*Mass0 + fit[1]*Inertia0 + fit[2]
    elif order ==2:
        A = np.c_[np.ones(Mass.shape[0]), Mass, Inertia, Mass*Inertia, Mass**2, Inertia**2]
        fit, residual, rnk, s = scipy.linalg.lstsq(A, F)
        Z = np.dot(np.c_[np.ones(Mass.shape), Mass, Inertia, Mass*Inertia, Mass**2, Inertia**2], fit).reshape(Mass0.shape)
    
    print("fit coef: ", fit)
    # plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 6.0,
        'axes.labelpad': 15.0,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.3,
    }

    plt.rcParams.update(params)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={"projection": "3d"})
    ax.plot_surface(Mass0, Inertia0, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(Mass0, Inertia0, F, c='r', s=50)
    plt.xlabel('Mass(kg)')
    plt.ylabel('Inertia(kg.m2)')
    ax.set_zlabel('Time(s)')
    ax.set_title('Balance Time')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()
    pass

## 矩阵求解方法
def Opt2(M_arm, I_arm, F):
    Mass0, Inertia0 = np.meshgrid(M_arm, I_arm)
    Mass = Mass0.flatten()
    Inertia = Inertia0.flatten()
    order = 2
    if order ==1:
        A = np.c_[Mass, Inertia, np.ones(Mass.shape[0])]
        A = np.matrix(A)
        F = np.matrix(F)
        tmp = np.dot(A.T, A)
        tmp0 = np.linalg.inv(tmp)
        tmp1 = np.dot(tmp0, A.T)
        fit = np.dot(tmp1, F.T)
        errors = F.T - np.dot(A, fit)
        residual = np.linalg.norm(errors)

        Z = fit[0]*Mass0 + fit[1]*Inertia0 + fit[2]
    elif order ==2:
        A = np.c_[np.ones(Mass.shape[0]), Mass, Inertia, Mass*Inertia, Mass**2, Inertia**2]
        A = np.matrix(A)
        F = np.matrix(F)
        tmp = np.dot(A.T, A)
        tmp0 = np.linalg.inv(tmp)
        tmp1 = np.dot(tmp0, A.T)
        fit = np.dot(tmp1, F.T)
        errors = F.T - np.dot(A, fit)
        print(errors.shape)
        residual = np.linalg.norm(errors)
        Z = np.dot(np.c_[np.ones(Mass.shape), Mass, Inertia, Mass*Inertia, Mass**2, Inertia**2], fit).reshape(Mass0.shape)
    
    print("fit coef: ", fit)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={"projection": "3d"})
    ax.plot_surface(Mass0, Inertia0, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(Mass0, Inertia0, F, c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    ax.axis('tight')
    # plt.show()
    pass

if __name__ == "__main__":
    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    name1 = "ForceMap3-7-arm-cfun.pkl"
    name2 = "ForceMap3-7-noarm-cfun.pkl"
    calflag = 0
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # region: data load
    M_arm = np.asarray([3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5])
    I_arm = np.asarray([0.012, 0.015, 0.03, 0.04, 0.06, 0.07, 0.09])
    M_label = list(map(str, M_arm))
    I_label = list(map(str, I_arm))

    f1 = open(save_dir+name1,'rb')
    data1 = pickle.load(f1)
    f2 = open(save_dir+name2,'rb')
    data2 = pickle.load(f2)

    Fy = np.asarray(data1['Fy'])
    u_h = np.asarray(data1['u_h'])
    u_s = np.asarray(data1['u_s'])
    u_a = np.asarray(data1['u_a'])
    t_b = np.asarray(data1['t_b'])
    Pwcostfun1 = data1['P_J']
    # endregion


    Fy = Fy.flatten()
    u_h = u_h.flatten()
    u_s = u_s.flatten()
    u_a = u_a.flatten()
    t_b = t_b.flatten()
    
    if calflag == 0:
        print("="*50)
        print("Fy fitting 1")
        Opt(M_arm, I_arm, Fy)
        # print("Fy fitting 2")
        # Opt2(M_arm, I_arm, Fy)
        print("="*50)
        print("hip tor fitting 1")
        Opt(M_arm, I_arm, u_h)
        # print("hip tor fitting 2")
        # Opt2(M_arm, I_arm, u_h)
        print("="*50)
        print("shoulder tor fitting 1")
        Opt(M_arm, I_arm, u_s)
        # print("shoulder tor fitting 2")
        # Opt2(M_arm, I_arm, u_s)
        print("="*50)
        print("bal time fitting 1")
        Opt(M_arm, I_arm, t_b)
        # print("bal time fitting 2")
        # Opt2(M_arm, I_arm, t_b)
    else:
        # region create robot and NLP problem
        robot = Bipedal_hybrid()
        nonlinearOptimization = nlp(robot)
        # endregion
        q, F, t_b, u, t = nonlinearOptimization.solve_and_output(
            robot, StorePath=StorePath)
        print("="*50)
        print("casadi method")
        print("Mass, Inertia: ", q[-1][0], q[-1][1])
        print("hip, shd tor: ", u[-1][0], u[-1][1])
        print("Fy: ", F[-1])
        print("bal time: ", t_b[-1])

        print("="*50)
        print("Scipy method")
        SciOpt()
        pass
    pass