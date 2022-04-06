from this import d
import numpy as np
from numpy import NaN
from scipy import linalg
import scipy
import matplotlib.pyplot as plt
import os
import yaml
import time
import raisimpy as raisim
from Dynamics_MPC import RobotProperty
from matplotlib.pyplot import MultipleLocator
import pybullet as p
import pybullet_data as pd
from scipy.integrate import odeint


class LQR():
    def __init__(self, A, B, Q, R, n):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = n

    def lqr(self):
        """
        Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """

        ## solve discrete ricatti equation(for continuous is: solve_continuous_are)
        P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        # P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        # print(P)

        # compute LQR control gain
        K = scipy.linalg.inv(self.B.T @ P @ self.B + self.R) @ (self.B.T @ P @ self.A)
        # K = scipy.linalg.inv(self.R) @ self.B.T @ P

        # V = self.A.T @ P - K.T @ self.B.T @ P + P @ self.A - P @ self.B @ K
        # print("V is: ", V)
        # print(P)
        return -K

    def ilqr(self):
        """
        Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """

        ## solve discrete ricatti equation(for continuous is: solve_continuous_are)
        P = np.array([None] * (self.N + 1))
        P[self.N] = self.Q
        for i in range(self.N, 0, -1):
            # print(i)
            P[i-1] = self.Q + self.A.T @ P[i] @ self.A - (self.A.T @ P[i] @ self.B) @ np.linalg.pinv(
                self.R + self.B.T @ P[i] @ self.B) @ (self.B.T @ P[i] @ self.A)
        # print(P)

        # compute LQR control gain
        K = np.array([None] * self.N)
        for i in range(self.N):
            K[i] = -np.linalg.inv(self.B.T@P[i+1]@self.B+self.R)@self.B.T@P[i+1]@self.A

        return K

class DynamicsModel():
    def __init__(self, M, L, lc, I, dt, T):
        self.M = M
        self.L = L
        self.lc = lc
        self.I = I
        self.dt = dt
        self.T = T
        self.g = 9.8
        self.N = int(self.T / self.dt)
        pass

    def CartPole(self):   

        M = self.M[0]
        m = self.M[1]
        l = self.lc[0]
        I = self.I[0]
        b = 0
        dt = self.dt
        T = self.T
        n = int(T / dt)
        g = self.g
        
        p = (M + m)*I +M*m*l**2

        x0 = np.array([-1, 1.1, 0.0, 0.0]).reshape(-1, 1)
        xf = np.array([0, 0, 0, 0]).reshape(-1, 1)
        
        A = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, -m**2*g*l**2 / p, -(I + m* l**2)*b / p, 0],
                    [0, m*g*l*(M+m) / p, -m*l*b / p, 0]])
        A_bar = A * dt + np.diag([1, 1, 1, 1])
        
        
        B = np.array([[0], [0], [(I + m* l**2) / p], [-m*l/p]])
        B_bar = B * dt
        C = np.eye(4)
        D = np.zeros((4, 1))
        
        t1 = np.array([[M + m, m*l], [m*l, I+m*l**2]])
        t2 = np.array([[0, 0], [0, m*g*l]])
        t1_inv = scipy.linalg.pinv(t1)
        print(np.dot(t1_inv, t2))
        print("="*50)
        print("state equation A matrix is:")
        print(A)
        print("="*50)
        print("state equation B matrix is:")
        print(B)
        print("="*50)
        print("state equation A_bar matrix is:")
        print(A_bar)
        print("="*50)
        print("state equation B_bar matrix is:")
        print(B_bar)

        Q = np.diag([10, 10, 1, 1])
        R = np.array([1])
        return A_bar, B_bar, Q, R, n, x0, xf

    def DoubleInvertPend(self):
        M = self.M[0]
        m1 = self.M[1]
        m2 = self.M[2]
        L1 = self.L[0]
        L2 = self.L[1]
        # lc1 = self.lc[0]
        # lc2 = self.lc[1]
        lc1 = self.L[0]
        lc2 = self.L[1]
        I1 = self.I[0]
        I2 = self.I[1]

        M_matrix = np.array([[M+m1+m2,      (m1*lc1+m2*L1),           m2*lc2],
                            [m1*lc1+m2*L1,  I1+m1*lc1**2+m2*L1**2,    m2*L1*lc2],
                            [m2*lc2,        m2*L1*lc2,                I2+m2*lc2**2]])
        N_matrix = np.array([[0, 0, 0],
                            [0, (m1*lc1+m2*L1)*self.g, 0],
                            [0, 0, m2*lc2*self.g]])
        F_matrix = np.array([[1], [0], [0]])
        M_inv = scipy.linalg.pinv(M_matrix)

        ## 分块矩阵合并
        Temp1 = -np.dot(M_inv, N_matrix)
        A = np.block([[np.zeros((3,3)), np.eye(3)], [Temp1, np.zeros((3,3))]]) 
        temp2 = np.dot(M_inv, F_matrix)
        B = np.block([[np.zeros((3, 1))], [temp2]])

        ## discrete state matrix
        A_bar = A * self.dt + np.eye(6)
        B_bar = B * self.dt

        ## LQR gain matrix
        # Q = np.diag([5, 50, 50, 700, 700, 700])
        Q = np.diag([10, 10, 10, 1, 1, 1])
        R = np.array([1])

        print("="*50)
        print("Mass matrix is:")
        print(M_matrix)
        print("="*50)
        print("Gravity matrix is:")
        print(N_matrix)
        print("="*50)
        print("state equation A matrix is:")
        print(A)
        print("="*50)
        print("state equation B matrix is:")
        print(B)
        print("="*50)
        print("state equation A_bar matrix is:")
        print(A_bar)
        print("="*50)
        print("state equation B_bar matrix is:")
        print(B_bar)

        ## init and target state
        x0 = np.array([-0.001, 0.001, -0.001, 0.0, 0.0, 0.0]).reshape(-1, 1)
        xf = np.array([0, 0, 0, 0, 0.0, 0.0]).reshape(-1, 1)

        return A_bar, B_bar, Q, R, self.N, x0, xf

    def TwoLink(self):
        m1 = self.M[0]
        m2 = self.M[1]
        l1 = self.L[0]
        l2 = self.L[1]
        g = self.g

        M_mat = np.array([[m2*l1**2 + m2*l1*l2 + m1*l1**2/3 + m2*l2**2/3, m2*l2**2/3 + m2*l1*l2/2],
                          [m2*l2**2/3 + m2*l1*l2/2,                       m2*l2**2/3]])

        N_mat = np.array([[-(m1/2+m2)*g*l1 - m2*g*l2/2, -m2*g*l2/2],
                          [-m2*g*l2/2,                  -m2*g*l2/2]])

        F_matrix = np.array([[0], [1]])
        M_inv = scipy.linalg.inv(M_mat)

        ## 分块矩阵合并
        Temp1 = -np.dot(M_inv, N_mat)
        A = np.block([[np.zeros((2,2)), np.eye(2)], [Temp1, np.zeros((2,2))]]) 
        temp2 = np.dot(M_inv, F_matrix)
        B = np.block([[np.zeros((2, 1))], [temp2]])

        ## discrete state matrix
        A_bar = A * self.dt + np.eye(4)
        B_bar = B * self.dt

        ## LQR gain matrix
        Q = np.diag([100, 100, 1, 1])
        R = np.array([10])

        ## init and target state
        x0 = np.array([0.2, -0.4, 0.0, 0.0]).reshape(-1, 1)
        xf = np.array([0, 0, 0.0, 0.0]).reshape(-1, 1)

        return A_bar, B_bar, Q, R, self.N, x0, xf

    def TwoLink2ndDown(self):
        m1 = self.M[0]
        m2 = self.M[1]
        L1 = self.L[0]
        L2 = self.L[1]
        lc1 = self.lc[0]
        lc2 = self.lc[1]
        I1 = self.I[0]
        I2 = self.I[1]
        g = self.g

        M_mat = np.array([[I1+I2+m2*L1**2+m1*lc1**2+m2*lc2**2-2*L1*lc2*m2, I2+m2*lc2**2-L1*lc2*m2],
                          [I2+m2*lc2**2-L1*lc2*m2,                         I2+m2*lc2**2]])

        N_mat = np.array([[-L1*m2*g-lc1*m1*g+lc2*m2*g, lc2*m2*g],
                          [lc2*m2*g,                   lc2*m2*g]])

        F_matrix = np.array([[0], [1]])
        M_inv = scipy.linalg.inv(M_mat)

        ## 分块矩阵合并
        Temp1 = -np.dot(M_inv, N_mat)
        A = np.block([[np.zeros((2,2)), np.eye(2)], [Temp1, np.zeros((2,2))]]) 
        temp2 = np.dot(M_inv, F_matrix)
        B = np.block([[np.zeros((2, 1))], [temp2]])

        ## discrete state matrix
        A_bar = A * self.dt + np.eye(4)
        B_bar = B * self.dt

        print("="*50)
        print("Mass matrix is:")
        print(M_mat)
        print("="*50)
        print("Gravity matrix is:")
        print(N_mat)
        print("="*50)
        print("state equation A matrix is:")
        print(A_bar)
        print("="*50)
        print("state equation B matrix is:")
        print(B_bar)

        ## LQR gain matrix
        Q = np.diag([100, 10, 1, 1])
        R = np.array([0.1])

        ## init and target state
        x0 = np.array([0.2, 0.4, 0.0, 0.0]).reshape(-1, 1)
        xf = np.array([0, 0, 0.0, 0.0]).reshape(-1, 1)

        return A_bar, B_bar, Q, R, self.N, x0, xf

    def TripleInvertPend(self, ParamData):

        GetProperty = RobotProperty(ParamData)
        M_matrix = GetProperty.getMassLinearMatrix2()
        N_matrix = -GetProperty.getGravityLinearMatrix2()
        F_matrix = np.array([[0, 0], [1, 0], [0, 1]])
        M_inv = scipy.linalg.inv(M_matrix)

        ## 分块矩阵合并
        Temp1 = np.dot(M_inv, N_matrix)
        A = np.block([[np.zeros((3,3)), np.eye(3)], [Temp1, np.zeros((3,3))]]) 
        temp2 = np.dot(M_inv, F_matrix)
        B = np.block([[np.zeros((3, 2))], [temp2]])

        # print(M_matrix)

        # print(M_inv)
        # print(N_matrix)
        # print(A)
        # print(B)

        ## discrete state matrix
        A_bar = A * self.dt + np.eye(6)
        B_bar = B * self.dt
        print("="*50)
        print("state equation A matrix is:")
        print(A_bar)
        print("="*50)
        print("state equation B matrix is:")
        print(B_bar)

        ## LQR gain matrix
        Q = np.diag([100, 10, 10, 1, 1, 1])
        R = np.diag([0.1, 0.1])

        ## init and target state
        # x0 = np.array([0.16, 3.64, -0.5, 0.0, 0.0, 0.0]).reshape(-1, 1)
        x0 = np.array([0.05, 0.05, -0.05, 0.0, 0.0, 0.0]).reshape(-1, 1)

        xf = np.array([0, 0, 0, 0, 0.0, 0.0]).reshape(-1, 1)

        return A_bar, B_bar, Q, R, self.N, x0, xf

class DataProcess():
    def __init__(self, x, u, t, n):
        self.x = x
        self.u = u
        self.t = t
        self.N = n
        pass

    def DataPlot(self):
        fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        theta1 = self.x[:, 0]
        theta2 = self.x[:, 1]
        # theta3 = self.x[:, 2]
        Tor1 = self.u[:,0]
        # Tor2 = self.u[:,1]
        print(theta1.shape, Tor1.shape)

        ax1.plot(self.t, theta1, label='Theta 1')
        ax1.plot(self.t, theta2, label='Theta 2')
        # ax1.plot(self.t, theta3, label='Theta 3')
        ax1.set_ylabel('Theta Angular ', fontsize = 15)
        ax1.legend(loc='upper right', fontsize = 12)
        # y_major_locator=MultipleLocator(0.3)
        # ax1.yaxis.set_major_locator(y_major_locator)
        ax1.grid()  


        ax2.plot(self.t, Tor1, label='Toque 2')
        # ax2.plot(self.t, Tor2, label='Toque 3')
        ax2.set_ylabel('Joint Torque ', fontsize = 15)
        ax2.legend(loc='upper right', fontsize = 12)
        ax2.grid()

        ax3.plot(self.t, self.x[:, 2], label='Theta d 1')
        ax3.plot(self.t, self.x[:, 3], label='Theta d 2')
        # ax2.plot(self.t, theta3, label='Theta 3')
        ax3.set_ylabel('Theta Angular ', fontsize = 15)
        ax3.legend(loc='upper right', fontsize = 12)
        # y_major_locator=MultipleLocator(0.3)
        # ax3.yaxis.set_major_locator(y_major_locator)
        ax3.grid() 

        plt.show()

def DIPEqua(S, t, M, L):
    x, theta1, theta2, dx, dtheta1, dtheta2 = S
    dSdt = [dx,
            dtheta1,
            dtheta2,
            ]
    pass

def main():
    # ================== cartpole model ===================
    # cartpole model params
    M = np.array([0.5, 0.2])    # mass array
    L = 0.3                     # link len
    I = 0.018                   # interia
    dt = 0.01                   # calculate time span
    T = 10                      # cal time

    # get state matrix, coef matrix of LQR and init, target state of the model cartpole
    # CartPole = DynamicsModel(M, L, I, dt, T)
    # A_bar, B_bar, Q, R, n, x0, xf = CartPole.CartPole()

    # ================== double inverted pendulum model ===================
    ## double inverted pendulum model params
    M = np.array([0.5, 0.3, 0.4])   # mass array
    L = np.array([0.4, 0.4])        # link len
    lc = np.array([L[0]/2, L[1]/2])
    I = np.array([0.004, 0.0053])        # interia
    dt = 0.001          # calculate time span
    T = 8                          # cal time

    ## get state matrix, coef matrix of LQR and init, target state of the model DIP
    DIP_Model = DynamicsModel(M, L, lc, I, dt, T)
    A_bar, B_bar, Q, R, n, x0, xf = DIP_Model.DoubleInvertPend()

    # ================== Two Link model ===================
    ## double inverted pendulum model params
    M = np.array([0.5, 0.3])   # mass array
    L = np.array([0.4, 0.4])        # link len
    lc = np.array([L[0]/2, L[1]/2])
    I = np.array([0.0067, 0.004])        # interia
    # dt = 0.01          # calculate time span
    # T = 8                          # cal time

    ## get state matrix, coef matrix of LQR and init, target state of the model DIP
    # DIP_Model = DynamicsModel(M, L, lc, I, dt, T)
    # A_bar, B_bar, Q, R, n, x0, xf = DIP_Model.TwoLink()


    # ================== Triple inverted pendulum model ===================
    ## double inverted pendulum model params
    M = np.array([0.5, 0.2, 0.2])   # mass array
    L = np.array([0.5, 0.3])        # link len
    I = np.array([0.2, 0.2])        # interia
    dt = 0.01                       # calculate time span
    # T = 20                          # cal time
    ## /..../Simulation/model_raisim/DRMPC/SaTiZ_3D
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    ## get state matrix, coef matrix of LQR and init, target state of the model DIP
    # TIP = DynamicsModel(M, L, I, dt, T)
    # A_bar, B_bar, Q, R, n, x0, xf = TIP.TripleInvertPend(ParamData)

    ## get LQR control method and gain coef K
    Controller = LQR(A_bar, B_bar, Q, R, n)
    K = Controller.lqr()
    # iK = Controller.ilqr()

    ## initial state storage array
    # F = np.array([[0.0, 0.0]])
    F = np.array([[0.0]])
    x = np.array([x0])
    t = np.linspace(0, T, n)

    ## ode integrate


    ## state update and u cal
    for i in range(n):
        u = K @ x0
        # u = K[i] @ x0
        # print(u.T.shape, F.shape)
        x0 = (A_bar + B_bar @ K) @ x0
        # F = np.concatenate([F, u.T], axis = 0)
        F = np.concatenate([F, u], axis = 0)
        x = np.concatenate([x, [x0]], axis = 0)
        if i == 0:
            print(u.shape)
            print(x.shape)

    x = x[0:n]
    F = F[1:,]
    print(x.shape, F.shape)

    # K2 =iK[0]
    # K1 = K
    # for i in range(len(iK)-1):
    #     K1 = np.concatenate([K1, K], axis = 0)
    #     K2 = np.concatenate([K2, iK[i+1]], axis = 0)

    # plt.figure()
    # # K = [K]*len(iK)
    # # print(K.shape, iK.shape)
    # print(K1)
    # print(K2)
    # plt.plot(K1[:,3], label='K')
    # plt.plot(K2[:,3], label='iK')
    # plt.legend(loc='upper right', fontsize = 12)
    # plt.show()

    Visual = DataProcess(x, F, t, n)
    Visual.DataPlot()

def Sim_main():
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    raisim.World.setLicenseFile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/activation.raisim")
    # TIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/TIP.urdf"
    TIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/TIP copy.urdf"
    world = raisim.World()

    t_step = ParamData["Environment"]["t_step"] 
    sim_time = ParamData["Environment"]["sim_time"]
    world.setTimeStep(t_step)
    ground = world.addGround(0)

    gravity = world.getGravity()
    # world.setGravity([0, 0, 0])
    # gravity1 = world.getGravity()
    TIP = world.addArticulatedSystem(TIP_urdf)
    TIP.setName("TIP")
    print(TIP.getDOF())

    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)

    # ================== Triple inverted pendulum model ===================
    ## double inverted pendulum model params
    M = np.array([0.5, 0.2, 0.2])   # mass array
    L = np.array([0.8, 0.3, 0.3])        # link len
    I = np.array([0.02678, 0.00152, 0.00152])        # interia
    dt = ParamData["Environment"]["t_step"]          # calculate time span
    T = 20                          # cal time

    ## get state matrix, coef matrix of LQR and init, target state of the model DIP
    TIP_Model = DynamicsModel(M, L, I, dt, T)
    A_bar, B_bar, Q, R, N, x0, xf = TIP_Model.TripleInvertPend(ParamData)

    ## 特征值，特征向量求解
    eigenvalue, featurevector = np.linalg.eig(A_bar)
    print("="*50)
    print("A 特征值：", eigenvalue)

    jointNominalConfig = np.array([x0[0], x0[1], x0[2]])
    jointVelocityTarget = np.array([x0[3], x0[4], x0[5]])
    TIP.setGeneralizedCoordinate(jointNominalConfig)
    TIP.setGeneralizedVelocity(jointVelocityTarget)

    ## get LQR control method and gain coef K
    Controller = LQR(A_bar, B_bar, Q, R, N)
    K = Controller.lqr()
    # iK = Controller.ilqr()

    u = np.dot(K, x0)
    # print(x0)
    print("="*50)
    print("LQR K is :")
    print(K)
    # print(u, u[0][0], u[1][0])
    for i in range(N):
        time.sleep(0.2)
        JointPos, JointVel = TIP.getState()
        print("="*50)
        print("JointPos, JointVel", JointPos, JointVel)
        x0 = np.concatenate([JointPos, JointVel])
        # x0[1] = np.pi + x0[1]
        x0 = x0.reshape(-1, 1)
        u = np.dot(K, x0)
        print("Joint 1 and Joint 2 Torque:", u[0][0], u[1][0])

        TIP.setGeneralizedForce([0.0, u[0][0], u[1][0]])
        # if i > 120:
        #     break

        server.integrateWorldThreadSafe()
    pass

def TwoLinkSim():
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    raisim.World.setLicenseFile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/activation.raisim")
    # TIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/TIP.urdf"
    # DIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/DIP.urdf"
    # DIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/DIP_2ndDown.urdf"
    # DIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/DIP_car.urdf"
    DIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/SIP_car.urdf"
    world = raisim.World()

    t_step = ParamData["Environment"]["t_step"]
    sim_time = ParamData["Environment"]["sim_time"]
    world.setTimeStep(t_step)
    ground = world.addGround(0)

    gravity = world.getGravity()
    # world.setGravity([0, 0, 0])
    # gravity1 = world.getGravity()
    DIP = world.addArticulatedSystem(DIP_urdf)
    DIP.setName("DIP")
    print(DIP.getDOF())

    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)

    # ================== DIP 2nd link Down model ===================
    ## double inverted pendulum model params
    M = np.array([10, 1])   # mass array
    L = np.array([3, 0.5])        # link len
    lc = np.array([L[0]/2, L[1]/2])
    I = np.array([7.5, 0.0208])        # interia
    dt = ParamData["Environment"]["t_step"]          # calculate time span
    T = 20                          # cal timeodel = DynamicsModel(M, L, lc, I, dt, T)

    ## get state matrix, coef matrix of LQR and init, target state of the model DIP
    # DIP_Model = DynamicsModel(M, L, lc, I, dt, T)
    # A_bar, B_bar, Q, R, N, x0, xf = DIP_Model.TwoLink2ndDown()

    # ================== DIP cart model ===================
    ## double inverted pendulum model params
    M = np.array([0.5, 0.3, 0.4])   # mass array
    L = np.array([0.2, 0.2])        # link len
    lc = np.array([L[0]/2, L[1]/2])
    I = np.array([0.001, 0.0013])        # interia
    dt = ParamData["Environment"]["t_step"]          # calculate time span
    T = 8                         # cal time

    ## get state matrix, coef matrix of LQR and init, target state of the model DIP
    # DIP_Model = DynamicsModel(M, L, lc, I, dt, T)
    # A_bar, B_bar, Q, R, N, x0, xf = DIP_Model.DoubleInvertPend()

    # ================== SIP cart model ===================
    ## double inverted pendulum model params
    M = np.array([0.5, 0.3])   # mass array
    L = np.array([0.4])        # link len
    lc = np.array([L[0]/2])
    I = np.array([0.004])        # interia
    dt = ParamData["Environment"]["t_step"]          # calculate time span
    T = 6                        # cal time
    
    ## get state matrix, coef matrix of LQR and init, target state of the model DIP
    SIP_Model = DynamicsModel(M, L, lc, I, dt, T)
    A_bar, B_bar, Q, R, N, x0, xf = SIP_Model.CartPole()


    ## 特征值，特征向量求解
    # eigenvalue, featurevector = np.linalg.eig(A_bar)
    # print("="*50)
    # print("A 特征值：", eigenvalue)

    # jointNominalConfig = np.array([x0[0], x0[1], x0[2]])
    # jointVelocityTarget = np.array([x0[3], x0[4], x0[5]])
    jointNominalConfig = np.array([x0[0], x0[1]])
    jointVelocityTarget = np.array([x0[2], x0[3]])
    DIP.setGeneralizedCoordinate(jointNominalConfig)
    DIP.setGeneralizedVelocity(jointVelocityTarget)
    
    ## get LQR control method and gain coef K
    Controller = LQR(A_bar, B_bar, Q, R, N)
    K = Controller.lqr()
    # iK = Controller.ilqr()

    u = np.dot(K, x0)
    # print(x0)
    print("="*50)
    print("LQR K is :")
    print(K)

    F = np.array([[0.0]])
    x = np.array([x0])
    t = np.linspace(0, T, N)
    # print(u, u[0][0], u[1][0])
    for i in range(N):
        time.sleep(0.01)
        JointPos, JointVel = DIP.getState()
        x0 = np.concatenate([JointPos, JointVel])
        # x0[1] = np.pi + x0[1]
        x0 = x0.reshape(-1, 1)
        u = np.dot(K, x0)
        if i <= 1:
            print("="*50)
            print("JointPos, JointVel", JointPos, JointVel)
            print("Joint 1 and Joint 2 Torque:", u)

        F = np.concatenate([F, u], axis = 0)
        x = np.concatenate([x, [x0]], axis = 0)

        DIP.setGeneralizedForce([u[0][0], 0, 0])
        # if i > 120:
        #     break

        server.integrateWorldThreadSafe()
    x = x[0:N]
    F = F[1:,]
    print(x.shape)

    Visual = DataProcess(x, F, t, N)
    Visual.DataPlot()
    pass

def TwoLinkBullet():
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # TIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/TIP.urdf"
    # DIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/DIP.urdf"
    # DIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/DIP_2ndDown.urdf"
    DIP_urdf = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/DIP_car.urdf"
    print(DIP_urdf)
    p.connect(p.GUI)
    p.setGravity(0,0,-9.81)
    DIPid=p.loadURDF(DIP_urdf,basePosition=[0.0,0,0])

    # ================== DIP cart model ===================
    ## double inverted pendulum model params
    M = np.array([0.5, 0.3, 0.4])   # mass array
    L = np.array([0.4, 0.4])        # link len
    lc = np.array([L[0]/2, L[1]/2])
    I = np.array([0.004, 0.0053])        # interia
    dt = ParamData["Environment"]["t_step"]          # calculate time span
    # dt = 1 / 240          # calculate time span
    T = 6                          # cal time

    ## get state matrix, coef matrix of LQR and init, target state of the model DIP
    DIP_Model = DynamicsModel(M, L, lc, I, dt, T)
    A_bar, B_bar, Q, R, N, x0, xf = DIP_Model.DoubleInvertPend()

    ## get LQR control method and gain coef K
    Controller = LQR(A_bar, B_bar, Q, R, N)
    K = Controller.lqr()
    # iK = Controller.ilqr()

    print("="*50)
    print("LQR K is :")
    print(K)

    p.setTimeStep(dt)
    p.resetJointState(DIPid, 1, -0.1)
    p.resetJointState(DIPid, 2, 0.01)
    p.resetJointState(DIPid, 3, -0.01)
    p.setJointMotorControlArray(DIPid, [0, 1, 2, 3], p.VELOCITY_CONTROL, forces=[0, 0,  0, 0])
    print(p.getPhysicsEngineParameters())

    F = np.array([[0.0]])
    x = np.array([x0])
    t = np.linspace(0, T, N)
    for i in range(N):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        JointState = p.getJointStates(DIPid, [1, 2, 3])
        x0 = np.array([JointState[0][0], JointState[1][0], JointState[2][0],
                       JointState[0][1], JointState[1][1], JointState[2][1]])
        ApplyTor = np.array([JointState[0][3],JointState[1][3], JointState[2][3]])
        # x0 = np.concatenate([JointPos, JointVel])
        x0 = x0.reshape(-1, 1)
        u = np.dot(K, x0)
        if i <= 1:
            print("="*50)
            print("JointPos, JointVel", x0)
            print("Apply torque: ", ApplyTor)
            print("cal torque: ", u)
            # print()

        F = np.concatenate([F, u], axis = 0)
        x = np.concatenate([x, [x0]], axis = 0)
        p.setJointMotorControl2(DIPid, 1, p.TORQUE_CONTROL, force = u[0][0])
        p.stepSimulation()
        # time.sleep(0.0001)

    
    x = x[0:N]
    F = F[1:,]
    print(x.shape)

    Visual = DataProcess(x, F, t, N)
    Visual.DataPlot()

if __name__ == "__main__":
    # main()
    # Sim_main()
    TwoLinkSim()
    # TwoLinkBullet()