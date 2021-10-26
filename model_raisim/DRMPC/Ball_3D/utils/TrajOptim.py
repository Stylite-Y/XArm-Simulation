import numpy as np
from numpy.core.defchararray import array 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import LinearConstraint, Bounds
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ========================
def TrajOptim(time_set, PosInit, VelInit, PosTar, VelTar):
    # ========================
    # boundary condition init
    PosLowerbound = [-1.5, -1.5, 0.5]
    ForceLowerbound = [-500.0, -500.0, -500.0]
    PosUpperbound = [1.5, 1.5, 1.0]
    ForceUpperbound = [500.0, 500.0, 0.0]

    # theta 11, 12, 21, 22, 13, 31 solve
    Theta = np.zeros((3, 6))
    Theta[0, 0] = PosInit[0]
    Theta[0, 1] = VelInit[0]
    Theta[1, 0] = PosInit[1]
    Theta[1, 1] = VelInit[1]
    Theta[2, 0] = PosInit[2]
    Theta[2, 1] = VelInit[2]

    # ========================
    # optimization for traj optimize
    # params
    T_f = time_set
    NSamples = 10
    t_step = T_f / NSamples
    print(t_step)

    # ========================
    # Boundary condition
    LowerThetaBnd = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    UpperThetaBnd = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,  np.inf])
    Boundary = Bounds(LowerThetaBnd, UpperThetaBnd)
    # print(Boundary)

    # ========================
    # equality contraints
    JacobianCoef_eq = np.zeros((6, 12))
    
    JacobianCoef_eq[0, ] = np.array([T_f ** 2, T_f ** 3, T_f ** 4, T_f ** 5,
                                     0.0,      0.0,      0.0,      0.0,
                                     0.0,      0.0,      0.0,      0.0])
    JacobianCoef_eq[1, ] = np.array([0.0,      0.0,      0.0,      0.0,
                                     T_f ** 2, T_f ** 3, T_f ** 4, T_f ** 5,
                                     0.0,      0.0,      0.0,      0.0])
    JacobianCoef_eq[2, ] = np.array([0.0,      0.0,      0.0,      0.0,
                                     0.0,      0.0,      0.0,      0.0,
                                     T_f ** 2, T_f ** 3, T_f ** 4, T_f ** 5])
    JacobianCoef_eq[3, ] = np.array([2 * T_f, 3 * T_f ** 2, 4 * T_f ** 3, 5 * T_f ** 4,
                                     0.0,      0.0,          0.0,          0.0,
                                     0.0,      0.0,          0.0,          0.0])
    JacobianCoef_eq[4, ] = np.array([0.0,      0.0,          0.0,          0.0,
                                     2 * T_f, 3 * T_f ** 2, 4 * T_f ** 3, 5 * T_f ** 4,
                                     0.0,      0.0,          0.0,          0.0])
    JacobianCoef_eq[5, ] = np.array([0.0,      0.0,          0.0,          0.0,
                                     0.0,      0.0,          0.0,          0.0,
                                     2 * T_f, 3 * T_f ** 2, 4 * T_f ** 3, 5 * T_f ** 4])

    EqCntConstant = np.array([Theta[0, 0] + Theta[0, 1] * T_f, Theta[1, 0] + Theta[1, 1] * T_f, Theta[2, 0] + Theta[2, 1] * T_f,
                              Theta[0, 1], Theta[1, 1], Theta[2, 1]])
    LowerBnd_eq = np.concatenate([PosTar, VelTar])
    UpperBnd_eq = np.concatenate([PosTar, VelTar])
    
    LowerBnd_eq = LowerBnd_eq - EqCntConstant
    UpperBnd_eq = UpperBnd_eq - EqCntConstant
    # print("# ========================")
    # print("equal contraints: ", JacobianCoef_eq, LowerBnd_eq, UpperBnd_eq, EqCntConstant)

    # ========================
    # inequality contraints
    JacobianCoef_ineq = np.zeros((6 * NSamples, 12))
    # print(JacobianCoef_ineq.shape)
    LowerBnd_ineq = np.zeros((6 * NSamples))
    UpperBnd_ineq = np.zeros((6 * NSamples))
    # print("# ========================")
    for i in range(0, NSamples):
        # print("dicrete time index: ", i)
        t_now = i * t_step
        # displacement boundry
        LowerBnd_ineq[i] = PosLowerbound[0] - Theta[0, 0] - Theta[0, 0] * t_now
        UpperBnd_ineq[i] = PosUpperbound[0] - Theta[0, 0] - Theta[0, 1] * t_now
        JacobianCoef_ineq[i] = np.array([t_now ** 2, t_now ** 3, t_now ** 4, t_now ** 5,
                                        0.0,      0.0,      0.0,      0.0,
                                        0.0,      0.0,      0.0,      0.0])
    
        LowerBnd_ineq[i + NSamples] = PosLowerbound[1] - Theta[1, 0] - Theta[1, 1] * t_now
        UpperBnd_ineq[i + NSamples] = PosUpperbound[1] - Theta[1, 0] - Theta[1, 1] * t_now
        JacobianCoef_ineq[i + NSamples] = np.array([0.0,      0.0,      0.0,      0.0,
                                        t_now ** 2, t_now ** 3, t_now ** 4, t_now ** 5,
                                        0.0,      0.0,      0.0,      0.0])
    
        LowerBnd_ineq[i + 2 * NSamples] = PosLowerbound[2] - Theta[2, 0] - Theta[2, 1] * t_now
        UpperBnd_ineq[i + 2 * NSamples] = PosUpperbound[2] - Theta[2, 0] - Theta[2, 1] * t_now
        JacobianCoef_ineq[i + 2 * NSamples] = np.array([0.0,      0.0,      0.0,      0.0,
                                        0.0,      0.0,      0.0,      0.0,
                                        t_now ** 2, t_now ** 3, t_now ** 4, t_now ** 5])
    # force boundry
    
        LowerBnd_ineq[i + 3 * NSamples] = ForceLowerbound[0]
        UpperBnd_ineq[i + 3 * NSamples] = ForceUpperbound[0]
        JacobianCoef_ineq[i + 3 * NSamples] = np.array([2.0, 6 * t_now, 12 * t_now ** 2, 20 * t_now ** 3,
                                        0.0, 0.0,       0.0,             0.0,
                                        0.0, 0.0,       0.0,             0.0])
    
        LowerBnd_ineq[i + 4 * NSamples] = ForceLowerbound[1]
        UpperBnd_ineq[i + 4 * NSamples] = ForceUpperbound[1]
        JacobianCoef_ineq[i + 4 * NSamples] = np.array([0.0, 0.0,       0.0,             0.0,
                                        2.0, 6 * t_now, 12 * t_now ** 2, 20 * t_now ** 3,
                                        0.0, 0.0,       0.0,             0.0])
    
        LowerBnd_ineq[i + 5 * NSamples] = ForceLowerbound[2]
        UpperBnd_ineq[i + 5 * NSamples] = ForceUpperbound[2]
        JacobianCoef_ineq[i + 5 * NSamples] = np.array([0.0, 0.0,       0.0,             0.0,
                                        0.0, 0.0,       0.0,             0.0,
                                        2.0, 6 * t_now, 12 * t_now ** 2, 20 * t_now ** 3,])


    # print("# ========================")
    # print("inequal constraints: ", JacobianCoef_ineq[50:60]) 
    # print("inequal constraints: ", UpperBnd_ineq) 
    LowerBnd = np.concatenate([LowerBnd_eq, LowerBnd_ineq])
    UpperBnd = np.concatenate([UpperBnd_eq, UpperBnd_ineq])

    JacobianCoef = np.concatenate([JacobianCoef_eq, JacobianCoef_ineq], axis = 0)
    LinearCnt = LinearConstraint(JacobianCoef, LowerBnd, UpperBnd)
    # print(JacobianCoef.shape, LowerBnd.shape, UpperBnd.shape)
    # print(JacobianCoef[0:6], LowerBnd[0:6], UpperBnd[0:6])

    # ========================
    # objective function
    ObjFunc = lambda x: 4 * (x[0] ** 2 + x[4] ** 2 + x[8] ** 2) / 5 + 12 * (x[0] * x[1] + x[4] * x[5] + x[8] * x[9]) / 25 + \
                        16 * (x[0] * x[2] + x[4] * x[6] + x[8] * x[10]) / 125 + 4 * (x[0] * x[3] + x[4] * x[7] + x[8] * x[11]) / 125 + \
                        12 * (x[1] ** 2 + x[5] ** 2 + x[9] ** 2) / 125 + 36 * (x[1] * x[2] + x[5] * x[6] + x[9] * x[10]) / 625 + 48 * (x[1] * x[3] + x[5] * x[7] + x[9] * x[11]) / 3125 + \
                        144 * (x[2] ** 2 + x[6] ** 2 + x[10] ** 2) / 15625 + 16 * (x[2] * x[3] + x[6] * x[7] + x[10] * x[11]) / 3125 + \
                        16 * (x[3] ** 2 + x[7] ** 2 + x[11] ** 2) / 21875

    # initial guess
    # x0 = np.array([1.68, -124.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, -24.76, 0.0, 0.0])
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # ========================
    # solver
    Res = minimize(ObjFunc, x0, method = 'trust-constr', constraints=[LinearCnt], options={'verbose': 1, 'disp': False }, bounds=Boundary)
    Theta_Res = np.array(Res.x)
    print(Res)
    print(Theta_Res)

    # data process

    Theta_Res.reshape(3, 4)
    for i in range(3):
        index = i * 4
        Theta[i, 2:6] = Theta_Res[index:(index + 4)]

    print("Theta Result: ", Theta)

    x_tf = Theta[0, 0] + Theta[0, 1] * T_f + Theta[0, 2] * T_f ** 2 + Theta[0, 3] * T_f ** 3 + Theta[0, 4] * T_f ** 4 + Theta[0, 5] * T_f ** 5
    y_tf = Theta[1, 0] + Theta[1, 1] * T_f + Theta[1, 2] * T_f ** 2 + Theta[1, 3] * T_f ** 3 + Theta[1, 4] * T_f ** 4 + Theta[1, 5] * T_f ** 5
    z_tf = Theta[2, 0] + Theta[2, 1] * T_f + Theta[2, 2] * T_f ** 2 + Theta[2, 3] * T_f ** 3 + Theta[2, 4] * T_f ** 4 + Theta[2, 5] * T_f ** 5
    print ("Theata Result Verify: ", x_tf, y_tf, z_tf)
    
    return Theta

def DataPostProcess(time_set, ThetaCoef):
    T = np.linspace(0.0, time_set, 100)
    x = ThetaCoef[0, 0] + ThetaCoef[0, 1] * T + ThetaCoef[0, 2] * T ** 2 + ThetaCoef[0, 3] * T ** 3 + ThetaCoef[0, 4] * T ** 4 + ThetaCoef[0, 5] * T ** 5
    y = ThetaCoef[1, 0] + ThetaCoef[1, 1] * T + ThetaCoef[1, 2] * T ** 2 + ThetaCoef[1, 3] * T ** 3 + ThetaCoef[1, 4] * T ** 4 + ThetaCoef[1, 5] * T ** 5
    z = ThetaCoef[2, 0] + ThetaCoef[2, 1] * T + ThetaCoef[2, 2] * T ** 2 + ThetaCoef[2, 3] * T ** 3 + ThetaCoef[2, 4] * T ** 4 + ThetaCoef[2, 5] * T ** 5

    vx = ThetaCoef[0, 1] + 2 * ThetaCoef[0, 2] * T + 3 * ThetaCoef[0, 3] * T ** 2 + 4 * ThetaCoef[0, 4] * T ** 3 + 5 * ThetaCoef[0, 5] * T ** 4
    vy = ThetaCoef[1, 1] + 2 * ThetaCoef[1, 2] * T + 3 * ThetaCoef[1, 3] * T ** 2 + 4 * ThetaCoef[1, 4] * T ** 3 + 5 * ThetaCoef[1, 5] * T ** 4
    vz = ThetaCoef[2, 1] + 2 * ThetaCoef[2, 2] * T + 3 * ThetaCoef[2, 3] * T ** 2 + 4 * ThetaCoef[2, 4] * T ** 3 + 5 * ThetaCoef[2, 5] * T ** 4

    fx = 2 * ThetaCoef[0, 2] + 6 * ThetaCoef[0, 3] * T + 12 * ThetaCoef[0, 4] * T ** 2 + 20 * ThetaCoef[0, 5] * T ** 3
    fy = 2 * ThetaCoef[1, 2] + 6 * ThetaCoef[1, 3] * T + 12 * ThetaCoef[1, 4] * T ** 2 + 20 * ThetaCoef[1, 5] * T ** 3
    fz = 2 * ThetaCoef[2, 2] + 6 * ThetaCoef[2, 3] * T + 12 * ThetaCoef[2, 4] * T ** 2 + 20 * ThetaCoef[2, 5] * T ** 3

    plt.figure()
    plt.subplot(311)
    plt.scatter(T, x, label = 'x-axis motion trajectory')
    plt.scatter(T, y, label = 'y-axis motion trajectory')
    plt.scatter(T, z, label = 'z-axis motion trajectory')
    plt.ylabel('Position (m)', fontsize = 15)
    plt.legend(loc='upper right', fontsize = 15)
    plt.title('Trajectory Optimization', fontsize = 20)
    plt.subplot(312)
    plt.scatter(T, vx, label = 'x-axis motion velocity')
    plt.scatter(T, vy, label = 'y-axis motion velocity')
    plt.scatter(T, vz, label = 'z-axis motion velocity')
    plt.ylabel('Velocity (m/s)', fontsize = 15)
    plt.legend(loc='upper right', fontsize = 15)
    plt.subplot(313)
    plt.scatter(T, fx, label = 'x-axis Force')
    plt.scatter(T, fy, label = 'y-axis Force')
    plt.scatter(T, fz, label = 'z-axis Force')
    plt.xlabel('Time (s)', fontsize = 15)
    plt.ylabel('Force (N)', fontsize = 15)
    plt.legend(loc='upper right', fontsize = 15)

    plt.figure()
    plt.scatter(x, z)
    plt.xlabel('X Position (m)', fontsize = 15)
    plt.ylabel('Z Position (m)', fontsize = 15)
    plt.title('X-Z plant Trajectory', fontsize = 20)
    plt.show()


if __name__ == "__main__":
    time_set = 0.2
    PosInit = [0.53, 0.0, 0.5]
    PosTar = [0.4, 0.0, 0.5]
    VelInit = [4.0, 0.0, 5.0]
    VelTar = [-10.4, 0.0, -6.0]

    ## [-7.68874710e-01 -6.07641871e-08  5.00449803e-01] [-1.02984454e+01 -5.33386431e-07  5.96381068e+00]
    ## -0.7 0.2 3.4285714285714284 6.857142857142857 -6.
    # PosInit = [0.77, 0.0, 0.5]
    # PosTar = [-0.7, 0.2, 0.5]
    # VelInit = [-10.3, 0.0, 5.96]
    # VelTar = [3.43, 6.86, -6.0]

    # [-0.31124933  0.97921309  0.50045149] [3.42515859 6.86529762 5.96387037]
    # -0.09999999999999998 1.0 3.4285714285714284 -6.857142857142857 -6.0
    # PosInit = [-0.31, 0.98, 0.5]
    # PosTar = [-0.1, 1.0, 0.5]
    # VelInit = [3.425, 6.86, 5.96]
    # VelTar = [3.42, -6.86, -6.0]

    ThetaCoef = TrajOptim(time_set, PosInit, VelInit, PosTar, VelTar)
    DataPostProcess(time_set, ThetaCoef)

