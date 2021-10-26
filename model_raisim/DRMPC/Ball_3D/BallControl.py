from math import e
import os
import sys
import numpy as np
from numpy.lib import index_tricks
import raisimpy as raisim
import datetime
import time
import yaml
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from xbox360controller import Xbox360Controller

import do_mpc
import matplotlib as mpl
from casadi import *
from casadi.tools import *
from do_mpc.tools.timer import Timer
from matplotlib import cm
from scipy.spatial.transform import Rotation 


sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/utils")
print(os.path.abspath(os.path.dirname(__file__))) # get current file path
# from ParamsCalculate import ControlParamCal
import visualization
import FileSave

from Dribble_model import template_model
from Dribble_mpc import template_mpc
from Dribble_simulator import template_simulator
from TrajOptim import TrajOptim


# xbox = Xbox360Controller(0, axis_threshold=0.02)
def tanh_sig(x):
    return 0.5 + 0.5 * np.tanh(1000 * x)

def DataPlot(Data):

    BallPosition = Data['BallPos']
    BallVelocity = Data['BallVel']
    ExternalForce = Data['ExternalForce']
    Point1Pos = Data['Point1Pos']
    Point1Vel = Data['Point1Vel']
    ResForce = Data['ResForce']
    T = Data['time']
    SumForce = ExternalForce

    plt.figure()
    plt.title('Ball motion in zx plane', fontsize = 20)

    plt.subplot(311)
    plt.plot(T, BallPosition[:, 0], label='Ball x-axis Position')
    plt.plot(T, BallPosition[:, 1], label='Ball y-axis Position')
    # plt.plot(T, BallVelocity[:, 0], label='Ball x-axis Velocity')
    plt.plot(T, BallPosition[:, 2], label='Ball z-axis Position')
    # plt.plot(T, line2, label='highest Velocity')
    plt.axis([0, max(T)*1.05, -max(BallPosition[:, 2])*2, max(BallPosition[:, 2])*2])
    plt.xlabel('time (s)')
    plt.ylabel('Ball Position (m)', fontsize = 15)
    plt.legend(loc='upper right', fontsize = 15)


    plt.subplot(312)
    plt.plot(T, BallVelocity[:, 0], label='Ball x-axis Velocity')
    plt.plot(T, BallVelocity[:, 1], label='Ball y-axis Velocity')
    plt.plot(T, BallVelocity[:, 2], label='Ball z-axis Velocity')

    plt.axis([0, max(T)*1.05, -max(BallVelocity[:, 2])*2, max(BallVelocity[:, 2])*2])
    plt.xlabel('time (s)')
    plt.ylabel('Velocity (m/s)', fontsize = 15)
    plt.legend(loc='upper right', fontsize = 15)

    plt.subplot(313)
    plt.plot(T, ExternalForce[:, 0], label='Ball x-axis Force')
    plt.plot(T, ExternalForce[:, 1], label='Ball y-axis Force')
    plt.plot(T, ExternalForce[:, 2], label='Ball z-axis Force')

    plt.axis([0, max(T)*1.05, -max(ExternalForce[:, 0])*2.5, max(ExternalForce[:, 0])*2.5])
    plt.xlabel('time (s)', fontsize = 15)
    plt.ylabel('Force (N)', fontsize = 15)
    plt.legend(loc='upper right', fontsize = 15)

    plt.figure()
    plt.scatter(BallPosition[:, 0], BallPosition[:, 2], label='X-Z plane Ball motion trajectory')
    ConPoint = []
    for i in range(len(BallVelocity[:, 2])):
        if i > 0 and BallPosition[i, 2] < 0.5 and (BallVelocity[i, 2] * BallVelocity[i-1, 2]) < 0:
            ConPoint.append(BallPosition[i, 0])
            plt.scatter(BallPosition[i, 0], 0.15, c = 'r')
    print("contact position is: ", ConPoint)
    x_ticks = np.arange(-1.5, 1.0, 0.1)
    plt.xticks(x_ticks) 
    plt.xlabel('x-axis position (m)', fontsize = 15)
    plt.ylabel('z-axis position (m)', fontsize = 15)
    # plt.legend(loc='upper right')
    plt.title('X-Z plane Ball motion trajectory', fontsize = 20)

    # plt.figure()
    # plt.plot(BallPosition[:, 0], ExternalForce[:, 0], label='Ball x-axis Pos-Force')
    # plt.plot(BallPosition[:, 0], ExternalForce[:, 1], label='Ball y-axis Pos-Force')
    # plt.plot(BallPosition[:, 0], ExternalForce[:, 2], label='Ball z-axis Pos-Force')
    # plt.xlabel('Position (m)')
    # plt.ylabel('Force (N)')
    # plt.axis([-0.8, 0.8, -300, 250])
    # plt.legend(loc='upper right')
    # plt.title('Ball Pos-Force trajectory', fontsize = 20)

    plt.figure()
    plt.scatter(BallPosition[:, 0], BallPosition[:, 1], label='X-Y plane Ball motion trajectory', cmap='inferno')
    plt.xlabel('x-axis position (m)', fontsize = 15)
    plt.ylabel('y-axis position (m)', fontsize = 15)
    # plt.legend(loc='upper right')
    plt.title('X-Y plane Ball motion trajectory', fontsize = 20)

    # plt.figure()
    # Num = len(Point1Vel)
    # index = np.linspace(0, Num, Num)
    # plt.subplot(211)
    # # Point1Vel[0, 1] = 6
    # # plt.scatter(index, Point1Pos[:, 0], label='Point 1 x-axis pos motion trajectory')
    # plt.scatter(index, Point1Pos[:, 1], label='Point 1 y-axis pos motion trajectory')
    # # plt.scatter(index, Point1Pos[:, 2], label='Point 1 z-axis pos motion trajectory')

    # plt.xlabel('Period', fontsize = 15)
    # plt.ylabel('axis position (m)', fontsize = 15)
    # # plt.axis([-0.5, Num + 0.5, -1.5, 1.5])
    # plt.legend(loc='upper right', fontsize = 15)

    # plt.subplot(212)
    # # plt.scatter(index, Point1Vel[:, 0], label='Point 1 x-axis vel motion trajectory')
    # plt.scatter(index, Point1Vel[:, 1], label='Point 1 y-axis vel motion trajectory')
    # # plt.scatter(index, Point1Vel[:, 2], label='Point 1 z-axis vel motion trajectory')

    # plt.xlabel('Period', fontsize = 15)
    # plt.ylabel('axis velocity (m)', fontsize = 15)
    # # plt.axis([-0.5, Num + 0.5, -1.5, 1.5])
    # plt.legend(loc='upper right', fontsize = 15)
    # # plt.title('Point1 motion trajectory', fontsize = 20)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(BallPosition[:, 0], BallPosition[:, 1], ExternalForce[:, 0], label='x-axis Force')
    # ax.plot(BallPosition[:, 0], BallPosition[:, 1], ExternalForce[:, 1], label='y-axis Force')
    # ax.plot(BallPosition[:, 0], BallPosition[:, 1], ExternalForce[:, 2], label='z-axis Force')
    # ax.plot(BallPosition[:, 0], BallPosition[:, 1], ResForce[:,0], label=' Resultant Force')
    # ax.set_xlabel('x-axis position (m)')
    # ax.set_ylabel('y-axis position (m)')
    # ax.set_zlabel('Force (N)')
    # ax.legend(loc='upper right')
    # ax.set_title('X-Y plane Force Trajectory', fontsize = 20)

    plt.show()

def RefTra(t):
    ## ref traj
    # Period = 0.5
    # N = Period / 0.0001
    # xtra = t * 2 * math.pi / N
    # ztra = np.cos(xtra)

    ## ref point
    xtra = 1
    return xtra

def TriCal(t_force, PosInit, VelInit, PosTar, VelTar):
    t = t_force
    pos_init = PosInit
    v_init = VelInit

    pos_tar = PosTar
    v_tar = VelTar

    b_x = np.array([pos_init[0], v_init[0], pos_tar[0], v_tar[0]])
    b_y = np.array([pos_init[1], v_init[1], pos_tar[1], v_tar[1]])
    b_z = np.array([pos_init[2], v_init[2], pos_tar[2], v_tar[2]])

    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, t, t ** 2, t ** 3], [0, 1, 2 * t, 3 * t ** 2]])

    x_coef = np.linalg.solve(A, b_x)
    y_coef = np.linalg.solve(A, b_y)
    z_coef = np.linalg.solve(A, b_z)
    print("pos and vel: ", PosInit, VelInit, PosTar, VelTar)
    print("x_coef, y_coef, z_coef: ", x_coef, y_coef, z_coef)
    
    return x_coef, y_coef, z_coef

def BallTest():
    for i in range(10000):
        time.sleep(0.05)
        # ball1.setExternalForce(0, [0, 0, 0.15], [5.0, 0.0, 0.0])
        GeneralizedCoordinate = ball1.getGeneralizedCoordinate()
        Quaternion = GeneralizedCoordinate[3:]
        Euler_matrix = Rotation.from_quat(Quaternion)
        Euler = Euler_matrix.as_euler('xyz', degrees=True)
        print(Quaternion)
        print(Euler)

        DeltaTheta = math.pi / 1000
        NormalForcc_Z = 9.81
        TangentialForce_X = 5.0 * np.sin(DeltaTheta * i)
        TangentialForce_Y = 5.0 * np.cos(DeltaTheta * i)
        Force_Z = NormalForcc_Z
        Force_X = TangentialForce_X
        Force_Y = TangentialForce_Y
        ball1.setExternalForce(0, [0.0, 0, 0.0], [0.0, 0.0, Force_Z])
        ball1.setExternalForce(0, [0.15, 0, 0.0], [Force_X, Force_Y, 0.0])

        # ball1.setExternalTorque(0, [0.0, 0.0, 5.0])

        server.integrateWorldThreadSafe()

    # for i in range(10000):
    #     time.sleep(0.01)
    #     ball1.setExternalForce(0, [0, 0, 0.15], [5.0, 0.0, 0.0])
    #     server.integrateWorldThreadSafe()

def MPCControl(Pos_init, Vel_init, xtra, ytra, v_xref, v_yref, index, mpc_flag):
    print("=========================================================================")
    show_animation = False
    store_results = False

    # model = Dribble_model()
    # mpc = Dribble_mpc(model, xtra, ytra, v_xref, v_yref, index)
    # mpc_endtime = datetime.datetime.now()
    # simulator = Dribble_simulator(model)

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # set variable of the dynamics system
    x_b = model.set_variable(var_type='_x', var_name='x_b', shape=(1, 1))
    y_b = model.set_variable(var_type='_x', var_name='y_b', shape=(1, 1))
    z_b = model.set_variable(var_type='_x', var_name='z_b', shape=(1, 1))

    # x2 = model.set_variable(var_type='_x', var_name='theta_dot', shape=(1, 1))
    dx_b = model.set_variable(var_type='_x', var_name='dx_b', shape=(1, 1))
    dy_b = model.set_variable(var_type='_x', var_name='dy_b', shape=(1, 1))
    dz_b = model.set_variable(var_type='_x', var_name='dz_b', shape=(1, 1))
    u_x = model.set_variable(var_type='_u', var_name='u_x', shape=(1, 1))
    u_y = model.set_variable(var_type='_u', var_name='u_y', shape=(1, 1))
    u_z = model.set_variable(var_type='_u', var_name='u_z', shape=(1, 1))

    model.set_rhs('x_b', dx_b)
    model.set_rhs('y_b', dy_b)
    model.set_rhs('z_b', dz_b)
    dx_b_next = vertcat(
        tanh_sig(z_b - z_ref) * u_x / m,
    )
    dy_b_next = vertcat(
        tanh_sig(z_b - z_ref) * u_y / m,
    )
    dz_b_next = vertcat(
        -g + tanh_sig(z_b - z_ref) * u_z / m,
    )
    model.set_rhs('dx_b', dx_b_next)
    model.set_rhs('dy_b', dy_b_next)
    model.set_rhs('dz_b', dz_b_next)

    model.setup()


    estimator = do_mpc.estimator.StateFeedback(model)

    x0  = np.concatenate([Pos_init, Vel_init])
    x0 = x0.reshape(-1, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    simulator.x0 = x0
    
    mpc.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()

    mpc.reset_history()

    # starttime_pre = datetime.datetime.now()
    u0 = mpc.make_step(x0)
    # endtime_pre = datetime.datetime.now()
    # spendtime_pre = endtime_pre - starttime_pre
    # mpc_spendtime = endtime_pre - mpc_starttime
    # model_spendtime = model_endtime - mpc_starttime
    mpcst_spend = mpc_endtime - model_endtime
    # sim_spendtime = simulator_endtime - mpc_endtime
    # print("every mpc control time: ", mpc_spendtime)
    # print("every model time: ", model_spendtime)
    print("every mpc time: ", mpcst_spend)
    # print("every sim time: ", sim_spendtime)
    # print("every predicition time: ", spendtime_pre)

    # y_next = simulator.make_step(u0)
    # x0 = estimator.make_step(y_next)

    # j = 0
    # n_step = 3000
    # for k in range(n_step):
    #     u0 = mpc.make_step(x0)
    #     y_next = simulator.make_step(u0)
    #     x0 = estimator.make_step(y_next)

    # input('Press any key to exit.')

    # store_results = False 
    print("x0: ", x0)
    print("u: ", u0)
    flag = index
    return u0 

def TRI_DriControl(ParamData):

    TraPoint_x = np.array([-0.2, -0.4, 0.0])
    TraPoint_y = np.array([0.0, 0.4, 0.4])

    flag = 0
    xref_flag = 0
    z_ref = 0.5
    x_ref = 0.0
    x_coef = 0.0
    v_zref = -6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    v_xref = 6
    index = 0
    K_xd = 400
    K_zd = 300
    K_zvup = 5
    K_zvdown = 30

    BallPos, BallVel = ball1.getState()
    print("init ball pos: ", BallPos)

    # set ball, arm end foot, contact force and joint state saved array
    BallPosition = np.array([[0.0, 0.0, 0.0]])          # the pos and vel state of ball
    BallVelocity = np.array([[0.0, 0.0, 0.0]])       # the pos and vel state of endfoot
    ExternalForce = np.array([[0.0, 0.0, 0.0]])         # the calculate force and real contact force betweem ball an foot
    SumForce = np.array([[0.0]])
    T = np.array([0.0])

    Point1Pos = np.array([[0.0, 0.0, 0.0]])
    Point1Vel = np.array([[0.0, 0.0, 0.0]])

    Point2State = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    Point3State = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
    for i in range(20000):
        time.sleep(0.0001)

        BallPos, BallVel = ball1.getState()
        BallPos = BallPos[0:3]
        BallVel = BallVel[0:3]
        
        # print(TraPoint_x[index])
        # XForce, ZForce, flag, x_coef = ForceControl(BallPos, BallVel, flag, x_coef, TraPoint_x[index], TraPoint_y[index], z_ref, v_zref, v_xref, K_zd, K_xd, K_zvup, K_zvdown)
        if BallPos[2] > z_ref:
            if flag == 0:
                x_ref = BallPos[0]
                # x_coef = - BallVel[0] / np.abs(BallVel[0])
                if index == 0:
                    x_coef = -1
                    y_coef = 0
                    v_xref = 6
                    v_yref = 0
                    # Point1Vel = np.concatenate([Point1Vel, [BallVel]], axis = 0)
                elif index == 1:
                    x_coef = 1
                    y_coef = 1
                    v_xref = 3
                    v_yref = 6
                    # Point1Vel = np.concatenate([Point1Vel, [BallVel]], axis = 0)
                elif index == 2:
                    x_coef = 1
                    y_coef = -1
                    v_xref = 3
                    v_yref = 6
                    # Point1Vel = np.concatenate([Point1Vel, [BallVel]], axis = 0)

                flag = 1
                print("=====================================================================")
                print("phase index: ", index)
                print("init pos: ", BallPos)
                print("init vel: ", BallVel)
            # print("x_coef, y_coef: ", x_coef, y_coef)

            if BallVel[2] >= 0:
                zx_ratio = np.abs(BallVel[0] / BallVel[2])
                zy_ratio = np.abs(BallVel[1] / BallVel[2])
                if index == 0:
                    print("zx_ratio", zx_ratio)

                ZForce = - K_zd * (BallPos[2] - z_ref) - K_zvup * (BallVel[2])
                if index == 0 :
                    XForce = - x_coef * zx_ratio * ZForce
                    YForce = - zy_ratio * ZForce
                elif index == 1:
                    XForce = - x_coef * zx_ratio * ZForce
                    YForce = 0
                elif index == 2:
                    XForce = x_coef * zx_ratio * ZForce
                    YForce = - y_coef * zy_ratio * ZForce
            
                # print("x pos and vel is ", BallPos[0], BallVel[0])
                # print("x_coef is ", x_coef)
                # if index == 2:
                #     print(BallVel)
                #     print("up state: ", XForce, YForce, ZForce)

            elif BallVel[2] <= 0:
                if xref_flag == 0:
                    x_ref = BallPos[0] + x_coef * 0.1
                    y_ref = BallPos[1] + y_coef * 0.1

                    T_free = - (z_ref - 0.15) / v_zref
                    # v_xref = np.abs(x_ref - TraPoint_x[index]) / T_free
                    # v_yref = np.abs(y_ref - TraPoint_y[index]) / T_free
                    # i = 0
                    if index == 1:
                        Point1Pos = np.concatenate([Point1Pos, [BallPos]], axis = 0)
                    xref_flag = 1
                    
                    print("max height pos: ", BallPos)
                    print("max height vel: ", BallVel)
                    print("x_ref, y_ref, T_free: ", x_ref, y_ref, T_free)
                    print("v_xref, v_yref: ", v_xref, v_yref)
                # XForce = x_coef * (K_xd * (BallPos[0] - x_ref) + 10 * (BallVel[2] - x_coef * v_xref))
                # xtemp = np.abs(BallPos[0] - x_ref)
                ytemp = K_xd * np.abs(BallPos[1] - y_ref)
                # xvtemp = v_xref - x_coef * BallVel[0]
                # XForce = x_coef * (K_xd * (np.abs(BallPos[0] - x_ref)) + 50 * (v_xref - x_coef * BallVel[0]))
                # YForce = y_coef * (K_xd * (np.abs(BallPos[1] - y_ref)) + 60 * (v_yref - y_coef * BallVel[1]))
                XForce = x_coef * 50 * (v_xref - x_coef * BallVel[0])
                YForce = y_coef * 50 * (v_yref - y_coef * BallVel[1])
                ZForce = - K_zd * (BallPos[2] - z_ref) - K_zvdown * (BallVel[2]- v_zref)
                # print("x pos and vel is ", BallPos[0], BallVel[0])

                # if index == 2 and i == 0:
                #     print(BallPos[1])
                #     print("down state: ", XForce, YForce, ZForce)
        elif BallPos[2] <= z_ref:
            
            if flag == 1:
                if index == 1:
                    Point1Vel = np.concatenate([Point1Vel, [BallVel]], axis = 0)
                flag = 0
                index = index + 1
                xref_flag = 0
                if index == 3:
                    index = 0

                print("end pos: ", BallPos)
                print("end vel: ", BallVel)

            XForce = 0.0
            YForce = 0.0
            ZForce = 0.0

        ball1.setExternalForce(0, [0, 0, 0], [XForce, YForce, ZForce])

        t = i * t_step
        T = np.concatenate([T, [t]], axis = 0)
        BallPosition = np.concatenate([BallPosition, [BallPos]], axis = 0)
        BallVelocity = np.concatenate([BallVelocity, [BallVel]], axis = 0)
        ExternalForce = np.concatenate([ExternalForce, [[XForce, YForce, ZForce]]], axis = 0)
        sumF = np.sqrt(XForce**2 +  YForce**2 + ZForce**2)
        SumForce = np.concatenate([SumForce, [[sumF]]], axis = 0)
        world.integrate()

    T = T[1:,]
    Point1Pos = Point1Pos[1:,]
    Point1Vel = Point1Vel[1:,]
    BallPosition = BallPosition[1:,]
    BallVelocity = BallVelocity[1:,]
    ExternalForce = ExternalForce[1:,]
    SumForce = SumForce[1:,]

    Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'ResForce':SumForce, 'time': T, \
            "Point1Pos":Point1Pos, "Point1Vel":Point1Vel}
    # print(0)
    # return BallPosition, BallVelocity, ExternalForce, T
    return Data

def SetPoint_DriControl(ParamData):

    TraPoint_x = np.array([-0.2, -0.4, 0.0])
    TraPoint_y = np.array([0.0, 0.4, 0.4])

    flag = 0
    xref_flag = 0
    z_ref = 0.5
    x_ref = 0.0
    x_coef = 0.0
    v_zref = -6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    v_xref = 6
    index = 0
    K_xd = 400
    K_zd = 300
    K_zvup = 5
    K_zvdown = 30

    BallPos, BallVel = ball1.getState()
    print("init ball pos: ", BallPos)

    # set ball, arm end foot, contact force and joint state saved array
    BallPosition = np.array([[0.0, 0.0, 0.0]])          # the pos and vel state of ball
    BallVelocity = np.array([[0.0, 0.0, 0.0]])       # the pos and vel state of endfoot
    ExternalForce = np.array([[0.0, 0.0, 0.0]])         # the calculate force and real contact force betweem ball an foot
    SumForce = np.array([[0.0]])
    T = np.array([0.0])

    Point1Pos = np.array([[0.0, 0.0, 0.0]])
    Point1Vel = np.array([[0.0, 0.0, 0.0]])

    Point2State = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    Point3State = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
    for i in range(20000):
        time.sleep(0.001)

        BallPos, BallVel = ball1.getState()
        BallPos = BallPos[0:3]
        BallVel = BallVel[0:3]
        
        # print(TraPoint_x[index])
        # XForce, ZForce, flag, x_coef = ForceControl(BallPos, BallVel, flag, x_coef, TraPoint_x[index], TraPoint_y[index], z_ref, v_zref, v_xref, K_zd, K_xd, K_zvup, K_zvdown)
        if BallPos[2] > z_ref:
            if flag == 0:
                x_ref = BallPos[0]
                # x_coef = - BallVel[0] / np.abs(BallVel[0])
                if index == 0:
                    x_coef = -1
                    y_coef = 0

                    x_ref = TraPoint_x[index] - x_coef * 0.275
                    # x_ref = TraPoint_x[index] - x_coef * 0.2
                    y_ref = TraPoint_y[index] - y_coef * 0.1
                elif index == 1:
                    x_coef = 1
                    y_coef = 1

                    x_ref = TraPoint_x[index] - x_coef * 0.1
                    # y_ref = TraPoint_y[index] - y_coef * 0.2
                    y_ref = TraPoint_y[index] - y_coef * 0.275
                elif index == 2:
                    x_coef = 1
                    y_coef = -1

                    x_ref = TraPoint_x[index] - x_coef * 0.1
                    y_ref = TraPoint_y[index] - y_coef * 0.275
                    # y_ref = TraPoint_y[index] - y_coef * 0.2

                flag = 1
                print("=====================================================================")
                print("phase index: ", index)
                print("init pos: ", BallPos)
                print("init vel: ", BallVel)
            # print("x_coef, y_coef: ", x_coef, y_coef)

            if BallVel[2] >= 0:
                zx_ratio = np.abs(BallVel[0] / BallVel[2])
                zy_ratio = np.abs(BallVel[1] / BallVel[2])
                print("zx_ratio", zx_ratio)

                ZForce = - K_zd * (BallPos[2] - z_ref) - K_zvup * (BallVel[2])
                if index == 0 :
                    # XForce = - 300 * (np.abs(BallPos[0] - 0.3)) - 30 * np.abs(BallVel[0])
                    XForce = zx_ratio * ZForce
                    # if np.abs(BallVel[1]) > 0.1:
                    #     ZForce = - 50 * (BallPos[2] - z_ref) - K_zvup * (BallVel[2]) - 200 * (BallPos[1])
                    # else:
                    #     YForce = - zy_ratio * ZForce

                    YForce = - zy_ratio * ZForce
                    # YForce = 300 * (np.abs(BallPos[1])) + 50 * np.abs(BallVel[1])
                elif index == 1:
                    XForce = - x_coef * zx_ratio * ZForce
                    YForce = 0
                
                elif index == 2:
                    XForce = x_coef * zx_ratio * ZForce
                    YForce = - y_coef * zy_ratio * ZForce
                    # XForce = - 300 * (np.abs(BallPos[0] + 0.2)) - 30 * np.abs(BallVel[0])
                    # YForce = - 300 * (np.abs(BallPos[1] - 0.8)) -  50 * np.abs(BallVel[1])
            
                # print("x pos and vel is ", BallPos[0], BallVel[0])
                # print("x_coef is ", x_coef)
                # if index == 2:
                #     print(BallVel)
                #     print("up state: ", XForce, YForce, ZForce)

            elif BallVel[2] <= 0:
                if xref_flag == 0:
                    # x_ref = TraPoint_x[index] - x_coef * 0.3
                    # y_ref = TraPoint_y[index] - y_coef * 0.1

                    T_free = - (z_ref - 0.15) / v_zref
                    v_xref = np.abs(x_ref - TraPoint_x[index]) / T_free
                    v_yref = np.abs(y_ref - TraPoint_y[index]) / T_free
                    # i = 0
                    if index == 1:
                        Point1Pos = np.concatenate([Point1Pos, [BallPos]], axis = 0)
                    xref_flag = 1
                    
                    print("max height pos: ", BallPos)
                    print("max height vel: ", BallVel)
                    print("x_ref, y_ref, T_free: ", x_ref, y_ref, T_free)
                    print("v_xref, v_yref: ", v_xref, v_yref)
                # XForce = x_coef * (K_xd * (BallPos[0] - x_ref) + 10 * (BallVel[2] - x_coef * v_xref))
                # xtemp = np.abs(BallPos[0] - x_ref)
                ytemp = K_xd * np.abs(BallPos[1] - y_ref)
                # xvtemp = v_xref - x_coef * BallVel[0]
                XForce = x_coef * (K_xd * (np.abs(BallPos[0] - x_ref)) + 50 * (v_xref - x_coef * BallVel[0]))
                YForce = y_coef * (K_xd * (np.abs(BallPos[1] - y_ref)) + 60 * (v_yref - y_coef * BallVel[1]))
                # XForce = x_coef * 50 * (v_xref - x_coef * BallVel[0])
                # YForce = y_coef * 50 * (v_yref - y_coef * BallVel[1])
                ZForce = - K_zd * (BallPos[2] - z_ref) - K_zvdown * (BallVel[2]- v_zref)
                # print("x pos and vel is ", BallPos[0], BallVel[0])

                # if index == 2 and i == 0:
                #     print(BallPos[1])
                #     print("down state: ", XForce, YForce, ZForce)
        elif BallPos[2] <= z_ref:
            
            if flag == 1:
                if index == 1:
                    Point1Vel = np.concatenate([Point1Vel, [BallVel]], axis = 0)
                flag = 0
                index = index + 1
                xref_flag = 0
                if index == 3:
                    index = 0

                print("end pos: ", BallPos)
                print("end vel: ", BallVel)

            XForce = 0.0
            YForce = 0.0
            ZForce = 0.0

        ball1.setExternalForce(0, [0, 0, 0], [XForce, YForce, ZForce])

        t = i * t_step
        T = np.concatenate([T, [t]], axis = 0)
        BallPosition = np.concatenate([BallPosition, [BallPos]], axis = 0)
        BallVelocity = np.concatenate([BallVelocity, [BallVel]], axis = 0)
        ExternalForce = np.concatenate([ExternalForce, [[XForce, YForce, ZForce]]], axis = 0)
        sumF = np.sqrt(XForce**2 +  YForce**2 + ZForce**2)
        SumForce = np.concatenate([SumForce, [[sumF]]], axis = 0)
        world.integrate()

    T = T[1:,]
    Point1Pos = Point1Pos[1:,]
    Point1Vel = Point1Vel[1:,]
    BallPosition = BallPosition[1:,]
    BallVelocity = BallVelocity[1:,]
    ExternalForce = ExternalForce[1:,]
    SumForce = SumForce[1:,]

    Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'ResForce':SumForce, 'time': T, \
            "Point1Pos":Point1Pos, "Point1Vel":Point1Vel}
    # print(0)
    # return BallPosition, BallVelocity, ExternalForce, T
    return Data

def SetPoint_MPCControl(ParamData):

    TraPoint_x = np.array([-0.2, -0.5, 0.1])
    TraPoint_y = np.array([0.0, 0.6, 0.6])

    flag = 0
    m = 0.4
    z_ref = 0.5
    v_zref = -6.0
    v_xref = -6.0
    v_yref = -6.0
    dx_ref = 0.6
    dy_ref = 0.6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

    xtra = 0.0
    ytra = 0.0
    # v_xref = 6
    index = 0
    gravity = world.getGravity()
    g = gravity[0]

    BallPos, BallVel = ball1.getState()
    # print("init ball pos: ", BallPos)

    # set ball, arm end foot, contact force and joint state saved array
    BallPosition = np.array([[0.0, 0.0, 0.0]])          # the pos and vel state of ball
    BallVelocity = np.array([[0.0, 0.0, 0.0]])       # the pos and vel state of endfoot
    ExternalForce = np.array([[0.0, 0.0, 0.0]])         # the calculate force and real contact force betweem ball an foot
    SumForce = np.array([[0.0]])
    T = np.array([0.0])

    Point1Pos = np.array([[0.0, 0.0, 0.0]])
    Point1Vel = np.array([[0.0, 0.0, 0.0]])

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x_b = model.set_variable(var_type='_x', var_name='x_b', shape=(1, 1))
    y_b = model.set_variable(var_type='_x', var_name='y_b', shape=(1, 1))
    z_b = model.set_variable(var_type='_x', var_name='z_b', shape=(1, 1))
    dx_b = model.set_variable(var_type='_x', var_name='dx_b', shape=(1, 1))
    dy_b = model.set_variable(var_type='_x', var_name='dy_b', shape=(1, 1))
    dz_b = model.set_variable(var_type='_x', var_name='dz_b', shape=(1, 1))
    u_x = model.set_variable(var_type='_u', var_name='u_x', shape=(1, 1))
    u_y = model.set_variable(var_type='_u', var_name='u_y', shape=(1, 1))
    u_z = model.set_variable(var_type='_u', var_name='u_z', shape=(1, 1))

    model.set_rhs('x_b', dx_b)
    model.set_rhs('y_b', dy_b)
    model.set_rhs('z_b', dz_b)
    dx_b_next = vertcat(
        tanh_sig(z_b - z_ref) * u_x / m,
    )
    dy_b_next = vertcat(
        tanh_sig(z_b - z_ref) * u_y / m,
    )
    dz_b_next = vertcat(
        g + tanh_sig(z_b - z_ref) * u_z / m,
    )
    model.set_rhs('dx_b', dx_b_next)
    model.set_rhs('dy_b', dy_b_next)
    model.set_rhs('dz_b', dz_b_next)

    model.setup()
        
    for i in range(20000):
        time.sleep(0.001)

        BallPos, BallVel = ball1.getState()
        BallPos = BallPos[0:3]
        BallVel = BallVel[0:3]
        
        if BallPos[2] > z_ref:

            if flag == 0:
                if index == 0:
                    v_xref = dx_ref / z_ref * v_zref
                    v_yref = 0.0
                    xtra = TraPoint_x[index] + dx_ref
                    ytra = 0.0

                elif index == 1:
                    v_xref = - (dx_ref / 3) / z_ref * v_zref
                    v_yref = - (2 * dy_ref / 3)/ z_ref * v_zref
                    xtra = TraPoint_x[index] - dx_ref / 3
                    ytra = TraPoint_y[index] + dy_ref / 3
                    # break

                elif index == 2:
                    v_xref = - (dx_ref / 3) / z_ref * v_zref
                    v_yref = (2 * dy_ref / 3) / z_ref * v_zref
                    xtra = TraPoint_x[index] - dx_ref / 3
                    ytra = TraPoint_y[index] + (2 * dy_ref) / 3

                # # MPC controller setup
                # model = Dribble_model()
                # # model_endtime = datetime.datetime.now()
                # mpc = Dribble_mpc(model, xtra, ytra, v_xref, v_yref, v_zref)
                # simulator = Dribble_simulator(model)
                # estimator = do_mpc.estimator.StateFeedback(model)

                mpc = do_mpc.controller.MPC(model)

                setup_mpc = {
                    'n_horizon': 150,
                    't_step': 0.0005,
                }
                mpc.set_param(**setup_mpc)

                xq1 = 2000.0
                yq2 = 1000.0
                zq3 = 1000.0
                vxq1 = 2000.0
                vyq2 = 1000.0
                vzq3 = 2000.0
                r1 = 0.001
                r2 = 0.001
                r3 = 0.0001

                lterm = xq1 * (model.x['x_b'] - xtra) ** 2 + yq2 * (model.x['y_b'] - ytra) ** 2 + zq3 * (model.x['z_b'] - z_ref) ** 2 + \
                        vxq1 * (model.x['dx_b'] - v_xref) ** 2 + vyq2 * (model.x['dy_b'] - v_yref) ** 2 + vzq3 * (model.x['dz_b'] - v_zref) ** 2 + \
                        r1 * (model.u['u_x']) ** 2 + r2 * (model.u['u_y']) ** 2 + r3 * (model.u['u_z']) ** 2

                mterm = xq1 * (model.x['x_b'] - xtra) ** 2 + yq2 * (model.x['y_b'] - ytra) ** 2 + zq3 * (model.x['z_b'] - z_ref) ** 2 + \
                        vxq1 * (model.x['dx_b'] - v_xref) ** 2 + vyq2 * (model.x['dy_b'] - v_yref) ** 2 + vzq3 * (model.x['dz_b'] - v_zref) ** 2

                mpc.set_objective(mterm=mterm, lterm=lterm)

                mpc.bounds['lower', '_x', 'x_b'] = -1.5
                mpc.bounds['upper', '_x', 'x_b'] = 1.0

                mpc.bounds['lower', '_x', 'y_b'] = -0.5
                mpc.bounds['upper', '_x', 'y_b'] = 1.5

                mpc.bounds['lower', '_x', 'z_b'] = 0.0
                mpc.bounds['upper', '_x', 'z_b'] = 1.0

                mpc.bounds['lower', '_u', 'u_x'] = -500.0
                mpc.bounds['upper', '_u', 'u_x'] = 500.0

                mpc.bounds['lower', '_u', 'u_y'] = -500.0
                mpc.bounds['upper', '_u', 'u_y'] = 500.0

                mpc.bounds['lower', '_u', 'u_z'] = -500.0
                mpc.bounds['upper', '_u', 'u_z'] = 0.0

                mpc.setup()

                simulator = do_mpc.simulator.Simulator(model)
                simulator.set_param(t_step=0.0005)

                simulator.setup()

                estimator = do_mpc.estimator.StateFeedback(model)

                flag = 1
                print("=====================================================================")
                print("phase index: ", index)
                print("init pos: ", BallPos)
                print("init vel: ", BallVel)
            # print("x_coef, y_coef: ", x_coef, y_coef)
            
            x0  = np.concatenate([BallPos, BallVel])
            x0 = x0.reshape(-1, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
            simulator.x0 = x0
            
            mpc.x0 = x0
            estimator.x0 = x0

            mpc.set_initial_guess()

            mpc.reset_history()
            Force = mpc.make_step(x0)
            y_next = simulator.make_step(Force)
            x0 = estimator.make_step(y_next)

            XForce = Force[0, 0]
            YForce = Force[1, 0]
            ZForce = Force[2, 0]
           
            print("**********************************************************************************************")
            print("x0: ", x0)
            print("Force: ", Force[0, 0], Force[1, 0], Force[2, 0])
            print("Ball pos and vel is ", BallPos, BallVel)
            print("xtra, ytra, v_xref, v_yref, v_zref: ",  xtra, ytra, v_xref, v_yref, v_zref)
            
        elif BallPos[2] <= z_ref:
            
            if flag == 1:
                print("=======================================================================================")
                print("free Ball pos and vel is ", BallPos, BallVel)
                # break
                if index == 1:
                    Point1Vel = np.concatenate([Point1Vel, [BallVel]], axis = 0)
                flag = 0
                index = index + 1
                if index == 3:
                    index = 0

                print("end pos: ", BallPos)
                print("end vel: ", BallVel)

            XForce = 0.0
            YForce = 0.0
            ZForce = 0.0

        ball1.setExternalForce(0, [0, 0, 0], [XForce, YForce, ZForce])

        t = i * t_step
        T = np.concatenate([T, [t]], axis = 0)
        BallPosition = np.concatenate([BallPosition, [BallPos]], axis = 0)
        BallVelocity = np.concatenate([BallVelocity, [BallVel]], axis = 0)
        ExternalForce = np.concatenate([ExternalForce, [[XForce, YForce, ZForce]]], axis = 0)
        sumF = np.sqrt(XForce**2 +  YForce**2 + ZForce**2)
        SumForce = np.concatenate([SumForce, [[sumF]]], axis = 0)
        world.integrate()

    T = T[1:,]
    Point1Pos = Point1Pos[1:,]
    Point1Vel = Point1Vel[1:,]
    BallPosition = BallPosition[1:,]
    BallVelocity = BallVelocity[1:,]
    ExternalForce = ExternalForce[1:,]
    SumForce = SumForce[1:,]

    Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'ResForce':SumForce, 'time': T, \
            "Point1Pos":Point1Pos, "Point1Vel":Point1Vel}
    # print(0)
    # return BallPosition, BallVelocity, ExternalForce, T
    return Data

def TRI_MPCControl(ParamData):

    TraPoint_x = np.array([-0.1, -0.3, 0.1])
    # TraPoint_x = np.array([-0.2, -0.2, 0.0])
    TraPoint_y = np.array([0.0, 0.3, 0.3])

    flag = 0
    dx_ref = 0.6
    dy_ref = 0.6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

    xtra = 0.0
    ytra = 0.0
    index = 0

    # mpc controller params
    sim_t_step = ParamData["environment"]["t_step"]
    sim_time = ParamData["environment"]["sim_time"]
    t_force = ParamData["MPCController"]["t_force"]
    z_ref = ParamData["environment"]["z_ref"]
    v_zref = ParamData["environment"]["v_zref"]

    BallPos, BallVel = ball1.getState()
    # print("init ball pos: ", BallPos)

    # set ball, arm end foot, contact force and joint state saved array
    BallPosition = np.array([[0.0, 0.0, 0.0]])          # the pos and vel state of ball
    BallVelocity = np.array([[0.0, 0.0, 0.0]])       # the pos and vel state of endfoot
    ExternalForce = np.array([[0.0, 0.0, 0.0]])         # the calculate force and real contact force betweem ball an foot
    SumForce = np.array([[0.0]])
    T = np.array([0.0])

    Point1Pos = np.array([[0.0, 0.0, 0.0]])
    Point1Vel = np.array([[0.0, 0.0, 0.0]])
    # RefCoef = np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    RefCoef = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])

    for i in range(sim_time):
        time.sleep(0.005)

        BallPos, BallVel = ball1.getState()
        BallPos = BallPos[0:3]
        BallVel = BallVel[0:3]
        
        if BallPos[2] > z_ref:

            if flag == 0:
                if index == 0:
                    v_xref = dx_ref / (z_ref - 0.15) * v_zref
                    v_yref = 0.0
                    xtra = TraPoint_x[index] + dx_ref
                    ytra = 0.0
                    print("Ball pos and vel is ", BallPos, BallVel)
                    print("xtra, ytra, v_xref, v_yref, v_zref: ",  xtra, ytra, v_xref, v_yref, v_zref)

                elif index == 1:
                    v_xref = - (dx_ref / 3) / (z_ref - 0.15) * v_zref
                    v_yref = - (2 * dy_ref / 3)/ (z_ref - 0.15) * v_zref
                    xtra = TraPoint_x[index] - dx_ref / 3
                    ytra = TraPoint_y[index] - 2 * dy_ref / 3

                    # v_xref = - dx_ref / (z_ref - 0.15) * v_zref
                    # v_yref = 0.0
                    # xtra = TraPoint_x[index] - dx_ref
                    # ytra = 0.0
                    print("Ball pos and vel is ", BallPos, BallVel)
                    print("xtra, ytra, v_xref, v_yref, v_zref: ",  xtra, ytra, v_xref, v_yref, v_zref)
                    # break

                elif index == 2:
                    v_xref = - (dx_ref / 3) / (z_ref - 0.15) * v_zref
                    v_yref = (2 * dy_ref / 3) / (z_ref - 0.15) * v_zref
                    xtra = TraPoint_x[index] - dx_ref / 3
                    ytra = TraPoint_y[index] + (2 * dy_ref) / 3
                    print("Ball pos and vel is ", BallPos, BallVel)
                    print("xtra, ytra, v_xref, v_yref, v_zref: ",  xtra, ytra, v_xref, v_yref, v_zref)
                    # break

                flag = 1

            # if i_force == -1:
                PosTar = np.array([xtra, ytra, z_ref])
                VelTar = np.array([v_xref, v_yref, v_zref])
                # x_coef, y_coef, z_coef = TriCal(t_force, BallPos, BallVel, PosTar, VelTar)
                # Coef = np.array([[x_coef, y_coef, z_coef]])
                # i_init = i

                ThetaCoef = TrajOptim(t_force, BallPos, BallVel, PosTar, VelTar)
                RefCoef = np.concatenate([RefCoef, [ThetaCoef]], axis = 0)
            
                # # MPC controller setup
                model = template_model()
                # mpc = template_mpc(model, x_coef, y_coef, z_coef)
                mpc = template_mpc(model, ThetaCoef[0], ThetaCoef[1], ThetaCoef[2])
                simulator = template_simulator(model, sim_t_step)
                estimator = do_mpc.estimator.StateFeedback(model)

            x0  = np.concatenate([BallPos, BallVel])
            x0 = x0.reshape(-1, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
            simulator.x0 = x0
            
            mpc.x0 = x0
            estimator.x0 = x0

            mpc.set_initial_guess()

            Force = mpc.make_step(x0)
            y_next = simulator.make_step(Force)
            x0 = estimator.make_step(y_next)

            XForce = Force[0, 0]
            YForce = Force[1, 0]
            ZForce = Force[2, 0]
            print("**********************************************************************************************")
            print("Force: ", XForce, YForce, ZForce)
            print("Ball pos and vel is ", BallPos, BallVel)
            print("x_coef, y_coef, z_coef: ", ThetaCoef[0], ThetaCoef[1], ThetaCoef[2])
            print("xtra, ytra, v_xref, v_yref, v_zref: ",  xtra, ytra, v_xref, v_yref, v_zref)
            
        elif BallPos[2] <= z_ref:
            
            if flag == 1:
                print("=======================================================================================")
                print("free Ball pos and vel is ", BallPos, BallVel)
                # break
                if index == 1:
                    Point1Vel = np.concatenate([Point1Vel, [BallVel]], axis = 0)
                flag = 0
                index = index + 1
                if index == 3:
                    index = 0

                print("end pos: ", BallPos)
                print("end vel: ", BallVel)

            XForce = 0.0
            YForce = 0.0
            ZForce = 0.0

        ball1.setExternalForce(0, [0, 0, 0], [XForce, YForce, ZForce])

        t = i * t_step
        T = np.concatenate([T, [t]], axis = 0)
        BallPosition = np.concatenate([BallPosition, [BallPos]], axis = 0)
        BallVelocity = np.concatenate([BallVelocity, [BallVel]], axis = 0)
        ExternalForce = np.concatenate([ExternalForce, [[XForce, YForce, ZForce]]], axis = 0)
        sumF = np.sqrt(XForce**2 +  YForce**2 + ZForce**2)
        SumForce = np.concatenate([SumForce, [[sumF]]], axis = 0)
        world.integrate()

    RefCoef = RefCoef[1:,]
    T = T[1:,]
    Point1Pos = Point1Pos[1:,]
    Point1Vel = Point1Vel[1:,]
    BallPosition = BallPosition[1:,]
    BallVelocity = BallVelocity[1:,]
    ExternalForce = ExternalForce[1:,]
    SumForce = SumForce[1:,]
    TraPoint = np.concatenate([[TraPoint_x], [TraPoint_y]], axis = 0)

    Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'ResForce': SumForce, 'time': T, \
            "Point1Pos": Point1Pos, "Point1Vel": Point1Vel, "TraPoint": TraPoint, "RefTraCoef": RefCoef}
    # print(0)
    # return BallPosition, BallVelocity, ExternalForce, T
    return Data

def XboxDriControl(ParamData):

    flag = 0
    xref_flag = 0
    z_ref = 0.5
    x_ref = 0.0
    x_coef = 0.0
    v_zref = -6
    v_yref_init = 6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    v_xref_init = 6
    index = 0
    K_xd = 400
    K_zd = 300
    K_zvup = 5
    K_zvdown = 20
    button_b = 0

    BallPos, BallVel = ball1.getState()
    print("init ball pos: ", BallPos)

    # set ball, arm end foot, contact force and joint state saved array
    BallPosition = np.array([[0.0, 0.0, 0.0]])          # the pos and vel state of ball
    BallVelocity = np.array([[0.0, 0.0, 0.0]])       # the pos and vel state of endfoot
    ExternalForce = np.array([[0.0, 0.0, 0.0]])         # the calculate force and real contact force betweem ball an foot
    SumForce = np.array([[0.0]])
    T = np.array([0.0])

    for i in range(sim_time):
        time.sleep(0.0005)

        BallPos, BallVel = ball1.getState()
        BallPos = BallPos[0:3]
        BallVel = BallVel[0:3]

        v_xref = 6
        v_yref = 0

        if xbox.button_b.is_pressed and button_b == 0:
            button_b = 1
            # print(xbox.button_b.is_pressed)
        elif xbox.button_b.is_pressed and button_b == 1:
            button_b = 0

        # print(v_xref, v_yref, x_coef_down, y_coef_down)

        if BallPos[2] > z_ref:
            if flag == 0:
                if BallVel[1] == 0:
                    y_coef_up = 0
                else:
                    y_coef_up = - BallVel[1] / np.abs(BallVel[1])
                x_coef_up = - BallVel[0] / np.abs(BallVel[0])
                # y_coef_up = - BallVel[1] / np.abs(BallVel[1])
                x_coef_down = - BallVel[0] / np.abs(BallVel[0])
                y_coef_down = 0
                flag = 1
                print("=====================================================================")
                # print("phase index: ", index)
                print("init pos: ", BallPos)
                print("init vel: ", BallVel)
            # print("x_coef, y_coef: ", x_coef, y_coef)
            if BallVel[2] >= 0:
                zx_ratio = np.abs(BallVel[0] / BallVel[2])
                zy_ratio = np.abs(BallVel[1] / BallVel[2])

                ZForce = - K_zd * (BallPos[2] - z_ref) - K_zvup * (BallVel[2])
                XForce = - x_coef_up * zx_ratio * ZForce
                YForce = - y_coef_up * zy_ratio * ZForce
            
                # print("x pos and vel is ", BallPos[0], BallVel[0])
                # print("x_coef is ", x_coef)
                # if index == 2:
                #     print(BallVel)
                # print("up state: ", XForce, YForce, ZForce)

            elif BallVel[2] <= 0:

                if button_b == 1:
                    v_xratio = xbox.axis_r.x
                    v_yratio = - xbox.axis_r.y
                    v_xref = v_xref_init * np.abs(v_xratio)
                    v_yref = v_yref_init * np.abs(v_yratio)
                    if v_xref == 0:
                        x_coef_down = 0
                    else:
                        x_coef_down = np.abs(v_xratio) / v_xratio
                    if v_yref == 0:
                        y_coef_down = 0
                    else:
                        y_coef_down = np.abs(v_yratio) / v_yratio
                    print("v_xratio, v_yratio:", v_xratio, v_yratio)
                    print("v_xref, v_yref:", v_xref, v_yref)
                    print("x_coef_down, y_coef_down", x_coef_down, y_coef_down)
                    # xref_flag = 1
        
                # XForce = x_coef * (K_xd * (BallPos[0] - x_ref) + 10 * (BallVel[2] - x_coef * v_xref))
                # xvtemp = v_xref - x_coef * BallVel[0]
                # XForce = x_coef * (K_xd * (np.abs(BallPos[0] - x_ref)) + 30 * (v_xref - x_coef * BallVel[0]))
                # YForce = y_coef * (K_xd * (np.abs(BallPos[1] - y_ref)) + 30 * (v_yref - y_coef * BallVel[1]))
                XForce = x_coef_down * 20 * (v_xref - x_coef_down * BallVel[0])
                YForce = y_coef_down * 20 * (v_yref - y_coef_down * BallVel[1])
                ZForce = - K_zd * (BallPos[2] - z_ref) - K_zvdown * (BallVel[2]- v_zref)
                # print("x pos and vel is ", BallPos, BallVel)

                # if index == 2 and i == 0:
                #     print(BallPos[1])
                # print("down state: ", XForce, YForce, ZForce)
        elif BallPos[2] <= z_ref:

            if flag == 1:
                flag = 0
                xref_flag = 0

            # print("end pos: ", BallPos)
            # print("end vel: ", BallVel)

            XForce = 0.0
            YForce = 0.0
            ZForce = 0.0

        ball1.setExternalForce(0, [0, 0, 0], [XForce, YForce, ZForce])

        t = i * t_step
        T = np.concatenate([T, [t]], axis = 0)
        BallPosition = np.concatenate([BallPosition, [BallPos]], axis = 0)
        BallVelocity = np.concatenate([BallVelocity, [BallVel]], axis = 0)
        ExternalForce = np.concatenate([ExternalForce, [[XForce, YForce, ZForce]]], axis = 0)
        sumF = np.sqrt(XForce**2 +  YForce**2 + ZForce**2)
        SumForce = np.concatenate([SumForce, [[sumF]]], axis = 0)
        world.integrate()

    T = T[1:,]
    BallPosition = BallPosition[1:,]
    BallVelocity = BallVelocity[1:,]
    ExternalForce = ExternalForce[1:,]
    SumForce = SumForce[1:,]

    Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'ResForce':SumForce, 'time': T}
    # print(0)
    # return BallPosition, BallVelocity, ExternalForce, T
    return Data 


if __name__ == "__main__":
    # get params config data
    FilePath = os.path.dirname(os.path.abspath(__file__))
    ParamFilePath = FilePath + "/config/LISM_test.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # load activation file and urdf file
    raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
    # ball1_urdf_file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/urdf/ball.urdf"
    ball1_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/ball.urdf"
    # raisim world config setting
    world = raisim.World()

    # set simulation step
    t_step = ParamData["environment"]["t_step"] 
    sim_time = ParamData["environment"]["sim_time"]
    world.setTimeStep(t_step)
    ground = world.addGround(0, "floor")
    
    # set material collision property
    # world.setMaterialPairProp("rubber", "rub", 1, 0, 0)
    # world.setMaterialPairProp("rubber", "rub", 1.0, 0.85, 0.0001)     # ball rebound model test
    # world.setMaterialPairProp("default", "rub", 0.8, 1.0, 0.0001)
    # world.updateMaterialProp(raisim.MaterialManager(os.path.dirname(os.path.abspath(__file__)) + "/urdf/testMaterial.xml"))
    gravity = world.getGravity()

    world.setMaterialPairProp("default", "steel", 0.0, 1.0, 0.001)
    ball1 = world.addArticulatedSystem(ball1_urdf_file)
    print(ball1.getDOF())
    ball1.setName("ball1")
    gravity = world.getGravity()
    print(gravity)
    print(ball1.getGeneralizedCoordinateDim())

    jointNominalConfig = np.array([0.0, 0.0, 0.5,1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([4.0, 0.0, -5, 0.0, 0.0, 0.0])
    # jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Euler_matrix = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])
    Euler = Euler_matrix.as_euler('xyz', degrees=True)
    print(Euler)
    ball1.setGeneralizedCoordinate(jointNominalConfig)
    # print(ball1.getGeneralizedCoordinateDim())
    ball1.setGeneralizedVelocity(jointVelocityTarget)

    world.setMaterialPairProp("floor", "steel", 0.0, 1.0, 0.001)
    world.setMaterialPairProp("floor", "rub", 0.0, 1.0, 0.001)
 
    ## ======================= single object ====================
    # ball1 = world.addSphere(0.12, 0.8, "steel")
    # dummy_inertia = np.zeros([3, 3])
    # np.fill_diagonal(dummy_inertia, 0.1)
    # ball1 = world.addMesh(ball_file, 0.6, dummy_inertia, np.array([0, 0, 1]), 0.001, "rub")
    # ball1.setPosition(0, 0.0, 0.5)
    # ball1.setVelocity(1.0, 0.0, -5, 0.0, 0, 0)
    # BallPos = ball1.getPosition()
    # BallVel = ball1.getLinearVelocity()

    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)

    # # dribbling control
    # BallPosition, BallVelocity, ExternalForce, T = DriControl(ParamData)

    # Data = TRI_DriControl(ParamData)
    # Data = SetPoint_DriControl(ParamData)
    # Data = XboxDriControl(ParamData)
    # Data = SetPoint_MPCControl(ParamData)
    # mpc_stime = datetime.datetime.now()
    Data = TRI_MPCControl(ParamData)
    # mpc_endtime = datetime.datetime.now()
    # spendtime = mpc_endtime - mpc_stime
    # print("this mpc simulate time is: ", spendtime)
    # BallTest()

    # file save
    FileFlag = ParamData["environment"]["FileFlag"] 
    FileSave.DataSave(Data, ParamData, FileFlag)

    # # data visulization
    # Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'time': T}

    # DataPlot(Data)
    visualization.DataPlot(Data)
    visualization.RealCmpRef(Data)

    # print("force, ", ForceState[0:100, 1])

    server.killServer()