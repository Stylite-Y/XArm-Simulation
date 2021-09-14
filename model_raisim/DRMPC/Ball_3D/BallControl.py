import os
import sys
import numpy as np
from numpy.lib import index_tricks
import raisimpy as raisim
import datetime
import time
import yaml
import random
import shutil
import pickle
import matplotlib
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

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/utils")
print(os.path.abspath(os.path.dirname(__file__))) # get current file path
# from ParamsCalculate import ControlParamCal
# import visualization
# import FileSave

from Dribble_model import Dribble_model
from Dribble_mpc import Dribble_mpc
from Dribble_simulator import Dribble_simulator


# xbox = Xbox360Controller(0, axis_threshold=0.02)

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
    Period = 0.5
    N = Period / 0.0001
    xtra = t * 2 * math.pi / N
    ztra = np.cos(xtra)
    return xtra, ztra


def MPCControl(Pos_init, Vel_init, xtra, ztra, index):
    print("=========================================================================")
    show_animation = False
    store_results = False

    model = Dribble_model()
    mpc = Dribble_mpc(model, xtra, ztra, index)
    simulator = Dribble_simulator(model)

    estimator = do_mpc.estimator.StateFeedback(model)

    x0  = np.concatenate([Pos_init, Vel_init])
    x0 = x0.reshape(-1, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    simulator.x0 = x0
    
    mpc.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()

    mpc.reset_history()

    u0 = mpc.make_step(x0)
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
                elif index == 1:
                    x_coef = 1
                    y_coef = 1
                elif index == 2:
                    x_coef = 1
                    y_coef = -1

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

def SetPoint_MPCControl(ParamData):

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
                elif index == 1:
                    x_coef = 1
                    y_coef = 1
                elif index == 2:
                    x_coef = 1
                    y_coef = -1

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

                xtra, ztra = RefTra(i)
                Force = MPCControl(BallPos, BallVel, xtra, ztra, index)
                YForce = Force[1, 0]
                ZForce = Force[2, 0]
                XForce = Force[0, 0]
                print("**********************************************************************************************")
                print("Force: ", Force)
                print("Ball pos and vel is ", BallPos, BallVel)
            
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
                # XForce = x_coef * (K_xd * (np.abs(BallPos[0] - x_ref)) + 50 * (v_xref - x_coef * BallVel[0]))
                # YForce = y_coef * (K_xd * (np.abs(BallPos[1] - y_ref)) + 60 * (v_yref - y_coef * BallVel[1]))
                # XForce = x_coef * 50 * (v_xref - x_coef * BallVel[0])
                # YForce = y_coef * 50 * (v_yref - y_coef * BallVel[1])
                # ZForce = - K_zd * (BallPos[2] - z_ref) - K_zvdown * (BallVel[2]- v_zref)
                # print("x pos and vel is ", BallPos[0], BallVel[0])

                xtra, ztra = RefTra(i)
                Force = MPCControl(BallPos, BallVel, xtra, ztra, index)
                YForce = Force[1, 0]
                ZForce = Force[2, 0]
                XForce = Force[0, 0]
                print("**********************************************************************************************")
                print("Force: ", Force)
                print("Ball pos and vel is ", BallPos, BallVel)

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
    ground = world.addGround(0)
    
    # set material collision property
    # world.setMaterialPairProp("rubber", "rub", 1, 0, 0)
    world.setMaterialPairProp("rubber", "rub", 1.0, 0.85, 0.0001)     # ball rebound model test
    world.setMaterialPairProp("default", "rub", 0.8, 1.0, 0.0001)
    gravity = world.getGravity()

    world.setMaterialPairProp("default", "steel", 0.0, 1.0, 0.001)
    ball1 = world.addArticulatedSystem(ball1_urdf_file)
    print(ball1.getDOF())
    ball1.setName("ball1")
    gravity = world.getGravity()
    print(gravity)
    print(ball1.getGeneralizedCoordinateDim())

    jointNominalConfig = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([2.0, 0.0, -5, 0.0, 0.0, 0.0])
    ball1.setGeneralizedCoordinate(jointNominalConfig)
    # print(ball1.getGeneralizedCoordinateDim())
    ball1.setGeneralizedVelocity(jointVelocityTarget)

    world.setMaterialPairProp("default", "steel", 0.0, 1.0, 0.001)
    world.setMaterialPairProp("default", "rub", 0.0, 1.0, 0.001)

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
    Data = SetPoint_MPCControl(ParamData)

    # # file save
    # Data = FileSave.DataSave(BallState, EndFootState, ForceState, JointTorque, JointVelSaved, T, ParamData)

    # # data visulization
    # Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'time': T}

    DataPlot(Data)

    # print("force, ", ForceState[0:100, 1])

    server.killServer()