import os
import sys
import numpy as np
import math
import raisimpy as raisim
import time
import yaml
import pickle
import matplotlib.pyplot as plt
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
    T = Data['time']

    plt.figure()
    plt.title('Ball motion in zx plane', fontsize = 20)

    plt.subplot(311)
    plt.plot(T, BallPosition[:, 0], label='Ball x-axis Position')
    # plt.plot(T, BallVelocity[:, 0], label='Ball x-axis Velocity')
    plt.plot(T, BallPosition[:, 2], label='Ball z-axis Position')
    # plt.plot(T, line2, label='highest Velocity')
    plt.axis([0, max(T)*1.05, -max(BallPosition[:, 2])*2, max(BallPosition[:, 2])*2])
    plt.xlabel('time (s)')
    plt.ylabel('Ball Position (m)')
    plt.legend(loc='upper right')


    plt.subplot(312)
    plt.plot(T, BallVelocity[:, 0], label='Ball x-axis Velocity')
    plt.plot(T, BallVelocity[:, 2], label='Ball z-axis Velocity')

    plt.axis([0, max(T)*1.05, -max(BallVelocity[:, 2])*2, max(BallVelocity[:, 2])*2])
    plt.xlabel('time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc='upper right')

    plt.subplot(313)
    plt.plot(T, ExternalForce[:, 0], label='Ball x-axis Force')
    plt.plot(T, ExternalForce[:, 2], label='Ball z-axis Force')

    plt.axis([0, max(T)*1.05, -max(ExternalForce[:, 0])*2, max(ExternalForce[:, 0])*2])
    plt.xlabel('time (s)')
    plt.ylabel('Force (N)')
    plt.legend(loc='upper right')

    plt.figure()
    plt.scatter(BallPosition[:, 0], BallPosition[:, 2], label='Ball motion trajectory')
    plt.xlabel('x-axis position (m)', fontsize = 15)
    plt.ylabel('z-axis position (m)', fontsize = 15)
    # plt.legend(loc='upper right')
    plt.title('Ball motion trajectory', fontsize = 20)

    plt.figure()
    plt.plot(BallPosition[:, 0], ExternalForce[:, 0], label='Ball x-axis Pos-Force')
    plt.plot(BallPosition[:, 2], ExternalForce[:, 2], label='Ball z-axis Pos-Force')
    plt.xlabel('Position (m)')
    plt.ylabel('Force (N)')
    plt.axis([-0.8, 0.8, -200, 200])
    plt.legend(loc='upper right')
    plt.title('Ball Pos-Force trajectory', fontsize = 20)

    plt.show()

def RefTra(t):
    Period = 0.5
    N = Period / 0.0001
    xtra = t * 2 * math.pi / N
    ztra = np.cos(xtra)
    return xtra, ztra

def MPCControl(Pos_init, Vel_init, xtra, ztra):
    print("=========================================================================")
    show_animation = False
    store_results = False

    model = Dribble_model()
    mpc = Dribble_mpc(model, xtra, ztra)
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

def DriControl(ParamData):

    TraPoint_x = np.array([[-0.1, -0.3, 0.1]])
    TraPoint_y = np.array([[0.0, 0.2, 0.2]])

    flag = 0
    xref_flag = 0
    z_ref = 0.5
    x_ref = 0.0
    x_coef = 0.0
    v_zref = -6
    v_xref = 6
    index = 0
    K_xd = 500
    K_zd = 300
    K_zvup = 5
    K_zvdown = 15

    # set ball, arm end foot, contact force and joint state saved array
    BallPosition = np.array([[0.0, 0.0, 0.0]])          # the pos and vel state of ball
    BallVelocity = np.array([[0.0, 0.0, 0.0]])       # the pos and vel state of endfoot
    ExternalForce = np.array([[0.0, 0.0, 0.0]])         # the calculate force and real contact force betweem ball an foot
    T = np.array([0.0])

    for i in range(20000):
        time.sleep(0.0001)

        BallPos, BallVel = ball1.getState() 
        BallPos = BallPos[0:3]
        BallVel = BallVel[0:3]
        
        if index == 2:
            index = 0
        if BallPos[2] > z_ref:
            if flag == 0:
                x_ref = BallPos[0]
                x_coef = - BallVel[0] / np.abs(BallVel[0])
                flag = 1

            if BallVel[2] >= 0:
                zx_ratio = np.abs(BallVel[0] / BallVel[2])
                # zy_ratio = np.abs(BallVel[1] / BallVel[2])

                # ZForce = - K_zd * (BallPos[2] - z_ref) - K_zvup * (BallVel[2])
                # XForce = - x_coef * zx_ratio * ZForce
                # print("x pos and vel is ", BallPos[0], BallVel[0])
                # print("x_coef is ", x_coef)
                # print("up state: ", XForce, ZForce)
                xtra, ztra = RefTra(i)
                Force = MPCControl(BallPos, BallVel, xtra, ztra)
                YForce = Force[1, 0]
                ZForce = Force[2, 0]
                XForce = Force[0, 0]
                print("**********************************************************************************************")
                print("Force: ", Force)
                print("Ball pos and vel is ", BallPos, BallVel)

            elif BallVel[2] <= 0:
                if xref_flag == 0:
                    x_ref = BallPos[0] + x_coef * 0.1
                    xref_flag = 1
                # XForce = x_coef * (K_xd * (BallPos[0] - x_ref) + 10 * (BallVel[2] - x_coef * v_xref))
                # XForce = x_coef * 20 * (v_xref - x_coef * BallVel[0])
                # XForce = x_coef * (K_xd * (np.abs(BallPos[0] - x_ref)) + 20 * (v_xref - x_coef * BallVel[0]))
                # ZForce = - K_zd * (BallPos[2] - z_ref) - K_zvdown * (BallVel[2]- v_zref)
                xtra, ztra = RefTra(i)
                Force = MPCControl(BallPos, BallVel, xtra, ztra)
                XForce = Force[0, 0]
                YForce = Force[1, 0]
                ZForce = Force[2, 0]
                print("**********************************************************************************************")
                print("Force: ", Force)
                print("Ball pos and vel is ", BallPos, BallVel)

                # print("x pos and vel is ", BallPos[0], BallVel[0])
                # print("down state: ", XForce, ZForce)

        elif BallPos[2] <= z_ref:
            if flag == 1:
                flag = 0
            xref_flag = 0
            XForce = 0.0
            YForce = 0.0
            ZForce = 0.0
            print("Ball pos and vel is ", BallPos, BallVel)


        ball1.setExternalForce(0, [0, 0, 0], [XForce, YForce, ZForce])

        t = i * t_step
        T = np.concatenate([T, [t]], axis = 0)
        BallPosition = np.concatenate([BallPosition, [BallPos]], axis = 0)
        BallVelocity = np.concatenate([BallVelocity, [BallVel]], axis = 0)
        ExternalForce = np.concatenate([ExternalForce, [[XForce, 0.0, ZForce]]], axis = 0)

        world.integrate()

    T = T[1:,]
    BallPosition = BallPosition[1:,]
    BallVelocity = BallVelocity[1:,]
    ExternalForce = ExternalForce[1:,]
    Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'time': T}
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
    Data = DriControl(ParamData)

    # # file save
    # Data = FileSave.DataSave(BallState, EndFootState, ForceState, JointTorque, JointVelSaved, T, ParamData)

    # # data visulization
    # Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'time': T}

    DataPlot(Data)

    # print("force, ", ForceState[0:100, 1])

    server.killServer()