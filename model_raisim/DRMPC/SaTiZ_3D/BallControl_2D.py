from math import e
import os
import sys
import numpy as np
from numpy.core.numeric import NaN
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

def PaperSim(ParamData):
    """
    this part is to replicate the experiment of paper:
    S. Haddadin, K. Krieger, A. Albu-Schäffer and T. Lilge, "Exploiting Elastic Energy Storage for “Blind” Cyclic Manipulation: 
    Modeling, Stability Analysis, Control, and Experiments for Dribbling," in IEEE Transactions on Robotics
    """

    EnvParam = ParamData["environment"]
    t_steps = EnvParam["t_step"]

    ## ====================
    # ball control initial pos and vel setting
    jointNominalConfig = np.array([0.35, 0.08, 0.1,1.0, 0.0, 0.0, 0.0])
    # jointNominalConfig = np.array([0.34, -0.05, 0.5,1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([0.0, 0.40, -5.0, 0.0, 0.0, 0.0])
    ball1.setGeneralizedCoordinate(jointNominalConfig)
    ball1.setGeneralizedVelocity(jointVelocityTarget)

    ## ===================
    # Arm initial setting
    # ArmTest_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/LISM_Arm_hand.urdf"
    world.setMaterialPairProp("rub", "rub", 1.0, 0.5, 0.0001)
    DRArm = world.addArticulatedSystem(Arm_urdf_file)
    DRArm.setName("DRArm")
    print(DRArm.getGeneralizedCoordinateDim())

    jointNominalConfig_Arm = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    DRArm.setGeneralizedCoordinate(jointNominalConfig_Arm)

    ## ===================
    # control params
    t_flag = 0
    T_Period = 0.15
    A = 0.1
    z0 = 0.5
    D_err_sum = 0.0
    phi_err_sum = 0.0
    phi_err_now = 0.0

    # PID params for desired angle cal
    K_Bdes_p = 1
    K_Bdes_d = 0.0005
    K_Bdes_I = 0.3
    K_Gdes_p = 1
    K_Gdes_d = 0.0002
    K_Gdes_I = 0.2
    xhand_err_last = 0.0
    yhand_err_last = 0.0
    zhand_err_last = 0.0
    Betahand_err_last = 0.0
    Gammahand_err_last = 0.0
    
    # ref point
    x_Ball_des = 0.4
    y_Ball_des = 0.0
    phi_des = math.atan(0.08 / 0.35)
    Beta_des = 0.0
    Gamma_des = 0.0
    D_des = math.sqrt(0.35 ** 2 + 0.08 ** 2)


    ## ================
    # arm and ball state
    BallPos, BallVel = ball1.getState()
    JointPos, JointVel = DRArm.getState()
    FootFrameId = DRArm.getFrameIdxByName("lower_hand_y")
    FootPos = DRArm.getFramePosition(FootFrameId)
    FootVel = DRArm.getFrameVelocity(FootFrameId)
    # print(FootFrameId)

    ## ===================
    # Jacobin matrix
    Jacobian = DRArm.getDenseFrameJacobian("lower_hand_y")
    # print(Jacobian.shape, Jacobian)

    Dof = DRArm.getDOF()
    print("the dof of the arm is: ", Dof)
    HandId = DRArm.getBodyIdx("hand")
    OrientaHand = DRArm.getBodyOrientation(HandId)
    # print(OrientaHand)
    # help(raisim.ArticulatedSystem)

    for i in range(10000):
        time.sleep(0.008)

        ## =======================
        # arm and ball current state
        BallPos, BallVel = ball1.getState()
        JointPos, JointVel = DRArm.getState()
        FootFrameId = DRArm.getFrameIdxByName("lower_add")
        FootPos = DRArm.getFramePosition(FootFrameId)
        FootVel = DRArm.getFrameVelocity(FootFrameId)
        OrientaHand = DRArm.getBodyOrientation(HandId)
        phi_hand = math.atan(FootPos[1] / FootPos[0])
        Beta_hand = math.asin(- OrientaHand[2, 0])
        # if Beta_hand > 0:
        #     Beta_hand = -Beta_hand - 1.57
        Gamma_hand = math.atan2(OrientaHand[2, 1], OrientaHand[2, 2])
        
        ## ====================
        # contact detect
        ContactPoint = DRArm.getContacts()
        contact_flag = False
        for c in ContactPoint:
            contact_flag = c.getlocalBodyIndex() == DRArm.getBodyIdx("hand")
            if(contact_flag):
                break
            pass
        # print(contact_flag)

        ## ===================
        # d and phi cal of ball pos
        D_ball = math.sqrt(BallPos[0] ** 2 + BallPos[1] ** 2)
        phi_ball = math.atan(BallPos[1] / BallPos[0])
        dh = 0.6

        # arm desired position
        # x_hand_des = (D_ball - dh) * math.cos(phi_ball)
        # y_hand_des = (D_ball - dh) * math.sin(phi_ball)
        x_hand_des = BallPos[0] - 0.06
        y_hand_des = BallPos[1]
        if contact_flag and t_flag == 0:
            phi_err_last = phi_err_now
            phi_err_now = (phi_des - phi_ball)
            phi_err_sum = phi_err_now * t_steps + phi_err_sum
            D_now = D_ball
            D_err_now = D_des - D_now
            D_err_sum = D_err_sum + D_err_now * t_steps
            phi_end_des = math.atan(y_hand_des / x_hand_des)
            Beta_end_des = - (K_Bdes_p * (D_des - D_now) + K_Bdes_d * (phi_err_now / t_steps) + K_Bdes_I * (D_err_sum))
            Gamma_end_des = (K_Gdes_p * (phi_des - phi_ball) + K_Gdes_d * (phi_err_now / t_steps) + K_Gdes_I * (phi_err_sum))
            t_flag = 1
            # if Beta_end_des > 10:
            #     break
        elif contact_flag == False:
            Beta_end_des = 0.0
            Gamma_end_des = 0.0
            t_flag = 0

        print("======================================")
        print("D_ball D_des , D_err_now is: ", D_ball, D_des, D_des - D_ball, D_err_sum)
        print("phi_ball, phi_err_now, phi_des phi_err_sum is: ", phi_ball, phi_err_now, phi_des, phi_err_now / t_steps, phi_err_sum)
        print("Beta_des is: ", Beta_end_des)
        print("Gamma_des is: ", Gamma_end_des)
        print("FootPos BallPos is:", FootPos)

        ## ===================
        # Jacobin matrix 
        a11 = - EnvParam["UpperArmLength"] * np.cos(JointPos[1]) - EnvParam["LowerArmLength"] * np.cos(JointPos[1] + math.pi / 2 - JointPos[2])
        a12 = EnvParam["UpperArmLength"] * np.sin(JointPos[1]) + EnvParam["LowerArmLength"] * np.sin(JointPos[1] + math.pi / 2 - JointPos[2])
        a21 = EnvParam["LowerArmLength"] * np.cos(JointPos[1] + math.pi / 2 - JointPos[2])
        a22 = - EnvParam["LowerArmLength"] * np.sin(JointPos[1] + math.pi / 2 - JointPos[2])
        Jacobin_F = np.array([[a11, a12], 
                                [a21, a22]])
        JacobianVel = DRArm.getDenseFrameJacobian("lower_hand_y")
        JacobianRotate = DRArm.getDenseFrameRotationalJacobian("lower_hand_y")
        print("*****************************************")
        print("rotation matrix:              ", OrientaHand)
        print("Cal Jacobian is:              ", Jacobin_F.T)
        print("Position Jacobian is:         ", JacobianVel)
        print("Angular Jacobian is:          ", JacobianRotate)

        # arm z axis desired position
        t_z = ((i) * EnvParam["t_step"]) % T_Period
        if t_z <= 0.8 * T_Period:
            z_hand_des = A * math.sin(1.25 * math.pi * t_z / T_Period) + z0
            zvel_hand_des = 1.25 * math.pi * math.cos(1.25 * math.pi * t_z / T_Period) / T_Period
        else:
            z_hand_des = -0.25 * A * math.sin(5.0 * math.pi * t_z / T_Period) + z0
            zvel_hand_des = - 1.25 * math.pi * A * math.cos(5.0 * math.pi * t_z / T_Period) / T_Period
            
        ## ======================
        # PD controller Force cal
        xhand_err_now = x_hand_des - FootPos[0]
        delta_xhand_err = xhand_err_now - xhand_err_last
        xhand_err_last = xhand_err_now

        yhand_err_now = y_hand_des - FootPos[1]
        delta_yhand_err = yhand_err_now - yhand_err_last
        yhand_err_last = yhand_err_now

        zhand_err_now = z_hand_des - FootPos[2]
        delta_zhand_err = zhand_err_now - zhand_err_last
        zhand_err_last = zhand_err_now

        Betahand_err_now = Beta_end_des - Beta_hand
        delta_Betahand_err = Betahand_err_now - Betahand_err_last        
        Gammahand_err_now = Gamma_end_des - Gamma_hand
        delta_Gammahand_err = Gammahand_err_now - Gammahand_err_last
        print("Betahand_err_last, Betahand_err_now is: ", Betahand_err_last, Betahand_err_now)
        print("BetaHand err is:              ", delta_Betahand_err)
        Betahand_err_last = Betahand_err_now
        Gammahand_err_last = Gammahand_err_now
        # Kx = np.diag([1500, 1500, 1500, 500, 500])
        # Kd = np.diag([0.5, 0.5, 0.5, 0.1, 0.1])
        # pos = np.array([x_hand_des - FootPos[0], y_hand_des - FootPos[1], z_hand_des - FootPos[2], 
        #                 Beta_end_des - Beta_hand, Gamma_end_des - Gamma_hand])
        # vel_err = np.array([delta_xhand_err / t_steps,  delta_yhand_err / t_steps,  delta_zhand_err / t_steps, 
        #                     delta_Betahand_err / t_steps, delta_Gammahand_err / t_steps])
        # JacobianRotate = JacobianRotate[0:2]
        # print("Angular Jacobian is:          ", JacobianRotate)
        # Jacobian = np.concatenate((JacobianVel.T, JacobianRotate.T), axis = 1)
        # # F = np.dot(Kx, pos)+ np.dot(Kd, vel_err)
        # F = np.dot(Kx, pos)
        # Torque = np.dot(Jacobian, F)
        # Torque[4] = - Torque[4]
        # Torque[3] = - Torque[3]

        Kx_F = np.diag([1500, 1500, 3000])
        Kd_F = np.diag([50, 50, 100])
        pos = np.array([x_hand_des - FootPos[0], y_hand_des - FootPos[1], z_hand_des - FootPos[2]])
        vel_err = np.array([delta_xhand_err / t_steps,  delta_yhand_err / t_steps,  delta_zhand_err / t_steps])
        F = np.dot(Kx_F, pos) + np.dot(Kd_F, vel_err)
        Torque3 = np.dot(JacobianVel[0:3, 0:3].T, F)

        Kx_t = np.diag([100, 100])
        Kd_t = np.diag([1, 1])
        pos_t = np.array([Beta_end_des - Beta_hand, Gamma_end_des - Gamma_hand])
        vel_err_t = np.array([delta_Betahand_err / t_steps, delta_Gammahand_err / t_steps])
        Tor = np.dot(Kx_t, pos_t) + np.dot(Kd_t, vel_err_t)
        Torque4 = np.dot(JacobianRotate[0:2, 3:5].T, -Tor)

        print("*****************************************")
        print("the time is:                  ", t_z)
        print("Is the hand and ball contact: ", contact_flag)
        print("pos des is:                   ", x_hand_des, y_hand_des, z_hand_des)
        print("FootPos BallPos is:           ", FootPos)
        print("Beta_des and Gamma_des is:    ", Beta_end_des, Gamma_end_des)
        print("Beta_Hand, Gamma_Hand is:     ", Beta_hand, Gamma_hand)
        print("the kp pos is:                ", pos_t)
        print("the kd vel err is:            ", vel_err_t)
        print("the Force is:                 ", F)
        print("the Tor is:                 ", Tor)
        # print("The Jacobian is: ", Jacobian)
        print("the torque is :               ", Torque3)
        # print("the torque is :               ", Torque4)

        # if contact_flag:
        #     break

        DRArm.setGeneralizedForce([Torque3[0], Torque3[1], Torque3[2], Tor[0], Tor[1]])
                
        server.integrateWorldThreadSafe()

def ArmBall2DSim(ParamData):
    EnvParam = ParamData["environment"]
    t_steps = EnvParam["t_step"]

    ## ====================
    ## ball control initial pos and vel setting
    jointNominalConfig = np.array([0.4, 0.18, 0.1, 1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ball1.setGeneralizedCoordinate(jointNominalConfig)
    ball1.setGeneralizedVelocity(jointVelocityTarget)

    ## ===================
    ## Arm initial setting
    world.setMaterialPairProp("rub", "rub", 1.0, 0.5, 0.0001)
    DRArm = world.addArticulatedSystem(Arm_urdf_file)
    DRArm.setName("DRArm")
    print(DRArm.getGeneralizedCoordinateDim())

    jointNominalConfig_Arm = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    DRArm.setGeneralizedCoordinate(jointNominalConfig_Arm)
    Dof = DRArm.getDOF()
    print("the dof of the arm is: ", Dof)
    ## ==================
    ## control params


    ## ================
    ## arm and ball state
    BallPos, BallVel = ball1.getState()
    JointPos, JointVel = DRArm.getState()
    FootFrameId = DRArm.getFrameIdxByName("lower_hand_y")
    FootPos = DRArm.getFramePosition(FootFrameId)
    FootVel = DRArm.getFrameVelocity(FootFrameId)
    
    HandId = DRArm.getBodyIdx("hand")
    OrientaHand = DRArm.getBodyOrientation(HandId)
    # help(raisim.ArticulatedSystem)

    for i in range(10000):
        time.sleep(0.008)

        ## =======================
        ## arm and ball current state
        BallPos, BallVel = ball1.getState()
        JointPos, JointVel = DRArm.getState()
        FootFrameId = DRArm.getFrameIdxByName("lower_add")
        FootPos = DRArm.getFramePosition(FootFrameId)
        FootVel = DRArm.getFrameVelocity(FootFrameId)

        ## =======================
        ## arm rotation matrix and Jacobian matrix
        OrientaHand = DRArm.getBodyOrientation(HandId)
        phi_hand = math.atan(FootPos[1] / FootPos[0])
        Beta_hand = math.asin(- OrientaHand[2, 0])                                  # angle around y axis
        Gamma_hand = math.atan2(OrientaHand[2, 1], OrientaHand[2, 2])               # angle around x axis

        JacobianVel = DRArm.getDenseFrameJacobian("lower_hand_y")

        JacobianRotate = DRArm.getDenseFrameRotationalJacobian("lower_hand_y")
        
        ## ======================
        ## contact detect
        ContactPoint = DRArm.getContacts()
        contact_flag = False
        for c in ContactPoint:
            contact_flag = c.getlocalBodyIndex() == DRArm.getBodyIdx("hand")
            if(contact_flag):
                break
            pass

        ## ======================
        ## params print 
        print("## ====================================================================")
        print("Is ball and hand contact:                ", contact_flag)
        print("The Foot Postion and Vel:                ", FootPos, FootVel)
        print("The Ball Postion and Vel:                ", BallPos, BallVel)
        print("Arm rotation matrix:                     ", JacobianRotate)
        print("Arm Jacobian vel matrix:                 ", JacobianVel)
        print("Arm Jacobian rotation matrix:            ", JacobianRotate)
        

        server.integrateWorldThreadSafe()



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
    ball2_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/ball2.urdf"
    hand1_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/handtest.urdf"
    Arm_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/LISM_Arm_plate.urdf"
    ArmHand_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/LISM_Arm_hand.urdf"
    # raisim world config setting
    world = raisim.World()

    # set simulation step
    t_step = ParamData["environment"]["t_step"] 
    sim_time = ParamData["environment"]["sim_time"]
    world.setTimeStep(t_step)
    ground = world.addGround(0)
    
    #======================
    # material collision property of change direction of force is applied
    # world.setMaterialPairProp("rubber", "rub", 1, 0, 0)
    world.setMaterialPairProp("default", "rub", 1.0, 0.85, 0.0001)     # ball rebound model test
    # world.setMaterialPairProp("rub", "rub", 0.52, 0.8, 0.001, 0.61, 0.01)
    # world.updateMaterialProp(raisim.MaterialManager(os.path.dirname(os.path.abspath(__file__)) + "/urdf/testMaterial.xml"))
    # help(world)

    gravity = world.getGravity()
    ball1 = world.addArticulatedSystem(ball1_urdf_file)
    ball1.setName("ball1")
    gravity = world.getGravity()
    world.setGravity([0, 0, 0])
    gravity1 = world.getGravity()
    print(gravity, gravity1)

    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)

 
    # PaperSim(ParamData)
    ArmBall2DSim(ParamData)

    # file save
    # FileFlag = ParamData["environment"]["FileFlag"] 
    # FileSave.DataSave(Data, ParamData, FileFlag)

    # # data visulization
    # Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'time': T}

    # DataPlot(Data)
    # visualization.DataPlot(Data)
    # visualization.RealCmpRef(Data)

    # print("force, ", ForceState[0:100, 1])

    server.killServer()