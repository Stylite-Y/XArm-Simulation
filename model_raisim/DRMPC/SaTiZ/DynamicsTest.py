from math import e
import os
import sys
import numpy as np
from numpy.core.numeric import NaN
import raisimpy as raisim
import datetime
import time
import yaml
import math
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt

from casadi import *
from casadi.tools import *
from do_mpc.tools.timer import Timer
from matplotlib import cm


sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/utils")
print(os.path.abspath(os.path.dirname(__file__))) # get current file path
# from ParamsCalculate import ControlParamCal
import visualization
import FileSave

def ColorSpan(ContPointTime, ColorId, axes):
    for i in range(len(ContPointTime) + 1):
        mod = i % 2
        if i == 0:
            axes.axvspan(-2, ContPointTime[i], facecolor=ColorId[mod])
        elif i == len(ContPointTime):
            axes.axvspan(ContPointTime[i-1], 20, facecolor=ColorId[mod])
        else:
            axes.axvspan(ContPointTime[i-1], ContPointTime[i], facecolor=ColorId[mod])

def DataPlot(Data):

    BallPosition = Data['BallPos']
    BallVelocity = Data['BallVel']
    FootPosition = Data['FootPos']
    FootVelocity = Data['FootVel']
    EndForce = Data['EndForce']
    JointTorque = Data['JointTorque']
    T = Data['time']
    ContPointTime = Data['ContPointTime']
    print(ContPointTime)

    """
    ball pos vel and force plot,
    background color block span setting
    """
    fig, axes = plt.subplots(4,1)
    ax1=axes[0]
    ax2=axes[1]
    ax3=axes[2]
    ax4=axes[3]
    
    # ColorId = ['#CDF0EA', '#FEFAEC']
    # ColorId = ['#DEEDF0', '#FFF5EB']
    # ColorId = ['#C5ECBE', '#FFEBBB']
    # ColorId = ['#A7D7C5', '#F7F4E3']
    ColorId = ['#E1F2FB', '#FFF5EB']

    ax1.plot(T, FootPosition[:, 0], label='Foot x-axis Position')
    ax1.plot(T, FootPosition[:, 1], label='Foot y-axis Position')
    # plt.plot(T, BallVelocity[:, 0], label='Ball x-axis Velocity')
    ax1.plot(T, FootPosition[:, 2], label='Foot z-axis Position')
    # plt.plot(T, line2, label='highest Velocity')
    ax1.axis([0, max(T)*1.05, 0.0, max(FootPosition[:, 2])*1.5])
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Foot-Pos (m)', fontsize = 15)
    ax1.legend(loc='upper right', fontsize = 15)
    ColorSpan(ContPointTime, ColorId, ax1)
    print(len(ContPointTime))
    
    ax2.plot(T, FootVelocity[:, 0], label='Foot x-axis Velocity')
    ax2.plot(T, FootVelocity[:, 1], label='Foot y-axis Velocity')
    ax2.plot(T, FootVelocity[:, 2], label='Foot z-axis Velocity')

    N_FootVel = FootVelocity[500:, 2]
    ax2.axis([0, max(T)*1.05, -max(N_FootVel)*1.5, max(N_FootVel)*1.5])
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Foot-Vel (m/s)', fontsize = 15)
    ax2.legend(loc='upper right', fontsize = 15)
    ColorSpan(ContPointTime, ColorId, ax2)

    ColorId = ['#CDF0EA', '#FEFAEC']

    ax3.plot(T, EndForce[:, 0], label='End Effector x-axis Force')
    ax3.plot(T, EndForce[:, 1], label='End Effector y-axis Force')
    ax3.plot(T, EndForce[:, 2], label='End Effector z-axis Force')
    ColorSpan(ContPointTime, ColorId, ax3)

    N_EndForce = EndForce[500:, 1]
    ax3.axis([0, max(T)*1.05, -max(N_EndForce)*1.5, max(N_EndForce)*1.5])
    ax3.set_xlabel('time (s)', fontsize = 15)
    ax3.set_ylabel('Force (N)', fontsize = 15)
    ax3.legend(loc='lower right', fontsize = 15)

    ax4.plot(T, JointTorque[:, 0], label='Joint 1 Torque')
    ax4.plot(T, JointTorque[:, 1], label='Joint 2 Torque')
    ax4.plot(T, JointTorque[:, 2], label='Joint 3 Torque')
    ax4.plot(T, JointTorque[:, 3], label='Joint 4 Torque')
    ax4.plot(T, JointTorque[:, 4], label='Joint 5 Torque')

    P_Torque = JointTorque[500:, 0]
    N_Torque = JointTorque[500:, 2]

    ax4.axis([0, max(T)*1.05, min(N_Torque)* 1.2, max(P_Torque)* 1.2])
    ax4.set_xlabel('time (s)', fontsize = 15)
    ax4.set_ylabel('Torque (N.m)', fontsize = 15)
    ax4.legend(loc='lower right', fontsize = 15)
    y_major_locator=MultipleLocator(1.5)
    ax4.yaxis.set_major_locator(y_major_locator)
    ColorSpan(ContPointTime, ColorId, ax4)

    # """
    # double y axis setting
    # """
    # ax42 = ax4.twinx()
    # ax42.set_ylabel('Hand-Ball Pos (m)', fontsize = 15)
    # ax42.plot(T, FootPosition[:, 2], label='Foot z-axis Position')
    # # ax42.plot(T, BallPosition[:, 2], label='Ball z-axis Position')
    # ax42.tick_params(axis='y')
    # ax42.set_ylim(-0.7, 0.7)
    # ax42.legend(loc='lower left', fontsize = 15)
    # # fig.tight_layout()
    plt.show()

# xbox = Xbox360Controller(0, axis_threshold=0.02)



def PaperSim(ParamData):
    """
    this part is to replicate the experiment of paper:
    S. Haddadin, K. Krieger, A. Albu-Schäffer and T. Lilge, "Exploiting Elastic Energy Storage for “Blind” Cyclic Manipulation: 
    Modeling, Stability Analysis, Control, and Experiments for Dribbling," in IEEE Transactions on Robotics
    """

    EnvParam = ParamData["environment"]
    t_steps = EnvParam["t_step"]
    PaperSimParam = ParamData["PaperSim"]

    ## ====================
    # ball control initial pos and vel setting
    jointNominalConfig = np.array([5.0, 1.0, 0.3, 1.0, 0.0, 0.0, 0.0])
    # jointNominalConfig = np.array([0.34, -0.05, 0.5,1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([0.0, 0.0, -4.0, 0.0, 0.0, 0.0])
    ball1.setGeneralizedCoordinate(jointNominalConfig)
    ball1.setGeneralizedVelocity(jointVelocityTarget)

    ## ===================
    # Arm initial setting
    # ArmTest_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/LISM_Arm_hand.urdf"
    world.setMaterialPairProp("rub", "rub", 1.0, 0.95, 0.0001)
    DRArm = world.addArticulatedSystem(Arm_urdf_file)
    DRArm.setName("DRArm")
    print(DRArm.getGeneralizedCoordinateDim())

    jointNominalConfig_Arm = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    DRArm.setGeneralizedCoordinate(jointNominalConfig_Arm)

    # world.updateMaterialProp(raisim.MaterialManager(os.path.dirname(os.path.abspath(__file__)) + "/urdf/testMaterial.xml"))

    ## ===================
    # control params
    T_Period = PaperSimParam['T_Period']
    A = PaperSimParam['A']
    z0 = PaperSimParam['z0']
    index = 0

    # PID params for desired angle cal
    K_Bdes_p = PaperSimParam['K_Bdes_p']
    K_Bdes_d = PaperSimParam['K_Bdes_d']
    K_Gdes_p = PaperSimParam['K_Gdes_p']
    K_Gdes_d = PaperSimParam['K_Gdes_d']
    K_Bdes_I = 0.3
    K_Gdes_I = 0.1
    xhand_err_last = 0.0
    yhand_err_last = 0.0
    zhand_err_last = 0.0
    Betahand_err_last = 0.0
    Gammahand_err_last = 0.0
    
    # ref point
    phi_des = math.atan(0.34 / 0.18)
    Beta_end_des = 0.0
    Gamma_end_des = 0.0
    D_des = math.sqrt(0.34 ** 2 + 0.18 ** 2)

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

    i_off = 4 * T_Period * math.asin((0.5 - z0) / A) / 5

    BallPosition = np.array([[0.0, 0.0, 0.0]])          # the pos and vel state of ball
    BallVelocity = np.array([[0.0, 0.0, 0.0]])       # the pos and vel state of endfoot
    FootPosition = np.array([[0.0, 0.0, 0.0]])          # the pos and vel state of ball
    FootVelocity = np.array([[0.0, 0.0, 0.0]])       # the pos and vel state of endfoot
    EndForce = np.array([[0.0, 0.0, 0.0]])         # the calculate force and real contact force betweem ball an foot
    JointTorque = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    SumForce = np.array([[0.0]])
    T = np.array([0.0])
    ContPointTime = np.array([0.0])

    for i in range(3000):
        time.sleep(0.002)

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
        Gamma_hand = math.atan2(OrientaHand[2, 1], OrientaHand[2, 2])

        ## ===================
        # d and phi cal of ball pos
        D_ball = math.sqrt(BallPos[0] ** 2 + BallPos[1] ** 2)
        D_foot = math.sqrt(FootPos[0] ** 2 + FootPos[1] ** 2)
        phi_ball = math.atan(BallPos[1] / BallPos[0])
        phi_foot = math.atan(FootPos[1] / FootPos[0])

        # arm desired position
        # x_hand_des = (D_ball - dh) * math.cos(phi_ball)
        # y_hand_des = (D_ball - dh) * math.sin(phi_ball)
        x_hand_des = 0.34
        y_hand_des = 0.18
        Beta_end_des = 0.0
        Gamma_end_des = 0.0
            
        ContactTime = i * t_steps
        ConFlag = 0.2 * index
        ConFlag = round(ConFlag, 1)
        ContactTime = round(ContactTime, 1)
        if ContactTime == ConFlag:
            index = index + 1
            ContPointTime = np.concatenate([ContPointTime, [ContactTime]], axis = 0)

        # print("phi_ball, phi_err_now, phi_des phi_err_sum is: ", phi_foot, phi_err_now, phi_des, phi_err_now / t_steps, phi_err_sum)
        # print("Beta_des is: ", Beta_end_des)
        # print("Gamma_des is: ", Gamma_end_des)
        # print("FootPos BallPos is:", FootPos)
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
        

        # arm z axis desired position
        t_z = ((i + i_off) * EnvParam["t_step"]) % T_Period
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
        # print("Betahand_err_last, Betahand_err_now is: ", Betahand_err_last, Betahand_err_now)
        # print("BetaHand err is:              ", delta_Betahand_err)
        Betahand_err_last = Betahand_err_now
        Gammahand_err_last = Gammahand_err_now

        Kx_F = np.diag(PaperSimParam['Kp_pos'])
        Kd_F = np.diag(PaperSimParam['Kd_pos'])
        pos = np.array([x_hand_des - FootPos[0], y_hand_des - FootPos[1], z_hand_des - FootPos[2]])
        vel_err = np.array([delta_xhand_err / t_steps,  delta_yhand_err / t_steps,  delta_zhand_err / t_steps])
        F = np.dot(Kx_F, pos) + np.dot(Kd_F, vel_err)
        Torque3 = np.dot(JacobianVel[0:3, 0:3].T, F)

        Kx_t = np.diag(PaperSimParam['Kp_angle'])
        Kd_t = np.diag(PaperSimParam['Kd_angle'])
        pos_t = np.array([Beta_end_des - Beta_hand, Gamma_end_des - Gamma_hand])
        vel_err_t = np.array([delta_Betahand_err / t_steps, delta_Gammahand_err / t_steps])
        Tor = np.dot(Kx_t, pos_t) + np.dot(Kd_t, vel_err_t)
        Torque4 = np.dot(JacobianRotate[0:2, 3:5].T, -Tor)
        DRArm.setGeneralizedForce([Torque3[0], Torque3[1], Torque3[2], Tor[0], Tor[1]])

        # if i < 10:
        #     print("*****************************************")
        #     print("the time is:                  ", t_z)
        #     print("pos des is:                   ", x_hand_des, y_hand_des, z_hand_des)
        #     print("Foot state is:                ", FootPos, FootVel)
        #     print("Ball state:                   ", BallPos[0:3], BallVel[0:3])
        #     print("Beta_des and Gamma_des is:    ", Beta_end_des, Gamma_end_des)
        #     print("Beta_Hand, Gamma_Hand is:     ", Beta_hand, Gamma_hand)
        #     print("the kp pos is:                ", pos, pos_t)
        #     print("the kd vel err is:            ", vel_err, vel_err_t)
        #     print("the Force is:                 ", F)
        #     print("the Tor is:                   ", Tor)        
        #     print("the torque is :               ", Torque3)
        #     print("The Jacobian is:              ", JacobianVel[0:3, 0:3].T)

        

        t = i * t_steps
        T = np.concatenate([T, [t]], axis = 0)
        BallPosition = np.concatenate([BallPosition, [BallPos[0:3]]], axis = 0)
        BallVelocity = np.concatenate([BallVelocity, [BallVel[0:3]]], axis = 0)
        FootPosition = np.concatenate([FootPosition, [FootPos]], axis = 0)
        FootVelocity = np.concatenate([FootVelocity, [FootVel]], axis = 0)
        EndForce = np.concatenate([EndForce, [F]], axis = 0)
        JointTorque = np.concatenate([JointTorque, [[Torque3[0], Torque3[1], Torque3[2], Tor[0], Tor[1]]]], axis = 0)
        # break
        server.integrateWorldThreadSafe()

    T = T[1:,]
    print(ContPointTime, T)
    ContPointTime = ContPointTime[1:,]
    BallPosition = BallPosition[1:,]
    BallVelocity = BallVelocity[1:,]
    FootPosition = FootPosition[1:,]
    FootVelocity = FootVelocity[1:,]
    EndForce = EndForce[1:,]
    JointTorque = JointTorque[1:,]

    Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, "FootPos": FootPosition, "FootVel": FootVelocity,\
            "JointTorque":JointTorque,  'EndForce': EndForce, 'time': T, "ContPointTime": ContPointTime}

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
    # world.updateMaterialProp(raisim.MaterialManager(os.path.dirname(os.path.abspath(__file__)) + "/urdf/testMaterial.xml"))
    # help(world)

    gravity = world.getGravity()
    ball1 = world.addArticulatedSystem(ball1_urdf_file)
    ball1.setName("ball1")
    gravity = world.getGravity()
    # world.setGravity([0, 0, 0])
    gravity1 = world.getGravity()
    print(gravity, gravity1)

    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)

 
    Data = PaperSim(ParamData)
  
    DataPlot(Data)

    server.killServer()