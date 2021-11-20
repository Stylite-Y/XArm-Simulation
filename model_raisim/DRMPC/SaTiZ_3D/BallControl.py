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

def ArmTest(ParamData):
    EnvParam = ParamData["environment"]

    ## ====================
    # ball control initial pos and vel setting
    jointNominalConfig = np.array([0.34, 0.18, 0.2,1.0, 0.0, 0.0, 0.0])
    # jointNominalConfig = np.array([0.34, -0.05, 0.5,1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([0.0, 0.0, -5, 0.0, 0.0, 0.0])
    ball1.setGeneralizedCoordinate(jointNominalConfig)
    ball1.setGeneralizedVelocity(jointVelocityTarget)

    ## ===================
    # Arm initial setting
    world.setMaterialPairProp("rub", "rub", 1.0, 0.5, 0.0001)
    Arm = world.addArticulatedSystem(Arm_urdf_file)
    Arm.setName("Arm")
    print(Arm.getGeneralizedCoordinateDim())

    jointNominalConfig_Arm = np.array([0.0, 0.0, -1.57])
    Arm.setGeneralizedCoordinate(jointNominalConfig_Arm)

    flag = 0
    N = 1000
    k = 0
    Theta = np.linspace(math.pi / 2, 2 * math.pi / 3, N)
    for i in range(20000):
        time.sleep(0.02)
        gravity = world.getGravity()
        ## ====================
        # contact detect
        ContactPoint = Arm.getContacts()        
        contact_flag = False
        for c in ContactPoint:
            contact_flag = c.getlocalBodyIndex() == Arm.getBodyIdx("lower_r")
            if(contact_flag):
                # print("Contact position in the world frame: ", ContactPoint[0].getNormal())
                break
            pass

        ## ====================
        # foot and arm pos 
        FootFrameId = Arm.getFrameIdxByName("toe_fr_joint")
        JointPos, JointVel = Arm.getState()
        FootPos = Arm.getFramePosition(FootFrameId)
        FootVel = Arm.getFrameVelocity(FootFrameId)
        BallPos, BallVel = ball1.getState()

        jointPgain = np.zeros(Arm.getDOF())
        jointDgain = np.zeros(Arm.getDOF())
        Arm.setPdGains(jointPgain, jointDgain)

        ## ===================
        # Jacobin matrix 
        a11 = - EnvParam["UpperArmLength"] * np.cos(JointPos[1]) - EnvParam["LowerArmLength"] * np.cos(JointPos[1] + JointPos[2])
        a12 = EnvParam["UpperArmLength"] * np.sin(JointPos[1]) + EnvParam["LowerArmLength"] * np.sin(JointPos[1] + JointPos[2])
        a21 = - EnvParam["LowerArmLength"] * np.cos(JointPos[1] + JointPos[2])
        a22 = EnvParam["LowerArmLength"] * np.sin(JointPos[1] + JointPos[2])
        Jacobin_F = np.array([[a11, a12], 
                                [a21, a22]])
        ## ==================
        # params output
        print("============================================================")
        print("is the ball and foot contact: ", contact_flag)
        print("Torque flag:                  ", flag==1)
        print("Ball position:                ", BallPos[:3])
        print("Ball velocity:                ", BallVel[:3])
        print("Foot position:                ", FootPos)
        print("Foot velocity:                ", FootVel)
        print("Joint angle:                  ", JointPos)
        print("Joint angular vel:            ", JointVel)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        if contact_flag:
            ## =================================
            # force direction change precisely
            # PosTar = np.array([xtra, ytra, z_ref])
            # VelTar = np.array([v_xref, v_yref, v_zref])
            # x_coef, y_coef, z_coef = TriCal(t_force, BallPos, BallVel, PosTar, VelTar)
            # TriCoef = np.array([[x_coef, y_coef, z_coef]])
            
            # normal force cal
            # F_nx = 2 * x_coef[2] + 6 * x_coef[3] * i + 12 * x_coef[4] * i ** 2
            # F_ny = 2 * y_coef[2] + 6 * y_coef[3] * i + 12 * y_coef[4] * i ** 2
            # F_nz = 2 * z_coef[2] + 6 * z_coef[3] * i + 12 * z_coef[4] * i ** 2

            # F_nom = 500
            # d = math.sqrt((BallPos[1] - FootPos[1]) ** 2 + (BallPos[2] - FootPos[2]) ** 2)
            # F_nx = - F_nom * (FootPos[0] - BallPos[0]) / d
            # F_ny = - F_nom * (FootPos[1] - BallPos[1]) / d
            # F_nz = - F_nom * (FootPos[2] - BallPos[2]) / d
            
            # # tangential force cal
            # dt = 0.005
            # dt_step = dt / EnvParam["t_step"]
            # if flag == 0:
            #     i_contact = i
            #     flag = 1
            # d = math.sqrt((BallPos[1] - FootPos[1]) ** 2 + (BallPos[2] - FootPos[2]) ** 2)
            # if FootPos[1] > BallPos[1]:
            #     Theta_now = math.asin((FootPos[2] - BallPos[2]) / d)
            # else:
            #     Theta_now = math.pi - math.asin((FootPos[2] - BallPos[2]) / d)
            # if (i - i_contact) % dt_step == 0:
            #     DTheta = Theta[k + 1] - Theta_now
            #     Beta = 2 * DTheta / (dt ** 2)
            #     F_t = 0.4 * Beta / (0.15 + 0.0275)

            # F_t_yz = - F_t * (FootPos[2] - BallPos[2]) / d
            # F_t_z = F_t * (FootPos[1] - BallPos[1]) / d

            # F_x = 0.0
            # F_y = F_ny + F_t_yz
            # F_z = F_nz + F_t_z

            # # use Jocabin convert End Force to  Joint Torque
            # Torque_1_n = F_y * np.abs(0.7 - FootPos[2])
            # Torque_2_n = Jacobin_F[0, 0] * F_x + Jacobin_F[0, 1] * (F_z - F_y * np.abs(0.7 - FootPos[2]) / np.abs(0.1 - FootPos[1]))
            # Torque_3_n = Jacobin_F[1, 0] * F_x + Jacobin_F[1, 1] * (F_z - F_y * np.abs(0.7 - FootPos[2]) / np.abs(0.1 - FootPos[1])) 
            # Arm.setGeneralizedForce([Torque_1_n, Torque_2_n, Torque_3_n])
            # print("cal i and k:                  ", i - i_contact, k)
            # print("Jocabin matrix:               ", Jacobin_F)
            # print("force angular now and target: ", [Theta_now, Theta[k], Theta[k + 1]])
            # print("tangential cal params:        ", [Theta_now, DTheta, Beta, F_t])
            # print("force cal params:             ", [F_t_yz, F_ny, F_t_z, F_nz])
            # print("Torque cal params:            ", [Torque_1_n, Torque_2_n, Torque_3_n])
            
            # if (i + 1 - i_contact) % dt_step == 0:
            #     k = k + 1
            #     if k == 2:
            #         break

            # =================================
            # force direction change test
            F_nom = 150
            d = math.sqrt((BallPos[1] - FootPos[1]) ** 2 + (BallPos[2] - FootPos[2]) ** 2)
            F_nom_x = - F_nom * (FootPos[0] - BallPos[0]) / d
            F_nom_y = - F_nom * (FootPos[1] - BallPos[1]) / d
            F_nom_z = - F_nom * (FootPos[2] - BallPos[2]) / d

            F_t = 50
            # F_t_xy = - F_t * (BallPos[1] - BallPos[1]) / d
            # F_t_xz = - F_t * (BallPos[2] - BallPos[2]) / d
            F_t_yz = - F_t * (FootPos[2] - BallPos[2]) / d
            F_t_z = F_t * (FootPos[1] - BallPos[1]) / d

            F_x = 0.0
            F_y = F_nom_y + F_t_yz
            F_z = F_nom_z + F_t_z

            Torque_1_n = F_y * np.abs(0.7 - FootPos[2])
            Torque_2_n = Jacobin_F[0, 0] * F_x + Jacobin_F[0, 1] * (F_z - F_y * np.abs(0.7 - FootPos[2]) / np.abs(0.1 - FootPos[1]))
            Torque_3_n = Jacobin_F[1, 0] * F_x + Jacobin_F[1, 1] * (F_z - F_y * np.abs(0.7 - FootPos[2]) / np.abs(0.1 - FootPos[1])) 

            Arm.setGeneralizedForce([Torque_1_n, Torque_2_n, Torque_3_n])
            if flag == 0:
                flag = 1
        else:
            if flag == 0 or (flag == 1 and BallPos[2] < 0.17):
                # Torque1 = - 5 * (JointPos[0] - 0.0) - 1 * (JointVel[0])
                # Torque2 = - 5 * (JointPos[1] - 0.0) - 1 * (JointVel[1])
                # Torque3 = - 10 * (JointPos[2] + 1.57) - 1 * (JointVel[2])
                # Arm.setGeneralizedForce([Torque1, Torque2, Torque3])
                Arm.setGeneralizedForce([0, 0, 0])
                jointNominalConfig = np.array([0, 0.0, -1.57])
                jointVelocityTarget = np.zeros([Arm.getDOF()])
                jointPgain = np.array([100, 100, 100])
                jointDgain = np.array([1.5, 1.5, 1.5])
                
                Arm.setPdGains(jointPgain, jointDgain)
                Arm.setPdTarget(jointNominalConfig, jointVelocityTarget)
                flag = 0
            # elif flag == 1 and BallPos[2] > 0.2 and k == 1:
            #     break
            # elif flag == 1 and BallPos[2] < 0.17:
            #     Arm.setGeneralizedForce([0, 0, 0])
            #     jointNominalConfig = np.array([0, 0.0, -1.57])
            #     jointVelocityTarget = np.zeros([Arm.getDOF()])
            #     jointPgain = np.array([40, 40, 40])
            #     jointDgain = np.array([0.8, 0.8, 0.8])
                
            #     Arm.setPdGains(jointPgain, jointDgain)
            #     Arm.setPdTarget(jointNominalConfig, jointVelocityTarget)
            #     flag = 0
            #     print(2)
                # break
            else:
                F_nom = 500
                d = math.sqrt((BallPos[1] - FootPos[1]) ** 2 + (BallPos[2] - FootPos[2]) ** 2)
                F_nom_x = - F_nom * (FootPos[0] - BallPos[0]) / d
                F_nom_y = - F_nom * (FootPos[1] - BallPos[1]) / d
                F_nom_z = - F_nom * (FootPos[2] - BallPos[2]) / d

                Torque_1_n = F_nom_y * np.abs(0.7 - FootPos[2])
                Torque_2_n = Jacobin_F[0, 0] * 0.0 + Jacobin_F[0, 1] * (F_nom_z - F_nom_y * np.abs(0.7 - FootPos[2]) / np.abs(0.1 - FootPos[1]))
                Torque_3_n = Jacobin_F[1, 0] * 0.0 + Jacobin_F[1, 1] * (F_nom_z - F_nom_y * np.abs(0.7 - FootPos[2]) / np.abs(0.1 - FootPos[1]))

                Arm.setGeneralizedForce([Torque_1_n, Torque_2_n, Torque_3_n])
                print("Torque noncontact:            ", [Torque_1_n, Torque_2_n, Torque_3_n])
                # if k == 1:
                #     break            

        server.integrateWorldThreadSafe()

def BallTest(ParamData):
    ## ====================
    # world env setting
    world.setMaterialPairProp("rub", "rub", 1.0, 0.2, 0.0001)
    flag = 0

    ## ====================
    # ball control initial pos and vel setting
    jointNominalConfig = np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([0.0, 0.0, -5, 0.0, 0.0, 0.0])
    ball1.setGeneralizedCoordinate(jointNominalConfig)
    ball1.setGeneralizedVelocity(jointVelocityTarget)

    ## ===================
    # ball2 initial setting
    ball2 = world.addArticulatedSystem(ball2_urdf_file)
    ball2.setName("ball2")
    jointNominalConfig_2 = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ball2.setGeneralizedCoordinate(jointNominalConfig_2)
    ball2.setGeneralizedVelocity(jointVelocityTarget_2)

    for i in range(10000):
        time.sleep(0.1)
        gravity = world.getGravity()

        ContactPoint = ball2.getContacts()

        # wether the contact occurs berween ball and arm end foot
        contact_flag = False
        for c in ContactPoint:
            contact_flag = c.getlocalBodyIndex() == ball2.getBodyIdx("ball")
            if(contact_flag):
                # print("Contact position in the world frame: ", ContactPoint[0].getNormal())
                break
            pass

        BallPos, BallVel = ball1.getState()
        BallPos_2, BallVel_2 = ball2.getState()
        print("is the ball and foot contact: ", contact_flag)
        print("Ball position:                ", BallPos)
        print("Ball velocity:                ", BallVel)

        if contact_flag:
            flag = 1
            F_nom = 50
            d = math.sqrt((BallPos[1] - BallPos_2[1]) ** 2 + (BallPos[2] - BallPos_2[2]) ** 2)
            F_nom_x = - F_nom * (BallPos_2[0] - BallPos[0]) / d
            F_nom_y = - F_nom * (BallPos_2[1] - BallPos[1]) / d
            F_nom_z = - F_nom  * (BallPos_2[2] - BallPos[2]) / d

            F_t = 5
            # F_t_xy = - F_t * (BallPos[1] - BallPos[1]) / d
            # F_t_xz = - F_t * (BallPos[2] - BallPos[2]) / d
            F_t_yz = - F_t * (BallPos_2[2] - BallPos[2]) / d
            F_t_z = F_t * (BallPos_2[1] - BallPos[1]) / d
            # ball2.setExternalForce(0, [0.0, 0, 0.0], [F_nom_x, 0.0, F_nom_z])
            # ball2.setExternalForce(0, [0.0, 0, 0.0], [F_t_xz, 0.0, F_t_z])
            ball2.setExternalForce(0, [0.0, 0, 0.0], [0.0, F_nom_y + F_t_yz, F_nom_z + F_t_z])
            # ball2.setExternalForce(0, [0.0, 0, 0.0], [0.0, F_t_yz, F_t_z])

        else:
            if flag == 0:
                ball2.setExternalForce(0, [0.0, 0, 0.0], [0.0, 0.0, 0.0])

            else:
                F_nom = 100
                d = math.sqrt((BallPos[1] - BallPos_2[1]) ** 2 + (BallPos[2] - BallPos_2[2]) ** 2)
                F_nom_x = - F_nom * (BallPos_2[0] - BallPos[0]) / d
                F_nom_y = - F_nom * (BallPos_2[1] - BallPos[1]) / d
                F_nom_z = - F_nom  * (BallPos_2[2] - BallPos[2]) / d
                # ball2.setExternalForce(0, [0.0, 0, 0.0], [F_nom_x, 0.0, F_z])
                ball2.setExternalForce(0, [0.0, 0, 0.0], [0.0, F_nom_y, F_nom_z])
                # ball2.setExternalForce(0, [0.0, 0, 0.0], [F_nom_x, F_nom_y, 0.0])

        server.integrateWorldThreadSafe()

def BallHandTest():
    R = 0.2
    r = 0.15
    N = 100
    k = 0
    alpha = np.linspace(0, math.pi / 6, N)
    ## ====================
    # world env setting
    world.setMaterialPairProp("rub", "rub", 1.0, 0.0, 0.0001)
    flag = 0

    ## ====================
    # ball control initial pos and vel setting
    jointNominalConfig = np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([0.0, 0.0, -5, 0.0, 0.0, 0.0])
    ball1.setGeneralizedCoordinate(jointNominalConfig)
    ball1.setGeneralizedVelocity(jointVelocityTarget)

    ## ====================
    # arm control initial pos and vel setting  
    hand1 = world.addArticulatedSystem(hand1_urdf_file) 
    hand1.setName("hand1")
    # print(hand1.getDOF())

    jointNominalConfig_hand = np.array([0.0, 0.0, 0.0])
    hand1.setGeneralizedCoordinate(jointNominalConfig_hand)
    
    world.setMaterialPairProp("steel", "rub", 1.0, 0.2, 0.001)


    for i in range(10000):
        time.sleep(0.02)
        # ball1.setExternalForce(0, [0, 0, 0.15], [5.0, 0.0, 0.0])
        # GeneralizedCoordinate = ball1.getGeneralizedCoordinate()
        # Quaternion = GeneralizedCoordinate[3:]
        # Euler_matrix = Rotation.from_quat(Quaternion)
        # Euler = Euler_matrix.as_euler('xyz', degrees=True)
        # print(Quaternion)
        # print(Euler)
        gravity = world.getGravity()
        # ===================
        # arm contact
        ContactPoint = hand1.getContacts()
        contact_flag = False
        for c in ContactPoint:
            contact_flag = c.getlocalBodyIndex() == hand1.getBodyIdx("hand")
            if(contact_flag):
                break
            pass
        print(contact_flag)
        # help(raisim.contact.Contact)
        if(contact_flag) and k < 100:
            # ===================
            # hand d and theta calculation
            Beta = math.acos((R - r) * math.sin(alpha[N  -1]) / R)
            # Beta = math.acos((R - r) * math.sin(alpha[k]) / R)
            Theta = math.pi / 2 - Beta
            D = R * math.sin(Beta) - (R - r) * math.cos(alpha[N  -1])
            # D = R * math.sin(Beta) - (R - r) * math.cos(alpha[k])
            Dis_p = D - r
            arc_p = math.sqrt((D - r * math.cos(alpha[N  -1])) ** 2 + (r * math.sin(alpha[N  -1])) ** 2)
            # arc_p = math.sqrt((D - r * math.cos(alpha[k])) ** 2 + (r * math.sin(alpha[k])) ** 2)

            print("=================================")
            print("Force direction angle:                             ", alpha[k] * 180 / math.pi)
            print("Distance between hadn Vertex and centre of sphere: ", D)
            print("Distance of hand and contact point is:             ", arc_p)
            print("Hand rotation angle:                               ", Theta * 180 / math.pi)
            print("Distance of hand move in z axis:                   ", Dis_p)
            print("---------------------------------")

            # nomial force for balance
            Normal_F = 500
            F_pri = - np.abs(Normal_F * math.sin(alpha[N  -1]))
            # F_pri = - np.abs(Normal_F * math.sin(alpha[k]))
            Torque = Normal_F * arc_p

            Ballforce = ball1.getGeneralizedForce()
            ContactPos = ContactPoint[0].get_position()
            alpha_cal = math.atan(ContactPos[0] / (ContactPos[2] - 0.5))
            Normal_Cont = ContactPoint[0].getNormal()
            alpha_cal2 = math.atan(Normal_Cont[0] / (Normal_Cont[2]))
            print("Is the Ball contact with hand:                     ", contact_flag)
            print("Contact position in the world frame:               ", ContactPos)
            print("Contact normal in the world frame:                 ", Normal_Cont)
            print("External force angle cal of the ball:              ", alpha_cal * 180 / math.pi, alpha_cal2 * 180 / math.pi)
            print("External force of the ball:                        ", Ballforce)

            # ===================
            # xy pos and force for hand apply force
            jointNominalConfig = np.array([0.0, Dis_p, Theta])
            hand1.setGeneralizedCoordinate(jointNominalConfig)
            hand1.setGeneralizedForce([10, F_pri, Torque])

            k = k + 1

        else:
            hand1.setGeneralizedCoordinate(np.array([0.0, 0.0, 0.0]))
            hand1.setGeneralizedVelocity(np.array([0.0, 0.0, 0.0]))

            # hand1.setGeneralizedForce([F_pri, Torque])
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
    Arm_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/LISM_Arm_sim.urdf"
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

    # ArmTest(ParamData)
    # BallTest(ParamData)
    BallHandTest()

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