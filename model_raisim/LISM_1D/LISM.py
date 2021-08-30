import os
import sys
import numpy as np
import raisimpy as raisim
import datetime
import time
import yaml
import random
import shutil
import pickle

sys.path.append("./utils")
from ParamsCalculate import ControlParamCal

def EnvConfig():
    a = 1

def FileSave(BallState, EndFootState, ForceState, JointTorque, JointVel, T, ParamData):
    today=datetime.date.today()
    data = {'BallState': BallState, 'EndFootState': EndFootState, 'ForceState': ForceState, 'JointTorque': JointTorque, 'JointVel':JointVel, 'time': T}
    # print(data['state'][0:100, 0])
    x_ref = ParamData["controller"]["x_ref"]
    v_ref = ParamData["controller"]["v_ref"]
    v0 = ParamData["controller"]["v_int"]
    dx = round(ParamData["controller"]["x_top"] - x_ref, 2)
    f_up = ParamData["controller"]["f_up"]

    name = str(today) + '-x_ref_' + str(x_ref) + '-v0_'+ str(v0) + '-vref_' + str(v_ref) + '-dx_' + str(dx) + '-f1_' + str(f_up) + '.pkl'
    pathDir = './data/'
    print(name)

    if os.path.exists(pathDir + name):
        name = str(today) + '-x_ref_' + str(x_ref) + '-v0_'+ str(v0) + '-vref_' + str(v_ref) + '-dx_' + str(dx) + '-f1_' + str(f_up) + '-' + str(random.randint(0,100)) + '.pkl'

    with open(pathDir + name, 'wb') as f:
        pickle.dump(data, f)

def DriControl(ParamData):

    EnvParam = ParamData["environment"]
    CtlParam = ParamData["controller"]
    g = -gravity[2]
    flag = 0

    BallState = np.array([[0.0, 0.0]])
    EndFootState = np.array([[0.0, 0.0]])
    ForceState = np.array([[0.0, 0.0]])
    JointTorque = np.array([[0.0, 0.0]])
    JointVelSaved = np.array([[0.0, 0.0]])
    T = np.array([0.0])

    for i in range(50000):
        time.sleep(0.0001)
        # if i == 0:
        #     server.startRecordingVideo("v10_with-x_1x.mp4")

        BallPos = LISM.getFramePosition(BallFrameId)
        BallVel = LISM.getFrameVelocity(BallFrameId)
        JointPos, JointVel = LISM.getState()
        FootPos = LISM.getFramePosition(FootFrameId)
        FootVel = LISM.getFrameVelocity(FootFrameId)

        jointPgain = np.array([0, 0, 0])
        jointDgain = np.array([0, 0, 0])
        LISM.setPdGains(jointPgain, jointDgain)

        # pos, vel and force data get
        t = i * EnvParam["t_step"]
        T = np.concatenate([T, [t]], axis = 0)
        BallState = np.concatenate([BallState, [[BallPos[2], BallVel[2]]]], axis = 0)
        EndFootState = np.concatenate([EndFootState, [[FootPos[2], FootVel[2]]]], axis = 0)
        # JointVelSaved = np.concatenate([JointVelSaved, [[JointVel[1], JointVel[2]]]], axis = 0)

        ContactPoint = LISM.getContacts()

        contact_flag = False
        for c in ContactPoint:
            contact_flag = c.getlocalBodyIndex() == LISM.getBodyIdx("lower_r")
            if(contact_flag):
                break
            pass
        
        if EnvParam["con_flag"] == 0 and contact_flag:
            EnvParam["con_flag"] = 1
            # print("the contact vel of ball", BallVel[2])

        if BallPos[2] >= (CtlParam["x_ref"]-0.1275) and EnvParam["con_flag"] == 1:
        
            if BallVel[2] > 0 and EnvParam["fun_flag"] == 0:
                dx_up = CtlParam["x_top"] - FootPos[2]
                if (dx_up * 2 * g) >= (BallVel[2] ** 2):
                    raise FloatingPointError("calculate Error: init velocity is too small or the heaving height is too high!")
                else:
                    k_vir, f_down = ControlParamCal().ParamsCal(FootPos[2], BallVel[2], CtlParam, g)
                EnvParam["fun_flag"] = 1
            
                print("contact vel of ball ", FootPos[2], BallVel[2])
                print("the k_vir and f_up, f_down is ", k_vir, CtlParam["f_up"], f_down)

            jointPgain = np.zeros(LISM.getDOF())
            jointDgain = np.zeros(LISM.getDOF())

            LISM.setPdGains(jointPgain, jointDgain)

            JointPos, JointVec = LISM.getState()
            FootPos = LISM.getFramePosition(FootFrameId)
            FootVel = LISM.getFrameVelocity(FootFrameId)

            ## Force kinematics
            # jacobian matrix of force transmission
            a11 = - EnvParam["UpperArmLength"] * np.cos(JointPos[1]) - EnvParam["LowerArmLength"] * np.cos(JointPos[1] + JointPos[2])
            a12 = EnvParam["UpperArmLength"] * np.sin(JointPos[1]) + EnvParam["LowerArmLength"] * np.sin(JointPos[1] + JointPos[2])
            a21 = - EnvParam["LowerArmLength"] * np.cos(JointPos[1] + JointPos[2])
            a22 = EnvParam["LowerArmLength"] * np.sin(JointPos[1] + JointPos[2])
            Jacobin_F = np.array([[a11, a12], 
                                [a21, a22]])
            # print(Jacobin_F)
            # if BallVel[2] < 0.1 and BallVel[2] > 0:
            #     print("the highest ball and foot pos: ", BallPos[2] ,FootPos[2])

            if BallVel[2] > 0 and contact_flag:
                # ContactPointVel = LISM.getContactPointVel(ContactPoint[0].getlocalBodyIndex())
                ContactPointVel = ContactPoint[0].getImpulse()
                ContactForce = ContactPointVel / t_step
                EndForce = - k_vir * (FootPos[2] - CtlParam["x_ref"]) - CtlParam["f_up"]
                # EndForce = - 2000 * (FootPos[2] - FootPosInit[2])
                EndForce_x = - CtlParam["K_virx"] * (FootPos[0] - FootPosInit[0])
                # print("1", ContactForce[2])
            # elif BallVel[2] > 0 and BallPos[2] > 0.6:
            #     EndForce = - 10000 * (FootPos[2] - 0.4) - CtlParam["f_up"]
            #     # EndForce = - 2000 * (FootPos[2] - FootPosInit[2])
            #     EndForce_x = - 10000 * (FootPos[0] - FootPosInit[0])

            elif BallVel[2] <= 0  and contact_flag == True:
                EndForce = - k_vir * (FootPos[2] - CtlParam["x_ref"]) - f_down
                EndForce_x = - CtlParam["K_virx"] * (FootPos[0] - FootPosInit[0])
                ContactPointVel = ContactPoint[0].getImpulse()
                ContactForce = ContactPointVel / t_step
                # print("contact foot pos: ", FootPos[2], EndForce)
                # print("contact ball pos: ", BallPos[2])
                # print("2")
                # print("k_vir", k_vir)


            # elif BallVel[2] <= 0 and contact_flag == False:
            #     k_vir2 = 2000
            #     EndForce = - k_vir2 * (FootPos[2] - 0.4) - f_down
            #     EndForce_x = - 10000 * (FootPos[0] - FootPosInit[0])
                # print("nocontact foot pos: ", FootPos[2], EndForce)
                # print("nocontact ball pos: ", BallPos[2])
                # print("k_vir2", k_vir2)

            JointForce_z = EndForce
            JointForce_x = EndForce_x

            Torque_1 = (Jacobin_F[0, 0] * JointForce_x + Jacobin_F[0, 1] * JointForce_z)
            Torque_2 = (Jacobin_F[1, 0] * JointForce_x + Jacobin_F[1, 1] * JointForce_z)

            Torque_1_z = (Jacobin_F[0, 1] * JointForce_z)
            Torque_2_z = (Jacobin_F[1, 1] * JointForce_z)
            Torque_1_x = (Jacobin_F[0, 0] * JointForce_x)
            Torque_2_x = (Jacobin_F[1, 0] * JointForce_x)
            # print(EndForce, Torque_1, Torque_2)
            flag = 1
            LISM.setGeneralizedForce([0, Torque_1, Torque_2])
            # print("ball vel, foot vel: ", BallVel[2], FootVel[2])
            # print("foot pso: ", FootPos)
            # print("torque: ", EndForce, Torque_1_z, Torque_2_z)
            # print(Torque_1_z, Torque_2_z, Torque_1_x, Torque_2_x)
            # print("============================")

            # force data save
            ForceState = np.concatenate([ForceState, [[EndForce, ContactForce[2]]]], axis = 0)
            JointTorque = np.concatenate([JointTorque, [[Torque_1_z, Torque_2_z]]], axis = 0)
            JointVelSaved = np.concatenate([JointVelSaved, [[JointVel[1], JointVel[2]]]], axis = 0)
            
        elif BallPos[2] < (CtlParam["x_ref"]-0.1275):
            if flag == 1:
                print("the leave pos and vel of ball", FootPos[2], BallVel[2])
                print("*********************************************************")
            LISM.setGeneralizedForce([0, 0, 0])
            jointNominalConfig = np.array([0, 0.0, -1.57])
            jointVelocityTarget = np.zeros([LISM.getDOF()])
            # jointPgain = np.array([0, 1000, 1000])
            # jointDgain = np.array([0, 10, 10])

            jointPgain = np.array([0, 40, 40])
            jointDgain = np.array([0, 0.8, 0.8])
            
            LISM.setPdGains(jointPgain, jointDgain)
            LISM.setPdTarget(jointNominalConfig, jointVelocityTarget)
            
            force = LISM.getGeneralizedForce()
            ForceState = np.concatenate([ForceState, [[0.0, 0.0]]], axis = 0)
            # JointTorque = np.concatenate([JointTorque, [[force[1], force[2]]]], axis = 0)
            JointTorque = np.concatenate([JointTorque, [[0.0, 0.0]]], axis = 0)
            JointVelSaved = np.concatenate([JointVelSaved, [[0.0, 0.0]]], axis = 0)
            
            # print("Joint position: ", JointPos)
            # print("joint force: ", force)

            flag = 0
            EnvParam["con_flag"] = 0
            EnvParam["fun_flag"] = 0

        else:
            ForceState = np.concatenate([ForceState, [[0.0, 0.0]]], axis = 0)
            JointTorque = np.concatenate([JointTorque, [[0.0, 0.0]]], axis = 0)
            JointVelSaved = np.concatenate([JointVelSaved, [[0.0, 0.0]]], axis = 0)
        
        world.integrate()

    return BallState, EndFootState, ForceState, JointTorque, JointVelSaved, T


if __name__ == "__main__":
    # get params data
    FilePath = os.path.dirname(os.path.abspath(__file__))
    ParamFilePath = FilePath + "/config/LISM_test.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # load activation file and urdf file
    raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
    LISM_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/LISM_Arm_sim.urdf"

    # raisim world config setting
    world = raisim.World()
    t_step = ParamData["environment"]["t_step"]
    print(type(t_step))
    world.setTimeStep(t_step)
    ground = world.addGround(0)
    world.setDefaultMaterial(1, 1, 1)
    world.setMaterialPairProp("rubber", "rub", 1, 0, 0)
    gravity = world.getGravity()

    # load simulate arm model
    LISM = world.addArticulatedSystem(LISM_urdf_file)
    LISM.setName("LISM")
    # print(LISM.getDOF()

    # init pos and vel setting of the model
    jointNominalConfig = np.array([0.07, 0, -1.57])
    jointVelocityTarget = np.array([ParamData["controller"]["v_int"], 0, 0])
    LISM.setGeneralizedCoordinate(jointNominalConfig)
    LISM.setGeneralizedVelocity(jointVelocityTarget)
    LISM.setControlMode(raisim.ControlMode.PD_PLUS_FEEDFORWARD_TORQUE)

    JointPosInit, JointVelInit = LISM.getState()

    # get the frame id of the ball and ender of the arm
    FootFrameId = LISM.getFrameIdxByName("toe_fr_joint")
    FootPosInit = LISM.getFramePosition(FootFrameId)
    print(FootPosInit)
    BallFrameId = LISM.getFrameIdxByName("base_ball")
    BallPosInit = LISM.getFramePosition(BallFrameId)
    # print(BallFrameId)

    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)

    # dribbling control
    BallState, EndFootState, ForceState, JointTorque, JointVelSaved, T = DriControl(ParamData)

    # file save
    FileSave(BallState, EndFootState, ForceState, JointTorque, JointVelSaved, T, ParamData)

    # print("force, ", ForceState[0:100, 1])

    server.killServer()