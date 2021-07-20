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
from xbox360controller import Xbox360Controller

sys.path.append("./utils")
from ParamsCalculate import ControlParamCal
import visualization
import FileSave

# xbox = Xbox360Controller(0, axis_threshold=0.02)

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

    for i in range(EnvParam["sim_time"]):
        # v_ref = xbox.trigger_r.value * (-4) - 3
        # v_ref = xbox.trigger_r.value * (-7) - 5

        v_ref = CtlParam["v_ref"]
        time.sleep(0.0005)
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
            print("the contact vel of ball", BallVel[2])

        # if contact_flag == 1 and BallVel[2] >= CtlParam["v_ref"]:
        if contact_flag == 1 and BallVel[2] >= v_ref:
    
            # print("the contact vel of ball", BallVel[2])
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
            
            if BallVel[2] > 0:
            # ContactPointVel = LISM.getContactPointVel(ContactPoint[0].getlocalBodyIndex())
                ContactPointVel = ContactPoint[0].getImpulse()
                ContactForce = ContactPointVel / t_step
                EndForce = - CtlParam["K_virz"] * (FootPos[2] - CtlParam["x_ref_FD"]) - CtlParam["K_errv_up"] * (FootVel[2] - 0)
                # EndForce = - 2000 * (FootPos[2] - FootPosInit[2])
                EndForce_x = - CtlParam["K_virx"] * (FootPos[0] - FootPosInit[0])

            if BallVel[2] <= 0:
            # ContactPointVel = LISM.getContactPointVel(ContactPoint[0].getlocalBodyIndex())
                ContactPointVel = ContactPoint[0].getImpulse()
                ContactForce = ContactPointVel / t_step
                # EndForce = - CtlParam["K_virz"] * (FootPos[2] - CtlParam["x_ref_FD"]) - CtlParam["K_errv_down"] * (FootVel[2] - CtlParam["v_ref"])
                EndForce = - CtlParam["K_virz"] * (FootPos[2] - CtlParam["x_ref_FD"]) - CtlParam["K_errv_down"] * (FootVel[2] - v_ref)
                # EndForce = - 2000 * (FootPos[2] - FootPosInit[2])
                EndForce_x = - CtlParam["K_virx"] * (FootPos[0] - FootPosInit[0])

            JointForce_z = EndForce
            JointForce_x = EndForce_x

            Torque_1 = (Jacobin_F[0, 0] * JointForce_x + Jacobin_F[0, 1] * JointForce_z)
            Torque_2 = (Jacobin_F[1, 0] * JointForce_x + Jacobin_F[1, 1] * JointForce_z)

            Torque_1_z = (Jacobin_F[0, 1] * JointForce_z)
            Torque_2_z = (Jacobin_F[1, 1] * JointForce_z)
            flag = 1
            LISM.setGeneralizedForce([0, Torque_1, Torque_2])
            # print("ball vel, foot vel: ", BallVel[2], FootVel[2])
            # print("foot pso: ", FootPos)
            # print("torque: ", EndForce, Torque_1_z, Torque_2_z)
            # print("============================")

            # force data save
            ForceState = np.concatenate([ForceState, [[EndForce, ContactForce[2]]]], axis = 0)
            JointTorque = np.concatenate([JointTorque, [[Torque_1_z, Torque_2_z]]], axis = 0)
            JointVelSaved = np.concatenate([JointVelSaved, [[JointVel[1], JointVel[2]]]], axis = 0)
            
        # elif BallPos[2] < (CtlParam["x_ref"]-0.1275):
        else:
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

        # else:
        #     ForceState = np.concatenate([ForceState, [[0.0, 0.0]]], axis = 0)
        #     JointTorque = np.concatenate([JointTorque, [[0.0, 0.0]]], axis = 0)
        #     JointVelSaved = np.concatenate([JointVelSaved, [[0.0, 0.0]]], axis = 0)
        
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
    # world.setDefaultMaterial(1, 1, 1)
    world.setMaterialPairProp("rubber", "rub", 1, 0, 0)
    world.setMaterialPairProp("default", "rub", 1, 0.85, 0.0001)
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
    Data = FileSave.DataSave(BallState, EndFootState, ForceState, JointTorque, JointVelSaved, T, ParamData)

    visualization.DataProcess(Data)

    # print("force, ", ForceState[0:100, 1])

    server.killServer()