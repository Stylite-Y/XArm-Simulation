import os
import numpy as np
from numpy.core.fromnumeric import ptp
import raisimpy as raisim
import time

raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
# LISM_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/urdf/black_panther.urdf"
LISM_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/LISM_Arm_sim.urdf"

world = raisim.World()
t_step = 0.00005
world.setTimeStep(t_step)
ground = world.addGround(0)

world.setDefaultMaterial(1, 1, 1)
world.setMaterialPairProp("rubber", "rub", 1, 0, 0)

gravity = world.getGravity()
# print(g)

LISM = world.addArticulatedSystem(LISM_urdf_file)
LISM.setName("LISM")
# print(LISM.getDOF()

jointNominalConfig = np.array([-0.1, 0, -1.57])
jointVelocityTarget = np.array([-10, 0, 0])
LISM.setGeneralizedCoordinate(jointNominalConfig)
LISM.setGeneralizedVelocity(jointVelocityTarget)
LISM.setControlMode(raisim.ControlMode.PD_PLUS_FEEDFORWARD_TORQUE)

JointPosInit, JointVelInit = LISM.getState()
AbadFrameId = LISM.getFrameIdxByName("abad_upper_r")
AbadPos = LISM.getFramePosition(AbadFrameId)

FootFrameId = LISM.getFrameIdxByName("toe_fr_joint")
FootPosInit = LISM.getFramePosition(FootFrameId)
print(FootPosInit)
print(AbadFrameId)


BallFrameId = LISM.getFrameIdxByName("base_ball")
BallPosInit = LISM.getFramePosition(BallFrameId)
print(BallPosInit)
print(BallFrameId)


server = raisim.RaisimServer(world)
server.launchServer(8080)

UpperArmLength = 0.2
LowerArmLength = 0.2
flag = 0
m = 1
con_flag = 0

f1 = 25
fun_flag = 0
# x_init = 0.4   # init position
# v_init = 10        # init velocity
x_top = 0.6 # dribbling height of ball
v_ref = -15     # desired velocity
mass = 0.5
x_ref = 0.35
g = -gravity[2]
# k_vir = 1000
# f2 = 500

def ParamsCal(x0, v0, x_top, xref, vref, f1):
    dx_up = x_top - x0
    dx_down = x_top - xref

    k_vir = (mass * v0 ** 2 - 2 * mass * g * dx_up - 2 * f1 * dx_up) / ((x_top - xref) ** 2 - (x0 - xref) ** 2)
    f2 = (mass * vref ** 2 - 2 * mass * g * dx_down - k_vir * dx_down ** 2) / (2 * dx_down)

    print("dx_up and dx_down is ", dx_up, dx_down)
    if k_vir < 0:
        raise ValueError('invalid value: k_vir is negative, can not sqrt:')
    return k_vir, f2

for i in range(500000):
    time.sleep(0.001)
    # if i == 0:
    #     server.startRecordingVideo("v10_with-x_1x.mp4")

    BallPos = LISM.getFramePosition(BallFrameId)
    BallVel = LISM.getFrameVelocity(BallFrameId)
    JointPos, JointVec = LISM.getState()
    FootPos = LISM.getFramePosition(FootFrameId)
    FootVel = LISM.getFrameVelocity(FootFrameId)

    jointPgain = np.array([0, 0, 0])
    jointDgain = np.array([0, 0, 0])
    LISM.setPdGains(jointPgain, jointDgain)

    ContactPoint = LISM.getContacts()

    contact_flag = False
    for c in ContactPoint:
        contact_flag = c.getlocalBodyIndex() == LISM.getBodyIdx("lower_r")
        if(contact_flag):
            break
        pass
    
    if con_flag == 0 and contact_flag:
        con_flag = 1
        # print("the contact vel of ball", BallVel[2])

    if BallPos[2] >= (x_ref-0.1275) and con_flag == 1:
    
        if BallVel[2] > 0 and fun_flag == 0:
            dx_up = x_top - FootPos[2]
            if (dx_up * 2 * g) >= (BallVel[2] ** 2):
                raise FloatingPointError("calculate Error: init velocity is too small or the heaving height is too high!")
            else:
                k_vir, f2 = ParamsCal(FootPos[2], BallVel[2], x_top, x_ref, v_ref, f1)
            fun_flag = 1
        
            print("contact point pos and vel of ball ", FootPos[2], BallVel[2])
            print("the k_vir and f1, f2 is ", k_vir, f1, f2)

        jointPgain = np.zeros(LISM.getDOF())
        jointDgain = np.zeros(LISM.getDOF())

        LISM.setPdGains(jointPgain, jointDgain)

        JointPos, JointVec = LISM.getState()
        FootPos = LISM.getFramePosition(FootFrameId)
        FootVel = LISM.getFrameVelocity(FootFrameId)

        ## Force kinematics
        # jacobian matrix of force transmission
        a11 = - UpperArmLength * np.cos(JointPos[1]) - LowerArmLength * np.cos(JointPos[1] + JointPos[2])
        a12 = UpperArmLength * np.sin(JointPos[1]) + LowerArmLength * np.sin(JointPos[1] + JointPos[2])
        a21 = - LowerArmLength * np.cos(JointPos[1] + JointPos[2])
        # print(np.cos(JointPos[1] + JointPos[2]))
        a22 = LowerArmLength * np.sin(JointPos[1] + JointPos[2])
        Jacobin_F = np.array([[a11, a12], 
                              [a21, a22]])
        # print(Jacobin_F)
        if BallVel[2] <0.1 and BallVel[2] > 0:
            print("the highest ball and foot pos: ", BallPos[2] ,FootPos[2])

        if BallVel[2] > 0 and contact_flag:
            # ContactPointVel = LISM.getContactPointVel(ContactPoint[0].getlocalBodyIndex())
            ContactPointVel = ContactPoint[0].getImpulse()
            ContactForce = ContactPointVel / t_step
            EndForce = - k_vir * (FootPos[2] - x_ref) - f1
            # EndForce = - 2000 * (FootPos[2] - FootPosInit[2])
            EndForce_x = - 10000 * (FootPos[0] - FootPosInit[0])
            # EndForce_x = 0
            # EndForce = 800
            # print("1", ContactForce[2])
            # LISM.setGeneralizedForce([50, 50])
        # elif BallVel[2] > 0 and BallPos[2] > 0.6:
        #     EndForce = - 10000 * (FootPos[2] - 0.4) - f1
        #     # EndForce = - 2000 * (FootPos[2] - FootPosInit[2])
        #     EndForce_x = - 10000 * (FootPos[0] - FootPosInit[0])

        elif BallVel[2] <= 0  and contact_flag == True:

            # EndForce = 2000 * (FootPos[2] - FootPosInit[2]) + 150
            EndForce = - k_vir * (FootPos[2] - x_ref) - f2
            EndForce_x = - 10000 * (FootPos[0] - FootPosInit[0])
            # print("contact foot pos: ", FootPos[2], EndForce)
            # print("contact ball pos: ", BallPos[2])
            # print("2")
            # print("k_vir", k_vir)


        # elif BallVel[2] <= 0 and contact_flag == False:
        #     k_vir2 = 2000
        #     EndForce = - k_vir2 * (FootPos[2] - 0.4) - f2
        #     # EndForce = - 2000 * (FootPos[2] - FootPosInit[2])
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
        # torque = LISM.getGeneralizedForce()
        # print("ball vel, foot vel: ", BallVel[2], FootVel[2])
        # print("foot pso: ", FootPos)
        # print("torque: ", EndForce, Torque_1_z, Torque_2_z)
        # print(Torque_1_z, Torque_2_z, Torque_1_x, Torque_2_x)
        # print("============================")
        # LISM.setGeneralizedForce([200, 200])

    elif BallPos[2] < 0.3:
        if flag == 1:
            print("the leave pos and vel of ball", FootPos[2], BallVel[2])
            print("*********************************************************")
        LISM.setGeneralizedForce([0, 0, 0])
        jointNominalConfig = np.array([0, 0.0, -1.57])
        jointVelocityTarget = np.zeros([LISM.getDOF()])
        jointPgain = np.array([0, 10000, 10000])
        jointDgain = np.array([0, 100, 100])
        LISM.setPdGains(jointPgain, jointDgain)
        LISM.setPdTarget(jointNominalConfig, jointVelocityTarget)
        # force = LISM.getGeneralizedForce()
        # print("Joint position: ", JointPos)
        # print("joint force: ", force)
        # print("PD", m)
        m = m + 1

        # jointNominalConfig = np.array([0, 0.0, -1.57])
        # jointVelocityTarget = np.zeros([LISM.getDOF()])
        # LISM.setGeneralizedCoordinate(jointNominalConfig)
        flag = 0
        con_flag = 0
        fun_flag = 0

    # server.integrateWorldThreadSafe()

    # if i == 20000:
    #     raisim. stopRecordingVideo()
    world.integrate()

server.killServer()