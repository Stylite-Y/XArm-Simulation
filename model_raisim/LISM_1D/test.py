
import os
import numpy as np
import raisimpy as raisim
import time

raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
# arm_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/urdf/black_panther.urdf"
# arm_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/LISM_Arm_sim.urdf"
arm_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/Arm_test.urdf"

world = raisim.World()
world.setTimeStep(0.00005)
ground = world.addGround(0, "steel")

g = world.getGravity()
# print(g)

# pin1 = world.addSphere(0.1, 0.8, "steel")
# pin1.setAppearance("1,0,0,1")
# pin1.setPosition(0.335, 0.18, 1)
# pin1.setBodyType(raisim.BodyType.DYNAMIC)

world.setMaterialPairProp("steel", "steel", 0.8, 0.95, 0.001)

arm = world.addArticulatedSystem(arm_urdf_file)
arm.setName("arm")
print(arm.getDOF())

# jointNominalConfig = np.array([-1.0472, 2.0944])
jointNominalConfig = np.array([0.785, -1.57])
jointVelocityTarget = np.zeros([arm.getDOF()])
arm.setGeneralizedCoordinate(jointNominalConfig)
# jointPgain = np.ones(arm.getDOF()) * 100.0
# jointDgain = np.ones(arm.getDOF()) * 1.0

# arm.setPdGains(jointPgain, jointDgain)
# arm.setPdTarget(jointNominalConfig, jointVelocityTarget)

x_0, v_0 = arm.getState()
# print(x_0)
# frameid = arm.getFrameIdxByName("base_link1")
# print(frameid)

# bodyid = arm.getBodyIdx("link2")
# dim = arm.getGeneralizedCoordinateDim()
# print(dim)

frameid = arm.getFrameIdxByName("link2_ball")
foot_pos_0 = arm.getFramePosition(frameid)
print(foot_pos_0)

frameid2 = arm.getFrameIdxByName("base_link1")
foot_pos_1 = arm.getFrameOrientation(frameid2)
print(foot_pos_1)

server = raisim.RaisimServer(world)
server.launchServer(8080) 

# time.sleep(200)
# world.integrate1()
K = -20 * g[2]
UpperArmLength = 0.3
LowerArmLength = 0.3
for i in range(500000):
    time.sleep(0.001)
    if i % 100 == 0 and i <=2000:
        foot_pos_1 = arm.getFrameOrientation(frameid2)
        print(foot_pos_1)
        print("======================================")

    JointPos, JointVec = arm.getState()
    foot_pos = arm.getFramePosition(frameid)
    foot_vec = arm.getFrameVelocity(frameid)

    EndForce = K * (- foot_pos[2] + foot_pos_0[2])
    EndForce_x = - 1 * (foot_pos[0] - foot_pos_0[0])
    JointForce_z = EndForce
    JointForce_x = EndForce_x
    
    a11 = - UpperArmLength * np.cos(JointPos[0]) - LowerArmLength * np.cos(JointPos[0] + JointPos[1])
    a12 = UpperArmLength * np.sin(JointPos[0]) + LowerArmLength * np.sin(JointPos[0] + JointPos[1])
    a21 = - LowerArmLength * np.cos(JointPos[0] + JointPos[1])
    a22 = LowerArmLength * np.sin(JointPos[0] + JointPos[1])
    Jacobin_F = np.array([[a11, a12], 
                          [a21, a22]])

    Torque_1 = Jacobin_F[0, 0] * JointForce_x + Jacobin_F[0, 1] * JointForce_z
    Torque_2 = Jacobin_F[1, 0] * JointForce_x + Jacobin_F[1, 1] * JointForce_z

    arm.setGeneralizedForce([Torque_1, Torque_2])


    # print(JointPos)
    # print(foot_pos)
    # print(EndForce, JointForce_x, JointForce_z)
    # print(Torque_1,Torque_2)
    # if i==1000:
    #   exit()

    # if i <= 2000:
    #   # print(foot_pos_0[2], foot_pos[2], foot_vec[2], EndForce)
    #   print(foot_pos_0[2], foot_pos[2], foot_vec[2])

    # server.integrateWorldThreadSafe()
    world.integrate()

server.killServer()


# physicsClient = p.connect(p.GUI)

# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# p.setGravity(0, 0, -10)
# planeId = p.loadURDF("plane.urdf")
# cubeStartPos = [0, 0, 1]
# cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# boxId = p.loadURDF("/home/stylite-y/Documents/Master/Manipulator/Simulation/model_raisim/urdf/urdf/black_panther.urdf", cubeStartPos, cubeStartOrientation)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)

# useRealTimeSimulation = 0

# if (useRealTimeSimulation):
#   p.setRealTimeSimulation(1)

# while 1:
#   if (useRealTimeSimulation):
#     p.setGravity(0, 0, -10)
#     sleep(0.01)  # Time in seconds.
#   else:
#     p.stepSimulation()