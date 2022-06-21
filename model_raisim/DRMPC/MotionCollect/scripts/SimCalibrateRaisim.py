import numpy as np
import time
import os
import yaml
import raisimpy as raisim
import datetime
import matplotlib.pyplot as plt

# UrdfPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/DIP.urdf"
UrdfPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/human2_175.urdf"
# Datapath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/walk_boneglocal.calc"
Datapath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/walk_bvhlocal2.calc"

posture_data = []

# read the data from document
with open(Datapath, 'r') as f:      #通过使用with...as...不用手动关闭文件。当执行完内容后，自动关闭文件
    for i in range(6):               #跳过前五行
        next(f)
    for lines in f.readlines():       #依次读取每行
        posture_data.append(list(map(float, lines.split())))

#总帧数
numFrames = len(posture_data)

raisim.World.setLicenseFile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/activation.raisim")
# region raisim ecvsetting
world = raisim.World()
world.setTimeStep(1/120)
ground = world.addGround(0)

world.setGravity([0, 0, 0])
gravity = world.getGravity()
print(gravity)
Human = world.addArticulatedSystem(UrdfPath)
Human.setName("Human")
print(Human.getDOF())

# raisim world server setting
server = raisim.RaisimServer(world)
server.launchServer(8080)
# endregion

base = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
jointNominalConfig = np.concatenate((base, np.array([1.0, 0.0, 0.0, 0.0]*17)), axis=0)
jointVelocityTarget = np.array([0.0]*60)
# jointNominalConfig = np.array([0.0, 0.0, 2.0, 1, 0.0, 0.0, 0, 0.0, 1.57])
# jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
Human.setGeneralizedCoordinate(jointNominalConfig)
Human.setGeneralizedVelocity(jointVelocityTarget)
print(jointNominalConfig.shape)

for j in range(300, 700):
    Posture_data_current1 = np.array([posture_data[i][7], posture_data[i][8], posture_data[i][9],posture_data[i][6]])
    pass

for i in range(300, 1500):
    time.sleep(0.1)
    Posture_data = base
    for j in range(17):
        data = np.array([posture_data[i][j*16 + 6], posture_data[i][j*16 + 7], posture_data[i][j*16 + 8], posture_data[i][j*16 + 9]])
        Posture_data = np.concatenate((Posture_data, data), axis=0)
    pass

    # right leg
    # base = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # foreore = np.array([1.0, 0.0, 0.0, 0.0]*1)
    # Posture_data_current2 = [posture_data[i][23], posture_data[i][24],posture_data[i][25], posture_data[i][22]]
    # jointNominalConfig = np.concatenate((base, foreore), axis=0)
    # jointNominalConfig = np.concatenate((jointNominalConfig, Posture_data_current2), axis=0)
    # jointNominalConfig = np.concatenate((jointNominalConfig, np.array([1.0, 0.0, 0.0, 0.0]*15)), axis=0)

    # right arm
    # base = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # foreore = np.array([1.0, 0.0, 0.0, 0.0]*8)
    # Posture_data_current9 = [posture_data[i][134], posture_data[i][135], posture_data[i][136], posture_data[i][137]]
    # jointNominalConfig = np.concatenate((base, foreore), axis=0)
    # jointNominalConfig = np.concatenate((jointNominalConfig, Posture_data_current9), axis=0)
    # jointNominalConfig = np.concatenate((jointNominalConfig, np.array([1.0, 0.0, 0.0, 0.0]*8)), axis=0)

    # Posture_data_current14 = [posture_data[i][198], posture_data[i][199], posture_data[i][200], posture_data[i][201]]
    # base = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # foreore = np.array([1.0, 0.0, 0.0, 0.0]*12)
    # jointNominalConfig = np.concatenate((base, foreore), axis=0)
    # jointNominalConfig = np.concatenate((jointNominalConfig, Posture_data_current14), axis=0)
    # jointNominalConfig = np.concatenate((jointNominalConfig, np.array([1.0, 0.0, 0.0, 0.0]*4)), axis=0)

    Human.setGeneralizedCoordinate(Posture_data)
    server.integrateWorldThreadSafe()
    pass