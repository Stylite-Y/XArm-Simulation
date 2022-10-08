import pybullet as p
import pybullet_data
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml

# get params config data
FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ParamFilePath = FilePath + "/config/default_cfg.yaml"
ParamFile = open(ParamFilePath, "r", encoding="utf-8")
ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

# load activation file and urdf file
# raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
Human_urdf_file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/urdf/IOReal/Bipedal_V2_v.urdf"

# connect the GUI

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)

planeId = p.loadURDF("plane.urdf")

cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF(Human_urdf_file,cubeStartPos, cubeStartOrientation)

for i in range (1000):
    p.stepSimulation()
    time.sleep(1./240.)