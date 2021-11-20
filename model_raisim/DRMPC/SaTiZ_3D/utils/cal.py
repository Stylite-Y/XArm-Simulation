import os
import numpy as np
from numpy.core.fromnumeric import ptp
import raisimpy as raisim
import time
import sys
import datetime
import matplotlib
import matplotlib.pyplot as plt

from xbox360controller import Xbox360Controller
xbox = Xbox360Controller(0, axis_threshold=0.02)
# v_ref = xbox.trigger_r.value * (-4) - 3
# v_ref = xbox.trigger_r.value * (-7) - 5

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/utils")
print(os.path.abspath(os.path.dirname(__file__))) # get current file path
from ParamsCalculate import ControlParamCal
import visualization
import FileSave

raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
ball1_urdf_file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/urdf/ball.urdf"
# ball_file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/urdf/meshes/ball/ball.obj"
# ball1_urdf_file = "/home/stylite-y/Documents/Raisim/raisim_workspace/raisimLib/rsc/anymal/urdf/anymal.urdf"

print(ball1_urdf_file)
world = raisim.World()
ground = world.addGround(0)
t_step = 0.0001
world.setTimeStep(t_step)
gravity = world.getGravity()
# print(1)

ball1 = world.addArticulatedSystem(ball1_urdf_file)
print(ball1.getDOF())
ball1.setName("ball1")
gravity = world.getGravity()
print(gravity)
print(ball1.getGeneralizedCoordinateDim())

jointNominalConfig = np.array([0.0, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0])
jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ball1.setGeneralizedCoordinate(jointNominalConfig)
# print(ball1.getGeneralizedCoordinateDim())
ball1.setGeneralizedVelocity(jointVelocityTarget)

world.setMaterialPairProp("default", "steel", 0.0, 0.85, 0.001)
world.setMaterialPairProp("default", "rub", 0.0, 0.85, 0.001)
# ball1 = world.addSphere(0.12, 0.8, "steel")
# dummy_inertia = np.zeros([3, 3])
# np.fill_diagonal(dummy_inertia, 0.1)
# ball1 = world.addMesh(ball_file, 0.6, dummy_inertia, np.array([0, 0, 1]), 0.001, "rub")
# ball1.setPosition(0, 0.0, 0.5)
# ball1.setVelocity(1.0, 0.0, -5, 0.0, 0, 0)

server = raisim.RaisimServer(world)
server.launchServer(8080)

Vel = 3
for i in range(50000):
    time.sleep(0.0005)

    x_vel = 3 * xbox.axis_r.x
    y_vel = 3 * xbox.axis_r.y
    # ball1.setExternalForce(0, [0, 0, 0], [XForce, YForce, 0.0])
    jointVelocityTarget = np.array([x_vel, y_vel, 0.0, 0.0, 0.0, 0.0])
    ball1.setGeneralizedVelocity(jointVelocityTarget)

    world.integrate()

server.killServer()