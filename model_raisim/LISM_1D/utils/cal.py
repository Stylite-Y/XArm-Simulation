import os
import numpy as np
from numpy.core.fromnumeric import ptp
import raisimpy as raisim
import time

raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
# LISM_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/urdf/black_panther.urdf"
world = raisim.World()
ground = world.addGround(0)
world.setTimeStep(0.001)

world.setMaterialPairProp("default", "steel", 0.0, 0.85, 3)

ball1 = world.addSphere(0.1498, 0.8, "steel")
ball1.setPosition(0, 0.0, 5)

ball2 = world.addSphere(0.1499, 0.8, "steel")
ball2.setPosition(0.8, 0.0, 5)

server = raisim.RaisimServer(world)
server.launchServer(8080)
for i in range(500000):
    time.sleep(0.001)
    server.integrateWorldThreadSafe()


server.killServer()