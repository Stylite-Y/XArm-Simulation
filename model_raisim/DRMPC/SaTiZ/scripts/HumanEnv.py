import numpy as np
import raisimpy as raisim
import os
import datetime
import time
import yaml
import matplotlib.pyplot as plt


def WeightsLift():
    jointNominalConfig = np.array([0.0, 0.0, 1.03, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,     # wx, wy
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,               # rsx, rsy, re, lsx, lsy, le
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    ShoulderTraj = np.linspace(0, 1.57, 500)

    hand = Human.getBodyIdx("Formarm_L")
    ball1 = world.addSphere(0.08, 3.0, "steel")
    ball1.setPosition(0, 0.0, 1.0)
    world.addStiffWire(Human, 0, (0.08, -0.22, -0.23), ball1, 0, np.zeros(3), 0.5)

    for i in range(500000):
        time.sleep(0.01)
        jointTar = np.array([0.0, 0.0, 0.0]) 
        if i <500:
            jointTar = np.concatenate((jointTar, [ShoulderTraj[i]]))
            jointTar = np.concatenate((jointTar, np.zeros(12)))

        else:
            jointTar = np.concatenate((jointTar, [ShoulderTraj[-1]]))
            jointTar = np.concatenate((jointTar, np.zeros(12)))
            
        HumanPos, HumanVel = Human.getState()
        HumanPos = HumanPos[7:]
        HumanVel = HumanVel[6:]
        tor = 100*(jointTar - HumanPos) + (0-HumanVel)
        tor = np.concatenate((np.zeros(6), tor))
        Human.setGeneralizedForce(tor)
        server.integrateWorldThreadSafe()
    pass


if __name__ == "__main__":
    # get params config data
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/default_cfg.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # load activation file and urdf file
    # raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
    Human_urdf_file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/urdf/IOReal/Bipedal_V2_v2.urdf"
    print(Human_urdf_file)
    # raisim world config setting
    world = raisim.World()

    # set simulation step
    # t_step = ParamData["environment"]["t_step"] 
    # sim_time = ParamData["environment"]["sim_time"]
    world.setTimeStep(0.001)
    ground = world.addGround(0)

    gravity = world.getGravity()
    print(gravity)
    Human = world.addArticulatedSystem(Human_urdf_file)
    Human.setName("Human")
    print(Human.getDOF())

    # world.setGravity([0, 0, 0])
    # gravity1 = world.getGravity() 

    ## ====================
    # ball control initial pos and vel setting
    jointNominalConfig = np.array([0.0, 0.0, 1.03, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,     # wx, wy
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,               # rsx, rsy, re, lsx, lsy, le
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    # jointNominalConfig = np.array([0.34, -0.05, 0.5,1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,     
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,          
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Human.setGeneralizedCoordinate(jointNominalConfig)
    Human.setGeneralizedVelocity(jointVelocityTarget)
    
    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)

    WeightsLift()

    server.killServer()