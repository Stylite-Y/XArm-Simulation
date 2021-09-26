import numpy as np
import sys
import do_mpc
from casadi import *  # symbolic library CasADi
import datetime
import raisimpy as raisim
import yaml
import time
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from template_mpc import template_mpc
from template_simulator import template_simulator
from template_model import template_model

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/utils")
print(os.path.abspath(os.path.dirname(__file__))) # get current file path
# from ParamsCalculate import ControlParamCal
# import visualization
# import FileSave

# from Dribble_model import template_model
# from Dribble_mpc import template_mpc
# from Dribble_simulator import template_simulator

def SetPoint_MPCControl(ParamData):

    """ User settings: """
    show_animation = True
    store_animation = False
    store_results = False

    """
    Get configured do-mpc modules:
    """

    model = template_model()
    simulator = template_simulator(model)
    mpc = template_mpc(model)
    estimator = do_mpc.estimator.StateFeedback(model)

    """
    Set initial state
    """

    # x0 = np.array([0.0, 1.0, 0.0, 0.0]).reshape(-1, 1)
    # simulator.x0 = x0

    # mpc.x0 = x0
    # estimator.x0 = x0

    # mpc.set_initial_guess()

    """
    Run MPC main loop:
    """
    # time_list = []
    # fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
    # plt.ion()
    n_steps = 2400
    for k in range(n_steps):
        
        time.sleep(0.0001)

        BallPos, BallVel = ball1.getState()
        BallPos = BallPos[0:3]
        BallVel = BallVel[0:3]
        PosInit = BallPos[0:2]
        VelInit = BallVel[0:2]

        x0  = np.concatenate([PosInit, VelInit])
        x0 = x0.reshape(-1, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        simulator.x0 = x0
        
        mpc.x0 = x0
        estimator.x0 = x0

        mpc.set_initial_guess()

        u0 = mpc.make_step(x0)
        # y_next = simulator.make_step(u0)
        # x0 = estimator.make_step(y_next)
        print("=====================================================")
        print("vel and force is: ", VelInit, u0)
        XForce = u0[0, 0]
        YForce = u0[1, 0]
        ball1.setExternalForce(0, [0, 0, 0], [XForce, YForce, 0.0])
        world.integrate()
        # if show_animation:

        #     graphics.plot_results(t_ind=k)
        #     graphics.plot_predictions(t_ind=k)
        #     graphics.reset_axes()
        #     plt.show()
        #     plt.pause(0.01)


    # graphics.plot_predictions(t_ind=0)

    show_animation = True

    mpl.rcParams['font.size'] = 18
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['axes.grid'] = True

    fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
    plt.ion()
    graphics.plot_results()
    graphics.reset_axes()
    plt.show()

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

    # raisim world config setting
    world = raisim.World()

    # set simulation step
    t_step = ParamData["environment"]["t_step"] 
    sim_time = ParamData["environment"]["sim_time"]
    world.setTimeStep(0.01)
    ground = world.addGround(0)
    
    # set material collision property
    # world.setMaterialPairProp("rubber", "rub", 1, 0, 0)
    world.setMaterialPairProp("rubber", "rub", 1.0, 0.85, 0.0001)     # ball rebound model test
    world.setMaterialPairProp("default", "rub", 0.8, 1.0, 0.0001)
    gravity = world.getGravity()

    world.setMaterialPairProp("default", "steel", 0.0, 1.0, 0.001)
    ball1 = world.addArticulatedSystem(ball1_urdf_file)
    print(ball1.getDOF())
    ball1.setName("ball1")
    gravity = world.getGravity()
    print(gravity)
    print(ball1.getGeneralizedCoordinateDim())

    pin1 = world.addSphere(0.1, 0.8)
    pin1.setAppearance("1,0,0,1.0")
    pin1.setPosition(0.75, 0.0, 0.0)

    jointNominalConfig = np.array([0.0, 0.7, 0.15, 1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ball1.setGeneralizedCoordinate(jointNominalConfig)
    # print(ball1.getGeneralizedCoordinateDim())
    ball1.setGeneralizedVelocity(jointVelocityTarget)

    world.setMaterialPairProp("default", "steel", 0.0, 1.0, 0.001)
    world.setMaterialPairProp("default", "rub", 0.0, 1.0, 0.001)

    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)


    Data = SetPoint_MPCControl(ParamData)

    # print("force, ", ForceState[0:100, 1])

    server.killServer()


