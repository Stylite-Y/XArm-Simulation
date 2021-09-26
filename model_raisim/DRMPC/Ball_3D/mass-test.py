import numpy as np
import sys
import do_mpc
from casadi import *  # symbolic library CasADi
import datetime
import raisimpy as raisim
import yaml
import time
import math

def xytri(t):
    r = 1.0
    T = 1
    omga = 2 * math.pi / T
    x = r * sin(omga * t)
    y = r * cos(omga * t)
    # print(x, y)
    return x, y

def SetPoint_MPCControl(ParamData):

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    """
    dynamics model:
    x1_dot = x2
    x2_dot = u
    """

    v_xref = -16
    v_yref = -16
    x_ref = 1.4

    g = 10
    m = 0.5     # kg, mass of the ball
    # set variable of the dynamics system
    x_b = model.set_variable(var_type='_x', var_name='x_b', shape=(1, 1))
    y_b = model.set_variable(var_type='_x', var_name='y_b', shape=(1, 1))
    # x2 = model.set_variable(var_type='_x', var_name='theta_dot', shape=(1, 1))
    dx_b = model.set_variable(var_type='_x', var_name='dx_b', shape=(1, 1))
    dy_b = model.set_variable(var_type='_x', var_name='dy_b', shape=(1, 1))
    ux = model.set_variable(var_type='_u', var_name='ux', shape=(1, 1))
    uy = model.set_variable(var_type='_u', var_name='uy', shape=(1, 1))

    xtraj = model.set_variable(var_type='_tvp', var_name='xtraj')
    # ytraj = model.set_variable(var_type='_tvp', var_name='ytraj')

    # rhs
    model.set_rhs('x_b', dx_b)
    model.set_rhs('y_b', dy_b)

    model.set_rhs('dx_b', ux / m)
    model.set_rhs('dy_b', uy / m)
    # model.set_rhs('dx_b', u)

    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 50,
        't_step': 0.1,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    q1 = 1000
    q2 = 1000
    r = 0.001
    lterm = q1 * (x_b - model.tvp['xtraj']) ** 2
    mterm = lterm
    
    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(ux=1e-4, uy=1e-4)

    mpc.bounds['lower', '_x', 'x_b'] = -1000
    mpc.bounds['upper', '_x', 'x_b'] = 1000
    mpc.bounds['lower', '_x', 'y_b'] = -1000
    mpc.bounds['upper', '_x', 'y_b'] = 1000

    mpc.bounds['lower', '_u', 'ux'] = -1000
    mpc.bounds['upper', '_u', 'ux'] = 1000
    mpc.bounds['lower', '_u', 'uy'] = -1000
    mpc.bounds['upper', '_u', 'uy'] = 1000

    m_ = 1 * 1e-4 * np.array([0.5, 0.4, 0.6])
    # mpc.set_rterm(u=1e-2)
    xtra = [0.0]
    ytra = [0.0]
    for i in range(5000):
        t_pre = i * setup_mpc['t_step']
        x, y = xytri(t_pre)
        xtra.append(x)
        ytra.append(y)
    
    tvp_template = mpc.get_tvp_template()
    def tvp_fun(t_ind):
        horizon = setup_mpc['n_horizon']
        print(horizon)
        print(1)
        for k in range(horizon + 1):
            t_pre = t_ind + k * setup_mpc['t_step']
            # print(t_pre)
            # tvp_template['_tvp', k, 'xtraj'], tvp_template['_tvp', k, 'ytraj'] = xytri(t_pre)
            tvp_template['_tvp', k, 'xtraj'] = 0.4
            # tvp_template['_tvp', k, 'ytraj'] = 0.3

            if k == 0:
                print("tvp: ", len(tvp_template['_tvp']))
        # ind = int(t_ind/setup_mpc['t_step'])
        # tvp_template['_tvp', :, 'xtraj'] = vertsplit(xtra[ind:ind+setup_mpc['n_horizon']+1])
        # tvp_template['_tvp', :, 'ytraj'] = vertsplit(ytra[ind:ind+setup_mpc['n_horizon']+1])

        return tvp_template
            
    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=0.01)
    tvp_template = simulator.get_tvp_template()

    def tvp_fun(t_ind):
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()


    estimator = do_mpc.estimator.StateFeedback(model)

    # =======================================================
    # visualization
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm

    # print("===========initial mpc data===========")
    # print(mpc.data['_x', 'x_b'][0, 0])s
    # print(mpc.data['_x', 'x_b'])
    mpc.reset_history()

    n_step = 100
    for k in range(n_step):
        
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

        mpc.reset_history()

    # starttime_pre = datetime.datetime.now()
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
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

    jointNominalConfig = np.array([1.0, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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


