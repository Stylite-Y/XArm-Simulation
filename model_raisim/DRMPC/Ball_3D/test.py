import numpy as np
import sys
import do_mpc
from casadi import *  # symbolic library CasADi
import datetime
import raisimpy as raisim
import yaml
import time

def tanh_sig(x):
    return 0.5 + 0.5 * np.tanh(1000 * x)


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

show_animation = True

mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

def MPCControl():
    TraPoint_x = np.array([-0.2, -0.5, 0.1])
    TraPoint_y = np.array([0.0, 0.6, 0.6])

    flag = 0
    m = 0.4
    z_ref = 0.5
    v_zref = -6.0
    v_xref = -6.0
    v_yref = -6.0
    dx_ref = 0.6
    dy_ref = 0.6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

    sim_t_step = 0.001
    xtra = 0.0
    ytra = 0.0
    # v_xref = 6
    index = 0
    g = -9.8

    flag = 0
    pos_init = np.array([0.0, 0.0, 0.45])
    v_init = np.array([5, 0.0, -5])
    x0  = np.concatenate([pos_init, v_init])
    x0 = x0.reshape(-1, 1)   

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)
    x_b = model.set_variable(var_type='_x', var_name='x_b', shape=(1, 1))
    y_b = model.set_variable(var_type='_x', var_name='y_b', shape=(1, 1))
    z_b = model.set_variable(var_type='_x', var_name='z_b', shape=(1, 1))
    dx_b = model.set_variable(var_type='_x', var_name='dx_b', shape=(1, 1))
    dy_b = model.set_variable(var_type='_x', var_name='dy_b', shape=(1, 1))
    dz_b = model.set_variable(var_type='_x', var_name='dz_b', shape=(1, 1))
    u_x = model.set_variable(var_type='_u', var_name='u_x', shape=(1, 1))
    u_y = model.set_variable(var_type='_u', var_name='u_y', shape=(1, 1))
    u_z = model.set_variable(var_type='_u', var_name='u_z', shape=(1, 1))

    model.set_rhs('x_b', dx_b)
    model.set_rhs('y_b', dy_b)
    model.set_rhs('z_b', dz_b)
    dx_b_next = vertcat(
        tanh_sig(z_b - z_ref) * u_x / m,
    )
    dy_b_next = vertcat(
        tanh_sig(z_b - z_ref) * u_y / m,
    )
    dz_b_next = vertcat(
        g + tanh_sig(z_b - z_ref) * u_z / m,
    )
    model.set_rhs('dx_b', dx_b_next)
    model.set_rhs('dy_b', dy_b_next)
    model.set_rhs('dz_b', dz_b_next)

    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 150,
        't_step': sim_t_step,
        'n_robust': 1,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    xq1 = 2000.0
    yq2 = 1000.0
    zq3 = 1000.0
    vxq1 = 2000.0
    vyq2 = 1000.0
    vzq3 = 2000.0
    r1 = 0.001
    r2 = 0.001
    r3 = 0.0001

    lterm = xq1 * (model.x['x_b'] - xtra) ** 2 + yq2 * (model.x['y_b'] - ytra) ** 2 + zq3 * (model.x['z_b'] - z_ref) ** 2 + \
            vxq1 * (model.x['dx_b'] - v_xref) ** 2 + vyq2 * (model.x['dy_b'] - v_yref) ** 2 + vzq3 * (model.x['dz_b'] - v_zref) ** 2 + \
            r1 * (model.u['u_x']) ** 2 + r2 * (model.u['u_y']) ** 2 + r3 * (model.u['u_z']) ** 2

    mterm = xq1 * (model.x['x_b'] - xtra) ** 2 + yq2 * (model.x['y_b'] - ytra) ** 2 + zq3 * (model.x['z_b'] - z_ref) ** 2 + \
            vxq1 * (model.x['dx_b'] - v_xref) ** 2 + vyq2 * (model.x['dy_b'] - v_yref) ** 2 + vzq3 * (model.x['dz_b'] - v_zref) ** 2

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.bounds['lower', '_x', 'x_b'] = -1.5
    mpc.bounds['upper', '_x', 'x_b'] = 1.0

    mpc.bounds['lower', '_x', 'y_b'] = -0.5
    mpc.bounds['upper', '_x', 'y_b'] = 1.5

    mpc.bounds['lower', '_x', 'z_b'] = 0.0
    mpc.bounds['upper', '_x', 'z_b'] = 1.0

    mpc.bounds['lower', '_u', 'u_x'] = -500.0
    mpc.bounds['upper', '_u', 'u_x'] = 500.0

    mpc.bounds['lower', '_u', 'u_y'] = -500.0
    mpc.bounds['upper', '_u', 'u_y'] = 500.0

    mpc.bounds['lower', '_u', 'u_z'] = -500.0
    mpc.bounds['upper', '_u', 'u_z'] = 0.0

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=sim_t_step)

    simulator.setup()

    estimator = do_mpc.estimator.StateFeedback(model)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    simulator.x0 = x0

    mpc.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()
    mpc.reset_history()
    fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
    plt.ion()

    for i in range(20000):
        if flag == 0:
            if index == 0:
                v_xref = dx_ref / z_ref * v_zref
                v_yref = 0.0
                xtra = TraPoint_x[index] + dx_ref
                ytra = 0.0

            elif index == 1:
                v_xref = - (dx_ref / 3) / z_ref * v_zref
                v_yref = - (2 * dy_ref / 3)/ z_ref * v_zref
                xtra = TraPoint_x[index] - dx_ref / 3
                ytra = TraPoint_y[index] + dy_ref / 3
                # break

            elif index == 2:
                v_xref = - (dx_ref / 3) / z_ref * v_zref
                v_yref = (2 * dy_ref / 3) / z_ref * v_zref
                xtra = TraPoint_x[index] - dx_ref / 3
                ytra = TraPoint_y[index] + (2 * dy_ref) / 3

            flag = 1
        
        Force = mpc.make_step(x0)
        y_next = simulator.make_step(Force)
        x0 = estimator.make_step(y_next)

        print("**********************************************************************************************")
        print("x0: ", x0)
        print("Force: ", Force[0, 0], Force[1, 0], Force[2, 0])
        print("xtra, ytra, v_xref, v_yref, v_zref: ",  xtra, ytra, v_xref, v_yref, v_zref)

        if show_animation:
            # mpc_graphics.plot_results(t_ind=k)
            # mpc_graphics.plot_predictions(t_ind=k)
            # mpc_graphics.reset_axes()

            graphics.plot_results(t_ind=i)
            graphics.plot_predictions(t_ind=i)
            graphics.reset_axes()
            plt.show()
            plt.pause(0.01)


    # graphics.plot_predictions(t_ind=0)
    graphics.plot_results()
    graphics.reset_axes()
    plt.show()



def TriCal():
    t = 0.2
    pos_init = np.array([0.88, 0.0, 0.833])
    v_init = np.array([-0.8,  0.0, 0.075])

    pos_tar = np.array([0.4, 0.0, 0.5])
    v_tar = np.array([-7.2, 0.0, -6])

    b_x = np.array([pos_init[0], v_init[0], pos_tar[0], v_tar[0]])
    b_y = np.array([pos_init[1], v_init[1], pos_tar[1], v_tar[1]])
    b_z = np.array([pos_init[2], v_init[2], pos_tar[2], v_tar[2]])

    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, t, t ** 2, t ** 3], [0, 1, 2 * t, 3 * t ** 2]])

    x_coef = np.linalg.solve(A, b_x)
    y_coef = np.linalg.solve(A, b_y)
    z_coef = np.linalg.solve(A, b_z)

    t = np.linspace(0.0, 0.2, 50)
    t_test = 0.171
    x = x_coef[0] + x_coef[1] * t + x_coef[2] * t ** 2 + x_coef[3] * t ** 3
    x_test = x_coef[0] + x_coef[1] * t_test + x_coef[2] * t_test ** 2 + x_coef[3] * t_test ** 3
    vx = x_coef[1] + 2 * x_coef[2] * t + 3 * x_coef[3] * t ** 2
    vx_test = x_coef[1] + 2 * x_coef[2] * t_test + 3 * x_coef[3] * t_test ** 2
    ax = x_coef[2] *  2 + x_coef[3] * t * 6

    z = z_coef[0] + z_coef[1] * t + z_coef[2] * t ** 2 + z_coef[3] * t ** 3
    vz = z_coef[1] + 2 * z_coef[2] * t + 3 * z_coef[3] * t ** 2
    az = z_coef[2] * 2 + z_coef[3] * t * 6
    plt.figure(1)
    plt.plot(t, x)
    plt.plot(t, vx)
    # plt.plot(t, ax)
    plt.title('x')

    plt.figure(2)
    # plt.plot(t, vz)
    plt.plot(t, z)
    # plt.plot(t, az)
    plt.title('z')

    plt.show()

    print("x_coef, y_coef, z_coef: ", x_coef, y_coef, z_coef)
    print("t_test is: ", x_test, vx_test)
    return x_coef, y_coef, z_coef

if __name__ == "__main__":
    # MPCControl()
    TriCal()