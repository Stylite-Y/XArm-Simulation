import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import do_mpc

def logistic(x):
    return 0.01*np.exp(x*700)/(1+0.01*(np.exp(700*x)-1))

def tanh_sig(x):
    return 0.5 + 0.5 * np.tanh(1000 * x)

def Dribble_model():
    """
    dynamics model:
    x1_dot = x2
    x2_dot = -g - F/m
    """

    # v_ref = -5
    z_ref = 0.5
    z_reb = 0.15

    g = 9.8
    m = 0.4     # kg, mass of the ball
    K_xd = 500
    K_zd = 300
    K_zvup = 5
    K_zvdown = 15
    K_con = 1000

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # set variable of the dynamics system
    x_b = model.set_variable(var_type='_x', var_name='x_b', shape=(1, 1))
    y_b = model.set_variable(var_type='_x', var_name='y_b', shape=(1, 1))
    z_b = model.set_variable(var_type='_x', var_name='z_b', shape=(1, 1))

    # x2 = model.set_variable(var_type='_x', var_name='theta_dot', shape=(1, 1))
    dx_b = model.set_variable(var_type='_x', var_name='dx_b', shape=(1, 1))
    dy_b = model.set_variable(var_type='_x', var_name='dy_b', shape=(1, 1))
    dz_b = model.set_variable(var_type='_x', var_name='dz_b', shape=(1, 1))
    u_x = model.set_variable(var_type='_u', var_name='u_x', shape=(1, 1))
    u_y = model.set_variable(var_type='_u', var_name='u_y', shape=(1, 1))
    u_z = model.set_variable(var_type='_u', var_name='u_z', shape=(1, 1))

    # time vary reference
    # xtraj = model.set_variable(var_type='_tvp', var_name='xtraj')
    # ztraj = model.set_variable(var_type='_tvp', var_name='ztraj')
    # ytraj = model.set_variable(var_type='_tvp', var_name='ytraj')
    # TrajIndex = model.set_variable(var_type='_tvp', var_name='TrajIndex')

    # x_reb = model.set_variable(var_type='_p', var_name='x_reb', shape=(1, 1))
    # x_ref = model.set_variable(var_type='_p', var_name='x_ref', shape=(1, 1))

    # rhs
    model.set_rhs('x_b', dx_b)
    model.set_rhs('y_b', dy_b)
    model.set_rhs('z_b', dz_b)

    # dx_b_next = vertcat(
    #     u_x / m * tanh_sig(z_b - z_ref),
    #     u_y / m * tanh_sig(z_b - z_ref),
    #     -g + u_z / m * tanh_sig(z_b - z_ref),
    #     # -g + u_z / m * tanh_sig(z_b - z_ref) + (K_con * (-z_b + z_reb)) * tanh_sig(z_reb - z_b),

    #     # u_x / m,
    #     # u_y / m,
    #     # -g + u_z / m,
    # )

    dx_b_next = vertcat(
        u_x / m * tanh_sig(z_b - z_ref),
    )
    dy_b_next = vertcat(
        u_y / m * tanh_sig(z_b - z_ref),
    )
    dz_b_next = vertcat(
        -g + u_z / m * tanh_sig(z_b - z_ref),
    )
    model.set_rhs('dx_b', dx_b_next)
    model.set_rhs('dy_b', dy_b_next)
    model.set_rhs('dz_b', dz_b_next)

    model.setup()

    return model
