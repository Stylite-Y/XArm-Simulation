import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
# sys.path.append('../../')
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
    x_ref = -0.1
    x_reb = -0.47

    g = 10
    m = 0.5     # kg, mass of the ball
    k_con = 100000
    c_con = 5

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # set variable of the dynamics system
    x_b = model.set_variable(var_type='_x', var_name='x_b', shape=(1, 1))
    # x2 = model.set_variable(var_type='_x', var_name='theta_dot', shape=(1, 1))
    dx_b = model.set_variable(var_type='_x', var_name='dx_b', shape=(1, 1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))
    # x_reb = model.set_variable(var_type='_p', var_name='x_reb', shape=(1, 1))
    # x_ref = model.set_variable(var_type='_p', var_name='x_ref', shape=(1, 1))

    # rhs
    model.set_rhs('x_b', dx_b)

    dx_b_next = vertcat(
        -g + u / m * tanh_sig(x_b-x_ref) + (k_con * (-x_b + x_reb)) * tanh_sig(x_reb-x_b)
    )

    # dx_b_next = vertcat(
    #     -g + u / m * logistic(x_b-x_ref) + (k_con * (-x_b + x_reb)) * logistic(x_reb-x_b)
    # )

    model.set_rhs('dx_b', dx_b_next)

    model.setup()

    return model
