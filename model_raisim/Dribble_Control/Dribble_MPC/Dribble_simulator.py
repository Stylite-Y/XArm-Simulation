import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
# sys.path.append('../../')
import do_mpc


def Dribble_simulator(model):
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        # 'integration_tool': 'cvodes',
        # 'abstol': 1e-10,
        # 'reltol': 1e-10,
        't_step': 0.001
    }

    simulator.set_param(**params_simulator)

    # p_template = simulator.get_p_template()

    # def p_fun(t_now):
    #     p_template['x_ref'] = -0.1
    #     p_template['x_reb'] = -0.47
    #     return p_template

    # simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator
