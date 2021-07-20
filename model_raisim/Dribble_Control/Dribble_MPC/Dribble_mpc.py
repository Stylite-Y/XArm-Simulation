import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc

def Dribble_mpc(model):

    v_ref = -3
    x_ref = -0.1

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 50,
        't_step': 0.001,
        'n_robust': 1,
        'store_full_solution': True,  
        # 'open_loop': True,  
        # 'state_discretization': 'collocation',
        # 'collocation_type': 'radau',
        # 'collocation_deg': 4,
        # 'collocation_ni': 2,
    }
    mpc.set_param(**setup_mpc)


    q1 = 100
    q2 = 10000
    r = 0.1
    mterm = q1 * (model.x['x_b'] - x_ref) ** 2 + q2 * (model.x['dx_b'] - v_ref) ** 2
    lterm = q1 * (model.x['x_b'] - x_ref) ** 2 + q2 * (model.x['dx_b'] - v_ref) ** 2 + r * model.u['u'] ** 2
    # lterm =  q2 * (model.x['dx_b'] - v_ref) ** 2 + r * model.u['u'] ** 2

    # mterm = q2 * (model.x['dx_b'] - v_ref) ** 2
    # lterm = q1 * (model.x['x_b'] - x_ref) ** 2 + q2 * (model.x['dx_b'] - v_ref) ** 2 + r * mod


    mpc.set_objective(mterm=mterm, lterm=lterm)
    0
    mpc.set_rterm(u=1e-2)

    mpc.bounds['lower', '_x', 'x_b'] = -0.5
    mpc.bounds['upper', '_x', 'x_b'] = 0.5

    mpc.bounds['lower', '_u', 'u'] = -200
    mpc.bounds['upper', '_u', 'u'] = 0

    mpc.setup()

    return mpc
