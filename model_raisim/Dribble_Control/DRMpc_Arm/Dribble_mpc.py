import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc

from Dribble_model import tanh_sig

# =======parameter==========
k_vir = 560
f1 = 25
f2 = 150

v_ref = -15
x_ref = -0.1

# ======mpc function========
def F_TRA(x, v):
    u1 = - k_vir * (x - x_ref) - f1
    u2 = - k_vir * (x - x_ref) - f2

    # u_ref = (u1 * (0.5 + 0.5 * np.tanh(1000 * v)) + u2 * (0.5 + 0.5 * np.tanh(1000 * (- v)))) * (0.5 + 0.5 * np.tanh(1000 * (x - x_ref)))
    u_ref = (u1 * tanh_sig(v) + u2 * tanh_sig(-v))

    return u_ref

def Dribble_mpc(model):

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
    q2 = 1000
    r = 10
    u_tra = F_TRA(model.x['x_b'], model.x['dx_b'])

    mterm = q2 * (model.x['dx_b'] - v_ref) ** 2
    lterm = r * (model.u['u'] - u_tra) ** 2
    # lterm =  q2 * (model.x['dx_b'] - v_ref) ** 2 + r * model.u['u'] ** 2

    # mterm = q2 * (model.x['dx_b'] - v_ref) ** 2
    # lterm = q1 * (model.x['x_b'] - x_ref) ** 2 + q2 * (model.x['dx_b'] - v_ref) ** 2 + r * mod


    mpc.set_objective(mterm=mterm, lterm=lterm)
    
    mpc.set_rterm(u=1e-2)

    mpc.bounds['lower', '_x', 'x_b'] = -0.5
    mpc.bounds['upper', '_x', 'x_b'] = 0.5

    mpc.bounds['lower', '_u', 'u'] = -200
    mpc.bounds['upper', '_u', 'u'] = 0

    mpc.setup()

    return mpc
