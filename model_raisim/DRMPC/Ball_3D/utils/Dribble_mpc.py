from re import T
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
import math
import datetime

from Dribble_model import tanh_sig

# =======parameter==========
# k_vir = 560
# f1 = 25
# f2 = 150

# v_ref = -15
# x_ref = -0.2
# z_ref = 0.15
# v_zref = -6

# theta = np.linspace(0, 2*math.pi, 50)


# ======mpc function========
# def F_TRA(k_vir, x, v, v_ref, f1, f2):
#     u1 = - k_vir * (x - 0.5) - f1 
#     u2 = - k_vir * (x - 0.5) - f2 * (v - v_ref)
#     # u_ref = (u1 * (0.5 + 0.5 * np.tanh(1000 * v)) + u2 * (0.5 + 0.5 * np.tanh(1000 * (- v)))) * (0.5 + 0.5 * np.tanh(1000 * (x - x_ref)))
#     u_ref = (u1 * tanh_sig(v) + u2 * tanh_sig(-v))

#     return u_ref

# def X_TRA(xtra):
#     return xtra

# def Y_TRA(x):
#     theta = np.arcsin(x)
#     yref = np.cos(theta)
#     return yref

# def Z_TRA(xtra, ztra):
#     z_reftra = np.cos()
#     return z_reftra

def Dribble_mpc(model, sim_t_step, x_coef, y_coef, z_coef):

    mpc = do_mpc.controller.MPC(model)

    starttime = datetime.datetime.now()
    setup_mpc = {
        'n_horizon': 150,
        't_step': sim_t_step,
        # 'n_robust': 1,
        # 'store_full_solution': True,  
        # 'open_loop': True,  
        # 'state_discretization': 'collocation',
        # 'collocation_type': 'radau',
        # 'collocation_deg': 4,
        # 'collocation_ni': 2,
    }
    mpc.set_param(**setup_mpc)
    setp_endtime = datetime.datetime.now()

    xq1 = 50.0
    yq2 = 50.0
    zq3 = 50.0
    vxq1 = 150.0
    vyq2 = 100.0
    vzq3 = 100.0
    r1 = 1.0
    r2 = 1.0
    r3 = 1.0
#     v_zref = -6
    z_ref = 0.5

    # x_reftra = X_TRA(xtra)

    # z_reftra = Z_TRA(xtra, ztra)

    mterm = xq1 * (model.x['x_b'] - model.tvp['xtraj']) ** 2 + yq2 * (model.x['y_b'] - model.tvp['ytraj']) ** 2 + zq3 * (model.x['z_b'] - z_ref) ** 2
    lterm = xq1 * (model.x['x_b'] - model.tvp['xtraj']) ** 2 + yq2 * (model.x['y_b'] - model.tvp['ytraj']) ** 2 + zq3 * (model.x['z_b'] - z_ref) ** 2 \
             + r1 * (model.u['u_x']) ** 2 + r2 * (model.u['u_y']) ** 2 + r3 * (model.u['u_z']) ** 2


    # mterm = q1 * (model.x['x_b'] - model.tvp['xtraj']) ** 2 + q2 * (model.x['z_b'] - model.tvp['ztraj']) ** 2
    # lterm = q1 * (model.x['x_b'] - model.tvp['xtraj']) ** 2 + q2 * (model.x['z_b'] - model.tvp['ztraj']) ** 2
    #  + r1 * (model.u['u_x']) ** 2 + + r2 * (model.u['u_z']) ** 2


    # mterm = q2 * (model.x['z_b']) ** 2
    # lterm = q2 * (model.x['z_b']) ** 2
    # v_b = np.array([v_xref, v_yref, v_zref])
    setob_stime = datetime.datetime.now()
    # lterm = xq1 * (model.x['x_b'] - xtra) ** 2 + yq2 * (model.x['y_b'] - ytra) ** 2 + zq3 * (model.x['z_b'] - z_ref) ** 2 + \
    #         vxq1 * (model.x['dx_b'] - v_xref) ** 2 + vyq2 * (model.x['dy_b'] - v_yref) ** 2 + vzq3 * (model.x['dz_b'] - v_zref) ** 2 + \
    #         r1 * (model.u['u_x']) ** 2 + r2 * (model.u['u_y']) ** 2 + r3 * (model.u['u_z']) ** 2

    # mterm = xq1 * (model.x['x_b'] - xtra) ** 2 + yq2 * (model.x['y_b'] - ytra) ** 2 + zq3 * (model.x['z_b'] - z_ref) ** 2 + \
    # #         vxq1 * (model.x['dx_b'] - v_xref) ** 2 + vyq2 * (model.x['dy_b'] - v_yref) ** 2 + vzq3 * (model.x['dz_b'] - v_zref) ** 2 
    # lterm = vxq1 * (model.x['dx_b'] - v_xref) ** 2 + vyq2 * (model.x['dy_b'] - v_yref) ** 2 + vzq3 * (model.x['dz_b'] - v_zref) ** 2 + \
    #         r1 * (model.u['u_x']) ** 2 + r2 * (model.u['u_y']) ** 2 + r3 * (model.u['u_z']) ** 2

    # mterm = vxq1 * (model.x['dx_b'] - v_xref) ** 2 + vyq2 * (model.x['dy_b'] - v_yref) ** 2 + vzq3 * (model.x['dz_b'] - v_zref) ** 2

    mpc.set_objective(mterm=mterm, lterm=lterm)
#     setob_endtime = datetime.datetime.now()
    
    # mpc.set_rterm(u_x=1e-2, u_y=1e-2, u_z=1e-2)
#     setbound_stime = datetime.datetime.now()

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

    setbound_etime = datetime.datetime.now()

#     print("params set time: ", setp_endtime - starttime)
#     print("objective set time: ", setob_endtime - setob_stime)
#     print("bound set time: ", setbound_etime - setbound_stime)

    tvp_template = mpc.get_tvp_template()
    # Period = 0.5
    # N = Period / setup_mpc['t_step']

    def tvp_fun(t_ind):
        # ind = t_ind // setup_mpc['t_step']
        # tvp_template['_tvp',:, 'xtraj'] = xtra + ind * 2 * math.pi / N
        # tvp_template['_tvp',:, 'ztraj'] = 0.5 + 0.35 * np.cos(xtra)

        tvp_template['_tvp',:, 'xtraj'] = x_coef[0] + x_coef[1] * t_ind + x_coef[2] * t_ind ** 2 + x_coef[3] * t_ind ** 3
        tvp_template['_tvp',:, 'ytraj'] = x_coef[0] + y_coef[1] * t_ind + y_coef[2] * t_ind ** 2 + y_coef[3] * t_ind ** 3
        tvp_template['_tvp',:, 'ztraj'] = x_coef[0] + z_coef[1] * t_ind + z_coef[2] * t_ind ** 2 + z_coef[3] * t_ind ** 3

        return tvp_template
            
    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()
    setuptime = datetime.datetime.now()
    print("setup time: ", setuptime -setbound_etime)

    return mpc
