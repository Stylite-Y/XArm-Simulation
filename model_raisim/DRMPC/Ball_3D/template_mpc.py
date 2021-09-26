#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
import math

def xytri(t):
    r = 1.0
    T = 1
    omga = 2 * math.pi / T
    x = r * sin(omga * t)
    y = r * cos(omga * t)
    # print(x, y)
    return x, y


def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 100,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.04,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 3,
        'collocation_ni': 1,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'ma27'}
    }

    mpc.set_param(**setup_mpc)

    # mterm = 100*(model.aux['E_kin'] - model.aux['E_pot'])
    # lterm = (model.aux['E_kin'] - model.aux['E_pot'])+10*(model.x['pos']-model.tvp['pos_set'])**2 # stage cost

    q1 = 1000
    q2 = 1000
    r = 0.001
    lterm = q1 * (model.x['x_b'] - model.tvp['xtraj']) ** 2 + q2 * (model.x['y_b'] - model.tvp['ytraj']) ** 2
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

    tvp_template = mpc.get_tvp_template()

    # When to switch setpoint:
    t_switch = 4    # seconds
    ind_switch = t_switch // setup_mpc['t_step']

    def tvp_fun(t_ind):
        ind = t_ind // setup_mpc['t_step']
        # if ind <= ind_switch:
        #     tvp_template['_tvp',:, 'pos_set'] = -0.8
        # else:
            # tvp_template['_tvp',:, 'pos_set'] = 0.8
        for k in range(setup_mpc['n_horizon'] + 1):
            t_pre = t_ind + k * setup_mpc['t_step']
            # tvp_template['_tvp',k, 'xtraj'] = -0.8
            tvp_template['_tvp', k, 'xtraj'], tvp_template['_tvp', k, 'ytraj'] = xytri(t_pre)
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc
