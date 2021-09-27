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


def template_model():
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Certain parameters
    m0 = 0.6  # kg, mass of the cart
    m1 = 0.2  # kg, mass of the first rod
    m2 = 0.2  # kg, mass of the second rod
    L1 = 0.5  #m, length of the first rod
    L2 = 0.5  #m, length of the second rod
    l1 = L1/2
    l2 = L2/2
    J1 = (m1 * l1**2) / 3   # Inertia
    J2 = (m2 * l2**2) / 3   # Inertia

    # m1 = model.set_variable('_p', 'm1')
    # m2 = model.set_variable('_p', 'm2')

    g = 9.80665 # m/s^2, gravity

    h1 = m0 + m1 + m2
    h2 = m1*l1 + m2*L1
    h3 = m2*l2
    h4 = m1*l1**2 + m2*L1**2 + J1
    h5 = m2*l2*L1
    h6 = m2*l2**2 + J2
    h7 = (m1*l1 + m2*L1) * g
    h8 = m2*l2*g

    # Setpoint x:
    # pos_set = model.set_variable('_tvp', 'pos_set')

    m = 0.4
    # States struct (optimization variables):
    px_b = model.set_variable(var_type='_x', var_name='x_b', shape=(1, 1))
    y_b = model.set_variable(var_type='_x', var_name='y_b', shape=(1, 1))
    dx_b = model.set_variable(var_type='_x', var_name='dx_b', shape=(1, 1))
    dy_b = model.set_variable(var_type='_x', var_name='dy_b', shape=(1, 1))
    ux = model.set_variable(var_type='_u', var_name='ux', shape=(1, 1))
    uy = model.set_variable(var_type='_u', var_name='uy', shape=(1, 1))

    xtraj = model.set_variable(var_type='_tvp', var_name='xtraj')
    ytraj = model.set_variable(var_type='_tvp', var_name='ytraj')

    # Differential equations
    model.set_rhs('x_b', dx_b)
    model.set_rhs('y_b', dy_b)

    model.set_rhs('dx_b', ux / m)
    model.set_rhs('dy_b', uy / m)

    model.set_expression('tvp1', xtraj)
    model.set_expression('tvp2', ytraj)

    # Build the model
    model.setup()

    return model
