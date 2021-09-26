import numpy as np
import sys
import do_mpc
from casadi import *  # symbolic library CasADi
import datetime
import math

def xytri(t):
    r = 1.0
    T = 1
    omga = 2 * math.pi / T
    x = r * sin(omga * t)
    y = r * cos(omga * t)
    # print(x, y)
    return x, y

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

"""
dynamics model:
x1_dot = x2
x2_dot = u
"""

v_ref = -16
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
model.set_expression('tvp', xtraj)
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
lterm = q1 * (model.x['x_b'] - model.tvp['xtraj']) ** 2
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

# m_ = 1 * 1e-4 * np.array([0.5, 0.4, 0.6])
# mpc.set_rterm(u=1e-2)
# xtra = [0.0]
# ytra = [0.0]
# for i in range(5000):
#     t_pre = i * setup_mpc['t_step']
#     x, y = xytri(t_pre)
#     xtra.append(x)
#     ytra.append(y)
horizon = setup_mpc['n_horizon']
tvp_template = mpc.get_tvp_template()
def tvp_fun(t_ind):
    print(horizon)
    print(1)
    for k in range(horizon + 1):
        # t_pre = t_ind + k * setup_mpc['t_step']
        # print(t_pre)
        # if k == 0:
        #     print("tvp: ", tvp_template['_tvp'])
        # tvp_template['_tvp', k, 'xtraj'], tvp_template['_tvp', k, 'ytraj'] = xytri(t_pre)
        tvp_template['_tvp',k, 'xtraj'] = 0.4
        # tvp_template['_tvp', k, 'ytraj'] = 0.3
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

# mpc.reset_history()

x0 = np.array([0.0, 1.0, 0.0, 0.0]).reshape(-1, 1)
simulator.x0 = x0

mpc.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

# =======================================================
# visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

show_animation = True

# mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
# sim_graphics = do_mpc.graphics.Graphics(simulator.data)

# fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
# fig.align_ylabels()

# mpc_graphics.add_line(var_type='_x', var_name='x_b', axis=ax[0])
# mpc_graphics.add_line(var_type='_x', var_name='dx_b', axis=ax[0])
# mpc_graphics.add_line(var_type='_u', var_name='u', axis=ax[1])

# ax[0].set_ylabel('height of ball(m)')
# ax[1].set_ylabel('Force(N)')
# ax[1].set_xlabel('time(s)')

fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
plt.ion()

# print("===========initial mpc data===========")
# print(mpc.data['_x', 'x_b'][0, 0])s
# print(mpc.data['_x', 'x_b'])
mpc.reset_history()

n_step = 100
for k in range(n_step):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    if show_animation:
        # mpc_graphics.plot_results(t_ind=k)
        # mpc_graphics.plot_predictions(t_ind=k)
        # mpc_graphics.reset_axes()

        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True
# graphics.plot_predictions(t_ind=0)
graphics.plot_results()
graphics.reset_axes()
plt.show()

# print("===========initial mpc data===========")
# print(j)

print("===========controller mpc position data===========")
# print("current_time_opt: ", mpc.data['t_wall_S'])
# print("current_time_data: ", mpc.data['_time'])
# print(mpc.data['_x', 'x_b'])
print("===========controller mpc force data===========")
# print(mpc.data['_u', 'u'])
print("===========controller mpc veco data===========")
# print(mpc.data['_x', 'dx_b'])
# v = mpc.data['_x', 'dx_b']
# print(v[1, 0])
# print(type(v))
# _x = np.ones((1, 2))
# do_mpc.data.MPCData.update('_x': _x)
# np.concatenate((mpc.data['_x', 'dx_b'], [[16]]))
# print(mpc.data['_x', 'dx_b'])
# print(setup_mpc['t_step'])

input('Press any key to exit.')

# Store results:
# if store_results:
#     do_mpc.data.save_results([mpc, simulator], 'mk')
