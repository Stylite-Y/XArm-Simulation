import numpy as np
import sys
import do_mpc
from casadi import *  # symbolic library CasADi

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
# x2 = model.set_variable(var_type='_x', var_name='theta_dot', shape=(1, 1))
dx_b = model.set_variable(var_type='_x', var_name='dx_b', shape=(1, 1))
u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))

# rhs
model.set_rhs('x_b', dx_b)

# dx_b_next = vertcat(
#     -g - u / m
# )

model.set_rhs('dx_b', -g + u / m)
# model.set_rhs('dx_b', u)

model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 50,
    't_step': 0.03,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

r = 0.01
# mterm = (x_b - -1) ** 2 + dx_b ** 2
# lterm = (x_b - -1) ** 2 + dx_b ** 2 + r * u ** 2
# mterm = x_b ** 2 + dx_b ** 2
# lterm = x_b ** 2 + dx_b ** 2 + r * u ** 2
mterm = (dx_b - v_ref) ** 2
lterm = (dx_b - v_ref) ** 2 + r * u ** 2


mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(u=1e-4)

mpc.bounds['lower', '_x', 'x_b'] = -1000
mpc.bounds['upper', '_x', 'x_b'] = 1000

mpc.bounds['lower', '_u', 'u'] = -1000
mpc.bounds['upper', '_u', 'u'] = 1000

m_ = 1 * 1e-4 * np.array([0.5, 0.4, 0.6])

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=0.03)

# p_template = simulator.get_p_template()

# def p_fun(t_now):
#     # p_template['l'] = 1
#     # p_template['m'] = 1
#     return p_template

# simulator.set_p_fun(p_fun)

simulator.setup()


estimator = do_mpc.estimator.StateFeedback(model)

x0 = np.array([1, 30]).reshape(-1, 1)
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

mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

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
    # if x0[0, 0] <= x_ref:
    #     u0 = mpc.make_step(x0)
    #     y_next = simulator.make_step(u0)
    #     x0 = estimator.make_step(y_next)

    # elif x0[0, 0] > x_ref and x0[1, 0] > 0:
    #     # k = k + int(2 / setup_mpc['t_step'])
    #     # j = k
    #     x0 = np.array([1, -5]).reshape(-1, 1)
    #     simulator.x0 = x0
    #     mpc.x0 = x0 
    #     estimator.x0 = x0
    #     # t0 = k * setup_mpc['t_step']
    #     # t1 = np.linspace(t0, t0 + 1, int(1 / setup_mpc['t_step']))
    #     # v1 = x0[1, 0] - 16 * (t1 - t0)
    #     # s1 = x0[1, 0] * (t1 - t0) - 0.5 * 16 * (t1 - t0) ** 2
    #     # t2 = np.linspace(t0, t0 + 2, int(1 / setup_mpc['t_step']))
    #     # v2 = 16 * (t2 - 1 - t0)
    #     # s2 = 0.5 * 16 * (t2 - 1 - t0) ** 2

    current_time_data = mpc.data['_time']
    current_time_mpc = mpc.t0
    # print("current_time_data: ", current_time_data)
    print("t_ind: ", t_ind)
    # print("current_time_opt: ", mpc.data['t_wall_S'])
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


# graphics.plot_predictions(t_ind=0)
graphics.plot_results()
graphics.reset_axes()
plt.show()

# print("===========initial mpc data===========")
# print(j)

print("===========controller mpc position data===========")
print("current_time_opt: ", mpc.data['t_wall_S'])
print("current_time_data: ", mpc.data['_time'])
print(mpc.data['_x', 'x_b'])
print("===========controller mpc force data===========")
# print(mpc.data['_u', 'u'])
print("===========controller mpc veco data===========")
print(mpc.data['_x', 'dx_b'])
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