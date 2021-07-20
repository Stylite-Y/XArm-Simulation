import numpy as np
import math
import sys
import do_mpc
from casadi import *  # symbolic library CasADi

from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

"""
dynamics model:
x1_dot = x2
x2_dot = -g - F/m
"""

def logistic(x):
    return 0.01*np.exp(x*700)/(1+0.01*(np.exp(700*x)-1))

def tanh_sig(x):
    return 0.5 + 0.5 * np.tanh(1000 * x)

v_ref = -3
x_ref = -0.1
x_reb = -0.47

g = 10
m = 0.5     # kg, mass of the ball
k_con = 100000
c_con = 5

# set variable of the dynamics system
x_b = model.set_variable(var_type='_x', var_name='x_b', shape=(1, 1))
# x2 = model.set_variable(var_type='_x', var_name='theta_dot', shape=(1, 1))
dx_b = model.set_variable(var_type='_x', var_name='dx_b', shape=(1, 1))
u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))
# x_reb = model.set_variable(var_type='_p', var_name='x_reb', shape=(1, 1))
# x_ref = model.set_variable(var_type='_p', var_name='x_ref', shape=(1, 1))

# rhs
model.set_rhs('x_b', dx_b)

# dx_b_next = vertcat(
#     -g + u / m
# )

# print(type(x_b))
# dx_b_next = vertcat(
#     -g + u / m * (1 if (x_b > x_ref) else 0) + (k_con * (x_b - x_reb) + c_con * dx_b) * (1 if (x_b < x_reb) else 0)
# )

dx_b_next = vertcat(
    -g + u / m * tanh_sig(x_b-x_ref) + (k_con * (-x_b + x_reb)) * tanh_sig(x_reb-x_b)
)

# dx_b_next = vertcat(
#     -g + u / m * logistic(x_b-x_ref) + (k_con * (-x_b + x_reb)) * logistic(x_reb-x_b)
# )

model.set_rhs('dx_b', dx_b_next)

model.setup()

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
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)


q1 = 100
q2 = 10000
r = 0.1
mterm = q1 * (x_b - x_ref) ** 2 + q2 * (dx_b - v_ref) ** 2
lterm = q1 * (x_b - x_ref) ** 2 + q2 * (dx_b - v_ref) ** 2 + r * u ** 2

# mterm = q2 * (dx_b - v_ref) ** 2
# lterm = q2 * (dx_b - v_ref) ** 2 + r * u ** 2


mpc.set_objective(mterm=mterm, lterm=lterm)
0
mpc.set_rterm(u=1e-2)

mpc.bounds['lower', '_x', 'x_b'] = -0.5
mpc.bounds['upper', '_x', 'x_b'] = 0.5

mpc.bounds['lower', '_u', 'u'] = -500
mpc.bounds['upper', '_u', 'u'] = 0
# m_ = 1 * 1e-4 * np.array([0.5, 0.4, 0.6])

# ref_pos_1 = np.array([-0.1])
# ref_pos_2 = np.array([-0.47])

# mpc.set_uncertainty_values(
#     x_ref = ref_pos_1,
#     x_reb = ref_pos_2
# )

mpc.setup()

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

estimator = do_mpc.estimator.StateFeedback(model)

x0 = np.array([0, 0.8]).reshape(-1, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
simulator.x0 = x0

mpc.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

# =======================================================
# visualization

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import math 

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

mpc.reset_history()

j = 0
n_step = 3000
for k in range(n_step):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    # if show_animation:

    #     graphics.plot_results(t_ind=k)
    #     graphics.plot_predictions(t_ind=k)
    #     graphics.reset_axes()
    #     plt.show()
    #     plt.pause(0.01)


    # if x0[0, 0] < x_ref and x0[1, 0] < 0:
    #     # free flight time and height 
    #     i = k
    #     v_temp = x0[1, 0]
    #     x_temp = x0[0, 0]
    #     # t_k = k * setup_mpc['t_step']
    #     # v0 = mpc.data['_x', 'dx_b'][k - 2, 0]
    #     # x_0 = mpc.data['_x', 'x_b'][k - 2, 0]
    #     t1 = (2 * v_ref + math.sqrt(4 * v_ref ** 2 + 4 * g + 8 * g * x_ref)) / (2 * g)
    #     t1_sample = np.linspace(0, t1, int(t1/0.005))
    #     s1_sample = x_ref + v_ref * t1_sample - 0.5 * g * t1_sample ** 2
    #     v1 = v_ref - g * t1
    #     v2 = - 0.85 * v1
    #     t2 = (2 * v2 - math.sqrt(4 * v2 ** 2 - 4 * g - 8 * g * x_ref)) / (2 * g)
    #     t2_sample = np.linspace(t1, t1 + t2, int(t2/0.005))
    #     t2_temp = np.linspace(0, t2, int(t2/0.005))
    #     s2_sample = -0.5 + v2 * t2_temp - 0.5 * g * t2_temp ** 2
    #     v3 = v2 - g * t2
    #     t_free = t1 + t2
    #     # print(k)
    #     # k = k + int(t_free / setup_mpc['t_step'])
    #     # print("===========controller mpc position data===========")
    #     # print(k)
    #     # print(t_free)
    #     # print(v3)
    #     x0 = np.array([x_ref, v3]).reshape(-1, 1)
    #     simulator.x0 = x0
    #     mpc.x0 = x0 
    #     estimator.x0 = x0

    # #     # k = k + int(t_free / setup_mpc['t_step'])

    # elif x0[0, 0] >= x_ref:
    #     u0 = mpc.make_step(x0)
    #     y_next = simulator.make_step(u0)
    #     # if k == 75:
    #     #     m = x0
    #     #     n = u0
    #     # if k == 76:
    #     #     m2 = x0
    #     #     n2 = u0
    #     x0 = estimator.make_step(y_next)
        # if k == 75:
            # h = x0
        
        # if k == 76:
        #     h2 = x0

        # if show_animation:
        #     graphics.plot_results(t_ind=k)
        #     graphics.plot_predictions(t_ind=k)
        #     graphics.reset_axes()
        #     plt.show()
        #     plt.pause(0.01)

graphics.plot_predictions(t_ind=0)
graphics.plot_results()
graphics.reset_axes()
plt.show()

# print("===========controller mpc position data===========")
# print(mpc.data['_x', 'x_b'])
# print("===========controller mpc force data===========")
# print(mpc.data['_u', 'u'])
print("===========controller mpc veco data===========")
print(mpc.data['_x', 'dx_b'][1500:2000, 0])



input('Press any key to exit.')

store_results = True  

# if store_results:
#     do_mpc.data.save_results([mpc, simulator], 'BALL_robust_MPC_5-r0.1-u50')

# def update(t_ind):
#     print('Writing frame: {}.'.format(t_ind), end='\r')
#     graphics.plot_results(t_ind=t_ind)
#     graphics.plot_predictions(t_ind=t_ind)
#     graphics.reset_axes()
    
# n_steps = mpc.data['_time'].shape[0]


# anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

# gif_writer = ImageMagickWriter(fps=5)
# anim.save('anim_CSTR.gif', writer=gif_writer)