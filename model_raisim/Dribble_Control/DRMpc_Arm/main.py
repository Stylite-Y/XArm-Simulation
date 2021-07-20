import math
import pickle
import sys
import time

import do_mpc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
from casadi.tools import *
from do_mpc.tools.timer import Timer
from matplotlib import cm

sys.path.append('./')

from Dribble_model import Dribble_model
from Dribble_mpc import Dribble_mpc
from Dribble_simulator import Dribble_simulator

"""
dynamics model:
x1_dot = x2
x2_dot = -g - F/m
"""

def Dribble_MpcController():
    show_animation = True
    store_results = False

    model = Dribble_model()
    mpc = Dribble_mpc(model)
    simulator = Dribble_simulator(model)

    estimator = do_mpc.estimator.StateFeedback(model)

    x0 = np.array([-0.1, 10]).reshape(-1, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    simulator.x0 = x0

    mpc.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()

    # =======================================================
    # visualization

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

if __name__ == "__main__":
    Dribble_MpcController() 
