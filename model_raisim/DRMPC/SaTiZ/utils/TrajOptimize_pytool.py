import numpy as np 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Dynamics, Problem, Guess, Condition

class Ball:
    def __init__(self, x_init, v_init, x_des, v_des):
        self.mass = 0.4
        self.g = -9.81
        self.x_init = x_init
        self.v_init = v_init
        self.x_des = x_des
        self.v_des = v_des
        self.x_lowerbound = [-0.8, -0.8, 0.5]
        self.u_lowerbound = [-500.0, -500.0, -500.0]
        self.x_upperbound = [0.8, 0.8, 0.7]
        self.u_upperbound = [500.0, 500.0, 100.0]

    
def dynamics(prob, obj, section):
    x = prob.states(0, section)
    y = prob.states(1, section)
    z = prob.states(2, section)
    vx = prob.states(3, section)
    vy = prob.states(4, section)
    vz = prob.states(5, section)

    ux = prob.controls(0, section)
    uy = prob.controls(1, section)
    uz = prob.controls(2, section)

    dx = Dynamics(prob, section)
    dx[0] = vx
    dx[1] = vy
    dx[2] = vz
    dx[3] = ux / obj.mass
    dx[4] = uy / obj.mass
    dx[5] = uz / obj.mass + obj.g

    return dx()

def equality(prob, obj):
    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    z = prob.states_all_section(2)
    vx = prob.states_all_section(3)
    vy = prob.states_all_section(4)
    vz = prob.states_all_section(5)

    result = Condition()

    result.equal(x[0], obj.x_init[0])
    result.equal(y[0], obj.x_init[1])
    result.equal(z[0], obj.x_init[2])
    result.equal(vx[0], obj.v_init[0])
    result.equal(vy[0], obj.v_init[1])
    result.equal(vz[0], obj.v_init[2])

    result.equal(x[-1], obj.x_des[0])
    result.equal(y[-1], obj.x_des[1])
    result.equal(z[-1], obj.x_des[2])
    result.equal(vx[-1], obj.v_des[0])
    result.equal(vy[-1], obj.v_des[1])
    result.equal(vz[-1], obj.v_des[2])
    
    return result()

def inequality(prob, obj):
    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    z = prob.states_all_section(2)

    ux = prob.controls_all_section(0)
    uy = prob.controls_all_section(1)
    uz = prob.controls_all_section(2)

    result = Condition()

    result.lower_bound(x, obj.x_lowerbound[0])
    result.lower_bound(y, obj.x_lowerbound[1])
    result.lower_bound(z, obj.x_lowerbound[2])
    result.lower_bound(ux, obj.u_lowerbound[0])
    result.lower_bound(uy, obj.u_lowerbound[1])
    result.lower_bound(uz, obj.u_lowerbound[2])

    result.upper_bound(x, obj.x_upperbound[0])
    result.upper_bound(y, obj.x_upperbound[1])
    result.upper_bound(z, obj.x_upperbound[2])
    # result.upper_bound(ux, obj.u_upperbound[0])
    # result.upper_bound(uy, obj.u_upperbound[1])
    # result.upper_bound(uz, obj.u_upperbound[2])
    result.upper_bound(ux, 500.0)
    result.upper_bound(uy, 500.0)
    result.upper_bound(uz, 0.0)
    

    return result()

def cost(prob, obj):
    # ux = prob.controls_all_section(0)
    # uy = prob.controls_all_section(1)
    # uz = prob.controls_all_section(2)

    # return ux ** 2 + uy ** 2 + uz ** 2
    tf = prob.time_final(-1)
    return tf

def running_cost(prob, obj):
    ux = prob.controls_all_section(0)
    uy = prob.controls_all_section(1)
    uz = prob.controls_all_section(2)

    return ux ** 2 + uy ** 2 + uz ** 2

# ========================
def TrajOptim():
    time_init = [0.0, 0.2]
    n = [40]
    num_states = [6]
    num_controls = [3]
    max_iteration = 10

    #  [0.53, 0., 0.50038345] [4., 0., 5.00578796] [0.4 0., 0.5] [-10.28571429, 0. , -6.]
    x_init = [0.53, 0.0, 0.5]
    x_des = [0.4, 0.0, 0.5]
    v_init = [4.0, 0.0, 5.0]
    v_des = [-10.39, 0.0, -6.0]

    # ------------------------
    # set OpenGoddard class for algorithm determination
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    obj = Ball(x_init, v_init, x_des, v_des)

    # ========================
    # Initial parameter guess
    # x_init_guess = Guess.linear(prob.time_all_section, obj.x_init[0], obj.x_des[0])
    # y_init_guess = Guess.linear(prob.time_all_section, obj.x_init[1], obj.x_des[1])
    # z_init_guess = Guess.linear(prob.time_all_section, obj.x_init[2], obj.x_des[2])
    x_init_guess = Guess.cubic(prob.time_all_section, obj.x_init[0], 0.0, obj.x_des[0], 0.0)
    y_init_guess = Guess.cubic(prob.time_all_section, obj.x_init[1], 0.0, obj.x_des[1], 0.0)
    z_init_guess = Guess.cubic(prob.time_all_section, obj.x_init[2], 0.0, obj.x_des[2], 0.0)

    vx_init_guess = Guess.linear(prob.time_all_section, obj.v_init[0], obj.v_des[0])
    vy_init_guess = Guess.linear(prob.time_all_section, obj.v_init[1], obj.v_des[1])
    vz_init_guess = Guess.linear(prob.time_all_section, obj.v_init[2], obj.v_des[2])

    # ux_init_guess = Guess.linear(prob.time_all_section, obj.u_upperbound[0], obj.u_upperbound[0])
    # uy_init_guess = Guess.linear(prob.time_all_section, obj.u_upperbound[1], obj.u_upperbound[1])
    # uz_init_guess = Guess.linear(prob.time_all_section, obj.u_upperbound[2], obj.u_upperbound[2])

    ux_init_guess = Guess.cubic(prob.time_all_section, obj.u_upperbound[0], 0.0, 0.0, 0.0)
    uy_init_guess = Guess.cubic(prob.time_all_section, obj.u_upperbound[1], 0.0, 0.0, 0.0)
    uz_init_guess = Guess.cubic(prob.time_all_section, obj.u_upperbound[2], 0.0, 0.0, 0.0)

    prob.set_states_all_section(0, x_init_guess)
    prob.set_states_all_section(1, y_init_guess)
    prob.set_states_all_section(2, z_init_guess)
    prob.set_states_all_section(3, vx_init_guess)
    prob.set_states_all_section(4, vy_init_guess)
    prob.set_states_all_section(5, vz_init_guess)

    prob.set_controls_all_section(0, ux_init_guess)
    prob.set_controls_all_section(1, uy_init_guess)
    prob.set_controls_all_section(2, uz_init_guess)

    # ========================
    # Main Process
    # Assign problem to SQP solver
    prob.dynamics = [dynamics]
    prob.knot_states_smooth = []
    prob.cost = cost
    prob.running_cost = running_cost
    # prob.running_cost = None
    prob.equality = equality
    prob.inequality = inequality

    def display_func():
        return 0.0

    prob.solve(obj, display_func, ftol=1e-8)

    # ========================
    # Post Process
    # ------------------------
    # Convert parameter vector to variable
    # plt.close("all")
    # plt.ion()

    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    z = prob.states_all_section(2)
    vx = prob.states_all_section(3)
    vy = prob.states_all_section(4)
    vz = prob.states_all_section(5)

    ux = prob.controls_all_section(0)
    uy = prob.controls_all_section(1)
    uz = prob.controls_all_section(2)
    time = prob.time_update()

    print(uz)
    # ------------------------
    # Visualizetion
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time, vx, marker="o", label="vx")
    plt.plot(time, vy, marker="o", label="vy")
    plt.plot(time, vz, marker="o", label="vz")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.ylabel("velocity [m/s]")
    plt.legend(loc="best")

    plt.subplot(3, 1, 2)
    plt.plot(time, x, marker="o", label="x")
    plt.plot(time, y, marker="o", label="y")
    plt.plot(time, z, marker="o", label="z")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.ylabel("position [m]")
    plt.legend(loc="best")

    plt.subplot(3, 1, 3)
    plt.plot(time, ux, marker="o", label="ux")
    plt.plot(time, uy, marker="o", label="uy")
    plt.plot(time, uz, marker="o", label="uz")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("force [N]")
    plt.legend(loc="best")

    plt.figure()
    plt.plot(x, z, marker="o", label="trajectry")
    # plt.axhline(0, color="k")
    # plt.axhline(1, color="k")
    # plt.axvline(0, color="k")
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    # plt.axis('equal')
    x_ticks = np.arange(0.3, 0.9, 0.1)
    plt.xticks(x_ticks)
    z_ticks = np.arange(0.4, 0.9, 0.1)
    plt.yticks(z_ticks)
    plt.legend(loc="best")

    plt.show()

if __name__ == "__main__":
    TrajOptim()