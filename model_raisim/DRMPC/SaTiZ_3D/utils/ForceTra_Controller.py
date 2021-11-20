import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_bvp, solve_ivp
from scipy import optimize
import pickle
import os
import random

x_ref = -0.1
x_reb = -0.47
# f1 = 25
# f2 = 150
# k_vir = 560
k_con = 100000
g = 10
m = 0.5
n = 1


class DefFun(object):
    def __init__(self):
        self.m = 0.5
        self.g = 10

    def tanh_sig(self, x):
        return 0.5 + 0.5 * np.tanh(1000 * x)

    def F_TRA(self, x, v):
        u1 = - k_vir * (x - x_ref) - f1
        u2 = - k_vir * (x - x_ref) - f2

        u_ref = (u1 * tanh_sig(v) + u2 * tanh_sig(-v))

        return u_ref

class fun_solve(object):
    def __init__(self):
        self.m = 0.5
        self.g = 10

    def fun_down(t, x1, v1):
        beta_down = np.sqrt(2 * k_vir)
        a_down = x_ref - (2 * f2 + g) / (2 * k_vir)
        c1_down = x1 - a_down
        c2_down = (-v1) / beta_down
        return c1_down * np.cos(beta_down * t) + c2_down * np.sin(beta_down * t) + a_down - x_ref

    def fun_free(t, x2, v2):
        return x2 + v2 * t - 0.5 * g * t ** 2 - x_reb

    def fun_con(t, x3, v3):
        beta_con =  np.sqrt(2 * k_con)
        a_con = x_reb + (- g) / (2 * k_con)
        c1_con = x3 - a_con
        c2_con = v3 / beta_con
        return c1_con * np.cos(beta_con * t) + c2_con * np.sin(beta_con * t) + a_con - x_reb
        # return -beta_con * c1_con * np.sin(beta_con * t) + beta_con * c2_con * np.cos(beta_con * t)

class StateFun(object):
    def __init__(self, t, x, v, f1, f2, k_vir):
        self.m = 0.5
        self.g = 10
        self.t = t
        self.v0 = v
        self.x0 = x
        self.f1 = f1
        self.f2 = f2
        self.k_vir = k_vir

    def state_fun_up(self):
        beta_up = np.sqrt(2 * self.k_vir)
        a_up = x_ref - (2 * self.f1 + g) / (2 * self.k_vir)
        c1_up = self.x0 - a_up
        c2_up = (self.v0) / beta_up
        x_up = c1_up * np.cos(beta_up * self.t) + c2_up * np.sin(beta_up * self.t) + a_up
        v_up = - beta_up * c1_up * np.sin(beta_up * self.t) + beta_up * c2_up * np.cos(beta_up * self.t)
        
        return x_up, v_up

    def state_fun_down(self):
        beta_down = np.sqrt(2 * self.k_vir)
        a_down = x_ref - (2 * self.f2 + g) / (2 * self.k_vir)
        c1_down = self.x0 - a_down
        c2_down = self.v0 / beta_down
        # print(c2_down, c1_down)
        x_down = c1_down * np.cos(beta_down * self.t) + c2_down * np.sin(beta_down * self.t) + a_down
        v_down = - beta_down * c1_down * np.sin(beta_down * self.t) + beta_down * c2_down * np.cos(beta_down * self.t)
        
        return x_down, v_down

    def state_fun_free(self):
        x_free = self.x0 + self.v0 * self.t - 0.5 * g * self.t ** 2
        v_free = self.v0 - g * self.t

        return x_free, v_free

    def state_fun_bound(self):
        beta_con =  np.sqrt(2 * k_con)
        a_con = x_reb - g / (2 * k_con)
        c1_con = self.x0 - a_con
        c2_con = self.v0 / beta_con
        # print(a_con, c1_con, c2_con)
        x_con = c1_con * np.cos(beta_con * self.t) + c2_con * np.sin(beta_con * self.t) + a_con
        v_con = -beta_con * c1_con * np.sin(beta_con * self.t) + beta_con * c2_con * np.cos(beta_con * self.t)

        return x_con, v_con

def CurrentState(x, state_ball, u, t, cal_init, f1, f2, k_vir, i):
    
    if x[0, 2] != state_ball[i-1, 2]:
        cal_init[0] = state_ball[i-1, 0]
        cal_init[1] = state_ball[i-1, 1]
        cal_init[2] = t
    
    if x[0, 2] == 1:
        x[0, 0], x[0, 1] = StateFun(t - cal_init[2], cal_init[0], cal_init[1], f1, f2, k_vir).state_fun_up()

    elif x[0, 2] == 2:
        x[0, 0], x[0, 1] = StateFun(t - cal_init[2], cal_init[0], cal_init[1], f1, f2, k_vir).state_fun_down()

    elif x[0, 2] == 3:
        x[0, 0], x[0, 1] = StateFun(t - cal_init[2], cal_init[0], cal_init[1], f1, f2, k_vir).state_fun_free()

    elif x[0, 2] == 4:
        x[0, 0], x[0, 1] = StateFun(t - cal_init[2], cal_init[0], cal_init[1], f1, f2, k_vir).state_fun_bound()

    return x[0, 0], x[0, 1]

def ParamsCal(x0, v0, x_top, x_ref, v_ref, f1):
    dx_up = x_top - x0
    dx_down = x_top - x_ref

    k_vir = (m * v0 ** 2 - 2 * m * g * dx_up - 2 * f1 * dx_up) / (dx_up ** 2)
    f2 = (m * v_ref ** 2 - 2 * m * g * dx_down - k_vir * dx_down ** 2) / (2 * dx_down)

    if k_vir < 0:
        raise ValueError('invalid value: k_vir is negative, can not sqrt:')
    return k_vir, f2

def FigPlot(state_ball, u, t_sim):
    print(u.shape)
    print(state_ball.shape)
    # print(state_ball[1500:2500, 0])
    # print(state_ball[17000:18000, 1])
    # print(state_ball[500:1000, 2])


    # ================== plot ========================
    fig = plt.figure()
    fig.suptitle('Controll of Dribbling Ball', fontsize = 20)
    # p1, ax2, ax3 = fig.subplots(3, 1, sharex = True, sharey = False)
    # fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex = True, sharey = False)

    ax1 = plt.subplot(311)
    plt.tick_params(labelsize = 12)
    ax1_ytick = np.arange(-0.5, 0.5 , 0.1)
    plt.yticks(ax1_ytick)
    ax1.set_ylabel('Height (m)',fontsize=18)
    ax1.plot(t_sim, state_ball[:, 0])
    # plt.plot(t_sim, state_ball[:, 2])

    ax2 = plt.subplot(312)
    plt.tick_params(labelsize = 12)
    ax2_ytick = np.arange(-15, 16, 5)
    plt.yticks(ax2_ytick)
    ax2.set_ylabel('Velocity (m/s)', fontsize=18)
    ax2.plot(t_sim, state_ball[:, 1])
    # plt.plot(t_sim, state_ball[:, 2])
    
    ax3 = plt.subplot(313)
    plt.tick_params(labelsize = 12)
    ax3_ytick = np.arange(-600, 10 , 50)
    plt.yticks(ax3_ytick)
    ax3.set_ylabel('Force (N)', fontsize=18)
    ax3.set_xlabel('time (s)', fontsize=18)
    ax3.plot(t_sim, u)

    plt.show()

def FileSave(state_ball, u, t_sim, x0, dx, v0, v_ref):
    data = {'state': state_ball, 'time': t_sim, 'u': u}
    # print(data['state'][0:100, 0])

    name = 'x0_' + str(x0) + '-v0_'+ str(v0) + '-vref_' + str(v_ref) + '-dx_' + str(dx) + '.pkl'
    pathDir = './results/'

    if os.path.exists(pathDir + name):
        name = name + '-' + str(random.randint(0,100))

    # with open(pathDir + name, 'wb') as f:
    #     pickle.dump(data, f)


def Ball_Control():
    # ====================================== init params modified =======================================
    # simulation time setting
    i = 1
    sample_num = 0.0001
    sample_time = 2

    # other params
    fun_flag = 0
    flag = 0
    cal_init = [0, 0, 0]
    t_sim = np.linspace(0, sample_time, int(sample_time / sample_num))

    # init params
    x0 = -0.1   # init position
    v0 = 20        # init velocity
    x_top = 0.25 # dribbling height of ball
    v_ref = -15     # desired velocity

    x_hcon = 0.25

    # model params setting
    f1 = 0
    f2 = 0
    k_vir = 0

    # init condition setting
    state_ball = np.array([[x0, v0, flag]])
    x = np.array([[0.0, 0.0, 0.0]])
    u = np.array([0.0])
    # =====================================================================================================


    # ============================ motion process calculation and graphics ================================ 
    try:
        for t in np.arange(0, sample_time, sample_num):

            if  state_ball[i-1, 0] >= x_ref and state_ball[i-1, 1] > 0 and fun_flag == 0:
                
                f1 = 25
                dx_up = x_top - x0

                if (dx_up * 2 * g) >= (state_ball[i-1, 1] ** 2):
                    raise FloatingPointError("calculate Error: init velocity is too small or the heaving height is too high!")

                else:
                    k_vir, f2 = ParamsCal(state_ball[i-1, 0], state_ball[i-1, 1], x_top, x_ref, v_ref, f1)
                    print(state_ball[i-1, 0], state_ball[i-1, 1], f1, f2, k_vir)
                
                # print(state_ball[i-1, 0], state_ball[i-1, 1], f1, f2, k_vir)
                # print("the virtual stiffness and constant down force is respectively", k_vir, f2)
                # f1 = 25
                # f2 = f1 + 0.5 * m * (v_ref ** 2 - (state_ball[i-1, 1]) ** 2) / x_hcon
                # k_vir = (m * v_ref ** 2 - 2 * m * g * x_hcon - 2 * f2 * x_hcon) / (x_hcon ** 2)
                # print(state_ball[i-1, 0], state_ball[i-1, 1], f1, f2, k_vir)

                fun_flag = 1

            elif  state_ball[i-1, 0] >= x_ref and state_ball[i-1, 1] < 0 and fun_flag == 0:
                
                f1 = 25
                dx_down = state_ball[i-1, 0] - x_ref

                if (dx_down * 2 * g) >= (state_ball[i-1, 1] ** 2):
                    raise FloatingPointError("calculate Error: init velocity is too small or the heaving height is too high!")

                else:
                    f2 = 50
                    k_vir = (m * (v_ref ** 2 - state_ball[i-1, 1] ** 2) - 2 * m * g * dx_down - 2 * f2 * dx_down) / (dx_down ** 2)
                    print(dx_down, f2, k_vir)

                fun_flag = 1
            
            if state_ball[i-1, 0] >= x_ref and state_ball[i-1, 1] > 0:
                x[0, 2] = 1
                x[0, 0], x[0, 1] = CurrentState(x, state_ball, u, t, cal_init, f1, f2, k_vir, i)

                state_ball = np.concatenate([state_ball, x], axis = 0)
                f = - k_vir * (state_ball[i, 0] - x_ref) - f1
                u = np.concatenate([u, [f]])
                
                i = i + 1

            elif state_ball[i-1, 0] >= x_ref and state_ball[i-1, 1] <= 0:
                x[0, 2] = 2
                x[0, 0], x[0, 1] = CurrentState(x, state_ball, u, t, cal_init, f1, f2, k_vir, i)

                state_ball = np.concatenate([state_ball, x], axis = 0)
                f = - k_vir * (state_ball[i, 0] - x_ref) - f2
                u = np.concatenate([u, [f]])
                # CurrentState(x, state_ball, u, t, f1, f2, k_vir, i)

                if state_ball[i, 0] <= x_ref:
                    print("the v separate is ", state_ball[i, 1])

                i = i + 1

            elif state_ball[i-1, 0] < x_ref and state_ball[i-1, 0] > x_reb:
                x[0, 2] = 3
                x[0, 0], x[0, 1] = CurrentState(x, state_ball, u, t, cal_init, f1, f2, k_vir, i)

                state_ball = np.concatenate([state_ball, x], axis = 0)
                u = np.concatenate([u, [0.0]])
                # CurrentState(x, state_ball, u, t, f1, f2, k_vir, i)
                
                i = i + 1

            elif state_ball[i-1, 0] <= x_reb:
                x[0, 2] = 4
                x[0, 0], x[0, 1] = CurrentState(x, state_ball, u, t, cal_init, f1, f2, k_vir, i)

                state_ball = np.concatenate([state_ball, x], axis = 0)
                u = np.concatenate([u, [0.0]])
                # CurrentState(x, state_ball, u, t, f1, f2, k_vir, i)

                fun_flag = 0
                i = i + 1

        # data Interception
        state_ball = state_ball[1:,]
        u = u[1:]

        # dribbling ball fig plot
        FigPlot(state_ball, u, t_sim)

        dx = x_top - x_ref
        # data file save as .pkl
        FileSave(state_ball, u, t_sim, x0, dx, v0, v_ref)

    except ValueError as e:
        print("ValueError: k_vir is negative, can not sqrt, please decrease the value of F_up!")

    except ZeroDivisionError as e:
        print("ZeroDivisionError: k_vir is zero, can not be divided!")

    except FloatingPointError as e:
        print("FloatingPointError: init velocity is too small or the heaving height is too high!")

    finally:
        print("End")
    # =====================================================================================================

if __name__ == "__main__":
    
    Ball_Control()