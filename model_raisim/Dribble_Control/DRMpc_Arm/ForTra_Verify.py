import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_bvp, solve_ivp
from scipy import optimize

x_ref = -0.1
x_reb = -0.47
# f1 = 25
# f2 = 150
# k_vir = 560
k_con = 100000
g = 10
m = 0.5


class DefFun(object):
    def __init__(self):
        self.m = 0.5
        self.g = 10

    def tanh_sig(self, x):
        return 0.5 + 0.5 * np.tanh(1000 * x)

    def F_TRA(self, x, v):
        u1 = - k_vir * (x - x_ref) - f1
        u2 = - k_vir * (x - x_ref) - f2

        # u_ref = (u1 * (0.5 + 0.5 * np.tanh(1000 * v)) + u2 * (0.5 + 0.5 * np.tanh(1000 * (- v)))) * (0.5 + 0.5 * np.tanh(1000 * (x - x_ref)))
        u_ref = (u1 * tanh_sig(v) + u2 * tanh_sig(-v))

        return u_ref

class fun_solve(object):
    def __init__(self):
        self.m = 0.5
        self.g = 10

    def fun_down(self, t, x1, v1, f1, f2, k_vir):
        beta_down = np.sqrt(2 * k_vir)
        a_down = x_ref - (2 * f2 + g) / (2 * k_vir)
        c1_down = x1 - a_down
        c2_down = (-v1) / beta_down
        return c1_down * np.cos(beta_down * t) + c2_down * np.sin(beta_down * t) + a_down - x_ref

    def fun_free(self, t, x2, v2, f1, f2, k_vir):
        return x2 + v2 * t - 0.5 * g * t ** 2 - x_reb

    def fun_con(self, t, x3, v3, f1, f2, k_vir):
        beta_con =  np.sqrt(2 * k_con)
        a_con = x_reb + (- g) / (2 * k_con)
        c1_con = x3 - a_con
        c2_con = v3 / beta_con
        return c1_con * np.cos(beta_con * t) + c2_con * np.sin(beta_con * t) + a_con - x_reb
        # return -beta_con * c1_con * np.sin(beta_con * t) + beta_con * c2_con * np.cos(beta_con * t)

# class StateFun(object):
#     def __init__(self, t, x, v):
#         self.m = 0.5
#         self.g = 10
#         self.t = t
#         self.v0 = v
#         self.x0 = x

#     def state_fun_up(self):
#         beta_up = np.sqrt(2 * k_vir)
#         a_up = x_ref - (2 * f1 + g) / (2 * k_vir)
#         c1_up = self.x0 - a_up
#         c2_up = (self.v0) / beta_up
#         x_up = c1_up * np.cos(beta_up * self.t) + c2_up * np.sin(beta_up * self.t) + a_up
#         v_up = - beta_up * c1_up * np.sin(beta_up * self.t) + beta_up * c2_up * np.cos(beta_up * self.t)

#         # if self.t < 0.001:
#         #     print(self.t, x_up, v_up, self.x0, self.v0, c2_up)
        
#         return x_up, v_up

#     def state_fun_down(self):
#         beta_down = np.sqrt(2 * k_vir)
#         a_down = x_ref - (2 * f2 + g) / (2 * k_vir)
#         c1_down = self.x0 - a_down
#         c2_down = 0
#         # print(c2_down, c1_down)
#         x_down = c1_down * np.cos(beta_down * self.t) + c2_down * np.sin(beta_down * self.t) + a_down - x_ref
#         v_down = - beta_down * c1_down * np.sin(beta_down * self.t) + beta_down * c2_down * np.cos(beta_down * self.t)
        
#         return x_down, v_down

#     def state_fun_free(self):
#         x_free = self.x0 + self.v0 * self.t - 0.5 * g * self.t ** 2
#         v_free = self.v0 - g * self.t

#         return x_free, v_free

#     def state_fun_bound(self):
#         beta_con =  np.sqrt(2 * k_con)
#         a_con = x_reb - g / (2 * k_con)
#         c1_con = self.x0 - a_con
#         c2_con = self.v0 / beta_con
#         # print(a_con, c1_con, c2_con)
#         x_con = c1_con * np.cos(beta_con * self.t) + c2_con * np.sin(beta_con * self.t) + a_con
#         v_con = -beta_con * c1_con * np.sin(beta_con * self.t) + beta_con * c2_con * np.cos(beta_con * self.t)

#         return x_con, v_con

def Ball_Control():
    t = np.linspace(0, 4, 10000)

    f1 = 25
    f2 = f1 + 0.5 * m * (v_ref ** 2 - v0 ** 2) / x_hcon
    k_vir = (m * v_ref ** 2 - 2 * m * g * x_hcon - 2 * f2 * x_hcon) / (x_hcon ** 2)

    # ================ 上升阶段 ================
    beta_up = np.sqrt(2 * k_vir)
    a_up = x_ref - (2 * f1 + g) / (2 * k_vir)
    c1_up = x0 - a_up
    c2_up = (v0) / beta_up
    x_up = c1_up * np.cos(beta_up * t) + c2_up * np.sin(beta_up * t) + a_up
    # v_up = - beta_up * c1_up * np.sin(beta_up * t) + beta_up * c2_up * np.cos(beta_up * t)
    
    t1 = np.arctan(c2_up / c1_up) / beta_up
    x1 = c1_up * np.cos(beta_up * t1) + c2_up * np.sin(beta_up * t1) + a_up
    v1 = - beta_up * c1_up * np.sin(beta_up * t1) + beta_up * c2_up * np.cos(beta_up * t1)
    print("the end time, velo, height of up part", t1, v1, x1)

    # ================ 下降阶段 ================
    beta_down = np.sqrt(2 * k_vir)
    a_down = x_ref - (2 * f2 + g) / (2 * k_vir)
    c1_down = x1 - a_down
    c2_down = (v1) / beta_down
    # print(c2_down, c1_down)
    x_down = c1_down * np.cos(beta_down * t) + c2_down * np.sin(beta_down * t) + a_down - x_ref

    sol = optimize.root(fun_solve().fun_down, [0.02], args = (x1, v1, f1, f2, k_vir))
    # print(sol.x)
    t2 = sol.x[0]
    v2 = - beta_down * c1_down * np.sin(beta_down * t2) + beta_down * c2_down * np.cos(beta_down * t2)
    x2 = c1_down * np.cos(beta_down * t2) + c2_down * np.sin(beta_down * t2) + a_down
    print("the end time, velo, height of down part", t2, v2, x2)

    # ================ 自由飞行 ================
    x_free = x2 + v2 * t - 0.5 * g * t ** 2

    sol = optimize.root(fun_solve().fun_free, [0.2], args = (x2, v2, f1, f2, k_vir))
    # print(sol.x)
    t3 = sol.x[0]
    v3 = v2 - g * t3
    x3 = x2 + v2 * t3 - 0.5 * g * t3 ** 2
    print("the end time, velo, height of free part", t3, v3, x3)

    # ================ 地面接触 ================
    beta_con =  np.sqrt(2 * k_con)
    a_con = x_reb - g / (2 * k_con)
    c1_con = x3 - a_con
    c2_con = v3 / beta_con
    # print(a_con, c1_con, c2_con)
    x_con = c1_con * np.cos(beta_con * t) + c2_con * np.sin(beta_con * t) + a_con

    sol = optimize.root(fun_solve().fun_con, [0.005], args = (x3, v3, f1, f2, k_vir))
    # print(sol.x)
    t4 = sol.x[0]
    # t42 = np.arctan(c2_con / c1_con) / beta_con
    # print(t42, np.sin(np.pi / 6)) 
    v4 = -beta_con * c1_con * np.sin(beta_con * t4) + beta_con * c2_con * np.cos(beta_con * t4)
    x4 = c1_con * np.cos(beta_con * t4) + c2_con * np.sin(beta_con * t4) + a_con
    print("the end time, velo, height of bound part", t4, v4, x4)


    T = t1 + t2 + 2 * t3 + t4
    print("the total time of each period-", T)
    plt.show()


def ParamsCal():
    # print(x_init)
    dx_up = x_top - x_init
    dx_down = x_top - x_ref

    k_vir = (m * v0 ** 2 - 2 * m * g * dx_up - 2 * f1 * dx_up) / (dx_up ** 2)
    f2 = (m * v_ref ** 2 - 2 * m * g * dx_down - k_vir * dx_down ** 2) / (2 * dx_down)

    print("the up height, down height, virtual stiffness and constant down force is respectively", dx_up, dx_down, k_vir, f2)

if __name__ == "__main__":
    x_hcon = 0.1
    v_ref = 15
    x0 = -0.1
    v0 = 10

    x_top = 0.15
    x_init = -0.1
    f1 = 25

    # Ball_Control()

    ParamsCal()