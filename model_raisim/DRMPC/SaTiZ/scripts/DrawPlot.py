'''
该部分用于一些数据的可视化和展示作图
1. 2022.09.23:
        - 增加机械臂的质量、速度、负载的参数调研结果并可视化
'''
import numpy as np
from numpy import sin  as s
from numpy import cos as c
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
from scipy.optimize import curve_fit
import scipy.stats as st
import seaborn as sns

## robot arm mass velocity payload survey fig plot
def RobotArmSurvey():
    # UR5, KUKA, Franka, KINOVA, DLR, ABB, AUBO, yaskawa, OMRON, OB, Z1, nasa robonaut
    # LISM2, human, 新松， 节卡, 珞石
    Mass = [18.4, 22.0, 18.0, 7.2, 14.0, 9.5, 24.0, 45.0, 22.1, 22.0, 4.5, 35.0,
            23, 18.6, 21]
    Vel = [1.0, 2.5, 2.0, 0.5, 1.8, 1.5, 2.8, 3.0, 1.1, 2.0, 2.2, 2.0, 
            3.0, 2.5, 3.0]
    Human_m = [4.5]
    Human_v = [15.0]
    LISM_m = [5.8]
    LISM_v= [5.35]
    Payload = [5.0, 7.0, 3.0, 2.0, 14.0, 4.0, 5.0, 10.0, 6.0, 5.0, 3.0, 9.0,
            5.0, 3.0, 10.0, 5.0, 5.0, 3.0]

    f1 = np.polyfit(Mass, Vel, 1)
    p1 = np.poly1d(f1)
    vel_opt = p1(Mass)

    Mass = np.asarray(Mass)
    Vel = np.asarray(Vel)
    ax = sns.regplot(Mass, Vel)

    plt.style.use("science")
    # plt.style.use('fivethirtyeight')
    params = {
        'text.usetex': True,
        'image.cmap': 'inferno',
        'font.size': 18,
        'axes.labelsize': 25,
        'axes.titlesize': 25,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'legend.fontsize': 15,
        'axes.titlepad': 15.0,
        'axes.labelpad': 12.0,
        'figure.subplot.wspace': 0.2,
        'figure.subplot.hspace': 0.3,
    }
    mark_s = 250
    plt.rcParams.update(params)
    fig, axes = plt.subplots(1, 1, figsize=(12, 12))
    ax1 = axes

    ax1.scatter(Mass, Vel, s=150)
    ax.scatter(Mass, Vel, s=150)
    ax1.plot(Mass, vel_opt)
    ax1.scatter(Human_m, Human_v, s=mark_s, c="r", marker="s")
    ax1.scatter(LISM_m, LISM_v, s = mark_s, c="coral", marker="^")
    ax1.set_xlabel("Robot Arm Mass (Kg)")
    ax1.set_ylabel("Robot Arm Velocity (m/s)")
    ax1.grid()

    plt.show()
    pass

if __name__ == "__main__":
    RobotArmSurvey()
    pass