import os
import sys
import pickle
import random
import datetime
import numpy as np
import sympy as sp
from sympy import symbols, diff
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sympy.matrices.dense import matrix2numpy

def DynamicsEquation():
    Theta1, Theta2 = symbols('Theta1 Theta2')
    D_Theta1, D_Theta2 = symbols('D_Theta1 D_Theta2')
    DD_Theta1, DD_Theta2 = symbols('DD_Theta1 DD_Theta2')
    m1, m2 = symbols('m1, m2')
    L1, L2 = symbols('L1, L2')
    g = symbols('g')

    y0 = 1
    x0 = 1

    # 连杆绕中心轴的转动惯量
    I1 = 1 / 12 * m1 * L1 **2
    I2 = 1 / 12 * m2 * L2 **2

    # 质心位置坐标
    x1 = x0 + L1 / 2 * sp.sin(Theta1)
    y1 = y0 - L1 / 2 * sp.cos(Theta1)

    x2 = L1 * sp.cos(Theta1) + L2 / 2 * sp.cos(Theta1 + Theta2)
    y2 = L1 * sp.sin(Theta1) + L2 / 2 * sp.sin(Theta1 + Theta2)

    # 质心速度
    dx1 = diff(x1, Theta1) * D_Theta1
    dy1 = diff(y1, Theta1) * D_Theta1

    dx2 = sp.simplify(diff(x2, Theta1) * D_Theta1 + diff(x2, Theta2) * D_Theta2)
    dy2 = sp.simplify(diff(y2, Theta1) * D_Theta1 + diff(y2, Theta2) * D_Theta2)

    # 动能计算： 刚体动能 = 刚体质心平移动能 + 绕质心转动动能 = 1 / 2 * m * vc ** 2 + 1 / 2 * Ic * dtheta ** 2
    K1 = sp.simplify(1 / 2 * m1 * (dx1 ** 2 + dy1 ** 2) + 1 / 2 * I1 * D_Theta1 ** 2)
    K2 = sp.simplify(1 / 2 * m2 * (dx2 ** 2 + dy2 ** 2) + 1 / 2 * I2 * (D_Theta1 + D_Theta2) ** 2)

    # 势能计算
    P1 = 1 / 2 * m1 * g * L1 * sp.cos(Theta1)
    P2 = m2 * g * (L1 * sp.cos(Theta1) + L2 / 2 * sp.cos(Theta1 + Theta2))

    # 拉格朗日函数
    Lag = sp.simplify(K1 + K2 - P1 - P2)

    # 拉格朗日方程
    Lag_DTheta1 = sp.simplify(diff(Lag, D_Theta1))
    Lag_DTheta2 = sp.simplify(diff(Lag, D_Theta2))

    Tor1 = sp.simplify(diff(Lag_DTheta1, D_Theta1) * DD_Theta1 + diff(Lag_DTheta1, D_Theta2) * DD_Theta2 \
        + diff(Lag_DTheta1, Theta1) * D_Theta1 + + diff(Lag_DTheta1, Theta2) * D_Theta2\
        - diff(Lag, Theta1))

    Tor2 = sp.simplify(diff(Lag_DTheta2, D_Theta1) * DD_Theta1 + diff(Lag_DTheta2, D_Theta2) * DD_Theta2\
        + diff(Lag_DTheta2, Theta1) * D_Theta1 + + diff(Lag_DTheta2, Theta2) * D_Theta2\
        - diff(Lag, Theta2)) 
    
    Tor_bar_1 = sp.simplify(Tor1 / (m2 * L1 * L2))
    Tor_bar_2 = sp.simplify(Tor2 / (m2 * L1 * L2))
    print(Tor_bar_1)
    print(Tor_bar_2)

    return Tor_bar_1, Tor_bar_2

def DynamicsAnalysis():
    
    K = 1
    M = 2
    D = 1.0
    g = 9.8
    m2 = 0.6
    L2 = D / (K + 1)
    L1 = K * D / (K + 1)
    N = 1 / (m2 * L1 * L2)

    T_Period = 0.5
    w = 2 * np.pi / T_Period

    N_lag = 0.01
    N_freq = 2.5

    x_end, y_end = symbols('x_end, y_end')
    dx_end, dy_end = symbols('dx_end, dy_end')
    ddx_end, ddy_end = symbols('ddx_end, ddy_end')
    t = symbols('t')  
    print(0.25 * sp.sin(w * t) + L1)

    # 初始点位置
    # y0 = 0
    # x0 = 0

    # 关节位置坐标
    # x1 = x0 + L1 * sp.sin(Theta1)
    # y1 = y0 - L1 * sp.cos(Theta1)

    # x2 = x0 + L1 * sp.sin(Theta1) + L2 * sp.sin(Theta1 + Theta2)
    # y2 = y0 - L1 * sp.cos(Theta1) - L2 * sp.cos(Theta1 + Theta2)

    # 逆运动学
    Ctheta_end = x_end / sp.sqrt(x_end ** 2 + y_end ** 2)
    ThetaTraj_1 = - sp.acos((x_end ** 2 + y_end ** 2 + L1 ** 2 - L2 ** 2) * x_end / (2 * L1 * x_end * sp.sqrt(x_end ** 2 + y_end ** 2))) + sp.atan(y_end / x_end)
    ThetaTraj_2 = sp.acos((x_end ** 2 + y_end ** 2 + L2 ** 2 - L1 ** 2) * x_end / (2 * L2 * x_end * sp.sqrt(x_end ** 2 + y_end ** 2))) + sp.atan(y_end / x_end) - ThetaTraj_1

    Theta1 = sp.simplify(ThetaTraj_1)
    Theta2 = sp.simplify(ThetaTraj_2)
    vx_end = diff(x_end, t)

    # method 1
    D_Theta1 = sp.simplify(diff(Theta1, x_end) * dx_end + diff(Theta1, y_end) * dy_end)
    D_Theta2 = sp.simplify(diff(Theta2, x_end) * dx_end + diff(Theta2, y_end) * dy_end)

    DD_Theta1 = diff(D_Theta1, x_end) * dx_end + diff(D_Theta1, y_end) * dy_end + \
                diff(D_Theta1, dx_end) * ddx_end + diff(D_Theta1, dy_end) * ddy_end
    DD_Theta1 = sp.simplify(DD_Theta1)

    DD_Theta2 = diff(D_Theta2, x_end) * dx_end + diff(D_Theta2, y_end) * dy_end + \
                diff(D_Theta2, dx_end) * ddx_end + diff(D_Theta2, dy_end) * ddy_end
    DD_Theta2 = sp.simplify(DD_Theta2)

    # print(DD_Theta1)    

    # =====================================================
    ## 求导赋值
    ## 轨迹设定
    x_ref = 0.25 * sp.sin(w * t) + L1
    y_ref = 0 * t + L2

    dx_ref = diff(x_ref, t)
    dy_ref = diff(y_ref, t)

    ddx_ref = diff(dx_ref, t)
    ddy_ref = diff(dy_ref, t)

    # 末端轨迹赋值
    N_samples = int(1 / N_lag * N_freq)

    # 数据储存
    EndState = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])     # end pos, vel, accel
    JointState = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])     # joint 1 pos, vel, accel
    TorqueBar = np.array([[0.0, 0.0]])
    TorqueReal = np.array([[0.0, 0.0]])
    TorqueBarPart = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    T = np.array([0.0])
    PeriodPoint = np.array([0.0])

    for i in range(0, N_samples):
        t_now = N_lag * i

        if i % (T_Period / N_lag) == 0.0:
             PeriodPoint = np.concatenate([PeriodPoint, [t_now]], axis = 0)

        x_ref_val = x_ref.evalf(subs={t: t_now})
        y_ref_val = y_ref.evalf(subs={t: t_now})

        dx_ref_val = dx_ref.evalf(subs={t: t_now})
        dy_ref_val = dy_ref.evalf(subs={t: t_now})

        ddx_ref_val = ddx_ref.evalf(subs={t: t_now})
        ddy_ref_val = ddy_ref.evalf(subs={t: t_now})

        # 关节角度赋值
        Theta1_Val = Theta1.evalf(subs={x_end: x_ref_val, y_end: y_ref_val})
        Theta2_Val = Theta2.evalf(subs={x_end: x_ref_val, y_end: y_ref_val})

        D_Theta1_Val = D_Theta1.evalf(subs={x_end: x_ref_val, y_end: y_ref_val, dx_end: dx_ref_val, dy_end: dy_ref_val})
        D_Theta2_Val = D_Theta2.evalf(subs={x_end: x_ref_val, y_end: y_ref_val, dx_end: dx_ref_val, dy_end: dy_ref_val})

        DD_Theta1_Val = DD_Theta1.evalf(subs={x_end: x_ref_val, y_end: y_ref_val, dx_end: dx_ref_val, dy_end: dy_ref_val, \
                        ddx_end: ddx_ref_val, ddy_end: ddy_ref_val})
        DD_Theta2_Val = DD_Theta2.evalf(subs={x_end: x_ref_val, y_end: y_ref_val, dx_end: dx_ref_val, dy_end: dy_ref_val,  \
                        ddx_end: ddx_ref_val, ddy_end: ddy_ref_val})

        ## 雅克比矩阵
        Theta1_Val = float(Theta1_Val)
        Theta2_Val = float(Theta2_Val)
        dx_ref_val = float(dx_ref_val)
        Jacobian = [[- L1 * np.sin(Theta1_Val) - L2 * np.sin(Theta1_Val + Theta2_Val), - L2 * np.sin(Theta1_Val + Theta2_Val)],
                    [L1 * np.cos(Theta1_Val) + L2 * np.cos(Theta1_Val + Theta2_Val),   L2 * np.cos(Theta1_Val + Theta2_Val)]]
        Jacobian_inv = np.linalg.inv(Jacobian)

        D_Theta1_2 = Jacobian_inv[0][0] * dx_ref_val + Jacobian_inv[0][1] * 0.0
        D_Theta2_2 = Jacobian_inv[1][0] * dx_ref_val + Jacobian_inv[1][1] * 0.0
   

        # 动力学方程
        M_Theta = [[(1 / 3 * M + 1) * K + 1 / 3 * 1 / K + sp.cos(Theta2_Val), 1 / 3 * 1 / K + 1 / 2 * sp.cos(Theta2_Val)],
                [1 / 3 * 1 / K + 1 / 2 * sp.cos(Theta2_Val),               1 / 3 * 1 / K]]

        C_Theta = [[-D_Theta1_Val * D_Theta2_Val * sp.sin(Theta2_Val) - 1 / 2 * D_Theta2_Val ** 2 * sp.sin(Theta2_Val)],
                [1 / 2 * D_Theta1_Val ** 2 * sp.sin(Theta2_Val)]]

        G_Theta = [[-1 / 2 * (M + 1) * (K + 1) / D * g * sp.sin(Theta2_Val) - 1 / 2 * (K + 1) / (K * D) * sp.sin(Theta1_Val + Theta2_Val)],
                [- 1 / 2 * (K + 1) / (K * D) * sp.sin(Theta1_Val + Theta2_Val)]]
        

        # 力矩求解
        TorBar_1 = M_Theta[0][0] * DD_Theta1_Val + M_Theta[0][1] * DD_Theta2_Val + C_Theta[0][0] + G_Theta[0][0]
        TorBar_2 = M_Theta[1][0] * DD_Theta1_Val + M_Theta[1][1] * DD_Theta2_Val + C_Theta[1][0] + G_Theta[1][0]

        TorBar_1_M11 = M_Theta[0][0] * DD_Theta1_Val
        TorBar_1_M12 = M_Theta[0][1] * DD_Theta2_Val
        TorBar_2_M21 = M_Theta[1][0] * DD_Theta1_Val
        TorBar_2_M22 = M_Theta[1][1] * DD_Theta2_Val
        TorBar_1_C = C_Theta[0][0]
        TorBar_2_C = C_Theta[1][0]
        TorBar_1_G = G_Theta[0][0]
        TorBar_2_G = G_Theta[1][0]

        TorBar_1_M11 = float(TorBar_1_M11)
        TorBar_1_M12 = float(TorBar_1_M12)
        TorBar_2_M21 = float(TorBar_2_M21)
        TorBar_2_M22 = float(TorBar_2_M22)
        TorBar_1_C = float(TorBar_1_C)
        TorBar_2_C = float(TorBar_2_C)
        TorBar_1_G = float(TorBar_1_G)
        TorBar_2_G = float(TorBar_2_G)

        TorqueBarPart = np.concatenate([TorqueBarPart, [[TorBar_1_M11, TorBar_1_M12, TorBar_2_M21, TorBar_2_M22, TorBar_1_C, TorBar_2_C, TorBar_1_G, TorBar_2_G]]], axis = 0)

        Tor_1 = TorBar_1 / N
        Tor_2 = TorBar_2 / N
        print("#=========================================================================================")
        print("Time Now:",                          t_now)
        print("N is: ",                             N)
        print("End x-axis Pos, Vel, Accel",         x_ref_val, dx_ref_val, ddx_ref_val)
        print("End y-axis Pos, Vel, Accel",         y_ref_val, dy_ref_val, ddy_ref_val)
        print("Joint Pos",                          Theta1_Val, Theta2_Val)
        print("Joint Vel by Jacobian",              D_Theta1_2, D_Theta2_2)
        print("Joint Vel",                          D_Theta1_Val, D_Theta2_Val)
        print("Joint Accel",                        DD_Theta1_Val, DD_Theta2_Val) 
        print("Joint 1, 2 Toerque Bar:",            TorBar_1, TorBar_1)
        print("Joint 1, 2 Toerque:",                Tor_1, Tor_2)
        

        x_ref_val = float(x_ref_val)
        dx_ref_val = float(dx_ref_val)
        ddx_ref_val = float(ddx_ref_val)
        y_ref_val = float(y_ref_val)
        dy_ref_val = float(dy_ref_val)
        ddy_ref_val = float(ddy_ref_val)

        Theta1_Val = float(Theta1_Val)
        Theta2_Val = float(Theta2_Val)
        D_Theta1_Val = float(D_Theta1_Val)
        D_Theta2_Val = float(D_Theta2_Val)
        DD_Theta1_Val = float(DD_Theta1_Val)
        DD_Theta2_Val = float(DD_Theta2_Val)
        # print(x_ref_val)

        TorBar_1 = float(TorBar_1)
        TorBar_2 = float(TorBar_2)
        Tor_1 = float(Tor_1)
        Tor_2 = float(Tor_2)
        print(Tor_1, Tor_2)

        EndState = np.concatenate([EndState, [[x_ref_val, y_ref_val, dx_ref_val, dy_ref_val, ddx_ref_val, ddy_ref_val]]], axis = 0)
        JointState = np.concatenate([JointState, [[Theta1_Val, Theta2_Val, D_Theta1_Val, D_Theta2_Val, DD_Theta1_Val, DD_Theta2_Val]]], axis = 0)
        TorqueBar =  np.concatenate([TorqueBar, [[TorBar_1, TorBar_2]]], axis = 0)
        TorqueReal =  np.concatenate([TorqueReal, [[Tor_1, Tor_2]]], axis = 0)
        T = np.concatenate([T, [t_now]], axis = 0)

    EndState = EndState[1:,]
    JointState = JointState[1:,]
    TorqueBar = TorqueBar[1:,]
    TorqueReal = TorqueReal[1:,]
    TorqueBarPart = TorqueBarPart[1:,]
    T = T[1:,]

    Data = {"EndState": EndState, "JointState": JointState, "TorqueBar": TorqueBar, "TorqueReal": TorqueReal, "T": T, "PeriodPoint": PeriodPoint, "TorqueBarPart": TorqueBarPart}

    return Data, K, M, D, m2, T_Period


def FileSave(Data, K, M, D, m2, T_Period):
    today=datetime.date.today()
    RandNum = random.randint(0,100)
    
    name = str(today) + '-M_' + str(M) + '-K_'+ str(K) + '-D_' + str(D) +'-m2_' + str(m2) + '-T_' + str(T_Period) + '.pkl'
    pathDir = './data/'
    print(name)

    # 如果目录不存在，则创建
    todaytime = str(today)
    save_dir = os.path.join(pathDir, todaytime)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 数据文件保存
    if os.path.exists(os.path.join(save_dir, name)):
        name = str(today) + '-M_' + str(M) + '-K_'+ str(K) + '-D_' + str(D) +'-m2_' + str(m2) + '-T_' + str(T_Period) + '-' + str(RandNum) + '.pkl'

    with open(os.path.join(save_dir, name), 'wb') as f:
        pickle.dump(Data, f)

def ColorSpan(ContPointTime, ColorId, axes):
    for i in range(len(ContPointTime) + 1):
        mod = i % 2
        if i == 0:
            axes.axvspan(-2, ContPointTime[i], facecolor=ColorId[mod])
        elif i == len(ContPointTime):
            axes.axvspan(ContPointTime[i-1], 20, facecolor=ColorId[mod])
        else:
            axes.axvspan(ContPointTime[i-1], ContPointTime[i], facecolor=ColorId[mod])

def DataPlot(Data1, Data2, Data3, Data4):
    EndState1 = Data1["EndState"]
    JointState1 = Data1["JointState"]
    TorqueBar1 = Data1["TorqueBar"]
    TorqueReal1 = Data1["TorqueReal"]
    T1 = Data1["T"]
    PeriodPoint1 = Data1["PeriodPoint"]

    EndState2 = Data2["EndState"]
    JointState2 = Data2["JointState"]
    TorqueBar2 = Data2["TorqueBar"]
    TorqueReal2 = Data2["TorqueReal"]
    T2 = Data2["T"]
    PeriodPoint2 = Data2["PeriodPoint"]

    EndState3 = Data3["EndState"]
    JointState3 = Data3["JointState"]
    TorqueBar3 = Data3["TorqueBar"]
    TorqueReal3 = Data3["TorqueReal"]
    T3 = Data3["T"]
    PeriodPoint3 = Data3["PeriodPoint"]

    EndState4 = Data4["EndState"]
    JointState4 = Data4["JointState"]
    TorqueBar4 = Data4["TorqueBar"]
    TorqueReal4 = Data4["TorqueReal"]
    T4 = Data4["T"]
    PeriodPoint4 = Data4["PeriodPoint"]

    fig, axes = plt.subplots(2,2, dpi=100,figsize=(12,10))
    ax1=axes[0][0]
    ax2=axes[0][1]
    ax3=axes[1][0]
    ax4=axes[1][1]
    
    # ColorId = ['#CDF0EA', '#FEFAEC']
    # ColorId = ['#DEEDF0', '#FFF5EB']``
    # ColorId = ['#C5ECBE', '#FFEBBB']
    # ColorId = ['#A7D7C5', '#F7F4E3']
    ColorId = ['#E1F2FB', '#FFF5EB']

    ax1.plot(T1, TorqueBar1[:, 0], label='m2 = 0.4')
    ax1.plot(T2, TorqueBar2[:, 0], label='m2 = 0.6')
    ax1.plot(T3, TorqueBar3[:, 0], label='m2 = 0.8')
    ax1.plot(T4, TorqueBar4[:, 0], label='m2 = 1.0')
    # plt.plot(T, line2, label='highest Velocity')
    N_FootVel = TorqueBar4[:, 0]
    ax1.axis([0, max(T1)*1.05, min(N_FootVel)*1.2, max(N_FootVel)*0.8])
    # ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Angular Velocity ', fontsize = 15)
    ax1.legend(loc='upper right', fontsize = 12)
    ax1.set_title("Joint 1", fontsize = 20)
    ColorSpan(PeriodPoint1, ColorId, ax1)
    # print(len(PeriodPoint))
    
    ax2.plot(T1, TorqueBar1[:, 1], label='m2 = 0.4')
    ax2.plot(T2, TorqueBar2[:, 1], label='m2 = 0.6')
    ax2.plot(T3, TorqueBar3[:, 1], label='m2 = 0.8')
    ax2.plot(T4, TorqueBar4[:, 1], label='m2 = 1.0')

    N_FootVel = TorqueBar4[:, 1]
    ax2.axis([0, max(T1)*1.05, min(N_FootVel)*1.2, max(N_FootVel)*1.2])
    # ax2.set_xlabel('time (s)')
    # ax2.set_ylabel('Foot-Vel (m/s)', fontsize = 15)
    ax2.legend(loc='upper right', fontsize = 12)
    ax2.set_title("Joint 2", fontsize = 20)
    ColorSpan(PeriodPoint1, ColorId, ax2)

    ColorId = ['#CDF0EA', '#FEFAEC']

    ax3.plot(T1, TorqueReal1[:, 0], label='m2 = 0.4')
    ax3.plot(T2, TorqueReal2[:, 0], label='m2 = 0.6')
    ax3.plot(T3, TorqueReal3[:, 0], label='m2 = 0.8')
    ax3.plot(T4, TorqueReal4[:, 0], label='m2 = 1.0')

    ColorSpan(PeriodPoint1, ColorId, ax3)

    N_EndForce = TorqueReal1[:, 0]
    ax3.axis([0, max(T1)*1.05, min(N_EndForce)*1.1, max(TorqueBar1[:, 0])*0.5])
    ax3.set_xlabel('time (s)', fontsize = 15)
    ax3.set_ylabel('Joint Torque', fontsize = 15)
    ax3.legend(loc='lower right', fontsize = 12)

    ax4.plot(T1, TorqueReal1[:, 1], label='m2 = 0.4')
    ax4.plot(T2, TorqueReal2[:, 1], label='m2 = 0.6')
    ax4.plot(T3, TorqueReal3[:, 1], label='m2 = 0.8')
    ax4.plot(T4, TorqueReal4[:, 1], label='m2 = 1.0')

    P_Torque = TorqueReal1[:, 1]

    ax4.axis([0, max(T1)*1.05, min(P_Torque)* 1.2, max(P_Torque)* 1.2])
    ax4.set_xlabel('time (s)', fontsize = 15)
    # ax4.set_ylabel('Torque (N.m)', fontsize = 15)
    ax4.legend(loc='lower right', fontsize = 12)
    # y_major_locator=MultipleLocator(1.5)
    # ax4.yaxis.set_major_locator(y_major_locator)
    ColorSpan(PeriodPoint1, ColorId, ax4)

    plt.show()

def TorPartPlot(Data):

    TorqueBarPart = Data["TorqueBarPart"]
    PeriodPoint = Data["PeriodPoint"]
    TorqueBar = Data["TorqueBar"]
    T = Data["T"]

    fig, axes = plt.subplots(1,2, dpi=100,figsize=(15,7))
    ax1=axes[0]
    ax2=axes[1]
    
    # ColorId = ['#CDF0EA', '#FEFAEC']
    # ColorId = ['#DEEDF0', '#FFF5EB']
    # ColorId = ['#C5ECBE', '#FFEBBB']
    # ColorId = ['#A7D7C5', '#F7F4E3']
    ColorId = ['#E1F2FB', '#FFF5EB']

    TorqueBar_1 = TorqueBar[:, 0] + 0.5 * (max(TorqueBar[:, 0] - min(TorqueBar[:, 0]))) - max(TorqueBar[:, 0])
    Torque_M_1 = TorqueBarPart[:, 0] + 0.5 * (max(TorqueBarPart[:, 0] - min(TorqueBarPart[:, 0]))) - max(TorqueBarPart[:, 0])
    ax1.plot(T, TorqueBar[:, 0] + 0.5 * (max(TorqueBar[:, 0] - min(TorqueBar[:, 0]))) - max(TorqueBar[:, 0]) , label='Total Torque')
    ax1.plot(T, TorqueBarPart[:, 0] + 0.5 * (max(TorqueBarPart[:, 0] - min(TorqueBarPart[:, 0]))) - max(TorqueBarPart[:, 0]), label='Interia Force M11')
    ax1.plot(T, TorqueBarPart[:, 1] + 0.5 * (max(TorqueBarPart[:, 1] - min(TorqueBarPart[:, 1]))) - max(TorqueBarPart[:, 1]), label='Interia Force M12')
    ax1.plot(T, TorqueBarPart[:, 4] + 0.5 * (max(TorqueBarPart[:, 4] - min(TorqueBarPart[:, 4]))) - max(TorqueBarPart[:, 4]), label='Coriolis-Centrifugal force')
    ax1.plot(T, TorqueBarPart[:, 6] + 0.5 * (max(TorqueBarPart[:, 6] - min(TorqueBarPart[:, 6]))) - max(TorqueBarPart[:, 6]), label='Gravity')
    
    # plt.plot(T, line2, label='highest Velocity')
    N_Torq1 = Torque_M_1
    P_Torq1 = Torque_M_1
    ax1.axis([0, max(T), min(N_Torq1)*1.1, max(P_Torq1)*1.1])
    ax1.tick_params(labelsize = 20)
    ax1.set_xlabel('time (s)', fontsize = 25)
    ax1.set_ylabel('Joint Torque', fontsize = 25)
    ax1.legend(loc='lower right', fontsize = 12)
    y_major_locator=MultipleLocator(25)
    # ax1.yaxis.set_major_locator(y_major_locator)
    ax1.set_title("Joint 1", fontsize = 25)
    ColorSpan(PeriodPoint, ColorId, ax1)


    Torque_M_2 = TorqueBarPart[:, 3] + 0.5 * (max(TorqueBarPart[:, 3] - min(TorqueBarPart[:, 3]))) - max(TorqueBarPart[:, 3])
    ax2.plot(T, TorqueBar[:, 1] + 0.5 * (max(TorqueBar[:, 1] - min(TorqueBar[:, 1]))) - max(TorqueBar[:, 1]), label='Total Torque')
    ax2.plot(T, TorqueBarPart[:, 2] + 0.5 * (max(TorqueBarPart[:, 2] - min(TorqueBarPart[:, 2]))) - max(TorqueBarPart[:, 2]), label='Interia Force M21')
    ax2.plot(T, TorqueBarPart[:, 3] + 0.5 * (max(TorqueBarPart[:, 3] - min(TorqueBarPart[:, 3]))) - max(TorqueBarPart[:, 3]), label='Interia Force M22')
    ax2.plot(T, TorqueBarPart[:, 5] + 0.5 * (max(TorqueBarPart[:, 5] - min(TorqueBarPart[:, 5]))) - max(TorqueBarPart[:, 5]), label='Coriolis-Centrifugal force')
    ax2.plot(T, TorqueBarPart[:, 7] + 0.5 * (max(TorqueBarPart[:, 7] - min(TorqueBarPart[:, 7]))) - max(TorqueBarPart[:, 7]), label='Gravity')
    
    # plt.plot(T, line2, label='highest Velocity')
    N_Torq2 = Torque_M_2
    P_Torq2 = Torque_M_2
    ax2.axis([0, max(T), min(N_Torq2)*1.1, max(P_Torq2)*1.1], fontsize = 25)
    ax2.tick_params(labelsize = 20)
    ax2.set_xlabel('time (s)', fontsize = 25)
    # ax2.set_ylabel('Angular Velocity ', fontsize = 15)
    y_major_locator=MultipleLocator(10)
    # ax2.yaxis.set_major_locator(y_major_locator)
    ax2.legend(loc='lower right', fontsize = 12)
    ax2.set_title("Joint 2", fontsize = 25)
    ColorSpan(PeriodPoint, ColorId, ax2)

    plt.show()

if __name__ == "__main__":

    # Tor1, Tor2 = DynamicsEquation()
    flag = 1
    if flag == 0:
        Data, K, M, D, m2, T_Period = DynamicsAnalysis()
        FileSave(Data, K, M, D, m2, T_Period)
        TorPartPlot(Data)
    elif flag == 1:
        f1 = open(os.path.abspath(os.path.dirname(__file__)) + \
            '/data/2022-03-08/2022-03-08-M_2-K_1-D_1.0-m2_0.6-T_0.5.pkl','rb')
        Data = pickle.load(f1)
        TorPartPlot(Data)

    elif flag == 2:
        f1 = open(os.path.abspath(os.path.dirname(__file__)) + \
            '/data/2022-03-07/2022-03-07-M_2-K_1-D_1.0-m2_0.4.pkl','rb')
        f2 = open(os.path.abspath(os.path.dirname(__file__)) + \
            '/data/2022-03-07/2022-03-07-M_2-K_1-D_1.0-m2_0.6.pkl','rb')
        f3 = open(os.path.abspath(os.path.dirname(__file__)) + \
            '/data/2022-03-07/2022-03-07-M_2-K_1-D_1.0-m2_0.8.pkl','rb')
        f4 = open(os.path.abspath(os.path.dirname(__file__)) + \
            '/data/2022-03-07/2022-03-07-M_2-K_1-D_1.0-m2_1.0.pkl','rb')

        # f1 = open(os.path.abspath(os.path.dirname(__file__)) + \
        #     '/data/2022-03-07/2022-03-07-M_2-K_0.67-D_1.0-m2_0.6.pkl','rb')
        # f2 = open(os.path.abspath(os.path.dirname(__file__)) + \
        #     '/data/2022-03-07/2022-03-07-M_2-K_1-D_1.0-m2_0.6.pkl','rb')
        # f3 = open(os.path.abspath(os.path.dirname(__file__)) + \
        #     '/data/2022-03-07/2022-03-07-M_2-K_1.2-D_1.0-m2_0.6.pkl','rb')
        # f4 = open(os.path.abspath(os.path.dirname(__file__)) + \
        #     '/data/2022-03-07/2022-03-07-M_2-K_1.5-D_1.0-m2_0.6.pkl','rb')

        # f1 = open(os.path.abspath(os.path.dirname(__file__)) + \
        #     '/data/2022-03-07/2022-03-07-M_0.5-K_1-D_1.0-m2_0.6.pkl','rb')
        # f2 = open(os.path.abspath(os.path.dirname(__file__)) + \
        #     '/data/2022-03-07/2022-03-07-M_1-K_1-D_1.0-m2_0.6.pkl','rb')
        # f3 = open(os.path.abspath(os.path.dirname(__file__)) + \
        #     '/data/2022-03-07/2022-03-07-M_2-K_1-D_1.0-m2_0.6.pkl','rb')
        # f4 = open(os.path.abspath(os.path.dirname(__file__)) + \
        #     '/data/2022-03-07/2022-03-07-M_3-K_1-D_1.0-m2_0.6.pkl','rb')
        Data1 = pickle.load(f1)
        Data2 = pickle.load(f2)
        Data3 = pickle.load(f3)
        Data4 = pickle.load(f4)

        DataPlot(Data1, Data2, Data3, Data4)