import numpy as np
import math
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
import yaml
from matplotlib.collections import LineCollection

def phaseplot(x, dx, flag):
    dydx = np.cos(0.05 * (x[:-1] + x[1:]))  # first derivative
    print(dydx.shape)
    points = np.array([x, dx]).T.reshape(-1, 1, 2)
    temp = np.array([x, dx])
    # print("temp: ", temp)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # print("point: ",points)
    # print("segment: ", segments)
    # print(segments.shape)
    fig, axs = plt.subplots(1, 1)
    if flag == 1:
        fig.suptitle('Hip Joint Torque-Velocity phase space', fontsize = 20)

    elif flag == 2:
        fig.suptitle('Knee Joint Torque-Velocity phase space', fontsize = 20)
    norm = plt.Normalize(dydx.min(), dydx.max())  
    print(type(norm))
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)

    # axs.broken_barh([(-0.6, 0.13), (-0.47, 0.37), (-0.1, 0.2)], (-8, 16),
    #                facecolors=( '#f4c7ab', '#deedf0', '#fff5eb'))

    # axs.broken_barh([(-0.6, 0.13), (-0.47, 0.37), (-0.1, 1)], (-20, 40),
    #                facecolors=( '#fffbdf', '#c6ffc1', '#34656d'))

    xlim_min = 1.2 * min(x)
    xlim_max = 1.2 * max(x)
    ylim_min = 1.2 * min(dx)
    ylim_max = 1.2 * max(dx)
    axs.set_xlim(xlim_min, xlim_max)
    axs.set_ylim(-5, ylim_max)
    # axs.set_xlim(-10, 8)
    # axs.set_ylim(-5, 15)
    plt.xlabel('Joint Velocity (rad/s)')
    plt.ylabel('Joint Torque (N.m)')

def ScatterPhaseplot(x, dx, flag):
    norm = plt.Normalize(dx.min(), dx.max())
    norm_y = norm(dx)
    plt.figure()
    # fig, axs = plt.subplots(1, 1)
    plt.scatter(x, dx, c=norm_y, cmap='viridis')
    if flag == 1:
        plt.title('Hip Joint Torque-Velocity phase space', fontsize = 20)
    elif flag == 2:
        plt.title('Knee Joint Torque-Velocity phase space', fontsize = 20)

    xlim_min = 1.2 * min(x)
    xlim_max = 1.2 * max(x)
    ylim_min = 1.2 * min(dx)
    ylim_max = 1.2 * max(dx)
    # axs.set_xlim(xlim_min, xlim_max)
    # axs.set_ylim(-5, ylim_max)
    plt.axis([xlim_min, xlim_max, -5, ylim_max])
    plt.xlabel('Joint Velocity (rad/s)')
    plt.ylabel('Joint Torque (N.m)')

def Force_TPlot(ParamData, T):
    Params_1 = -ParamData[:, 0]
    Params_2 = ParamData[:, 1]

    plt.plot(T, Params_1, label='Designed Force')
    plt.plot(T, Params_2, label='Contact Force')

    plt.axis([0, max(T)*1.05, -max(Params_1)*0.5, max(Params_1)*1.5])
    plt.xlabel('time (s)')
    plt.ylabel('Contact force (N)')
    plt.legend(loc='upper right')

def State_TPlot(ParamData, Params_1, Params_2, T, flag):

    if flag == 1:
        line = np.ones([len(T)])
        line1 = ParamData["controller"]["v_ref"] * line
        # line2 = 4 * line
        plt.plot(T, Params_1, label='Ball Velocity')
        plt.plot(T, Params_2, label='Foot Velocity')
        plt.plot(T, line1, label='highest Velocity')
        # plt.plot(T, line2, label='highest Velocity')

        plt.axis([0, max(T)*1.05, -max(Params_1)*2, max(Params_1)*2])
        plt.xlabel('time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend(loc='upper right')

    else:
        line = np.ones([len(T)])
        line = 0.56 * line
        print(len(line))
        plt.plot(T, Params_1, label='Ball Position')
        plt.plot(T, Params_2, label='Foot Position')
        # plt.plot(T, line, label='highest Position')

        plt.axis([0, max(T)*1.05, -0.2, max(Params_1)*1.8])
        plt.xlabel('time (s)')
        plt.ylabel('Position (m)')
        plt.legend(loc='lower right')


def DataProcess(data):
    matplotlib.rcParams['font.size'] = 18
    matplotlib.rcParams['lines.linewidth'] = 2

    FilePath = os.path.abspath(os.getcwd())
    ParamFilePath = FilePath + "/config/LISM_test.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    JointVel_1 = data['JointVel'][:, 0]
    JointVel_2 = data['JointVel'][:, 1]
    JointTorque_1 = data['JointTorque'][:, 0]
    JointTorque_2 = data['JointTorque'][:, 1]

    BallState = data['BallState']
    EndFootState = data['EndFootState']


    JointVel_1 = np.array(JointVel_1.flatten())
    JointVel_2 = np.array(JointVel_2.flatten())
    JointTorque_1 = np.array(JointTorque_1.flatten())
    JointTorque_2 = np.array(JointTorque_2.flatten())

    ForceState = data['ForceState']
    T = data['time']
    print(len(ForceState[:, 1]))
    for i in range(0, len(ForceState[:, 1])):
        if ForceState[i, 1] > 1000:
            ForceState[i, 1] = ForceState[i + 1, 1] - (ForceState[i + 2, 1] - ForceState[i + 1, 1])

    print(JointTorque_1.shape)
    print(JointTorque_2.shape)
    # print(Ballvel.shape)
    # print(JointTorque_1[0:1000])


    # ============================================ data visualization ===============================================
    # plot joint torque-velocity figure
    ScatterPhaseplot(JointVel_1, JointTorque_1, 1)
    ScatterPhaseplot(JointVel_2, JointTorque_2, 2)

    # phaseplot(JointVel_1, JointTorque_1, 1)
    # phaseplot(JointVel_2, JointTorque_2, 2)    

    ## plot Ball and foot state figure
    plt.figure()

    # plot Ball and foot velocity figure
    plt.subplot(311)
    State_TPlot(ParamData, BallState[:, 1], EndFootState[:, 1], T, 1)

    # plot Ball and foot position figure
    plt.subplot(312)
    State_TPlot(ParamData, BallState[:, 0], EndFootState[:, 0], T, 0)

    # plot contact force and designed force figure
    plt.subplot(313)
    # print(ForceState[0:1000, 1])
    Force_TPlot(ForceState, T)

    plt.show()

if __name__ == "__main__":
    f = open('./data/2021-07-12-x_ref_0.4-v0_8-vref_-10-dx_0.2-f1_15.0.pkl','rb')
    data = pickle.load(f)
    DataProcess(data)