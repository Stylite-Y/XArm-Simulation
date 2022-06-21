from cProfile import label
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy import signal
from DataProcess import DataProcess

def AnaTest():
    ## Trunk, Uarm, Farm, ULeg, Lleg
    R = [0.1, 0.23, 0.035, 0.03, 0.07, 0.05]
    m = [36.4, 1.82, 1.358, 9.8, 3.5]
    l = [0.813, 0.3, 0.375, 0.406, 0.5075]

    I_t = np.diag([m[0]*(R[1]**2+R[0]**2)/12, m[0]*(R[1]**2+R[0]**2)/12, m[0]*(R[0]**2+l[0]**2)/12])
    I_ua = np.diag([m[1]*(3*R[2]**2+l[1]**2)/12, m[1]*R[2]**2/2, m[1]*(3*R[2]**2+l[1]**2)/12])
    I_fa = np.diag([m[2]*(3*R[3]**2+l[2]**2)/12, m[2]*R[3]**2/2, m[2]*(3*R[3]**2+l[2]**2)/12])
    I_ul = np.diag([m[3]*(3*R[4]**2+l[3]**2)/12, m[3]*R[4]**2/2, m[3]*(3*R[4]**2+l[3]**2)/12])
    I_ll = np.diag([m[4]*(3*R[5]**2+l[4]**2)/12, m[4]*R[5]**2/2, m[4]*(3*R[5]**2+l[4]**2)/12])

    filepath = os.path.dirname(os.path.dirname(__file__)) + '/data/walk_bvhlocal2.calc'
    print(filepath)

    # read the data from document
    Data = []
    with open(filepath, 'r') as f:
        for i in range(6):
            next(f)
        for lines in f.readlines():
            Data.append(list(map(float, lines.split())))
    a = Data[0][0:3]
    Data = np.asarray(Data)
    num = len(Data)
    t = np.linspace(0, 400, 400)

    RArmMomt = np.array([[0.0, 0.0, 0.0]])
    LArmMomt = np.array([[0.0, 0.0, 0.0]])
    RLegMomt = np.array([[0.0, 0.0, 0.0]])
    LLegMomt = np.array([[0.0, 0.0, 0.0]])
    TrunkMomt = np.array([[0.0, 0.0, 0.0]])
    for i in range(300, 700):
        ## right arm
        # upper arm
        index = 128
        RUArmTemp = I_ua @ Data[i][(index+13):(index+16)] + m[1]*np.cross(Data[i][(index):(index+3)], Data[i][(index+3):(index+6)])
        # forearn arm
        index = 144
        RFArmTemp = I_fa @ Data[i][(index+13):(index+16)] + m[2]*np.cross(Data[i][(index):(index+3)], Data[i][(index+3):(index+6)])
        RArmMomt=np.concatenate((RArmMomt, [RUArmTemp+RFArmTemp]), axis=0)

        ## left arm
        # upper arm
        index = 192
        LUArmMomt = I_ua @ Data[i][(index+13):(index+16)] + m[1]*np.cross(Data[i][(index):(index+3)], Data[i][(index+3):(index+6)])
        # forearn arm
        index = 208
        LFArmMomt = I_fa @ Data[i][(index+13):(index+16)] + m[2]*np.cross(Data[i][(index):(index+3)], Data[i][(index+3):(index+6)])
        LArmMomt=np.concatenate((LArmMomt, [LUArmMomt+LFArmMomt]), axis=0)

        ## right leg
        # upper leg
        index = 16
        RULegMomt = I_ul @ Data[i][(index+13):(index+16)] + m[3]*np.cross(Data[i][(index):(index+3)], Data[i][(index+3):(index+6)])
        # lower leg
        index = 32
        RLLegMomt = I_ll @ Data[i][(index+13):(index+16)] + m[4]*np.cross(Data[i][(index):(index+3)], Data[i][(index+3):(index+6)])
        RLegMomt=np.concatenate((RLegMomt, [RULegMomt+RLLegMomt]), axis=0)

        ## left leg
        # upper leg
        index = 64
        LULegMomt = I_ul @ Data[i][(index+13):(index+16)] + m[3]*np.cross(Data[i][(index):(index+3)], Data[i][(index+3):(index+6)])
        # lower leg
        index = 80
        LLLegMomt = I_ll @ Data[i][(index+13):(index+16)] + m[4]*np.cross(Data[i][(index):(index+3)], Data[i][(index+3):(index+6)])
        LLegMomt=np.concatenate((LLegMomt, [LULegMomt+LLLegMomt]), axis=0)

        ## trunk
        index = 0
        TrunkMomtTemp = I_t @ Data[i][(index+13):(index+16)] + m[3]*np.cross(Data[i][(index):(index+3)], Data[i][(index+3):(index+6)])
        TrunkMomt=np.concatenate((TrunkMomt,[TrunkMomtTemp]), axis=0)
        pass

    BodyMomt = LLegMomt + RLegMomt + TrunkMomt
    LegMomt = LLegMomt + RLegMomt

    BodyMomt = BodyMomt[1:,]
    RArmMomt = RArmMomt[1:,]
    LArmMomt = LArmMomt[1:,]
    ArmMomt = RArmMomt + LArmMomt
    TrunkMomt = TrunkMomt[1:,]
    LegMomt = LegMomt[1:,]

    fig, axes = plt.subplots(4,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]

    # ax1.plot(t, Data[300:700, 192])
    # ax1.plot(t, Data[300:700, 193])
    # ax1.plot(t, Data[300:700, 194])

    ax1.plot(t, BodyMomt[:, 0], label = "body")
    ax1.plot(t, RArmMomt[:, 0], label = "Right Arm")
    ax1.plot(t, LArmMomt[:, 0], label = "Left Arm")
    ax1.plot(t, ArmMomt[:, 0], label = "All Arm")
    ax1.legend()
    ax1.grid()

    ax2.plot(t, BodyMomt[:, 1], label = "body")
    ax2.plot(t, RArmMomt[:, 1], label = "Right Arm")
    ax2.plot(t, LArmMomt[:, 1], label = "Left Arm")
    ax2.plot(t, ArmMomt[:, 1], label = "All Arm")
    ax2.legend()
    ax2.grid()

    ax3.plot(t, BodyMomt[:, 2], label = "body")
    ax3.plot(t, RArmMomt[:, 2], label = "Right Arm")
    ax3.plot(t, LArmMomt[:, 2], label = "Left Arm")
    ax3.plot(t, ArmMomt[:, 2], label = "All Arm")
    ax3.legend()
    ax3.grid()

    ax4.plot(Data[0:1500, 0], label="x")
    ax4.plot(Data[0:1500, 1], label="y")
    ax4.plot(Data[0:1500, 2], label="z")
    ax4.legend()
    ax4.grid()
    plt.show()

# def LocalAxis(Data, m, time, i):
#     Pos_lh = np.array([[0.0, 0.0, 0.0]])
#     # for n in range(200,1000):
#     # for n in range(200,700):
#     for n in range(time[0], time[1]):
#     # for n in range(2000,2500):
#         QuadHip = np.array([Data[n][(m-1)*16+7], Data[n][(m-1)*16+8], Data[n][(m-1)*16+9], Data[n][(m-1)*16+6]])
#         P_ho = np.array([[Data[n][(m-1)*16], Data[n][(m-1)*16+1], Data[n][(m-1)*16+2]]])
#         RotMatrix = Rotation.from_quat(QuadHip)
#         rotation_m = RotMatrix.as_matrix()
#         P = -rotation_m.T @ P_ho.T
#         Trans_m = np.concatenate((rotation_m.T, P), axis = 1)
#         Trans_m = np.concatenate((Trans_m, np.array([[0, 0, 0, 1]])), axis = 0)
#         P_lo = np.array([[Data[n][(i-1)*16], Data[n][(i-1)*16+1], Data[n][(i-1)*16+2], 1]])
#         P_lh = Trans_m @ P_lo.T
#         if n==500 and i ==2:
#             print(P_ho)
#             print(P_lo)
#             print(QuadHip)
#             print(rotation_m)
#             print(P_lh)
#         P_lh = P_lh[0:3].T

#         Pos_lh = np.concatenate((Pos_lh, P_lh), axis = 0)
#     return Pos_lh

def LocalAxis(Data, rm0, m, time, i):
    Pos_lh = np.array([[0.0, 0.0, 0.0]])
    for n in range(time[0], time[1]):
        P_ho = np.array([Data[n][(m-1)*16], Data[n][(m-1)*16+1], Data[n][(m-1)*16+2]])
        P_lo = np.array([Data[n][(i-1)*16], Data[n][(i-1)*16+1], Data[n][(i-1)*16+2]])
        Pos_res = P_lo - P_ho
        Pos_res = rm0.T @ Pos_res
        Pos_lh = np.concatenate((Pos_lh, [Pos_res]), axis = 0)
        if n == 0 and i == 7:
            print(Pos_res)    
    return Pos_lh

def AnaBVH():
    # region: inertia params
    ## Trunk, Uarm, Farm, ULeg, Lleg
    R = [0.1, 0.23, 0.035, 0.03, 0.07, 0.05]
    m = [36.4, 1.82, 1.358, 9.8, 3.5]
    l = [0.813, 0.3, 0.375, 0.406, 0.5075]

    ## zyx
    # I_t = np.diag([m[0]*(R[1]**2+R[0]**2)/12, m[0]*(R[0]**2+l[0]**2)/12, m[0]*(R[1]**2+l[0]**2)/12])
    # I_ua = np.diag([m[1]*R[2]**2/2, m[1]*(3*R[2]**2+l[1]**2)/12, m[1]*(3*R[2]**2+l[1]**2)/12])
    # I_fa = np.diag([m[2]*R[3]**2/2, m[2]*(3*R[3]**2+l[2]**2)/12, m[2]*(3*R[3]**2+l[2]**2)/12])
    # I_ul = np.diag([m[3]*R[4]**2/2, m[3]*(3*R[4]**2+l[3]**2)/12, m[3]*(3*R[4]**2+l[3]**2)/12])
    # I_ll = np.diag([m[4]*R[5]**2/2, m[4]*(3*R[5]**2+l[4]**2)/12, m[4]*(3*R[5]**2+l[4]**2)/12])

    ## xyz
    I_t = np.diag([m[0]*(R[1]**2+l[0]**2)/12, m[0]*(R[0]**2+l[0]**2)/12, m[0]*(R[1]**2+R[0]**2)/12])
    I_ua = np.diag([m[1]*(3*R[2]**2+l[1]**2)/12, m[1]*(3*R[2]**2+l[1]**2)/12, m[1]*R[2]**2/2])
    I_fa = np.diag([m[2]*(3*R[3]**2+l[2]**2)/12, m[2]*(3*R[3]**2+l[2]**2)/12, m[2]*R[3]**2/2])
    I_ul = np.diag([m[3]*(3*R[4]**2+l[3]**2)/12, m[3]*(3*R[4]**2+l[3]**2)/12, m[3]*R[4]**2/2])
    I_ll = np.diag([m[4]*(3*R[5]**2+l[4]**2)/12, m[4]*(3*R[5]**2+l[4]**2)/12, m[4]*R[5]**2/2])

    ## yxz
    # I_t = np.diag([m[0]*(R[0]**2+l[0]**2)/12, m[0]*(R[1]**2+l[0]**2)/12, m[0]*(R[1]**2+R[0]**2)/12])
    # I_ua = np.diag([m[1]*R[2]**2/2, m[1]*(3*R[2]**2+l[1]**2)/12, m[1]*(3*R[2]**2+l[1]**2)/12])
    # I_fa = np.diag([m[2]*R[3]**2/2, m[2]*(3*R[3]**2+l[2]**2)/12, m[2]*(3*R[3]**2+l[2]**2)/12])
    # I_ul = np.diag([m[3]*(3*R[4]**2+l[3]**2)/12, m[3]*(3*R[4]**2+l[3]**2)/12, m[3]*R[4]**2/2])
    # I_ll = np.diag([m[4]*(3*R[5]**2+l[4]**2)/12, m[4]*(3*R[5]**2+l[4]**2)/12, m[4]*R[5]**2/2])
    # endregion

    # filepath = os.path.dirname(os.path.dirname(__file__)) + '/data/Calibra_Walk_yuan1-4500.calc'
    # filepath = os.path.dirname(os.path.dirname(__file__)) + '/data/Calibra_yuan-4000.calc'
    # filepath = os.path.dirname(os.path.dirname(__file__)) + '/data/Calibra_Walk_yuan4.calc'
    filepath = os.path.dirname(os.path.dirname(__file__)) + '/data/yuan/walk_arm_yuan.calc'
    # filepath = os.path.dirname(os.path.dirname(__file__)) + '/data/yuan/walk_noarm_yuan.calc'
    # filepath = os.path.dirname(os.path.dirname(__file__)) + '/data/yuan/walk_yuan2.calc'

    Data = []
    with open(filepath, 'r') as f:
        for i in range(6):
            next(f)
        for lines in f.readlines():
            Data.append(list(map(float, lines.split())))
    Data = np.asarray(Data)

    # time = [3000, 4000] # walk_yuan2
    time = [0, 4000] # celabri yuan1
    RefIndex = 7
    # walk yuan3 L
    # index = [184, 248]
    # index = [910, 980]

    # walk arm yuan
    # index = [611, 671]      # R
    # ticks = ['Right 1']
    index = [959, 1032]      # L
    ticks = ['Left 1']

    t = np.linspace(time[0], time[1]-1, time[1]-time[0]-1)
    tt = np.linspace(time[0], time[1], time[1]-time[0])
    Anidata = Data[3000:4000, :]
    # TestAni = DataProcess()
    # TestAni.animation(Anidata, t)

    # orientation matrix
    V_ore = Data[100, (11*16):(11*16+3)] - Data[100, (7*16):(7*16+3)]
    Ang_ore = np.arctan(np.abs(V_ore[0] / V_ore[1]))
    r = Rotation.from_euler('xyz', [0, 0, Ang_ore])
    quad0 = r.as_quat()
    rm0 = r.as_matrix()
    print(V_ore, Ang_ore, quad0)
    print(rm0)

    # region: 四元数到旋转矩阵
    P_lh1 = LocalAxis(Data, rm0, RefIndex, time, 1)
    P_lh2 = LocalAxis(Data, rm0, RefIndex, time, 2)
    P_lh3 = LocalAxis(Data, rm0, RefIndex, time, 3)
    P_lh4 = LocalAxis(Data, rm0, RefIndex, time, 4)
    P_lh5 = LocalAxis(Data, rm0, RefIndex, time, 5)
    P_lh6 = LocalAxis(Data, rm0, RefIndex, time, 6)
    P_lh7 = LocalAxis(Data, rm0, RefIndex, time, 7)
    P_lh8 = LocalAxis(Data, rm0, RefIndex, time, 8)
    P_lh9 = LocalAxis(Data, rm0, RefIndex, time, 9)
    P_lh10 = LocalAxis(Data, rm0, RefIndex, time, 10)
    P_lh11 = LocalAxis(Data, rm0, RefIndex, time, 11)
    P_lh12 = LocalAxis(Data, rm0, RefIndex, time, 12)
    P_lh13 = LocalAxis(Data, rm0, RefIndex, time, 13)
    P_lh14 = LocalAxis(Data, rm0, RefIndex, time, 14)
    P_lh15 = LocalAxis(Data, rm0, RefIndex, time, 15)
    P_lh16 = LocalAxis(Data, rm0, RefIndex, time, 16)
    P_lh17 = LocalAxis(Data,rm0,  RefIndex, time, 17)

    P_lh1 = P_lh1[1:,]
    P_lh2 = P_lh2[1:,]
    P_lh3 = P_lh3[1:,]
    P_lh4 = P_lh4[1:,]
    P_lh5 = P_lh5[1:,]
    P_lh6 = P_lh6[1:,]
    P_lh7 = P_lh7[1:,]
    P_lh8 = P_lh8[1:,]
    P_lh9 = P_lh9[1:,]
    P_lh10 = P_lh10[1:,]
    P_lh11 = P_lh11[1:,]
    P_lh12 = P_lh12[1:,]
    P_lh13 = P_lh13[1:,]
    P_lh14 = P_lh14[1:,]
    P_lh15 = P_lh15[1:,]
    P_lh16 = P_lh16[1:,]
    P_lh17 = P_lh17[1:,]
    P_lh1_17 = (P_lh1 + P_lh17)/2
    # endregion

    # region: Moment calculate
    # region: matrix
    RArmMomt = np.array([[0.0, 0.0, 0.0]])
    LArmMomt = np.array([[0.0, 0.0, 0.0]])
    RLegMomt = np.array([[0.0, 0.0, 0.0]])
    LLegMomt = np.array([[0.0, 0.0, 0.0]])
    TrunkMomt = np.array([[0.0, 0.0, 0.0]])

    ORArmMomt = np.array([[0.0, 0.0, 0.0]])
    OLArmMomt = np.array([[0.0, 0.0, 0.0]])
    ORLegMomt = np.array([[0.0, 0.0, 0.0]])
    OLLegMomt = np.array([[0.0, 0.0, 0.0]])
    OTrunkMomt = np.array([[0.0, 0.0, 0.0]])

    VRArmMomt = np.array([[0.0, 0.0, 0.0]])
    VLArmMomt = np.array([[0.0, 0.0, 0.0]])
    VRLegMomt = np.array([[0.0, 0.0, 0.0]])
    VLLegMomt = np.array([[0.0, 0.0, 0.0]])
    VTrunkMomt = np.array([[0.0, 0.0, 0.0]])

    V_lh1_m = np.array([[0.0, 0.0, 0.0]])
    V_lh2_m = np.array([[0.0, 0.0, 0.0]])
    V_lh3_m = np.array([[0.0, 0.0, 0.0]])
    V_lh4_m = np.array([[0.0, 0.0, 0.0]])
    V_lh5_m = np.array([[0.0, 0.0, 0.0]])
    V_lh6_m = np.array([[0.0, 0.0, 0.0]])
    V_lh7_m = np.array([[0.0, 0.0, 0.0]])
    V_lh8_m = np.array([[0.0, 0.0, 0.0]])
    V_lh9_m = np.array([[0.0, 0.0, 0.0]])
    V_lh10_m = np.array([[0.0, 0.0, 0.0]])
    V_lh11_m = np.array([[0.0, 0.0, 0.0]])
    V_lh12_m = np.array([[0.0, 0.0, 0.0]])
    V_lh13_m = np.array([[0.0, 0.0, 0.0]])
    V_lh14_m = np.array([[0.0, 0.0, 0.0]])
    V_lh15_m = np.array([[0.0, 0.0, 0.0]])
    V_lh16_m = np.array([[0.0, 0.0, 0.0]])
    V_lh17_m = np.array([[0.0, 0.0, 0.0]])

    A_lh1_m = np.array([[0.0, 0.0, 0.0]])
    A_lh2_m = np.array([[0.0, 0.0, 0.0]])
    A_lh3_m = np.array([[0.0, 0.0, 0.0]])
    A_lh5_m = np.array([[0.0, 0.0, 0.0]])
    A_lh6_m = np.array([[0.0, 0.0, 0.0]])
    A_lh9_m = np.array([[0.0, 0.0, 0.0]])
    A_lh10_m = np.array([[0.0, 0.0, 0.0]])
    A_lh13_m = np.array([[0.0, 0.0, 0.0]])
    A_lh14_m = np.array([[0.0, 0.0, 0.0]])

    F_RS_m = np.array([[0.0, 0.0, 0.0]])
    F_LS_m = np.array([[0.0, 0.0, 0.0]])
    F_RL_m = np.array([[0.0, 0.0, 0.0]])
    F_LL_m = np.array([[0.0, 0.0, 0.0]])

    # endregion
    for i in range(len(P_lh2)-1):
        dt = 1 / 120
        # region: vel
        V_lh1 = (P_lh1[i+1,] - P_lh1[i,]) / dt
        V_lh2 = (P_lh2[i+1,] - P_lh2[i,]) / dt
        V_lh3 = (P_lh3[i+1,] - P_lh3[i,]) / dt
        V_lh4 = (P_lh4[i+1,] - P_lh4[i,]) / dt
        V_lh5 = (P_lh5[i+1,] - P_lh5[i,]) / dt
        V_lh6 = (P_lh6[i+1,] - P_lh6[i,]) / dt
        V_lh7 = (P_lh7[i+1,] - P_lh7[i,]) / dt
        V_lh8 = (P_lh8[i+1,] - P_lh8[i,]) / dt
        V_lh9 = (P_lh9[i+1,] - P_lh9[i,]) / dt
        V_lh10 = (P_lh10[i+1,] - P_lh10[i,]) / dt
        V_lh11 = (P_lh11[i+1,] - P_lh11[i,]) / dt
        V_lh12 = (P_lh12[i+1,] - P_lh12[i,]) / dt
        V_lh13 = (P_lh13[i+1,] - P_lh13[i,]) / dt
        V_lh14 = (P_lh14[i+1,] - P_lh14[i,]) / dt
        V_lh15 = (P_lh15[i+1,] - P_lh15[i,]) / dt
        V_lh16 = (P_lh16[i+1,] - P_lh16[i,]) / dt
        V_lh17 = (P_lh17[i+1,] - P_lh17[i,]) / dt
        V_lh1_17 = (P_lh1_17[i+1,]-P_lh1_17[i,]) / dt

        V_lh1_m = np.concatenate((V_lh1_m, [V_lh1]), axis = 0)
        V_lh2_m = np.concatenate((V_lh2_m, [V_lh2]), axis = 0)
        V_lh3_m = np.concatenate((V_lh3_m, [V_lh3]), axis = 0)
        V_lh4_m = np.concatenate((V_lh4_m, [V_lh4]), axis = 0)
        V_lh5_m = np.concatenate((V_lh5_m, [V_lh5]), axis = 0)
        V_lh6_m = np.concatenate((V_lh6_m, [V_lh6]), axis = 0)
        V_lh7_m = np.concatenate((V_lh7_m, [V_lh7]), axis = 0)
        V_lh8_m = np.concatenate((V_lh8_m, [V_lh8]), axis = 0)
        V_lh9_m = np.concatenate((V_lh9_m, [V_lh9]), axis = 0)
        V_lh10_m = np.concatenate((V_lh10_m, [V_lh10]), axis = 0)
        V_lh11_m = np.concatenate((V_lh11_m, [V_lh11]), axis = 0)
        V_lh12_m = np.concatenate((V_lh12_m, [V_lh12]), axis = 0)
        V_lh13_m = np.concatenate((V_lh13_m, [V_lh13]), axis = 0)
        V_lh14_m = np.concatenate((V_lh14_m, [V_lh14]), axis = 0)
        V_lh15_m = np.concatenate((V_lh15_m, [V_lh15]), axis = 0)
        V_lh16_m = np.concatenate((V_lh16_m, [V_lh16]), axis = 0)
        V_lh17_m = np.concatenate((V_lh17_m, [V_lh17]), axis = 0)

        # endregion

        # Moment17 = I_t @ Data[i][(13):(16)] + m[0]*np.cross(P_lh1[i,:], V_lh1)
        Moment17 = I_t @ Data[i][(13):(16)] + m[0]*np.cross(P_lh1_17[i,:], V_lh1_17)
        OMoment17 = I_t @ Data[i][(13):(16)]
        VMoment17 = m[0]*np.cross(P_lh1_17[i,:], V_lh1_17)
        TrunkMomt = np.concatenate((TrunkMomt, [Moment17]), axis = 0)
        OTrunkMomt = np.concatenate((OTrunkMomt, [OMoment17]), axis = 0)
        VTrunkMomt = np.concatenate((VTrunkMomt, [VMoment17]), axis = 0)

        # right leg
        Moment2_1 = I_ul @ Data[i][(16*1+13):(16*1+16)] + m[3]*np.cross(P_lh2[i,:], V_lh2)
        Moment2_2 = I_ll @ Data[i][(16*2+13):(16*2+16)] + m[4]*np.cross(P_lh3[i,:], V_lh3)
        OMoment2_1 = I_ul @ Data[i][(16*1+13):(16*1+16)]
        OMoment2_2 = I_ll @ Data[i][(16*2+13):(16*2+16)]
        VMoment2_1 = m[3]*np.cross(P_lh2[i,:], V_lh2)
        VMoment2_2 = m[4]*np.cross(P_lh3[i,:], V_lh3)
        Moment2 = Moment2_1 + Moment2_2
        OMoment2 = OMoment2_1 + OMoment2_2
        VMoment2 = VMoment2_1 + VMoment2_2
        RLegMomt = np.concatenate((RLegMomt, [Moment2]), axis = 0)
        ORLegMomt = np.concatenate((ORLegMomt, [OMoment2]), axis = 0)
        VRLegMomt = np.concatenate((VRLegMomt, [VMoment2]), axis = 0)

        # left leg
        Moment3_1 = I_ul @ Data[i][(16*4+13):(16*4+16)] + m[3]*np.cross(P_lh5[i,:], V_lh5)
        Moment3_2 = I_ll @ Data[i][(16*5+13):(16*5+16)] + m[4]*np.cross(P_lh6[i,:], V_lh6)
        OMoment3_1 = I_ul @ Data[i][(16*4+13):(16*4+16)]
        OMoment3_2 = I_ll @ Data[i][(16*5+13):(16*5+16)]
        VMoment3_1 = m[3]*np.cross(P_lh5[i,:], V_lh5)
        VMoment3_2 = m[4]*np.cross(P_lh6[i,:], V_lh6)
        Moment3 = Moment3_1 + Moment3_2
        OMoment3 = OMoment3_1 + OMoment3_2
        VMoment3 = VMoment3_1 + VMoment3_2
        LLegMomt = np.concatenate((LLegMomt, [Moment3]), axis = 0)
        OLLegMomt = np.concatenate((OLLegMomt, [OMoment3]), axis = 0)
        VLLegMomt = np.concatenate((VLLegMomt, [VMoment3]), axis = 0)

        # right arm
        Moment9 = I_ua @ Data[i][(16*8+13):(16*8+16)] + m[1]*np.cross(P_lh9[i,:], V_lh9)
        Moment10 = I_fa @ Data[i][(16*9+13):(16*9+16)] + m[2]*np.cross(P_lh10[i,:], V_lh10)
        OMoment9 = I_ua @ Data[i][(16*8+13):(16*8+16)]
        OMoment10 = I_fa @ Data[i][(16*9+13):(16*9+16)]
        VMoment9 = m[1]*np.cross(P_lh9[i,:], V_lh9)
        VMoment10 = m[2]*np.cross(P_lh10[i,:], V_lh10)
        Moment910 = Moment9 + Moment10
        OMoment910 = OMoment9 + OMoment10
        VMoment910 = VMoment9 + VMoment10
        RArmMomt = np.concatenate((RArmMomt, [Moment910]), axis = 0)
        ORArmMomt = np.concatenate((ORArmMomt, [OMoment910]), axis = 0)
        VRArmMomt = np.concatenate((VRArmMomt, [VMoment910]), axis = 0)

        # left arm
        Moment13 = I_ua @ Data[i][(16*12+13):(16*12+16)] + m[1]*np.cross(P_lh13[i,:], V_lh13)
        Moment14 = I_fa @ Data[i][(16*13+13):(16*13+16)] + m[2]*np.cross(P_lh14[i,:], V_lh14)
        OMoment13 = I_ua @ Data[i][(16*12+13):(16*12+16)]
        OMoment14 = I_fa @ Data[i][(16*13+13):(16*13+16)]
        VMoment13 = m[1]*np.cross(P_lh13[i,:], V_lh13)
        VMoment14 = m[2]*np.cross(P_lh14[i,:], V_lh14)
        Moment1314 = Moment13 + Moment14
        OMoment1314 = OMoment13 + OMoment14
        VMoment1314 = VMoment13 + VMoment14
        LArmMomt = np.concatenate((LArmMomt, [Moment1314]), axis = 0)
        OLArmMomt = np.concatenate((OLArmMomt, [OMoment1314]), axis = 0)
        VLArmMomt = np.concatenate((VLArmMomt, [VMoment1314]), axis = 0)
    
    for k in range(1, len(V_lh1_m)-1):
        dt = 1 / 120
        A_lh1 = (V_lh1_m[k+1,:]-V_lh1_m[k,:])/dt
        A_lh2 = (V_lh2_m[k+1,:]-V_lh2_m[k,:])/dt
        A_lh3 = (V_lh3_m[k+1,:]-V_lh3_m[k,:])/dt
        A_lh5 = (V_lh5_m[k+1,:]-V_lh5_m[k,:])/dt
        A_lh6 = (V_lh6_m[k+1,:]-V_lh6_m[k,:])/dt
        A_lh9 = (V_lh9_m[k+1,:]-V_lh9_m[k,:])/dt
        A_lh10 = (V_lh10_m[k+1,:]-V_lh10_m[k,:])/dt
        A_lh13 = (V_lh13_m[k+1,:]-V_lh13_m[k,:])/dt
        A_lh14 = (V_lh14_m[k+1,:]-V_lh14_m[k,:])/dt

        A_lh1_m = np.concatenate((A_lh1_m, [A_lh1]), axis = 0)
        A_lh2_m = np.concatenate((A_lh2_m, [A_lh2]), axis = 0)
        A_lh3_m = np.concatenate((A_lh3_m, [A_lh3]), axis = 0)
        A_lh5_m = np.concatenate((A_lh5_m, [A_lh5]), axis = 0)
        A_lh6_m = np.concatenate((A_lh6_m, [A_lh6]), axis = 0)
        A_lh9_m = np.concatenate((A_lh9_m, [A_lh9]), axis = 0)
        A_lh10_m = np.concatenate((A_lh10_m, [A_lh10]), axis = 0)
        A_lh13_m = np.concatenate((A_lh13_m, [A_lh13]), axis = 0)
        A_lh14_m = np.concatenate((A_lh14_m, [A_lh14]), axis = 0)

        F_RS = m[1]*A_lh9 + m[2]*A_lh10
        F_LS = m[1]*A_lh13 + m[2]*A_lh14
        F_RL = m[3]*A_lh2 + m[4]*A_lh3
        F_LL = m[3]*A_lh5 + m[4]*A_lh6

        F_RS_m = np.concatenate((F_RS_m, [F_RS]), axis = 0)
        F_LS_m = np.concatenate((F_LS_m, [F_LS]), axis = 0)
        F_RL_m = np.concatenate((F_RL_m, [F_RL]), axis = 0)
        F_LL_m = np.concatenate((F_LL_m, [F_LL]), axis = 0)

        pass

    # endregion
    BodyMomt = LLegMomt + RLegMomt + TrunkMomt
    LegMomt = LLegMomt + RLegMomt
    
    # region: data get
    RArmMomt = RArmMomt[1:,]
    LArmMomt = LArmMomt[1:,]
    LLegMomt = LLegMomt[1:,]
    RLegMomt = RLegMomt[1:,]
    TrunkMomt = TrunkMomt[1:,]

    ORArmMomt = ORArmMomt[1:,]
    OLArmMomt = OLArmMomt[1:,]
    OLLegMomt = OLLegMomt[1:,]
    ORLegMomt = ORLegMomt[1:,]
    OTrunkMomt = OTrunkMomt[1:,]

    VRArmMomt = VRArmMomt[1:,]
    VLArmMomt = VLArmMomt[1:,]
    VLLegMomt = VLLegMomt[1:,]
    VRLegMomt = VRLegMomt[1:,]
    VTrunkMomt = VTrunkMomt[1:,]

    BodyMomt = LLegMomt + RLegMomt + TrunkMomt
    LegMomt = LLegMomt + RLegMomt
    ArmMomt = RArmMomt + LArmMomt
    WholeMomt = BodyMomt + ArmMomt
    # endregion

    # region: part Momt
    XF_RS_m = F_RS_m[1:, 0]
    YF_RS_m = F_RS_m[1:, 1]
    ZF_RS_m = F_RS_m[1:, 2]

    XF_LS_m = F_LS_m[1:, 0]
    YF_LS_m = F_LS_m[1:, 1]
    ZF_LS_m = F_LS_m[1:, 2]

    XF_RL_m = F_RL_m[1:, 0]
    YF_RL_m = F_RL_m[1:, 1]
    ZF_RL_m = F_RL_m[1:, 2]

    XF_LL_m = F_LL_m[1:, 0]
    YF_LL_m = F_LL_m[1:, 1]
    ZF_LL_m = F_LL_m[1:, 2]
    XBodyMomt = BodyMomt[:, 0]
    YBodyMomt = BodyMomt[:, 1]
    ZBodyMomt = BodyMomt[:, 2]
    XWholeMomt = WholeMomt[:, 0]
    YWholeMomt = WholeMomt[:, 1]
    ZWholeMomt = WholeMomt[:, 2]

    XTrunkMomt = TrunkMomt[:, 0]
    YTrunkMomt = TrunkMomt[:, 1]
    ZTrunkMomt = TrunkMomt[:, 2]
    XLegMomt = LegMomt[:, 0]
    YLegMomt = LegMomt[:, 1]
    ZLegMomt = LegMomt[:, 2]
    XArmMomt = ArmMomt[:, 0]
    YArmMomt = ArmMomt[:, 1]
    ZArmMomt = ArmMomt[:, 2]
    # endregion

    # region: 数据高频滤波
    b,a = signal.butter(4, 0.1, 'lowpass')
    XF_RS_m = signal.filtfilt(b, a, XF_RS_m)
    YF_RS_m = signal.filtfilt(b, a, YF_RS_m)
    ZF_RS_m = signal.filtfilt(b, a, ZF_RS_m)

    XF_LS_m = signal.filtfilt(b, a, XF_LS_m)
    YF_LS_m = signal.filtfilt(b, a, YF_LS_m)
    ZF_LS_m = signal.filtfilt(b, a, ZF_LS_m)

    XF_RL_m = signal.filtfilt(b, a, XF_RL_m)
    YF_RL_m = signal.filtfilt(b, a, YF_RL_m)
    ZF_RL_m = signal.filtfilt(b, a, ZF_RL_m)
    
    XF_LL_m = signal.filtfilt(b, a, XF_LL_m)
    YF_LL_m = signal.filtfilt(b, a, YF_LL_m)
    ZF_LL_m = signal.filtfilt(b, a, ZF_LL_m)
    
    XBodyMomt = signal.filtfilt(b, a, XBodyMomt)
    YBodyMomt = signal.filtfilt(b, a, YBodyMomt)
    ZBodyMomt = signal.filtfilt(b, a, ZBodyMomt)

    XWholeMomt = signal.filtfilt(b, a, XWholeMomt)
    YWholeMomt = signal.filtfilt(b, a, YWholeMomt)
    ZWholeMomt = signal.filtfilt(b, a, ZWholeMomt)

    # XTrunkMomt = signal.filtfilt(b, a, XTrunkMomt)
    # YTrunkMomt = signal.filtfilt(b, a, YTrunkMomt)
    # ZTrunkMomt = signal.filtfilt(b, a, ZTrunkMomt)

    XLegMomt = signal.filtfilt(b, a, XLegMomt)
    YLegMomt = signal.filtfilt(b, a, YLegMomt)
    ZLegMomt = signal.filtfilt(b, a, ZLegMomt)

    XRLegMomt = RLegMomt[:, 0]
    YRLegMomt = RLegMomt[:, 1]
    ZRLegMomt = RLegMomt[:, 2]
    # XRLegMomt = signal.filtfilt(b, a, XRLegMomt)
    # YRLegMomt = signal.filtfilt(b, a, YRLegMomt)
    # ZRLegMomt = signal.filtfilt(b, a, ZRLegMomt)

    XLLegMomt = LLegMomt[:, 0]
    YLLegMomt = LLegMomt[:, 1]
    ZLLegMomt = LLegMomt[:, 2]
    # XLLegMomt = signal.filtfilt(b, a, XLLegMomt)
    # YLLegMomt = signal.filtfilt(b, a, YLLegMomt)
    # ZLLegMomt = signal.filtfilt(b, a, ZLLegMomt)

    # XArmMomt = signal.filtfilt(b, a, XArmMomt)
    # YArmMomt = signal.filtfilt(b, a, YArmMomt)
    # ZArmMomt = signal.filtfilt(b, a, ZArmMomt)

    XRArmMomt = RArmMomt[:, 0]
    YRArmMomt = RArmMomt[:, 1]
    ZRArmMomt = RArmMomt[:, 2]
    # XRArmMomt = signal.filtfilt(b, a, XRArmMomt)
    # YRArmMomt = signal.filtfilt(b, a, YRArmMomt)
    # ZRArmMomt = signal.filtfilt(b, a, ZRArmMomt)

    XLArmMomt = LArmMomt[:, 0]
    YLArmMomt = LArmMomt[:, 1]
    ZLArmMomt = LArmMomt[:, 2]
    # XLArmMomt = signal.filtfilt(b, a, XLArmMomt)
    # YLArmMomt = signal.filtfilt(b, a, YLArmMomt)
    # ZLArmMomt = signal.filtfilt(b, a, ZLArmMomt)
    
    # omega
    OXTrunkMomt = OTrunkMomt[:, 0]
    OYTrunkMomt = OTrunkMomt[:, 1]
    OZTrunkMomt = OTrunkMomt[:, 2]
    OXTrunkMomt = signal.filtfilt(b, a, OXTrunkMomt)
    OYTrunkMomt = signal.filtfilt(b, a, OYTrunkMomt)
    OZTrunkMomt = signal.filtfilt(b, a, OZTrunkMomt)

    OXRLegMomt = ORLegMomt[:, 0]
    OYRLegMomt = ORLegMomt[:, 1]
    OZRLegMomt = ORLegMomt[:, 2]
    OXRLegMomt = signal.filtfilt(b, a, OXRLegMomt)
    OYRLegMomt = signal.filtfilt(b, a, OYRLegMomt)
    OZRLegMomt = signal.filtfilt(b, a, OZRLegMomt)

    OXLLegMomt = OLLegMomt[:, 0]
    OYLLegMomt = OLLegMomt[:, 1]
    OZLLegMomt = OLLegMomt[:, 2]
    OXLLegMomt = signal.filtfilt(b, a, OXLLegMomt)
    OYLLegMomt = signal.filtfilt(b, a, OYLLegMomt)
    OZLLegMomt = signal.filtfilt(b, a, OZLLegMomt)

    OXRArmMomt = ORArmMomt[:, 0]
    OYRArmMomt = ORArmMomt[:, 1]
    OZRArmMomt = ORArmMomt[:, 2]
    OXRArmMomt = signal.filtfilt(b, a, OXRArmMomt)
    OYRArmMomt = signal.filtfilt(b, a, OYRArmMomt)
    OZRArmMomt = signal.filtfilt(b, a, OZRArmMomt)

    OXLArmMomt = OLArmMomt[:, 0]
    OYLArmMomt = OLArmMomt[:, 1]
    OZLArmMomt = OLArmMomt[:, 2]
    OXLArmMomt = signal.filtfilt(b, a, OXLArmMomt)
    OYLArmMomt = signal.filtfilt(b, a, OYLArmMomt)
    OZLArmMomt = signal.filtfilt(b, a, OZLArmMomt)
    
    # Vel
    VXTrunkMomt = VTrunkMomt[:, 0]
    VYTrunkMomt = VTrunkMomt[:, 1]
    VZTrunkMomt = VTrunkMomt[:, 2]
    VXTrunkMomt = signal.filtfilt(b, a, VXTrunkMomt)
    VYTrunkMomt = signal.filtfilt(b, a, VYTrunkMomt)
    VZTrunkMomt = signal.filtfilt(b, a, VZTrunkMomt)

    VXRLegMomt = VRLegMomt[:, 0]
    VYRLegMomt = VRLegMomt[:, 1]
    VZRLegMomt = VRLegMomt[:, 2]
    VXRLegMomt = signal.filtfilt(b, a, VXRLegMomt)
    VYRLegMomt = signal.filtfilt(b, a, VYRLegMomt)
    VZRLegMomt = signal.filtfilt(b, a, VZRLegMomt)

    VXLLegMomt = VLLegMomt[:, 0]
    VYLLegMomt = VLLegMomt[:, 1]
    VZLLegMomt = VLLegMomt[:, 2]
    VXLLegMomt = signal.filtfilt(b, a, VXLLegMomt)
    VYLLegMomt = signal.filtfilt(b, a, VYLLegMomt)
    VZLLegMomt = signal.filtfilt(b, a, VZLLegMomt)

    VXRArmMomt = VRArmMomt[:, 0]
    VYRArmMomt = VRArmMomt[:, 1]
    VZRArmMomt = VRArmMomt[:, 2]
    VXRArmMomt = signal.filtfilt(b, a, VXRArmMomt)
    VYRArmMomt = signal.filtfilt(b, a, VYRArmMomt)
    VZRArmMomt = signal.filtfilt(b, a, VZRArmMomt)

    VXLArmMomt = VLArmMomt[:, 0]
    VYLArmMomt = VLArmMomt[:, 1]
    VZLArmMomt = VLArmMomt[:, 2]
    VXLArmMomt = signal.filtfilt(b, a, VXLArmMomt)
    VYLArmMomt = signal.filtfilt(b, a, VYLArmMomt)
    VZLArmMomt = signal.filtfilt(b, a, VZLArmMomt)

    XTrunkArmMomt = XTrunkMomt + XArmMomt
    YTrunkArmMomt = YTrunkMomt + YArmMomt
    ZTrunkArmMomt = ZTrunkMomt + ZArmMomt
    OZArmMomt = OZLArmMomt + OZRArmMomt
    OZLegMomt = OZLLegMomt + OZRLegMomt
    # endregion

    # region    
    ZArm = []
    ZLeg = []
    ZTrunk = []
    for i in range(len(index)-1):
        temp1 = ZArmMomt[index[i]:index[i+1]]
        Armtemp = np.sum(temp1)
        ZArm.append(Armtemp)

        temp2= ZLegMomt[index[i]:index[i+1]]
        Legtemp = np.sum(temp2)
        ZLeg.append(Legtemp)

        temp3= ZTrunkMomt[index[i]:index[i+1]]
        Trunktemp = np.sum(temp3)
        ZTrunk.append(Trunktemp)
    # endregion

    # region: axis test
    # fig, axes = plt.subplots(2,1, dpi=100,figsize=(12,10))
    # ax1 = axes[0]
    # ax2 = axes[1]

    # TestData = P_lh1
    # ax1.plot(tt, TestData[:,0], label="x")
    # ax1.plot(tt, TestData[:,1], label="y")
    # ax1.plot(tt, TestData[:,2], label="z")
    # ax1.set_ylabel('Position (local coord)', fontsize = 15)
    # ax1.set_title('Position', fontsize = 25)
    # ax1.legend()
    # ax1.grid()

    # index = 1
    # print("quad is :")
    # print(Data[100, (index-1)*16+6:(index-1)*16+10])
    # ax2.plot(tt, Data[time[0]:time[1],(index-1)*16], label="x")
    # # ax2.plot(tt, Data[time[0]:time[1],(index)*16], label="x2")
    # ax2.plot(tt, Data[time[0]:time[1],(index-1)*16+1], label="y")
    # # ax2.plot(tt, Data[time[0]:time[1],(index)*16+1], label="y2")
    # ax2.plot(tt, Data[time[0]:time[1],(index-1)*16+2], label="z")
    # # ax2.plot(tt, Data[time[0]:time[1],(index)*16+2], label="z2")
    # ax2.set_ylabel('Position (global coord)', fontsize = 15)
    # ax2.legend()
    # ax2.grid()

    # plt.show()
    # endregion

    # region: sum of predioc
    # fig, ax = plt.subplots(1,1, dpi=100,figsize=(12,10))
    # xticks = np.arange(len(index)-1)
    # ax.bar(xticks, ZArm, width=0.25, label="Arm")
    # ax.bar(xticks + 0.25, ZLeg, width=0.25, label="Leg")
    # ax.bar(xticks + 0.5, ZTrunk, width=0.25, label="Trunk")

    # ax.set_title("Sum of Angular Momt", fontsize=20)
    # ax.set_xlabel("Period", fontsize=15)
    # ax.set_ylabel("Angular Momt (kg.m2/s)", fontsize=15)
    # ax.axhline(0, color ='black', lw = 2)
    # ax.legend()
    # ax.grid(axis='y')
    # ax.set_xticks(xticks + 0.25)
    # ax.set_xticklabels(ticks)
    # ax.yaxis.set_tick_params(labelsize = 12)
    # ax.xaxis.set_tick_params(labelsize = 12)
    # endregion

    # region: Moment data visualization
    fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    lw = 3  # linewidth
    # ax4 = axes[3]

    # ax1.plot(t, Data[300:700, 192])
    # ax1.plot(t, Data[300:700, 193])
    # ax1.plot(t, Data[300:700, 194])

    ax1.plot(t, YTrunkMomt, label = "Trunk", linewidth=lw)
    # ax1.plot(t, OXTrunkMomt, label = "OTrunk")
    ax1.plot(t, YLegMomt, label = "All leg", linewidth=lw)
    # ax1.plot(t, XRLegMomt, label = "Right leg")
    # ax1.plot(t, XLLegMomt, label = "Left leg")
    # ax1.plot(t, XRArmMomt, label = "Right arm")
    # ax1.plot(t, XLArmMomt, label = "Left arm")
    # ax1.plot(t, OXRArmMomt, label = "ORight arm")
    # ax1.plot(t, OXLArmMomt, label = "OLeft arm")
    ax1.plot(t, YArmMomt, label = "All Arm", linewidth=lw)
    # ax1.plot(t, XBodyMomt, label = "whole body except arm")
    # ax1.plot(t, XWholeMomt, label = "whole body")
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax1.set_ylabel('Angular Momt (X-aixs)', fontsize = 15)
    ax1.set_title('Angular Momentum (kg.m2/s)', fontsize = 25)
    ax1.axvline(time[0]+index[0], color ='red', lw = 1)
    ax1.axvline(time[0]+index[1], color ='red', lw = 1)
    ax1.legend()
    ax1.grid()
    
    ax2.plot(t, XTrunkMomt, label = "Trunk", linewidth=lw)
    # ax2.plot(t, OYTrunkMomt, label = "OTrunk")
    ax2.plot(t, XLegMomt, label = "All leg", linewidth=lw)
    ax2.plot(t, XArmMomt, label = "All Arm", linewidth=lw)
    # ax2.plot(t, YArmMomt, label = " Arm")
    # ax2.plot(t, OYRArmMomt, label = "OR Arm")
    # ax2.plot(t, OYLArmMomt, label = "OL Arm")
    # ax2.plot(t, YBodyMomt, label = "whole body except arm")
    # ax2.plot(t, YWholeMomt, label = "whole body")
    ax2.set_ylabel('Angular Momt (Y-aixs)', fontsize = 15)
    ax2.axvline(time[0]+index[0], color ='red', lw = 1)
    ax2.axvline(time[0]+index[1], color ='red', lw = 1)
    ax2.legend()
    ax2.grid()

    ax3.plot(t, ZTrunkMomt, label = "Trunk", linewidth=lw)
    # ax3.plot(t, ZLegMomt, label = "All leg", linewidth=lw)
    ax3.plot(t, ZArmMomt, label = "All Arm", linewidth=lw)
    # ax3.plot(t, OZTrunkMomt, label = "OTrunk")
    # ax3.plot(t, VZTrunkMomt, label = "VTrunk")
    ax3.plot(t, ZRLegMomt, label = "R Leg", linewidth=lw)
    ax3.plot(t, ZLLegMomt, label = "L Leg", linewidth=lw)
    # ax3.plot(t, ZRArmMomt, label = "Right Arm")
    # ZRArmLeg = ZLLegMomt + ZRArmMomt
    # ax3.plot(t, ZLLegMomt + ZRArmMomt, label = "Right Arm + leg")
    # ax3.plot(t, ZRLegMomt + ZLArmMomt, label = "left Arm + leg")
    # ax3.plot(t, ZLArmMomt, label = "Left Arm")
    # ax3.plot(t, VZRArmMomt, label = "VRight Arm")
    # ax3.plot(t, VZLArmMomt, label = "VLeft Arm")
    # ax3.plot(t, ZTrunkArmMomt, label = "Trunk + Arm", linewidth=lw)
    # ax3.plot(t, ZBodyMomt, label = "Trunk + leg", linewidth=lw)
    # ax3.plot(t, ZWholeMomt, label = "whole body")
    ax3.set_ylabel('Angular Momt (Z-aixs)', fontsize = 15)
    # yuan
    ax3.axvline(time[0]+index[0], color ='red', lw = 1)
    ax3.axvline(time[0]+index[1], color ='red', lw = 1)

    ax3.legend()
    ax3.grid()
    ax3.set_xlabel("time(s)", fontsize = 15)
    plt.show()
    # endregion

    # region: Vel data visualization
    fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    ax1.plot(t, V_lh1_m[1:, 0], label = "body", linewidth=lw)
    ax1.plot(t, V_lh4_m[1:, 0], label = "leg", linewidth=lw)
    ax1.plot(t, V_lh10_m[1:, 0], label = "right arm", linewidth=lw)
    ax1.plot(t, V_lh14_m[1:, 0], label = "left arm", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax1.set_ylabel('Velocity (X-aixs)', fontsize = 15)
    ax1.set_title('Velocity (m/s)', fontsize = 25)
    ax1.axvline(time[0]+index[0], color ='red', lw = 1)
    ax1.axvline(time[0]+index[1], color ='red', lw = 1)
    ax1.legend()
    ax1.grid()
    
    ax2.plot(t, V_lh1_m[1:, 1], label = "body", linewidth=lw)
    ax2.plot(t, V_lh4_m[1:, 1], label = "leg", linewidth=lw)
    ax2.plot(t, V_lh10_m[1:, 1], label = "right arm", linewidth=lw)
    ax2.plot(t, V_lh14_m[1:, 1], label = "left arm", linewidth=lw)
    ax2.axvline(time[0]+index[0], color ='red', lw = 1)
    ax2.axvline(time[0]+index[1], color ='red', lw = 1)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax2.set_ylabel('Velocity (Y-aixs)', fontsize = 15)
    ax2.legend()
    ax2.grid()

    ax3.plot(t, V_lh1_m[1:, 2], label = "body", linewidth=lw)
    ax3.plot(t, V_lh4_m[1:, 2], label = "leg", linewidth=lw)
    ax3.plot(t, V_lh11_m[1:, 2], label = "right arm", linewidth=lw)
    ax3.plot(t, V_lh15_m[1:, 2], label = "left arm", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax3.set_ylabel('Velocity (Z-aixs)', fontsize = 15)
    ax3.axvline(time[0]+index[0], color ='red', lw = 1)
    ax3.axvline(time[0]+index[1], color ='red', lw = 1)
    ax3.legend()
    ax3.grid()
    ax3.set_xlabel("time(s)", fontsize = 15)
    plt.show()
    # endregion
    
    # region: angular data visualization
    fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    ax1.plot(tt, Data[1:, 13], label = "body", linewidth=lw)
    ax1.plot(tt, Data[1:, 2*16+13], label = "leg", linewidth=lw)
    ax1.plot(tt, Data[1:, 9*16+13], label = "right arm", linewidth=lw)
    ax1.plot(tt, Data[1:, 13*16+13], label = "left arm", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax1.set_ylabel('angular Velocity (X-aixs)', fontsize = 15)
    ax1.set_title('angular Velocity (rad/s)', fontsize = 25)
    ax1.axvline(time[0]+index[0], color ='red', lw = 1)
    ax1.axvline(time[0]+index[1], color ='red', lw = 1)
    ax1.legend()
    ax1.grid()
    
    ax2.plot(tt, Data[1:, 14], label = "body", linewidth=lw)
    ax2.plot(tt, Data[1:, 2*16+14], label = "leg", linewidth=lw)
    ax2.plot(tt, Data[1:, 9*16+14], label = "right arm", linewidth=lw)
    ax2.plot(tt, Data[1:, 13*16+14], label = "left arm", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax2.set_ylabel('angular Velocity (Y-aixs)', fontsize = 15)
    ax2.axvline(time[0]+index[0], color ='red', lw = 1)
    ax2.axvline(time[0]+index[1], color ='red', lw = 1)
    ax2.legend()
    ax2.grid()

    ax3.plot(tt, Data[1:, 15], label = "body", linewidth=lw)
    ax3.plot(tt, Data[1:, 2*16+15], label = "leg", linewidth=lw)
    ax3.plot(tt, Data[1:, 9*16+15], label = "right arm", linewidth=lw)
    ax3.plot(tt, Data[1:, 13*16+15], label = "left arm", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax3.set_ylabel('angular Velocity (Z-aixs)', fontsize = 15)
    ax3.axvline(time[0]+index[0], color ='red', lw = 1)
    ax3.axvline(time[0]+index[1], color ='red', lw = 1)
    ax3.legend()
    ax3.grid()
    ax3.set_xlabel("time(s)", fontsize = 15)
    plt.show()
    # endregion
    
    # region: Force data visualization
    fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    ax1.plot(t[:-1], XF_RS_m, label = "Right shoulder", linewidth=lw)
    ax1.plot(t[:-1], XF_LS_m, label = "Left shoulder", linewidth=lw)
    ax1.plot(t[:-1], XF_RL_m, label = "Right leg", linewidth=lw)
    ax1.plot(t[:-1], XF_LL_m, label = "Left leg", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax1.set_ylabel('Force (X-aixs)', fontsize = 15)
    ax1.set_title('Force (kg.m2/s)', fontsize = 25)
    ax1.axvline(time[0]+index[0], color ='red', lw = 1)
    ax1.axvline(time[0]+index[1], color ='red', lw = 1)
    ax1.legend()
    ax1.grid()
    
    ax2.plot(t[:-1], YF_RS_m, label = "Right shoulder", linewidth=lw)
    ax2.plot(t[:-1], YF_LS_m, label = "Left shoulder", linewidth=lw)
    ax2.plot(t[:-1], YF_RL_m, label = "Right leg", linewidth=lw)
    ax2.plot(t[:-1], YF_LL_m, label = "Left leg", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax2.set_ylabel('Force (Y-aixs)', fontsize = 15)
    ax2.axvline(time[0]+index[0], color ='red', lw = 1)
    ax2.axvline(time[0]+index[1], color ='red', lw = 1)
    ax2.legend()
    ax2.grid()

    ax3.plot(t[:-1], ZF_RS_m, label = "Right shoulder", linewidth=lw)
    ax3.plot(t[:-1], ZF_LS_m, label = "Left shoulder", linewidth=lw)
    ax3.plot(t[:-1], ZF_RL_m, label = "Right leg", linewidth=lw)
    ax3.plot(t[:-1], ZF_LL_m, label = "Left leg", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax3.set_ylabel('Force (Z-aixs)', fontsize = 15)
    ax3.axvline(time[0]+index[0], color ='red', lw = 1)
    ax3.axvline(time[0]+index[1], color ='red', lw = 1)
    ax3.legend()
    ax3.grid()
    ax3.set_xlabel("time(s)", fontsize = 15)
    # plt.show()
    # endregion

    # region: Force data visualization
    fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    ax1.plot(t[:-1], F_RS_m[1:, 0], label = "Right shoulder", linewidth=lw)
    ax1.plot(t[:-1], F_LS_m[1:, 0], label = "Left shoulder", linewidth=lw)
    ax1.plot(t[:-1], F_RL_m[1:, 0], label = "Right leg", linewidth=lw)
    ax1.plot(t[:-1], F_LL_m[1:, 0], label = "Left leg", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax1.set_ylabel('Force (X-aixs)', fontsize = 15)
    ax1.set_title('Force (kg.m2/s)', fontsize = 25)
    ax1.axvline(time[0]+index[0], color ='red', lw = 1)
    ax1.axvline(time[0]+index[1], color ='red', lw = 1)
    ax1.legend()
    ax1.grid()
    
    ax2.plot(t[:-1], F_RS_m[1:, 1], label = "Right shoulder", linewidth=lw)
    ax2.plot(t[:-1], F_LS_m[1:, 1], label = "Left shoulder", linewidth=lw)
    ax2.plot(t[:-1], F_RL_m[1:, 1], label = "Right leg", linewidth=lw)
    ax2.plot(t[:-1], F_LL_m[1:, 1], label = "Left leg", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax2.set_ylabel('Force (Y-aixs)', fontsize = 15)
    ax2.axvline(time[0]+index[0], color ='red', lw = 1)
    ax2.axvline(time[0]+index[1], color ='red', lw = 1)
    ax2.legend()
    ax2.grid()

    ax3.plot(t[:-1], F_RS_m[1:, 2], label = "Right shoulder", linewidth=lw)
    ax3.plot(t[:-1], F_LS_m[1:, 2], label = "Left shoulder", linewidth=lw)
    ax3.plot(t[:-1], F_RL_m[1:, 2], label = "Right leg", linewidth=lw)
    ax3.plot(t[:-1], F_LL_m[1:, 2] ,label = "Left leg", linewidth=lw)
    # ax1.axvline(2022, color ='red', lw = 2) 
    ax3.set_ylabel('Force (Z-aixs)', fontsize = 15)
    ax3.axvline(time[0]+index[0], color ='red', lw = 1)
    ax3.axvline(time[0]+index[1], color ='red', lw = 1)
    ax3.legend()
    ax3.grid()
    ax3.set_xlabel("time(s)", fontsize = 15)
    plt.show()
    # endregion
    pass

if __name__ == "__main__":
    AnaBVH()
    # AnaTest()
    pass