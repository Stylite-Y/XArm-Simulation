from enum import Flag
import numpy as np
from numpy import sin  as s
from numpy import cos as c
import os
import numpy
import matplotlib.animation as animation
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from matplotlib.pyplot import MultipleLocator
import time

class DataProcess():
    def __init__(self) :
        ## Trunk, Uarm, Farm, ULeg, Lleg
        self.l = [0.813, 0.3, 0.375, 0.406, 0.5075]
        pass
    def LocalAxis(Data, i):
        Pos_lh = np.array([[0.0, 0.0, 0.0]])
        # for n in range(200,1000):
        # for n in range(200,700):
        for n in range(0,500):
        # for n in range(2000,2500):
            # QuadHip = np.array([Data[n][7], Data[n][8], Data[n][9],Data[n][6]])
            # P_ho = np.array([[Data[n][0], Data[n][1], Data[n][2]]])
            m = 7
            QuadHip = np.array([Data[n][(m-1)*16+7], Data[n][(m-1)*16+8], Data[n][(m-1)*16+9], Data[n][(m-1)*16+6]])
            P_ho = np.array([[Data[n][(m-1)*16], Data[n][(m-1)*16+1], Data[n][(m-1)*16+2]]])
            RotMatrix = Rotation.from_quat(QuadHip)
            rotation_m = RotMatrix.as_matrix()
            P = -rotation_m.T @ P_ho.T
            Trans_m = np.concatenate((rotation_m.T, P), axis = 1)
            Trans_m = np.concatenate((Trans_m, np.array([[0, 0, 0, 1]])), axis = 0)
            P_lo = np.array([[Data[n][(i-1)*16], Data[n][(i-1)*16+1], Data[n][(i-1)*16+2], 1]])
            P_lh = Trans_m @ P_lo.T
            if n==200 and i ==1:
                print(QuadHip)
                print(P_lh)
            P_lh = P_lh[0:3].T

            Pos_lh = np.concatenate((Pos_lh, P_lh), axis = 0)
        return Pos_lh

    def animation(self, Data, t, saveflag=False):
        ## kinematic equation
        Hip = Data[:, 0:3]
        RUL = Data[:, 16*1+0:16*1+3]
        RLL = Data[:, 16*2+0:16*2+3]
        RF = Data[:, 16*3+0:16*3+3]
        LUL = Data[:, 16*4+0:16*4+3]
        LLL = Data[:, 16*5+0:16*5+3]
        LF = Data[:, 16*6+0:16*6+3]
        RS = Data[:, 16*7+0:16*7+3]
        RUA = Data[:, 16*8+0:16*8+3]
        RFA = Data[:, 16*9+0:16*9+3]
        RH = Data[:, 16*10+0:16*10+3]
        LS = Data[:, 16*11+0:16*11+3]
        LUA = Data[:, 16*12+0:16*12+3]
        LFA = Data[:, 16*13+0:16*13+3]
        LH = Data[:, 16*14+0:16*14+3]
        Head = Data[:, 16*15+0:16*15+3]
        Body = Data[:, 16*16+0:16*16+3]
        AllData = np.array([Hip, RUL, RLL, RF, LUL, LLL, LF, RS, RUA, RFA, RH, LS, LUA, LFA, LH, Body, Head])
        print(Head.shape, AllData.shape)
        history_len = 100

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(autoscale_on=False, projection='3d')
        # ax.set_aspect('equal')
        ax.set(xlim=(-0.6, 1.2), ylim=(-0.6, 1.2), zlim=(-0.6, 1.2),)
        ax.set_xlabel('X axis ', fontsize = 15)
        ax.set_ylabel('Y axis ', fontsize = 15)
        ax.set_zlabel('Z axis ', fontsize = 15)
        ax.xaxis.set_tick_params(labelsize = 12)
        ax.yaxis.set_tick_params(labelsize = 12)
        ax.zaxis.set_tick_params(labelsize = 12)
        ax.grid()
        ax.view_init(-162, -80)

        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, 17))
        # set up trajectory lines
        lines = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
        # set up points
        pts = sum([ax.plot([], [], [], 'o', c=c) for c in colors], [])
        # set up lines which create the stick figures
        stick_defines = [(0, 1), (1, 2), (2, 3),
                        (0, 4), (4, 5), (5, 6),
                        (0, 7), (7, 8), (8, 9), (9, 10),
                        (0, 11), (11, 12), (12, 13), (13, 14),
                        (0, 15), (15, 16)]
        stick_lines = [ax.plot([], [], [], 'k-')[0] for _ in stick_defines]

        def animate(i):
            # we'll step two time-steps per frame.  This leads to nice results.
            # i = (5 * i) % x_t.shape[1]

            for line, pt, xi in zip(lines, pts, AllData):
                x, y, z = xi[:i].T # note ordering of points to line up with true exogenous registration (x,z,y)
                pt.set_data(x[-1:], y[-1:])
                pt.set_3d_properties(z[-1:])

            for stick_line, (sp, ep) in zip(stick_lines, stick_defines):
                stick_line._verts3d = AllData[[sp,ep], i, :].T.tolist()

            # ax.view_init(30, 0.3 * i)
            fig.canvas.draw()
            return pts + stick_lines

        # line1, = ax.plot(Hip[0][0], Hip[0][1], Hip[0][2], 'o-', lw=3,markersize=8)
        # line2, = ax.plot(Hip[0][0], Hip[0][1], Hip[0][2], 'o-', lw=3,markersize=8)
        # line3, = ax.plot(Hip[0][0], Hip[0][1], Hip[0][2], 'o-', lw=3,markersize=8)
        # line4, = ax.plot(Hip[0][0], Hip[0][1], Hip[0][2], 'o-', lw=3,markersize=8)
        # line5, = ax.plot(Hip[0][0], Hip[0][1], Hip[0][2], 'o-', lw=3,markersize=8)
        # line1, = ax.plot([], [], [], 'o-', lw=3,markersize=8)
        # line2, = ax.plot([], [], [], 'o-', lw=3,markersize=8)
        # line3, = ax.plot([], [], [], 'o-', lw=3,markersize=8)
        # line4, = ax.plot([], [], [], 'o-', lw=3,markersize=8)
        # line5, = ax.plot([], [], [], 'o-', lw=3,markersize=8)
        # trace, = ax.plot([], [], '.-', lw=1, ms=1)
        # time_template = 'time = %.1fs'
        # time_text = ax.text(0.05, 0.9, 0.9,  '', transform=ax.transAxes, fontsize=15)
        # history_x, history_y, history_z = deque(maxlen=history_len), deque(maxlen=history_len)

        # def animate(i, Hip, RUL, RLL, RF, LUL, LLL, LF, RS, RUA, RFA, RH, LS, LUA, LFA, LH, Body, Head, line1, line2, line3, line4, line5):
        # def animate(i):
        #     thisx1 = [Hip[i][0], RUL[i][0], RLL[i][0], RF[i][0]]
        #     thisy1 = [Hip[i][1], RUL[i][1], RLL[i][1], RF[i][1]]
        #     thisz1 = [Hip[i][2], RUL[i][2], RLL[i][2], RF[i][2]]

        #     thisx2 = [Hip[i][0], LUL[i][0], LLL[i][0], LF[i][0]]
        #     thisy2 = [Hip[i][1], LUL[i][1], LLL[i][1], LF[i][1]]
        #     thisz2 = [Hip[i][2], LUL[i][2], LLL[i][2], LF[i][2]]

        #     thisx3 = [Hip[i][0], RS[i][0], RUA[i][0], RFA[i][0], RH[i][0]]
        #     thisy3 = [Hip[i][1], RS[i][1], RUA[i][1], RFA[i][1], RH[i][1]]
        #     thisz3 = [Hip[i][2], RS[i][2], RUA[i][2], RFA[i][2], RH[i][2]]

        #     thisx4 = [Hip[i][0], LS[i][0], LUA[i][0], LFA[i][0], LH[i][0]]
        #     thisy4 = [Hip[i][1], LS[i][1], LUA[i][1], LFA[i][1], LH[i][1]]
        #     thisz4 = [Hip[i][2], LS[i][2], LUA[i][2], LFA[i][2], LH[i][2]]

        #     thisx5 = [Hip[i][0], Body[i][0], Head[i][0]]
        #     thisy5 = [Hip[i][1], Body[i][1], Head[i][1]]
        #     thisz5 = [Hip[i][2], Body[i][2], Head[i][2]]

        #     line1.set_data(thisx1, thisy1)
        #     line1.seset_3d_propertiest(thisz1)
        #     line2.set_data(thisx2, thisy2)
        #     line2.seset_3d_propertiest(thisz2)
        #     line3.set_data(thisx3, thisy3)
        #     line3.seset_3d_propertiest(thisz3)
        #     line4.set_data(thisx4, thisy4)
        #     line4.seset_3d_propertiest(thisz4)
        #     line5.set_data(thisx5, thisy5)
        #     line5.seset_3d_propertiest(thisz5)

        #     # trace.set_data(history_x, history_y)
        #     # trace.set_alpha(alpha)
        #     # time_text.set_text(time_template % (i*self.dt))
        #     # return line, trace, time_text
        #     return line1, line2, line3, line4, line5

        # ani = animation.FuncAnimation(fig, animate, len(t), 
        #     fargs=(Hip, RUL, RLL, RF, LUL, LLL, LF, RS, RUA, RFA, RH, LS, LUA, LFA, LH, Body, Head, line1, line2, line3, line4, line5), 
        #     interval=1/120, blit=True)

        ani = animation.FuncAnimation(fig, animate, len(t), interval=1, blit=True)

        ## animation save to gif
        # date = self.date
        # name = self.name + ".gif"

        # savename = self.save_dir + date + name

        # if saveflag:
        #     ani.save(savename, writer='pillow', fps=72)

        plt.show()

        pass
