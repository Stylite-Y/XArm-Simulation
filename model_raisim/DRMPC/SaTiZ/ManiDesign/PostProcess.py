from cProfile import label
from posixpath import dirname
from re import I
import numpy as np
from numpy import sin  as s
from numpy import cos as c
import os
import numpy
import yaml
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
from ruamel.yaml import YAML
import pickle


class DataProcess():
    def __init__(self, cfg, robot,theta, q, dq, ddq, u, F, t, savepath,save_flag):
        self.cfg = cfg
        self.robot = robot
        self.dt = robot.dt
        # self.arm_M = arm_M
        # self.arm_I = arm_I
        self.theta = theta
        self.q = q
        self.dq = dq
        self.ddq = ddq
        self.u = u
        # self.F = F
        self.t = t
        self.savepath = savepath
        self.save_flag = save_flag

        # self.ML = self.cfg["Optimization"]["MaxLoop"] / 1000
        # self.Tp = self.cfg['Controller']['Tp']
        # self.Nc = self.cfg['Controller']['Nc']
        # self.T = self.cfg['Controller']['T']
        # self.dt = self.cfg['Controller']['dt']
        # self.PostarCoef = self.cfg["Optimization"]["CostCoef"]["postarCoef"]
        # self.TorqueCoef = self.cfg["Optimization"]["CostCoef"]["torqueCoef"]
        # self.DTorqueCoef = self.cfg["Optimization"]["CostCoef"]["DtorqueCoef"]
        # self.VeltarCoef = cfg["Optimization"]["CostCoef"]["VeltarCoef"]
        # self.m = cfg['Robot']['Mass']['mass']
        # self.I = cfg['Robot']['Mass']['inertia']

        self.save_dir, self.name, self.date = self.DirCreate()
        # self.Inertia_main, self.Inertia_coupling, self.Corialis, self.Gravity = self.ForceCal()
        pass

    def DirCreate(self, method_choice=2):
        trackingCoeff = self.cfg["Optimization"]["CostCoeff"]["trackingCoeff"]
        VelCoeff = self.cfg["Optimization"]["CostCoeff"]["VelCoeff"]
        powerCoeff = self.cfg["Optimization"]["CostCoeff"]["powerCoeff"]
        forceCoeff = self.cfg["Optimization"]["CostCoeff"]["forceCoeff"]
        smoothCoeff = self.cfg["Optimization"]["CostCoeff"]["smoothCoeff"]
        impactCoeff = self.cfg["Optimization"]["CostCoeff"]["ImpactCoeff"]
        Vt = self.cfg["Controller"]["Target"]
        Tp = self.cfg["Controller"]["Period"]
        # Tst = self.cfg["Controller"]["Stance"]
        dt = self.cfg["Controller"]["dt"]
        theta = round(self.theta, 3)
        
        date = time.strftime("%Y-%m-%d-%H-%M-%S")
        if method_choice==1:
            dirname = "-Traj-Tcf_"+str(trackingCoeff)+"-Pcf_"+str(powerCoeff)+"-Fcf_"+str(forceCoeff)+\
                        "-Scf_"+str(smoothCoeff)+"-Icf_"+str(impactCoeff)+"-Vt_"+str(Vt)+"-Tp_"+str(Tp)+"-Tst_"+str(Tst)
        if method_choice==2:
            dirname = "-Traj-Tcf_"+str(trackingCoeff)+"-Pcf_"+str(powerCoeff)+"-Fcf_"+str(forceCoeff)+\
                        "-Scf_"+str(smoothCoeff)+"-Icf_"+str(impactCoeff)+"-Vt_"+str(Vt)+"-Tp_"+str(Tp)+"-Ang_"+str(theta)
            # dirname = "Iarm_"+str(self.arm_I)+"-Marm_"+str(self.arm_M)
        elif method_choice==3:
            dirname = "-MPC-Pos_"+str(self.PostarCoef[1])+"-Tor_"+str(self.TorqueCoef[1])+"-DTor_"+str(self.DTorqueCoef[1]) +"-Vel_"+str(self.VeltarCoef[1])\
                    +"-dt_"+str(self.dt)+"-T_"+str(self.T)+"-Tp_"+str(self.Tp)+"-Tc_"+str(self.Nc)+"-ML_"+str(self.ML)+ "k" 

        # dirname = "-mM_"+str(m_M)
        # dirname = "-Ir_"+str(I_r)
        save_dir = self.savepath + date + dirname+ "/"

        if self.save_flag:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
        return save_dir, dirname, date

    def DataPlot(self, saveflag=0):

        fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        ref = [0.04*0.2]*len(self.t)

        ax1.plot(self.t, self.q[:, 0], label="theta 1")
        # ax1.plot(t, q[:, 1], label="theta 2")
        ax1.plot(self.t, self.q[:, 2], label="theta 3")
        ax1.plot(self.t, ref, linestyle='--', color='r')

        ax11 = ax1.twinx()
        ax11.plot(self.t, self.q[:, 1], color='forestgreen', label="theta 2")
        # ax11.plot(t, q[:, 1], color='mediumseagreen', label="theta 2")
        ax11.legend(loc='lower right', fontsize = 12)
        ax11.yaxis.set_tick_params(labelsize = 12)

        ax1.set_ylabel('Angle ', fontsize = 15)
        ax1.xaxis.set_tick_params(labelsize = 12)
        ax1.yaxis.set_tick_params(labelsize = 12)
        ax1.legend(loc='upper right', fontsize = 12)
        ax1.grid()

        ax2.plot(self.t, self.dq[:, 0], label="theta 1 Vel")
        ax2.plot(self.t, self.dq[:, 1], label="theta 2 Vel")
        ax2.plot(self.t, self.dq[:, 2], label="theta 3 Vel")

        ax2.set_ylabel('Angular Vel ', fontsize = 15)
        ax2.xaxis.set_tick_params(labelsize = 12)
        ax2.yaxis.set_tick_params(labelsize = 12)
        ax2.legend(loc='upper right', fontsize = 12)
        ax2.grid()

        ax3.plot(self.t, self.u[:, 0], label="torque 1")
        ax3.plot(self.t, self.u[:, 1], label="torque 2")
        ax3.set_ylabel('Torque ', fontsize = 15)
        ax3.xaxis.set_tick_params(labelsize = 12)
        ax3.yaxis.set_tick_params(labelsize = 12)
        ax3.legend(loc='upper right', fontsize = 12)
        ax3.grid()

        date = self.date
        name = self.name + ".png"

        savename = self.save_dir + date + name

        if saveflag:
            plt.savefig(savename)
    
        plt.show()

    ## Three link
    def animation(self, fileflag, saveflag):
        from numpy import sin, cos
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque

        ## kinematic equation
        L0 = self.robot.L[0]
        L1 = self.robot.L[1]
        L2 = self.robot.L[2]
        L_max = L0+L1+L2
        x1 = L0*sin(self.q[:, 0])
        y1 = L0*cos(self.q[:, 0])
        x2 = L1*sin(self.q[:, 0] + self.q[:, 1]) + x1
        y2 = L1*cos(self.q[:, 0] + self.q[:, 1]) + y1
        x3 = L2*sin(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + x2
        y3 = L2*cos(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + y2

        history_len = 100
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-L0, L0), ylim=(-0.05, (L0+L1)*1.2))
        ax.set_aspect('equal')
        ax.set_xlabel('X axis ', fontsize = 20)
        ax.set_ylabel('Y axis ', fontsize = 20)
        ax.xaxis.set_tick_params(labelsize = 18)
        ax.yaxis.set_tick_params(labelsize = 18)
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=3,markersize=8)
        trace, = ax.plot([], [], '.-', lw=1, ms=1)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=15)
        history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

        def animate(i):
            thisx = [0, x1[i], x2[i], x3[i]]
            thisy = [0, y1[i], y2[i], y3[i]]

            if i == 0:
                history_x.clear()
                history_y.clear()

            history_x.appendleft(thisx[3])
            history_y.appendleft(thisy[3])

            alpha = (i / history_len) ** 2
            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            # trace.set_alpha(alpha)
            time_text.set_text(time_template % (i*self.dt))
            return line, trace, time_text
        
        ani = animation.FuncAnimation(
            fig, animate, len(self.t), interval=0.1, save_count = 30, blit=True)

        ## animation save to gif
        date = self.date
        name = "traj_ani" + ".gif"

        savename = self.save_dir +date+ name

        if saveflag:
            ani.save(savename, writer='pillow', fps=30)

        # plt.show()
        
        pass
    
    ## Two Link
    def animationTwoLink(self, fileflag, saveflag):
        from numpy import sin, cos
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque

        L = self.robot.L
        q1 = self.q[:,0]
        q2 = self.q[:,1]
        L0 = L[0]
        L1 = L[1]
        L_max = L0+L1
        x1 = L0*cos(q1)
        y1 = L0*sin(q1)
        x2 = L1*cos(q1 + q2) + x1
        y2 = L1*sin(q1 + q2) + y1

        history_len = 100
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-L_max, L_max), ylim=(-0.4, (L0+L1)))
        ax.set_aspect('equal')
        ax.set_xlabel('X axis ', fontsize = 20)
        ax.set_ylabel('Y axis ', fontsize = 20)
        ax.xaxis.set_tick_params(labelsize = 18)
        ax.yaxis.set_tick_params(labelsize = 18)
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=3,markersize=8)
        trace, = ax.plot([], [], '.-', lw=1, ms=1)
        time_template = 'time = %.2fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=15)
        history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

        def animate(i):
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]

            if i == 0:
                history_x.clear()
                history_y.clear()

            history_x.appendleft(thisx[2])
            history_y.appendleft(thisy[2])

            alpha = (i / history_len) ** 2
            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            # trace.set_alpha(alpha)
            time_text.set_text(time_template % (i*self.dt))
            return line, trace, time_text
        
        ani = animation.FuncAnimation(
            fig, animate, len(self.t), interval=0.1, save_count = 30, blit=True)

        ## animation save to gif
        date = self.date
        name = "traj_ani" + ".gif"

        savename = self.save_dir +date+ name

        if saveflag:
            ani.save(savename, writer='pillow', fps=30)

        # plt.show()
        
        pass

    ## Three link
    def animationFourLink(self, fileflag, saveflag):
        from numpy import sin, cos
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from collections import deque

        ## kinematic equation
        L0 = self.robot.L[0]
        L1 = self.robot.L[1]
        L2 = self.robot.L[2]
        L3 = self.robot.L[3]
        L_max = L0+L1+L2
        x1 = L0*sin(self.q[:, 0])
        y1 = L0*cos(self.q[:, 0])
        x2 = L1*sin(self.q[:, 0] + self.q[:, 1]) + x1
        y2 = L1*cos(self.q[:, 0] + self.q[:, 1]) + y1
        x3 = L2*sin(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + x2
        y3 = L2*cos(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]) + y2
        x4 = L3*sin(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]+self.q[:, 3]) + x3
        y4 = L3*cos(self.q[:, 0] + self.q[:, 1]+self.q[:, 2]+self.q[:, 3]) + y3

        history_len = 100
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-L0, L0), ylim=(-0.05, (L0+L1)*1.2))
        ax.set_aspect('equal')
        ax.set_xlabel('X axis ', fontsize = 20)
        ax.set_ylabel('Y axis ', fontsize = 20)
        ax.xaxis.set_tick_params(labelsize = 18)
        ax.yaxis.set_tick_params(labelsize = 18)
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=3,markersize=8)
        trace, = ax.plot([], [], '.-', lw=1, ms=1)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=15)
        history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

        def animate(i):
            thisx = [0, x1[i], x2[i], x3[i], x4[i]]
            thisy = [0, y1[i], y2[i], y3[i], y4[i]]

            if i == 0:
                history_x.clear()
                history_y.clear()

            history_x.appendleft(thisx[4])
            history_y.appendleft(thisy[4])

            alpha = (i / history_len) ** 2
            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            # trace.set_alpha(alpha)
            time_text.set_text(time_template % (i*self.dt))
            return line, trace, time_text
        
        ani = animation.FuncAnimation(
            fig, animate, len(self.t), interval=0.1, save_count = 30, blit=True)

        ## animation save to gif
        date = self.date
        name = "traj_ani" + ".gif"

        savename = self.save_dir +date+ name

        if saveflag:
            ani.save(savename, writer='pillow', fps=30)

        # plt.show()
        
        pass
    

    def ForceCal(self):
        robot = self.robot    # create robot
        # calculate force
        Inertia_main = []
        Inertia_coupling = []
        Corialis = []
        Gravity = []
        for i in range(len(self.t)):
            temp1, temp2 = robot.inertia_force2(self.q[i, :], self.ddq[i, :])
            Inertia_main.append(temp1)
            Inertia_coupling.append(temp2)
            Corialis.append(robot.Coriolis(self.q[i, :], self.dq[i, :]))
            Gravity.append(robot.Gravity(self.q[i, :]))
            if i ==0:
                print(self.q[0], self.dq[0], self.ddq[0])
                print(Inertia_main, Inertia_coupling, Corialis, Gravity)

        Inertia_main = np.asarray(Inertia_main)
        Inertia_coupling = np.asarray(Inertia_coupling)
        Corialis = np.asarray(Corialis)
        Gravity = np.asarray(Gravity)
        return Inertia_main, Inertia_coupling, Corialis, Gravity

    def ForceAnalysis(self, saveflag = 0):
        Inertia_main, Inertia_coupling, Corialis, Gravity = self.Inertia_main, self.Inertia_coupling, self.Corialis, self.Gravity

        temp = Inertia_main[0][0] + Inertia_coupling[0][0] + Corialis[0][0] + Gravity[0][0]
        print(temp, Inertia_main[0][0], Inertia_coupling[0][0], Corialis[0][0], Gravity[0][0])
        fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
        ax1.plot(self.t, Inertia_main[:, 0], label="Inertia_main")
        ax1.plot(self.t, Inertia_coupling[:, 0], label="Inertia_coupling")
        ax1.plot(self.t, Corialis[:, 0], label="Corialis")
        ax1.plot(self.t, Gravity[:, 0], label="Gravity")

        ax1.set_ylabel('Force ', fontsize = 15)
        ax1.xaxis.set_tick_params(labelsize = 12)
        ax1.yaxis.set_tick_params(labelsize = 12)
        ax1.legend(loc='upper right', fontsize = 12)
        ax1.grid()

        ax2.plot(self.t, Inertia_main[:, 1], label="Inertia_main")
        ax2.plot(self.t, Inertia_coupling[:, 1], label="Inertia_coupling")
        ax2.plot(self.t, Corialis[:, 1], label="Corialis")
        ax2.plot(self.t, Gravity[:, 1], label="Gravity")

        ax2.set_ylabel('Force ', fontsize = 15)
        ax2.xaxis.set_tick_params(labelsize = 12)
        ax2.yaxis.set_tick_params(labelsize = 12)
        ax2.legend(loc='upper right', fontsize = 12)
        ax2.grid()

        ax3.plot(self.t, Inertia_main[:, 2], label="Inertia_main")
        ax3.plot(self.t, Inertia_coupling[:, 2], label="Inertia_coupling")
        ax3.plot(self.t, Corialis[:, 2], label="Corialis")
        ax3.plot(self.t, Gravity[:, 2], label="Gravity")

        ax3.set_ylabel('Force ', fontsize = 15)
        ax3.xaxis.set_tick_params(labelsize = 12)
        ax3.yaxis.set_tick_params(labelsize = 12)
        ax3.legend(loc='upper right', fontsize = 12)
        ax3.grid()

        date = self.date
        name = self.name + "-Force.png"

        savename = self.save_dir + date + name

        if saveflag:
            plt.savefig(savename)
        plt.show()

        pass

    def PowerAnalysis(self, saveflag = 0):
        Inertia_main, Inertia_coupling, Corialis, Gravity = self.Inertia_main, self.Inertia_coupling, self.Corialis, self.Gravity
        power = np.asarray([self.u[:, i] * self.dq[:, i+1] for i in range(2)]).transpose()
        dq = np.asarray(self.dq)
        Inertia_main_p = Inertia_main * dq
        Inertia_coupling_p = Inertia_coupling * dq
        Corialis_p = Corialis * dq
        Gravity_p = Gravity * dq
        # print(power.shape)

        fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
        ax1.plot(self.t, Inertia_main_p[:, 0], label="Joint 0 Inertia_main Power")
        ax1.plot(self.t, Inertia_coupling_p[:, 0], label="Joint 0 Inertia_coupling Power")
        ax1.plot(self.t, Corialis_p[:, 0], label="Joint 0 Corialis Power")
        ax1.plot(self.t, Gravity_p[:, 0], label="Joint 0 Gravity Power")

        ax1.set_ylabel('Power ', fontsize = 15)
        ax1.xaxis.set_tick_params(labelsize = 12)
        ax1.yaxis.set_tick_params(labelsize = 12)
        ax1.legend(loc='upper right', fontsize = 12)
        ax1.grid()

        ax2.plot(self.t, power[:, 0], label="Joint 1 Torque Power")
        ax2.plot(self.t, Inertia_main_p[:, 1], label="Joint 1 Inertia_main Power")
        ax2.plot(self.t, Inertia_coupling_p[:, 1], label="Joint 1 Inertia_coupling Power")
        ax2.plot(self.t, Corialis_p[:, 1], label="Joint 1 Corialis Power")
        ax2.plot(self.t, Gravity_p[:, 1], label="Joint 1 Gravity Power")

        ax2.set_ylabel('Power ', fontsize = 15)
        ax2.xaxis.set_tick_params(labelsize = 12)
        ax2.yaxis.set_tick_params(labelsize = 12)
        ax2.legend(loc='upper right', fontsize = 12)
        ax2.grid()

        ax3.plot(self.t, power[:, 1], label="Joint 2 Torque Power")
        ax3.plot(self.t, Inertia_main_p[:, 2], label="Joint 2 Inertia_main Power")
        ax3.plot(self.t, Inertia_coupling_p[:, 2], label="Joint 2 Inertia_coupling Power")
        ax3.plot(self.t, Corialis_p[:, 2], label="Joint 2 Corialis Power")
        ax3.plot(self.t, Gravity_p[:, 2], label="Joint 2 Gravity Power")

        ax3.set_ylabel('Power ', fontsize = 15)
        ax3.xaxis.set_tick_params(labelsize = 12)
        ax3.yaxis.set_tick_params(labelsize = 12)
        ax3.legend(loc='upper right', fontsize = 12)
        ax3.grid()

        date = self.date
        name = self.name + "-Power.png"

        savename = self.save_dir + date + name

        if saveflag:
            plt.savefig(savename)
        # plt.show()
        pass

    def MomentumAnalysis(self,saveflag=0):
        from numpy import sin, cos

        L0 = self.robot.L[0]
        L1 = self.robot.L[1]
        L2 = self.robot.L[2]
        l0 = self.robot.l[0]
        l1 = self.robot.l[1]
        l2 = self.robot.l[2]
        I0 = self.robot.I[0]
        I1 = self.robot.I[1]
        I2 = self.robot.I[2]
        m0 = self.robot.m[0]
        m1 = self.robot.m[1]
        m2 = self.robot.m[2]
        q = self.q
        dq = self.dq
        
        vx0 = l0*cos(q[:, 0])*dq[:, 0]
        vy0 = -l0*sin(q[:, 0])*dq[:, 0]
        vx1 = L0*cos(q[:, 0])*dq[:, 0] + l1*cos(q[:, 0]+q[:, 1])*(dq[:, 0]+dq[:, 1])
        vy1 = -L0*sin(q[:, 0])*dq[:, 0] - l1*sin(q[:, 0]+q[:, 1])*(dq[:, 0]+dq[:, 1])
        vx2 = L0*cos(q[:, 0])*dq[:, 0] + L1*cos(q[:, 0]+q[:, 1])*(dq[:, 0]+dq[:, 1]) + l2*cos(q[:, 0]+q[:, 1]+q[:, 2])*(dq[:, 0]+dq[:, 1]+dq[:, 2])
        vy2 = -L0*sin(q[:, 0])*dq[:, 0] - L1*sin(q[:, 0]+q[:, 1])*(dq[:, 0]+dq[:, 1]) - l2*sin(q[:, 0]+q[:, 1]+q[:, 2])*(dq[:, 0]+dq[:, 1]+dq[:, 2])

        Momentum0=np.array([[0.0]])
        Momentum1=np.array([[0.0]])
        Momentum2=np.array([[0.0]])
        for i in range(len(self.t)-1):
            if i ==0:
                dv0 = np.sqrt(vx0[i+1]**2 + vy0[i+1]**2)*vx0[i+1]/np.abs(vx0[i+1]) - 0.0
                dv1 = np.sqrt(vx1[i+1]**2 + vy1[i+1]**2)*vx1[i+1]/np.abs(vx1[i+1]) - 0.0
                dv2 = np.sqrt(vx2[i+1]**2 + vy2[i+1]**2)*vx2[i+1]/np.abs(vx2[i+1]) - 0.0
            else:
                dv0 = np.sqrt(vx0[i+1]**2 + vy0[i+1]**2)*vx0[i+1]/np.abs(vx0[i+1]) - np.sqrt(vx0[i]**2 + vy0[i]**2)*vx0[i]/np.abs(vx0[i])
                dv1 = np.sqrt(vx1[i+1]**2 + vy1[i+1]**2)*vx1[i+1]/np.abs(vx1[i+1]) - np.sqrt(vx1[i]**2 + vy1[i]**2)*vx1[i]/np.abs(vx1[i])
                dv2 = np.sqrt(vx2[i+1]**2 + vy2[i+1]**2)*vx2[i+1]/np.abs(vx2[i+1]) - np.sqrt(vx2[i]**2 + vy2[i]**2)*vx2[i]/np.abs(vx2[i])
            temp0 = m0*dv0 + I0 * (dq[i+1][0]-dq[i][0])
            temp1 = m1*dv1 + I1 * (dq[i+1][1]-dq[i][1])
            temp2 = m2*dv2 + I2 * (dq[i+1][2]-dq[i][2])
            Momentum0 = np.concatenate([Momentum0, [[temp0]]], axis=0)
            Momentum1 = np.concatenate([Momentum1, [[temp1]]], axis=0)
            Momentum2 = np.concatenate([Momentum2, [[temp2]]], axis=0)
            
        Momentum0 = Momentum0[1:,]
        Momentum1 = Momentum1[1:,]
        Momentum2 = Momentum2[1:,]
        Impulse = self.u * self.dt
        Impulse = Impulse[0:len(Momentum0),]
        MomentumSum = Momentum0+Momentum1+Momentum2
        MomentumSum2 = Momentum0+Momentum1+Momentum2-Impulse[:,0].reshape(-1, 1)-Impulse[:,1].reshape(-1, 1)
        print(Momentum0.shape, Impulse.shape, Impulse[:,0].shape)

        ## test
        temp = vx0[1]*m0 + vx1[1]*m1 + vx2[1]*m2
        print(q[1], dq[1], temp, Momentum0[1], Impulse[0], self.u[0])

        fig, axes = plt.subplots(1,1, dpi=100,figsize=(12,10))
        axes.plot(self.t[0:len(Momentum0)], Momentum0, label="link 0 Momentum")
        axes.plot(self.t[0:len(Momentum0)], Momentum1, label="link 1 Momentum")
        axes.plot(self.t[0:len(Momentum0)], Momentum2, label="link 2 Momentum")
        # axes.plot(self.t[0:len(Momentum0)], MomentumSum, label="Sum of all link Momentum")
        # axes.plot(self.t[0:len(Momentum0)], MomentumSum2, label="Sum of Momentum and Impulse")
        # axes.plot(self.t, Impulse[:,0], label="joint 1 Impulse")
        # axes.plot(self.t, Impulse[:,1], label="joint 2 Impulse")

        axes.set_ylabel('Momentum ', fontsize = 15)
        axes.xaxis.set_tick_params(labelsize = 12)
        axes.yaxis.set_tick_params(labelsize = 12)
        axes.legend(loc='upper right', fontsize = 12)
        axes.grid()

        date = self.date
        name = self.name + "-Momentum.png"

        savename = self.save_dir + date + name

        if saveflag:
            plt.savefig(savename)
        # plt.show()
        pass

    def SupportForce(self, saveflag=0):
        Arm = self.robot
        Fx = []
        Fy = []
        print(len(self.t))
        for i in range(len(self.t)-1):
            AccF = Arm.SupportForce(self.q[i, :], self.dq[i, :], self.ddq[i, :])
            Fx.append(-AccF[0])
            Fy.append(-AccF[1])
        temp = self.t[0:len(self.t)-1]
        plt.figure()
        plt.plot(temp, Fx, label="Fx")
        plt.plot(temp, Fy, label="Fy")
        plt.legend(loc='upper right', fontsize = 12)
        plt.grid()
        date = self.date
        name = self.name + "-sF.png"

        savename = self.save_dir + date + name

        if saveflag:
            plt.savefig(savename)
        plt.show()

        pass

    def DataSave(self, saveflag, com_x, com_y, W_k, W_w, P_k, P_w, I_k, I_w, I_s, I_e):
        date = self.date
        name = self.name

        if saveflag:
            # np.save(self.save_dir+date+name+"-sol.npy",
            #         np.hstack((self.q, self.dq, self.ddq, self.u, self.F, self.t)))
            # # output the config yaml file
            # # with open(os.path.join(StorePath, date + name+"-config.yaml"), 'wb') as file:
            # #     yaml.dump(self.cfg, file)
            with open(self.save_dir+date+name+"-config.yaml", mode='w') as file:
                YAML().dump(self.cfg, file)
            # Data = {'F': self.F, 'u': self.u, "q": self.q, "dq": self.dq, "ddq": self.ddq, "t": self.t}
            Data = {'u': self.u, "q": self.q, "dq": self.dq, "ddq": self.ddq, "t": self.t, "com_x": com_x, "com_y": com_y,\
                    "W_k":W_k, "W_w":W_w, "P_k":P_k, "P_w":P_w,"I_k":I_k, "I_w":I_w, "I_s":I_s, "I_e":I_e}
            with open(os.path.join(self.save_dir, date+name+"-sol.pkl"), 'wb') as f:
                pickle.dump(Data, f)
            pass
        
        return self.save_dir
