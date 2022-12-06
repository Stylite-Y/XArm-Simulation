import numpy as np
import raisimpy as raisim
import os
import datetime
import time
import yaml
import matplotlib.pyplot as plt

## shoulder y
def WeightsLift():
    # jointNominalConfig = np.array([0.0, 0.0, 1.03, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,     # wx, wy
    #                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,               # rsx, rsy, re, lsx, lsy, le
    #                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    jointNominalConfig = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    Human.setGeneralizedCoordinate(jointNominalConfig)
    Human.setGeneralizedVelocity(jointVelocityTarget)

    ShoulderTraj = np.linspace(0, 1.57, 1000)

    hand = Human.getBodyIdx("Formarm_L")
    # ball1 = world.addSphere(0.08, 3.0, "steel")
    # ball1.setPosition(0, 0.0, 1.0)
    # world.addStiffWire(Human, 1, (0.08, -0.22, -0.23), ball1, 1, np.zeros(3), 0.5)

    PosError = 0
    Torque = np.array([[0.0, 0.0]])
    t = np.array([0.0])
    server.startRecordingVideo("sy.mp4")
    time.sleep(0.5)
    for i in range(4000):
        time.sleep(0.001)
        # jointTar = np.array([0.0, 0.0, 0.0]) 
        jointTar = np.array([0.0]) 
        # ShoulderTraj = np.pi/2-np.abs(np.pi/2*np.cos(i*0.001*np.pi))
        if i <1000:
            jointTar = np.concatenate((jointTar, [ShoulderTraj[i]]))
            jointTar = np.concatenate((jointTar, np.zeros(4)))
            traj = ShoulderTraj[i]
            # jointTar = np.concatenate((jointTar, np.zeros(12)))

        else:
            jointTar = np.concatenate((jointTar, [ShoulderTraj[-1]]))
            jointTar = np.concatenate((jointTar, np.zeros(4)))
            traj = ShoulderTraj[-1]
            # jointTar = np.concatenate((jointTar, np.zeros(12)))
            
        HumanPos, HumanVel = Human.getState()
        if i % 50 ==0 :
            print("="*50)
            print("Human Pos:", HumanPos[1])
            print("Desied Pos:", traj)

        # HumanPos = HumanPos[7:]
        # HumanVel = HumanVel[6:]
        PosError += (jointTar - HumanPos)*0.001

        # tor = 10*(jointTar - HumanPos) + 0.02*(0-HumanVel) + 0.5*PosError
        tor = 100*(jointTar - HumanPos)+ 10*(0-HumanVel)+ 50*PosError

        Torque = np.concatenate((Torque,[[tor[1], tor[2]]]), axis=0)
        t = np.concatenate((t,[i*0.001]))
        # tor = np.concatenate((np.zeros(6), tor))
        # tor = np.concatenate((np.zeros(6), tor))
        Human.setGeneralizedForce(tor)
        server.integrateWorldThreadSafe()
    server.stopRecordingVideo()
    
    u1 = Torque[1:,0]
    u2 = Torque[1:,1]
    tor_avg = np.sum(np.sqrt(u1**2)) / 4000
    tor_avg2 = np.sum(np.sqrt(u2**2)) / 4000
    print(tor_avg,tor_avg2)

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'image.cmap': 'inferno',
        'font.size': 20,
        'lines.linewidth': 3,
        'axes.labelsize': 30,
        'axes.titlesize': 50,
        'legend.fontsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.3,
    }

    plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    ax1 = axs

    ax1.plot(t[1:], Torque[1:,0], label = "shoulder Torque")
    ax1.plot(t[1:], Torque[1:,1], label = "elbow Torque")
    ax1.set_xlabel("t(s)")
    ax1.set_ylabel("Torque(N.m)")
    ax1.grid()
    ax1.legend()
    plt.show()
    pass

## elbow
def WeightsLift2():
    # jointNominalConfig = np.array([0.0, 0.0, 1.03, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,     # wx, wy
    #                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,               # rsx, rsy, re, lsx, lsy, le
    #                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    jointNominalConfig = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    Human.setGeneralizedCoordinate(jointNominalConfig)
    Human.setGeneralizedVelocity(jointVelocityTarget)

    ShoulderTraj = np.linspace(0, 1.57, 1000)

    hand = Human.getBodyIdx("Formarm_L")
    # ball1 = world.addSphere(0.08, 3.0, "steel")
    # ball1.setPosition(0, 0.0, 1.0)
    # world.addStiffWire(Human, 1, (0.08, -0.22, -0.23), ball1, 1, np.zeros(3), 0.5)

    PosError = 0
    Torque = np.array([[0.0, 0.0]])
    t = np.array([0.0])
    server.startRecordingVideo("elbow.mp4")
    time.sleep(0.5)

    for i in range(4000):
        time.sleep(0.002)
        # jointTar = np.array([0.0, 0.0, 0.0]) 
        jointTar = np.array([0.0, 0.0]) 
        # ShoulderTraj = np.pi/2-np.abs(np.pi/2*np.cos(i*0.001*np.pi))
        if i <1000:
            jointTar = np.concatenate((jointTar, [ShoulderTraj[i]]))
            jointTar = np.concatenate((jointTar, np.zeros(3)))
            traj = ShoulderTraj[i]
            # jointTar = np.concatenate((jointTar, np.zeros(12)))

        else:
            jointTar = np.concatenate((jointTar, [ShoulderTraj[-1]]))
            jointTar = np.concatenate((jointTar, np.zeros(3)))
            traj = ShoulderTraj[-1]
            # jointTar = np.concatenate((jointTar, np.zeros(12)))
            
        HumanPos, HumanVel = Human.getState()
        if i % 50 ==0 :
            print("="*50)
            print("Human Pos:", HumanPos[1])
            print("Desied Pos:", traj)

        # HumanPos = HumanPos[7:]
        # HumanVel = HumanVel[6:]
        PosError += (jointTar - HumanPos)*0.001

        # tor = 10*(jointTar - HumanPos) + 0.02*(0-HumanVel) + 0.5*PosError
        tor = 100*(jointTar - HumanPos)+ 10*(0-HumanVel)+ 50*PosError

        Torque = np.concatenate((Torque,[[tor[1], tor[2]]]), axis=0)
        t = np.concatenate((t,[i*0.001]))
        # tor = np.concatenate((np.zeros(6), tor))
        # tor = np.concatenate((np.zeros(6), tor))
        Human.setGeneralizedForce(tor)
        server.integrateWorldThreadSafe()
    server.stopRecordingVideo()
    
    u1 = Torque[1:,0]
    u2 = Torque[1:,1]
    tor_avg = np.sum(np.sqrt(u1**2)) / 4000
    tor_avg2 = np.sum(np.sqrt(u2**2)) / 4000
    print(tor_avg,tor_avg2)

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'image.cmap': 'inferno',
        'font.size': 20,
        'lines.linewidth': 3,
        'axes.labelsize': 30,
        'axes.titlesize': 50,
        'legend.fontsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.3,
    }

    plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    ax1 = axs

    ax1.plot(t[1:], Torque[1:,0], label = "shoulder Torque")
    ax1.plot(t[1:], Torque[1:,1], label = "elbow Torque")
    ax1.set_xlabel("t(s)")
    ax1.set_ylabel("Torque(N.m)")
    ax1.grid()
    ax1.legend()
    plt.show()
    pass

## shoulder abad
def WeightsLift3():
    # jointNominalConfig = np.array([0.0, 0.0, 1.03, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,     # wx, wy
    #                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,               # rsx, rsy, re, lsx, lsy, le
    #                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    jointNominalConfig = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    Human.setGeneralizedCoordinate(jointNominalConfig)
    Human.setGeneralizedVelocity(jointVelocityTarget)

    ShoulderTraj = np.linspace(0, 1.57, 1000)

    hand = Human.getBodyIdx("Formarm_L")
    # ball1 = world.addSphere(0.08, 3.0, "steel")
    # ball1.setPosition(0, 0.0, 1.0)
    # world.addStiffWire(Human, 1, (0.08, -0.22, -0.23), ball1, 1, np.zeros(3), 0.5)

    # server.startRecordingVideo("sx.mp4")
    PosError = 0
    Torque = np.array([[0.0, 0.0]])
    t = np.array([0.0])
    for i in range(4000):
        time.sleep(0.001)
        # jointTar = np.array([0.0, 0.0, 0.0]) 
        # jointTar = np.array([0.0, 0.0]) 
        # ShoulderTraj = np.pi/2-np.abs(np.pi/2*np.cos(i*0.001*np.pi))
        if i <1000:
            jointTar = np.array([-ShoulderTraj[i]])
            jointTar = np.concatenate((jointTar, np.zeros(5)))
            traj = ShoulderTraj[i]
            # jointTar = np.concatenate((jointTar, np.zeros(12)))

        else:
            jointTar = np.array([-ShoulderTraj[-1]])
            jointTar = np.concatenate((jointTar, np.zeros(5)))
            traj = ShoulderTraj[-1]
            # jointTar = np.concatenate((jointTar, np.zeros(12)))
            
        HumanPos, HumanVel = Human.getState()
        if i % 50 ==0 :
            print("="*50)
            print("Human Pos:", HumanPos[1])
            print("Desied Pos:", traj)

        # HumanPos = HumanPos[7:]
        # HumanVel = HumanVel[6:]
        PosError += (jointTar - HumanPos)*0.001

        # tor = 10*(jointTar - HumanPos) + 0.02*(0-HumanVel) + 0.5*PosError
        tor = 100*(jointTar - HumanPos)+ 10*(0-HumanVel) + 50*PosError

        Torque = np.concatenate((Torque,[[tor[0], tor[2]]]), axis=0)
        t = np.concatenate((t,[i*0.001]))
        # tor = np.concatenate((np.zeros(6), tor))
        # tor = np.concatenate((np.zeros(6), tor))
        Human.setGeneralizedForce(tor)
        server.integrateWorldThreadSafe()
    # server.stopRecordingVideo()
    u1 = Torque[1:,0]
    tor_avg = np.sum(np.sqrt(u1**2)) / 4000
    print(tor_avg)

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'image.cmap': 'inferno',
        'font.size': 20,
        'lines.linewidth': 3,
        'axes.labelsize': 30,
        'axes.titlesize': 50,
        'legend.fontsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.3,
    }

    plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    ax1 = axs

    ax1.plot(t[1:], Torque[1:,0], label = "shoulder Torque")
    # ax1.plot(t[1:], Torque[1:,1], label = "elbow Torque")
    ax1.set_xlabel("t(s)")
    ax1.set_ylabel("Torque(N.m)")
    ax1.grid()
    ax1.legend()
    plt.show()
    pass

def Throw():
    # jointNominalConfig = np.array([0.0, 0.0, 1.03, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,     # wx, wy
    #                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,               # rsx, rsy, re, lsx, lsy, le
    #                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    jointNominalConfig = np.array([0.0, 1.8, 1.57, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    Human.setGeneralizedCoordinate(jointNominalConfig)
    Human.setGeneralizedVelocity(jointVelocityTarget)

    # ShoulderTraj = np.linspace(2.1, 0, 1000)

    hand = Human.getBodyIdx("Formarm_L")
    pin3 = world.addSphere(0.04, 0.8)
    pin3.setAppearance("0,0,1,0.3")
    pin3.setPosition(-0.4, 0.2, 2.4)
    pin3.setBodyType(raisim.BodyType.STATIC)
    ball1 = world.addSphere(0.05, 1.0, "steel")
    ball1.setPosition(-0.4, 0.2, 1.85)
    world.addStiffWire(pin3, 0, np.zeros(3), ball1, 0, np.zeros(3), 0.55)

    PosError = 0
    Torque = np.array([[0.0, 0.0]])
    t = np.array([0.0])
    server.startRecordingVideo("throw.mp4")
    time.sleep(1.0)

    for i in range(1200):
        time.sleep(0.002)
        # jointTar = np.array([0.0, 0.0, 0.0]) 
        jointTar = np.array([0.0, 2.0]) 
        ShoulderTraj = np.abs(np.pi*3/5*np.cos(i*0.001*2*np.pi))
        ShoulderVel = 2*np.pi*np.pi*3/5*np.sin(i*0.001*2*np.pi)
        if i <1000:
            jointTar = np.concatenate((jointTar, [ShoulderTraj]))
            jointTar = np.concatenate((jointTar, np.zeros(3)))
            traj = ShoulderTraj
            # jointTar = np.concatenate((jointTar, np.zeros(12)))

        else:
            jointTar = np.concatenate((jointTar, [ShoulderTraj]))
            jointTar = np.concatenate((jointTar, np.zeros(3)))
            traj = ShoulderTraj
            # jointTar = np.concatenate((jointTar, np.zeros(12)))
            
        HumanPos, HumanVel = Human.getState()
        if i % 50 ==0 :
            print("="*50)
            print("Human Pos:", HumanPos[2])
            print("Human Vel:", HumanVel[2])
            print("Desied Pos:", traj)
            print("Desied Vel:", ShoulderVel)

        # HumanPos = HumanPos[7:]
        # HumanVel = HumanVel[6:]
        PosError += (jointTar - HumanPos)*0.001

        Veltar = np.array([0.0, 0.0, ShoulderVel, 0.0, 0.0, 0.0])

        # tor = 10*(jointTar - HumanPos) + 0.02*(0-HumanVel) + 0.5*PosError
        tor = 100*(jointTar - HumanPos)+ 20*(Veltar-HumanVel)+ 50*PosError
        Torque = np.concatenate((Torque,[[tor[1], tor[2]]]), axis=0)
        t = np.concatenate((t,[i*0.001]))
        # tor = np.concatenate((np.zeros(6), tor))
        # tor = np.concatenate((np.zeros(6), tor))
        Human.setGeneralizedForce(tor)
        server.integrateWorldThreadSafe()
    
    server.stopRecordingVideo()

    u1 = Torque[1:,1]
    tor_avg = np.sum(np.sqrt(u1**2)) / 4000
    print(tor_avg)

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'image.cmap': 'inferno',
        'font.size': 20,
        'lines.linewidth': 3,
        'axes.labelsize': 30,
        'axes.titlesize': 50,
        'legend.fontsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.3,
    }

    plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    ax1 = axs

    ax1.plot(t[1:], Torque[1:,0], label = "shoulder Torque")
    # ax1.plot(t[1:], Torque[1:,1], label = "elbow Torque")
    ax1.set_xlabel("t(s)")
    ax1.set_ylabel("Torque(N.m)")
    ax1.grid()
    ax1.legend()
    plt.show()
    pass


if __name__ == "__main__":
    # get params config data
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/default_cfg.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # load activation file and urdf file
    # raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
    # Human_urdf_file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/urdf/IOReal/Bipedal_V2_v2.urdf"
    Human_urdf_file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/urdf/IOReal/Bipedal_V2_arm.urdf"
    print(Human_urdf_file)
    # raisim world config setting
    world = raisim.World()

    # set simulation step
    # t_step = ParamData["environment"]["t_step"] 
    # sim_time = ParamData["environment"]["sim_time"]
    world.setTimeStep(0.001)
    ground = world.addGround(0)

    gravity = world.getGravity()
    print(gravity)
    Human = world.addArticulatedSystem(Human_urdf_file)
    Human.setName("Human")
    print(Human.getDOF())

    # world.setGravity([0, 0, 0])
    # gravity1 = world.getGravity() 

    ## ====================
    # ball control initial pos and vel setting
    jointNominalConfig = np.array([0.0, 0.0, 1.03, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,     # wx, wy
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,               # rsx, rsy, re, lsx, lsy, le
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # rabad, rh, rk, rf
    # jointNominalConfig = np.array([0.34, -0.05, 0.5,1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,     
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,          
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Human.setGeneralizedCoordinate(jointNominalConfig)
    # Human.setGeneralizedVelocity(jointVelocityTarget)
    
    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)

    WeightsLift()
    # WeightsLift2()
    # WeightsLift3()
    # Throw()

    server.killServer()