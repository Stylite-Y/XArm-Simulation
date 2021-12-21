import os
import sys
import math
import yaml
import time
import pickle
import random
import datetime
import numpy as np
import raisimpy as raisim
import matplotlib.pyplot as plt

def FileSave(T, BallVelocity, BallFroce, ball_y, ball_vy, miu_F, C_r, V_th):
    today=datetime.date.today()
    # print(data['state'][0:100, 0])
    RandNum = random.randint(0,100)
    name = str(today) + '-Vy_' + str(ball_vy) + '-miu_'+ str(miu_F) + '-Coef-Restitute_' + str(C_r) + '-V_th_' + str(V_th) + '.pkl'
    pathDir = './material_data/'
    # print(name)

    # 如果目录不存在，则创建
    todaytime = str(today)
    filename = 'Vy_' + str(ball_vy) + '-miu_'+ str(miu_F) + '-Coef-Restitute_' + str(C_r) + '-V_th_' + str(V_th)
    save_dir = os.path.join(pathDir, todaytime) + '/' + filename + '/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 数据文件保存
    if os.path.exists(os.path.join(save_dir, name)):
        name = str(today) + '-Vy_' + str(ball_vy) + '-miu_'+ str(miu_F) + '-Coef-Restitute_' + str(C_r) + '-V_th_' + str(V_th) + '.pkl'

    Data = {"BallVelocity": BallVelocity, "BallFroce": BallFroce, "time": T}

    with open(os.path.join(save_dir, name), 'wb') as f:
        pickle.dump(Data, f)

def FigPlot(T, BallVelocity, BallFroce, ball_vy, miu_F, C_r, V_th):
    fig, ax1 = plt.subplots()
    BallVel_y = BallVelocity[:, 0]
    BallVel_z = BallVelocity[:, 1]

    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Vel-y')
    p1, = ax1.plot(T, BallVel_y, label="y-axis velocity")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Vel-z', color = '#F97306')
    p2, = ax2.plot(T, BallVel_z, color = '#F97306', label="z-axis velocity")
    ax2.tick_params(axis='y', labelcolor='#F97306')
    fig.tight_layout() 
    ax1.legend(handles=[p1, p2])

    today=datetime.date.today()
    pathDir = './material_data/'
    todaytime = str(today)
    filename = 'Vy_' + str(ball_vy) + '-miu_'+ str(miu_F) + '-Coef-Restitute_' + str(C_r) + '-V_th_' + str(V_th)
    save_dir = os.path.join(pathDir, todaytime) + '/' + filename + '/'
    # print(pathDir, save_dir)
    name = str(today) + '-V_Fig' + '-Vy_' + str(ball_vy) + '-miu_'+ str(miu_F) + '-Coef-Restitute_' + str(C_r) + '-V_th_' + str(V_th) + '.png'
    plt.savefig(save_dir + name)
    plt.show()

def FigPlotForce(T, BallVelocity, BallFroce, ball_vy, miu_F, C_r, V_th):
    fig, ax1 = plt.subplots()
    BallForce_y = BallFroce[:, 0]
    BallForce_z = BallFroce[:, 1]

    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Force-y')
    p1, = ax1.plot(T, BallForce_y, label="y-axis Force")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Force-z', color = '#F97306')
    p2, = ax2.plot(T, BallForce_z, color = '#F97306', label="z-axis Force")
    ax2.tick_params(axis='y', labelcolor='#F97306')
    fig.tight_layout() 
    ax1.legend(handles=[p1, p2])

    today=datetime.date.today()
    pathDir = './material_data/'
    todaytime = str(today)
    filename = 'Vy_' + str(ball_vy) + '-miu_'+ str(miu_F) + '-Coef-Restitute_' + str(C_r) + '-V_th_' + str(V_th)
    save_dir = os.path.join(pathDir, todaytime) + '/' + filename + '/'
    # print(pathDir, save_dir)
    today=datetime.date.today()
    name = str(today) + '-Force_Fig' + '-Vy_' + str(ball_vy) + '-miu_'+ str(miu_F) + '-Coef-Restitute_' + str(C_r) + '-V_th_' + str(V_th) + '.png'
    plt.savefig(save_dir + name)
    plt.show()


def MaterialTest(ParamData, ball_y, ball_vy, miu_F, C_r, V_th):
    server = raisim.RaisimServer(world)
    server.launchServer(8080)

    EnvParam = ParamData["environment"]
    t_steps = EnvParam["t_step"]

    # sphere.setPosition(0, ball_y, 1)
    # sphere.setVelocity(0, ball_vy, -5, 0, 0, 0)
    # help(raisim.World)
    jointNominalConfig = np.array([0.0, ball_y, 1, 1.0, 0.0, 0.0, 0.0])
    # jointNominalConfig = np.array([0.34, -0.05, 0.5,1.0, 0.0, 0.0, 0.0])
    jointVelocityTarget = np.array([0.0, ball_vy, -5.0, 0.0, 0.0, 0.0])
    ball1.setGeneralizedCoordinate(np.array([ball_y, 1]))
    ball1.setGeneralizedVelocity(np.array([ball_vy, -5.0]))
    world.setMaterialPairProp("steel", "rubber", miu_F, C_r, V_th)
    world.set_contact_solver_parameters(1.0, 1.0, 1.0, 200, 1e-8)
    # world.updateMaterialProp(raisim.MaterialManager(os.path.dirname(os.path.abspath(__file__)) + "/urdf/testMaterial.xml"))
    

    BallVelocity = np.array([[0.0, 0.0]])
    BallFroce = np.array([[0.0, 0.0]])
    T = np.array([0.0])

    for i in range(5000):
        time.sleep(0.01)

        ## ======================
        ## contact detect
        ContactPoint = ball1.getContacts()
        contact_flag = False
        for c in ContactPoint:
            contact_flag = c.getlocalBodyIndex() == ball1.getBodyIdx("ball")
            if(contact_flag):
                break
            pass

        # BallVel = ball1.getLinearVelocity()
        BallPos, BallVel = ball1.getState()
        if contact_flag:
            ContactPointVel = ContactPoint[0].getImpulse()      # get contact impulse
            ContactForce = ContactPointVel / t_steps            # the contact force can be get by impluse / dt
            print(ContactForce)
        else:
            ContactForce = np.array([0.0, 0.0, 0.0])
        print(contact_flag)
        print(BallVel)
        print(BallPos[0:3])
        print(ContactForce)

        t = i * t_steps
        T = np.concatenate([T, [t]], axis = 0)
        BallVelocity = np.concatenate([BallVelocity, [[BallVel[0], BallVel[1]]]], axis = 0)
        BallFroce = np.concatenate([BallFroce, [[ContactForce[1], ContactForce[2]]]], axis = 0)

        server.integrateWorldThreadSafe()
    
    T = T[1:]
    BallVelocity = BallVelocity[1:, ]
    BallFroce = BallFroce[1:, ]

    server.killServer()
    FileSave(T, BallVelocity, BallFroce, ball_y, ball_vy, miu_F, C_r, V_th)
    FigPlot(T, BallVelocity, BallFroce, ball_vy, miu_F, C_r, V_th)
    FigPlotForce(T, BallVelocity, BallFroce, ball_vy, miu_F, C_r, V_th)
  

if __name__ == "__main__":
    # get params config data
    FilePath = os.path.dirname(os.path.abspath(__file__))
    ParamFilePath = FilePath + "/config/LISM_test.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # load activation file and urdf file
    raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
    ball1_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/ball2.urdf"
    world = raisim.World()

    # set simulation step
    t_step = ParamData["environment"]["t_step"] 
    sim_time = ParamData["environment"]["sim_time"]
    world.setTimeStep(0.005)
    ground = world.addGround(0, "steel")
    # sphere = world.addSphere(0.2, 1, "rubber")
    
    #======================
    # material collision property of change direction of force is applied
    # world.setMaterialPairProp("default", "rub", 1.0, 0.85, 0.0001)     # ball rebound model test
    # world.setMaterialPairProp("rub", "rub", 0.52, 0.8, 0.001, 0.61, 0.01)
    # world.updateMaterialProp(raisim.MaterialManager(os.path.dirname(os.path.abspath(__file__)) + "/urdf/testMaterial.xml"))
    # help(world)

    gravity = world.getGravity()
    ball1 = world.addArticulatedSystem(ball1_urdf_file)
    ball1.setName("ball1")
    # gravity = world.getGravity()

    # raisim world server setting
    # server = raisim.RaisimServer(world)
    # server.launchServer(8080)
    ball_y = np.array([0.0, 1, -1])
    ball_vy = np.array([0.0, -10.0, 5.0])
    miu_F = np.array([0.1, 0.3, 0.8])
    C_r = np.array([0.2, 0.5, 0.9])
    V_th = np.array([0.1, 2, 7])

    today=datetime.date.today()
    pathDir = './material_data/'
    todaytime = str(today)
    save_dir = os.path.join(pathDir, todaytime) + '/'
    print(pathDir, save_dir)

    # for i in range(0, len(ball_y)):
    #     for j in range(0, 3):
    #         for k in range(0, 3):
    #             for m in range(0, 3):
    #                 MaterialTest(ParamData, ball_y[i], ball_vy[i], miu_F[j], C_r[k], V_th[m])

    MaterialTest(ParamData, ball_y[2], ball_vy[2], miu_F[0], C_r[2], V_th[0])

    # file save
    # FileFlag = ParamData["environment"]["FileFlag"] 
    # FileSave.DataSave(Data, ParamData, FileFlag)

    # # data visulization
    # Data = {'BallPos': BallPosition, 'BallVel': BallVelocity, 'ExternalForce': ExternalForce, 'time': T}

    # DataPlot(Data)
    # visualization.DataPlot(Data)
    # visualization.RealCmpRef(Data)

    # print("force, ", ForceState[0:100, 1])