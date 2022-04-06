import os
import sys
import numpy as np
import datetime
import time
import yaml
import random
import shutil
import pickle

def ParamsCopy(save_dir, ParamData, FileFlag, RandNum):
    # 参数文件复制
    sim_t_step = ParamData["environment"]["t_step"]
    T_Period = ParamData["PaperSim"]["T_Period"]
    Amp = ParamData["PaperSim"]["A"]
    K_Bdes_p = ParamData["PaperSim"]["K_Bdes_p"]
    K_Gdes_p = ParamData["PaperSim"]["K_Gdes_p"]
    z0 = ParamData["PaperSim"]["z0"]
    # RandNum = random.randint(0,100)
    today=datetime.date.today()

    # set config params file pathdir and copy file name
    # FilePath = os.path.dirname(os.path.abspath(__file__))
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/LISM_test.yaml"
    print(ParamFilePath)
    if FileFlag == 1:
        ConfigFileName = str(today) + '-LISM' + '-Z_traj_FD' + '-tstep_' + str(sim_t_step) + '-TPeriod_'+ str(T_Period) + \
            '-Amp_' + str(Amp) + '-z0_' + str(z0) + '-K_Bdes_p_' + str(K_Bdes_p) + '-K_Gdes_p_' + str(K_Gdes_p)+ '.yaml'

    # 文件是否存在
    if os.path.exists(os.path.join(save_dir, ConfigFileName)):
        if FileFlag == 1:
            ConfigFileName = str(today) + '-LISM' + '-Z_traj_FD' + '-tstep_' + str(sim_t_step) + '-TPeriod_'+ str(T_Period) + \
            '-Amp_' + str(Amp) + '-z0_' + str(z0) + '-K_Bdes_p_' + str(K_Bdes_p) + '-K_Gdes_p_' + str(K_Gdes_p) + str(RandNum) + '.yaml'

    TargetFile = os.path.join(save_dir, ConfigFileName)
    shutil.copy(ParamFilePath, TargetFile)

def DataSave(Data, ParamData, FileFlag):
    today=datetime.date.today()
    # print(data['state'][0:100, 0])
    sim_t_step = ParamData["environment"]["t_step"]
    T_Period = ParamData["PaperSim"]["T_Period"]
    Amp = ParamData["PaperSim"]["A"]
    K_Bdes_p = ParamData["PaperSim"]["K_Bdes_p"]
    K_Gdes_p = ParamData["PaperSim"]["K_Gdes_p"]
    z0 = ParamData["PaperSim"]["z0"]
    RandNum = random.randint(0,100)
    if FileFlag == 1:
        name = str(today) + '-Z_traj_FD' + '-tstep_' + str(sim_t_step) + '-TPeriod_'+ str(T_Period) + \
            '-Amp_' + str(Amp) + '-z0_' + str(z0) + '-K_Bdes_p_' + str(K_Bdes_p) + '-K_Gdes_p_' + str(K_Gdes_p) + '.pkl'
    pathDir = './data/'
    print(name)

    # 如果目录不存在，则创建
    todaytime = str(today)
    save_dir = os.path.join(pathDir, todaytime)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 参数文件复制
    ParamsCopy(save_dir, ParamData, FileFlag, RandNum)

    # 数据文件保存
    if os.path.exists(os.path.join(save_dir, name)):
        if FileFlag == 1:
            name = str(today) + '-Z_traj_FD' + '-tstep_' + str(sim_t_step) + '-TPeriod_'+ str(T_Period) + \
            '-Amp_' + str(Amp) + '-z0_' + str(z0) + '-K_Bdes_p_' + str(K_Bdes_p) + '-K_Gdes_p_' + str(K_Gdes_p) + str(RandNum) + '.pkl'

    with open(os.path.join(save_dir, name), 'wb') as f:
        pickle.dump(Data, f)