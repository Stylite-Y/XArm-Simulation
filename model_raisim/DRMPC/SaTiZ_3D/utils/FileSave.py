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
    n_horizons = ParamData["MPCController"]["n_horizons"]
    t_force = ParamData["MPCController"]["t_force"]
    v_zref = ParamData["environment"]["v_zref"]
    xq = ParamData["MPCController"]["xq"]
    vxq = ParamData["MPCController"]["vxq"]
    uxr = ParamData["MPCController"]["uxr"]
    # RandNum = random.randint(0,100)
    today=datetime.date.today()

    # set config params file pathdir and copy file name
    # FilePath = os.path.dirname(os.path.abspath(__file__))
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/LISM_test.yaml"
    print(ParamFilePath)
    if FileFlag == 0:
        ConfigFileName = str(today) + '-LISM' + '-V' + '-tstep_' + str(sim_t_step) + '-horizon_'+ str(n_horizons) + '-tforce_' + str(t_force) + '-vzref_' + \
                        str(v_zref) + '-xq_' + str(xq) + '-vxq_' + str(vxq) + '-uxr_' + str(uxr) + '.yaml'
    elif FileFlag == 1:
        ConfigFileName = str(today) + '-LISM' + '-TRIGON' + '-tstep_' + str(sim_t_step) + '-horizon_'+ str(n_horizons) + '-tforce_' + str(t_force) + '-vzref_' + \
                        str(v_zref) + '-xq_' + str(xq) + '-vxq_' + str(vxq) + '-uxr_' + str(uxr) + '.yaml'

    # 文件是否存在
    if os.path.exists(os.path.join(save_dir, ConfigFileName)):
        if FileFlag == 0:
            ConfigFileName = str(today) + '-LISM' + '-V' + '-tstep_' + str(sim_t_step) + '-horizon_'+ str(n_horizons) + '-tforce_' + str(t_force) + '-vzref_' + \
                            str(v_zref) + '-xq_' + str(xq) + '-vxq_' + str(vxq) + '-uxr_' + str(uxr)  + '-' + str(RandNum) + '.yaml'
        elif FileFlag == 1:
            ConfigFileName = str(today) + '-LISM' + '-TRIGON' + '-tstep_' + str(sim_t_step) + '-horizon_'+ str(n_horizons) + '-tforce_' + str(t_force) + '-vzref_' + \
                            str(v_zref) + '-xq_' + str(xq) + '-vxq_' + str(vxq) + '-uxr_' + str(uxr)  + '-' + str(RandNum) + '.yaml'

    TargetFile = os.path.join(save_dir, ConfigFileName)
    shutil.copy(ParamFilePath, TargetFile)

def DataSave(Data, ParamData, FileFlag):
    today=datetime.date.today()
    # print(data['state'][0:100, 0])
    sim_t_step = ParamData["environment"]["t_step"]
    n_horizons = ParamData["MPCController"]["n_horizons"]
    t_force = ParamData["MPCController"]["t_force"]
    v_zref = ParamData["environment"]["v_zref"]
    xq = ParamData["MPCController"]["xq"]
    vxq = ParamData["MPCController"]["vxq"]
    uxr = ParamData["MPCController"]["uxr"]
    RandNum = random.randint(0,100)
    if FileFlag == 0:
        name = str(today) + '-V' + '-tstep_' + str(sim_t_step) + '-horizon_'+ str(n_horizons) + '-tforce_' + str(t_force) + '-vzref_' + str(v_zref) + '-xq_' + \
            str(xq) + '-vxq_' + str(vxq) + '-uxr_' + str(uxr) + '.pkl'
    elif FileFlag == 1:
        name = str(today) + '-TRIGON' + '-tstep_' + str(sim_t_step) + '-horizon_'+ str(n_horizons) + '-tforce_' + str(t_force) + '-vzref_' + str(v_zref) + '-xq_' + \
            str(xq) + '-vxq_' + str(vxq) + '-uxr_' + str(uxr) + '.pkl'
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
        if FileFlag == 0:
            name = str(today) + '-V' + '-tstep_' + str(sim_t_step) + '-horizon_'+ str(n_horizons) + '-tforce_' + str(t_force) + '-vzref_' + str(v_zref) + '-xq_' + \
               str(xq) + '-vxq_' + str(vxq) + '-uxr_' + str(uxr) + '-' + str(RandNum) + '.pkl'
        elif FileFlag == 1:
            name = str(today) + '-TRIGON' + '-tstep_' + str(sim_t_step) + '-horizon_'+ str(n_horizons) + '-tforce_' + str(t_force) + '-vzref_' + str(v_zref) + '-xq_' + \
                str(xq) + '-vxq_' + str(vxq) + '-uxr_' + str(uxr) + '-' + str(RandNum) + '.pkl'

    with open(os.path.join(save_dir, name), 'wb') as f:
        pickle.dump(Data, f)