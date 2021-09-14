import os
import sys
import numpy as np
import datetime
import time
import yaml
import random
import shutil
import pickle

def ParamsCopy(save_dir, ParamData):
    # 参数文件复制
    k_vir = ParamData["controller"]["K_virz"]
    Kd_up = ParamData["controller"]["K_errv_up"]
    Kd_down = ParamData["controller"]["K_errv_down"]
    RandNum = random.randint(0,100)
    today=datetime.date.today()

    # set config params file pathdir and copy file name
    # FilePath = os.path.dirname(os.path.abspath(__file__))
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/LISM_test.yaml"
    print(ParamFilePath)
    ConfigFileName = str(today) + '-LISM' + '-Kvir_' + str(k_vir) + '-Kd_up_' + str(Kd_up) + '-Kd_down_' + str(Kd_down)+ '.yaml'

    # 文件是否存在
    if os.path.exists(os.path.join(save_dir, ConfigFileName)):
        ConfigFileName = str(today) + '-LISM' + '-Kvir_' + str(k_vir) + '-Kd_up_' + str(Kd_up) + '-Kd_down_' + str(Kd_down)+ \
                        '-' + str(RandNum) + '.yaml'

    TargetFile = os.path.join(save_dir, ConfigFileName)
    shutil.copy(ParamFilePath, TargetFile)

def DataSave(BallState, EndFootState, ForceState, JointTorque, JointVel, T, ParamData):
    today=datetime.date.today()
    data = {'BallState': BallState, 'EndFootState': EndFootState, 'ForceState': ForceState, 'JointTorque': JointTorque, 'JointVel':JointVel, 'time': T}
    # print(data['state'][0:100, 0])
    x_ref = ParamData["controller"]["x_ref_FD"]
    v_ref = ParamData["controller"]["v_ref"]
    v0 = ParamData["controller"]["v_int"]
    k_vir = ParamData["controller"]["K_virz"]
    Kd_up = ParamData["controller"]["K_errv_up"]
    Kd_down = ParamData["controller"]["K_errv_down"]
    RandNum = random.randint(0,100)

    name = str(today) + '-x_ref_' + str(x_ref) + '-v0_'+ str(v0) + '-vref_' + str(v_ref) + '-K_' + str(k_vir) + '-Kd_up_' + \
           str(Kd_up) + '-Kd_down_' + str(Kd_down) + '.pkl'
    pathDir = './data/'
    print(name)

    # 如果目录不存在，则创建
    todaytime = str(today)
    save_dir = os.path.join(pathDir, todaytime)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 参数文件复制
    ParamsCopy(save_dir, ParamData)

    # 数据文件保存
    if os.path.exists(os.path.join(save_dir, name)):
        name = str(today) + '-x_ref_' + str(x_ref) + '-v0_'+ str(v0) + '-vref_' + '-K_' + str(k_vir) + '-Kd_up_' + \
               str(Kd_up) + '-Kd_down_' + str(Kd_down) + '-' + str(RandNum) + '.pkl'

    with open(os.path.join(save_dir, name), 'wb') as f:
        pickle.dump(data, f)

    return data