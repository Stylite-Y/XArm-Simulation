from cProfile import label
import numpy as np
from numpy import sin  as s
from numpy import cos as c
from casadi import *
from casadi.tools import *
import do_mpc
import os
import yaml
import matplotlib.pyplot as plt
import pickle
from matplotlib.pyplot import MultipleLocator


 
if __name__ =="__main__":
    ## get params config data
    ## /..../Simulation/model_raisim/DRMPC/SaTiZ_3D/scripts
    # FilePath1 = os.path.dirname(os.path.abspath(__file__))
    ## os.path.abspath(__file__): /.../DRMPC/SaTiZ_3D/scripts/Dynamics_MPC.py
    ## os.path.dirname(__file__): /.../DRMPC/SaTiZ_3D/scripts

    # /..../Simulation/model_raisim/DRMPC/SaTiZ_3D
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # DataFilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    f = open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + \
         '/results/001_results.pkl','rb')
    data = pickle.load(f)
    theta1 = data['mpc']['_x','theta1']
    theta2 = data['mpc']['_x','theta2']
    theta3 = data['mpc']['_x','theta3']
    t1 = data['mpc']['_u','t1']
    t2 = data['mpc']['_u','t2']

    print(theta1.shape)

    fig, axes = plt.subplots(3,1, dpi=100,figsize=(12,10))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    # ax1.plot(theta1, label='Theta 1')
    ax1.plot(theta2, label='Theta 2')
    ax1.plot(theta3, label='Theta 3')
    ax1.set_ylabel('Theta Angular ', fontsize = 15)
    ax1.legend(loc='upper right', fontsize = 12)
    y_major_locator=MultipleLocator(0.8)
    ax1.yaxis.set_major_locator(y_major_locator)
    ax1.grid()
    # ax1.set_title("Joint 1", fontsize = 20)   
    # 
    ax2.plot(theta1, label='Theta 1')
    ax2.set_ylabel('Theta Angular ', fontsize = 15)
    ax2.legend(loc='upper right', fontsize = 12)
    y_major_locator=MultipleLocator(0.04)
    ax2.yaxis.set_major_locator(y_major_locator)
    ax2.grid() 

    ax3.plot(t1, label='Toque 2')
    ax3.plot(t2, label='Toque 3')
    ax3.set_ylabel('Joint Torque ', fontsize = 15)
    ax3.legend(loc='upper right', fontsize = 12)
    ax3.grid()

    # ax2.set_title("Joint 1", fontsize = 20)

    plt.show()



