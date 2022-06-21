import pybullet as p
import pybullet_data
import os
import numpy as np
import matplotlib.pyplot as plt
from model import control_compute

# UrdfPath="G:\Master\DataCollect\MotionCollect\yu\code\CXK_vision_5.urdf"
# filepath="G:\Master\lab\Manipulator_Arm\Dynamics\data\walk2.calc"

UrdfPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/urdf/human2.urdf"
Datapath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/walk_bvhlocal2.calc"

posture_data = []

# read the data from document
with open(Datapath, 'r') as f:      #通过使用with...as...不用手动关闭文件。当执行完内容后，自动关闭文件
    for i in range(6):               #跳过前五行
        next(f)
    for lines in f.readlines():       #依次读取每行
        posture_data.append(list(map(float, lines.split())))

#总帧数
numFrames = len(posture_data)

# connect the GUI
p.connect(p.GUI)
p.setGravity(0, 0, 0)
p.setPhysicsEngineParameter(numSolverIterations=1)

# switch the path and load the inner plane urdf file
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# import the model written by URDF
p.setAdditionalSearchPath(os.getcwd())
basePosition = [0, 0, 0]
baseOrientation = p.getQuaternionFromEuler([0, 0, 0])
# GlobalScaling将对URDF模型应用比例因子
humanoid = p.loadURDF(UrdfPath, basePosition, baseOrientation, globalScaling=1)
# humanoid = p.loadURDF(UrdfPath)

for j in range(p.getNumJoints(humanoid)):   #p.getNumJoints得到关节数
    i = p.getJointInfo(humanoid, j)         #p.getJointInfo得到关节信息
    print("joint[", j, "].name=", i[1])     #i[1]jointIndex
    print("link [", j, "].name=", i[12])    #i[12]linkname
    print(i)

# change every joints' dynamics parameter and links' visual color
for j in range(p.getNumJoints(humanoid)):
    p.changeDynamics(humanoid, j, linearDamping=1, angularDamping=1)#线性阻尼和角阻尼
#     p.changeVisualShape(humanoid, j, rgbaColor=[1, 1, 1, 1])

# kp是用于位置控制的位置增益
kp = 1
# normalize
num_interval = 10
t = np.linspace(0, 1, num_interval)
print(t)

cyaw = 30
cpitch = -15
cdist = 2.5
while p.isConnected():
    p.setRealTimeSimulation(0)
    #帧数11000-12600
    for n in range(300, 1000):
        # print(n)
        #当前帧姿态数据
        Posture_data_current1 = [posture_data[n][7], posture_data[n][8], posture_data[n][9],posture_data[n][6]]
        #下一帧姿态数据
        Posture_data_next1 = [posture_data[n + 1][7], posture_data[n + 1][8], posture_data[n + 1][9], posture_data[n + 1][6]]
        #region test
        Posture_data_current2 = [posture_data[n][23], posture_data[n][24],posture_data[n][25],posture_data[n][22]]
        Posture_data_next2 = [posture_data[n + 1][23], posture_data[n + 1][24], posture_data[n + 1][25], posture_data[n + 1][22]]

        Posture_data_current3 = [posture_data[n][39], posture_data[n][40], posture_data[n][41], posture_data[n][38]]
        Posture_data_next3 = [posture_data[n + 1][39], posture_data[n + 1][40], posture_data[n + 1][41], posture_data[n + 1][38]]

        Posture_data_current4 = [posture_data[n][55], posture_data[n][56], posture_data[n][57], posture_data[n][54]]
        Posture_data_next4 = [posture_data[n + 1][55], posture_data[n + 1][56], posture_data[n + 1][57], posture_data[n + 1][54]]

        Posture_data_current5 = [posture_data[n][71], posture_data[n][72], posture_data[n][73], posture_data[n][70]]
        Posture_data_next5 = [posture_data[n + 1][71], posture_data[n + 1][72], posture_data[n + 1][73], posture_data[n + 1][70]]

        Posture_data_current6 = [posture_data[n][87], posture_data[n][88], posture_data[n][89], posture_data[n][86]]
        Posture_data_next6 = [posture_data[n + 1][87], posture_data[n + 1][88], posture_data[n + 1][89], posture_data[n + 1][86]]

        Posture_data_current7 = [posture_data[n][103], posture_data[n][104], posture_data[n][105], posture_data[n][102]]
        Posture_data_next7 = [posture_data[n + 1][103], posture_data[n + 1][104], posture_data[n + 1][105], posture_data[n + 1][102]]

        Posture_data_current8 = [posture_data[n][119], posture_data[n][120], posture_data[n][121], posture_data[n][118]]
        Posture_data_next8 = [posture_data[n + 1][119], posture_data[n + 1][120], posture_data[n + 1][121],posture_data[n + 1][118]]

        Posture_data_current9 = [posture_data[n][135], posture_data[n][136], posture_data[n][137], posture_data[n][134]]
        Posture_data_next9 = [posture_data[n + 1][135], posture_data[n + 1][136], posture_data[n + 1][137], posture_data[n + 1][134]]

        Posture_data_current10 = [posture_data[n][151], posture_data[n][152], posture_data[n][153], posture_data[n][150]]
        Posture_data_next10 = [posture_data[n + 1][151], posture_data[n + 1][152], posture_data[n + 1][153], posture_data[n + 1][150]]

        Posture_data_current11 = [posture_data[n][167], posture_data[n][168], posture_data[n][169], posture_data[n][166]]
        Posture_data_next11 = [posture_data[n + 1][167], posture_data[n + 1][168], posture_data[n + 1][169], posture_data[n + 1][166]]

        Posture_data_current12 = [posture_data[n][183], posture_data[n][184], posture_data[n][185], posture_data[n][182]]
        Posture_data_next12 = [posture_data[n + 1][183], posture_data[n + 1][184], posture_data[n + 1][185], posture_data[n + 1][182]]

        Posture_data_current13 = [posture_data[n][199], posture_data[n][200], posture_data[n][201], posture_data[n][198]]
        Posture_data_next13 = [posture_data[n + 1][199], posture_data[n + 1][200], posture_data[n + 1][201], posture_data[n + 1][198]]

        Posture_data_current14 = [posture_data[n][215], posture_data[n][216], posture_data[n][217], posture_data[n][214]]
        Posture_data_next14 = [posture_data[n + 1][215], posture_data[n + 1][216], posture_data[n + 1][217], posture_data[n + 1][214]]

        Posture_data_current15 = [posture_data[n][231], posture_data[n][232], posture_data[n][233], posture_data[n][230]]
        Posture_data_next15 = [posture_data[n + 1][231], posture_data[n + 1][232], posture_data[n + 1][233], posture_data[n + 1][230]]

        Posture_data_current16 = [posture_data[n][247], posture_data[n][248], posture_data[n][249], posture_data[n][246]]
        Posture_data_next16 = [posture_data[n + 1][247], posture_data[n + 1][248], posture_data[n + 1][249], posture_data[n + 1][246]]

        Posture_data_current17 = [posture_data[n][263], posture_data[n][264], posture_data[n][265], posture_data[n][262]]
        Posture_data_next17 = [posture_data[n + 1][263], posture_data[n + 1][264], posture_data[n + 1][265], posture_data[n + 1][262]]
        #endregion

        
        # Control = control_compute(humanoid)
        # testdata = [posture_data[300][7], posture_data[300][8], posture_data[300][9],posture_data[300][6]]
        # print(testdata)
        # # testdata1 = [posture_data[301][7], posture_data[301][8], posture_data[301][9],posture_data[301][6]]
        # p.setJointMotorControlMultiDof(humanoid, 1, p.POSITION_CONTROL, targetPosition=testdata, positionGain=kp)
        # p.stepSimulation()

        for j in range(len(t)):
            interval = t[j]  #定义当前帧和下一帧之间的插值
            Control = control_compute(humanoid)  # 调用“control compute”
            Control.compute1(Posture_data_current1, Posture_data_next1,interval)  # 获取每个关节的四元数以进行下一个位置控制
            Control.Position_control1(kp)  #每个关节的位置控制

            Control.compute2(Posture_data_current2, Posture_data_next2, interval)
            Control.Position_control2(kp)

            Control.compute3(Posture_data_current3, Posture_data_next3, interval)
            Control.Position_control3(kp)

            Control.compute4(Posture_data_current4, Posture_data_next4, interval)
            Control.Position_control4(kp)

            Control.compute5(Posture_data_current5, Posture_data_next5, interval)
            Control.Position_control5(kp)

            Control.compute6(Posture_data_current6, Posture_data_next6, interval)
            Control.Position_control6(kp)

            Control.compute7(Posture_data_current7, Posture_data_next7, interval)
            Control.Position_control7(kp)

            # Control.compute8(Posture_data_current8, Posture_data_next8, interval)
            # Control.Position_control8(kp)

            Control.compute9(Posture_data_current9, Posture_data_next9, interval)
            Control.Position_control9(kp)

            Control.compute10(Posture_data_current10, Posture_data_next10, interval)
            Control.Position_control10(kp)

            Control.compute11(Posture_data_current11, Posture_data_next11, interval)
            Control.Position_control11(kp)

            # Control.compute12(Posture_data_current12, Posture_data_next12, interval)
            # Control.Position_control12(kp)

            Control.compute13(Posture_data_current13, Posture_data_next13, interval)
            Control.Position_control13(kp)

            Control.compute14(Posture_data_current14, Posture_data_next14, interval)
            Control.Position_control14(kp)

            Control.compute15(Posture_data_current15, Posture_data_next15, interval)
            Control.Position_control15(kp)

            Control.compute16(Posture_data_current16, Posture_data_next16, interval)
            Control.Position_control16(kp)

            Control.compute17(Posture_data_current17, Posture_data_next17, interval)
            Control.Position_control17(kp)

        # 模拟过程中调整摄像机范围
        keys = p.getKeyboardEvents()
        if keys.get(100):  # D
            cyaw += 0.01
        if keys.get(97):  # A
            cyaw -= 0.01
        if keys.get(99):  # C
            cpitch += 0.01
        if keys.get(102):  # F
            cpitch -= 0.01
        if keys.get(122):  # Z
            cdist += .001
        if keys.get(120):  # X
            cdist -= .001
        cubePos, cubeOrn = p.getBasePositionAndOrientation(humanoid)
        p.resetDebugVisualizerCamera(cameraDistance=cdist, cameraYaw=cyaw, cameraPitch=cpitch,
                                         cameraTargetPosition=cubePos)
