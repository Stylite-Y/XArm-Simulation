import pybullet as p

# Def the joint_index
# These indexes number of joints are from the "Noitom Motion Capture" Instruction
# left_foot_index = 6

# def the joint index
# These joints' indexes are from the Incomplete_model urdf model
hips = 0
right_upleg = 2
right_leg = 3
right_foot = 4
left_upleg = 5
left_leg = 6
left_foot = 7
neck = 8
right_shoulder = 9
right_arm = 9
right_forearm = 11
right_hand = 12
left_shoulder = 13
left_arm = 14
left_forearm = 15
left_hand = 16
head = 17

help(p.getQuaternionSlerp)


class control_compute(object):
    def __init__(self, robotid):
       self.id = robotid
       base_init_pos = [0, 0, 2]
       base_init_orn = p.getQuaternionFromEuler([0, 0.0, 0])
       p.resetBasePositionAndOrientation(self.id, base_init_pos, base_init_orn)

    def compute1(self, Posture_data_current1, Posture_data_next1, interval):
         origin_hips = Posture_data_current1
         new_hips = Posture_data_next1
         self.hipsRot = p.getQuaternionSlerp(origin_hips, new_hips, interval)
        #  print(self.hipsRot)

    def Position_control1(self, kp):
         p.setJointMotorControlMultiDof(self.id, hips, p.POSITION_CONTROL, targetPosition=self.hipsRot,
                                           positionGain=kp)

         p.enableJointForceTorqueSensor(self.id, hips, 1)  # 添加关节传感器
         JointForceTorque = p.getJointState(self.id, hips)  # 计算关节力矩

         p.stepSimulation()
         return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute2(self, Posture_data_current2, Posture_data_next2, interval):
         origin_right_upleg = Posture_data_current2
         new_right_upleg = Posture_data_next2
         self.right_uplegRot = p.getQuaternionSlerp(origin_right_upleg, new_right_upleg, interval)
    def Position_control2(self, kp):
         p.setJointMotorControlMultiDof(self.id, right_upleg, p.POSITION_CONTROL, targetPosition=self.right_uplegRot,
                                           positionGain=kp)
         p.enableJointForceTorqueSensor(self.id, right_upleg, 1)  # 添加关节传感器
         JointForceTorque = p.getJointState(self.id, right_upleg)  # 计算关节力矩

         p.stepSimulation()
         return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute3(self, Posture_data_current3, Posture_data_next3, interval):
         origin_right_leg = Posture_data_current3
         new_right_leg = Posture_data_next3
         self.right_legRot = p.getQuaternionSlerp(origin_right_leg, new_right_leg, interval)
    def Position_control3(self, kp):
        p.setJointMotorControlMultiDof(self.id, right_leg, p.POSITION_CONTROL, targetPosition=self.right_legRot,
                                           positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, right_leg, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, right_leg)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute4(self, Posture_data_current4, Posture_data_next4, interval):
        origin_right_foot = Posture_data_current4
        new_right_foot = Posture_data_next4
        self.right_footRot = p.getQuaternionSlerp(origin_right_foot, new_right_foot, interval)
    def Position_control4(self, kp):
        p.setJointMotorControlMultiDof(self.id, right_foot, p.POSITION_CONTROL, targetPosition=self.right_footRot,
                                               positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, right_foot, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, right_foot)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute5(self, Posture_data_current5, Posture_data_next5, interval):
        origin_left_upleg = Posture_data_current5
        new_left_upleg = Posture_data_next5
        self.left_uplegRot = p.getQuaternionSlerp(origin_left_upleg, new_left_upleg, interval)
    def Position_control5(self, kp):
        p.setJointMotorControlMultiDof(self.id, left_upleg, p.POSITION_CONTROL,targetPosition=self.left_uplegRot,positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, left_upleg, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, left_upleg)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute6(self, Posture_data_current6, Posture_data_next6, interval):
        origin_left_leg = Posture_data_current6
        new_left_leg = Posture_data_next6
        self.left_legRot = p.getQuaternionSlerp(origin_left_leg, new_left_leg, interval)
    def Position_control6(self, kp):
        p.setJointMotorControlMultiDof(self.id, left_leg, p.POSITION_CONTROL, targetPosition=self.left_legRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, left_leg, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, left_leg)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute7(self, Posture_data_current7, Posture_data_next7, interval):
        # calculate the orientation for each joint in frame real displaced by quaternion
        origin_left_foot = Posture_data_current7
        new_left_foot = Posture_data_next7
        self.left_footRot = p.getQuaternionSlerp(origin_left_foot, new_left_foot, interval)
    def Position_control7(self, kp):
        # set the joint to the target position
        # Note:
        # p.setJointMotorControlMultiDof() this API is not in QuickStart.pdf, it is used for spherical joint control
        # The type of targetPosition is quaternion
        p.setJointMotorControlMultiDof(self.id, left_foot, p.POSITION_CONTROL, targetPosition=self.left_footRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, left_foot, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, left_foot)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute8(self, Posture_data_current8, Posture_data_next8, interval):
        origin_right_shoulder = Posture_data_current8
        new_right_shoulder = Posture_data_next8
        self.right_shoulderRot = p.getQuaternionSlerp(origin_right_shoulder, new_right_shoulder, interval)
    def Position_control8(self, kp):
       p.setJointMotorControlMultiDof(self.id, right_shoulder, p.POSITION_CONTROL, targetPosition=self.right_shoulderRot,
                                       positionGain=kp)
       p.enableJointForceTorqueSensor(self.id, right_shoulder, 1)  # 添加关节传感器
       JointForceTorque = p.getJointState(self.id, right_shoulder)  # 计算关节力矩

       p.stepSimulation()
       return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute9(self, Posture_data_current9, Posture_data_next9, interval):
        origin_right_arm = Posture_data_current9
        new_right_arm = Posture_data_next9
        self.right_armRot = p.getQuaternionSlerp(origin_right_arm, new_right_arm, interval)
    def Position_control9(self, kp):
        p.setJointMotorControlMultiDof(self.id, right_arm, p.POSITION_CONTROL, targetPosition=self.right_armRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, right_arm, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, right_arm)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute10(self, Posture_data_current10, Posture_data_next10, interval):
        origin_right_forearm = Posture_data_current10
        new_right_forearm = Posture_data_next10
        self.right_forearmRot = p.getQuaternionSlerp(origin_right_forearm, new_right_forearm, interval)
    def Position_control10(self, kp):
        p.setJointMotorControlMultiDof(self.id, right_forearm, p.POSITION_CONTROL, targetPosition=self.right_forearmRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, right_forearm, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, right_forearm)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute11(self, Posture_data_current11, Posture_data_next11, interval):
        origin_right_hand = Posture_data_current11
        new_right_hand = Posture_data_next11
        self.right_handRot = p.getQuaternionSlerp(origin_right_hand, new_right_hand, interval)
    def Position_control11(self, kp):
        p.setJointMotorControlMultiDof(self.id, right_hand, p.POSITION_CONTROL, targetPosition=self.right_handRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, right_hand, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, right_hand)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute12(self, Posture_data_current12, Posture_data_next12, interval):
        origin_left_shoulder = Posture_data_current12
        new_left_shoulder = Posture_data_next12
        self.left_shoulderRot = p.getQuaternionSlerp(origin_left_shoulder, new_left_shoulder, interval)
    def Position_control12(self, kp):
        p.setJointMotorControlMultiDof(self.id, left_shoulder, p.POSITION_CONTROL, targetPosition=self.left_shoulderRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, left_shoulder, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, left_shoulder)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute13(self, Posture_data_current13, Posture_data_next13, interval):
        origin_left_arm = Posture_data_current13
        new_left_arm = Posture_data_next13
        self.left_armRot = p.getQuaternionSlerp(origin_left_arm, new_left_arm, interval)
    def Position_control13(self, kp):
        p.setJointMotorControlMultiDof(self.id, left_arm, p.POSITION_CONTROL, targetPosition=self.left_armRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, left_arm, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, left_arm)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute14(self, Posture_data_current14, Posture_data_next14, interval):
        origin_left_forearm = Posture_data_current14
        new_left_forearm = Posture_data_next14
        self.left_forearmRot = p.getQuaternionSlerp(origin_left_forearm, new_left_forearm, interval)
    def Position_control14(self, kp):
        p.setJointMotorControlMultiDof(self.id, left_forearm, p.POSITION_CONTROL, targetPosition=self.left_forearmRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, left_forearm, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, left_forearm)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute15(self, Posture_data_current15, Posture_data_next15, interval):
        origin_left_hand = Posture_data_current15
        new_left_hand = Posture_data_next15
        self.left_handRot = p.getQuaternionSlerp(origin_left_hand, new_left_hand, interval)
    def Position_control15(self, kp):
        p.setJointMotorControlMultiDof(self.id, left_hand, p.POSITION_CONTROL, targetPosition=self.left_handRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, left_hand, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, left_hand)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute16(self, Posture_data_current16, Posture_data_next16, interval):
        origin_head = Posture_data_current16
        new_head = Posture_data_next16
        self.headRot = p.getQuaternionSlerp(origin_head, new_head, interval)
    def Position_control16(self, kp):
        p.setJointMotorControlMultiDof(self.id, head, p.POSITION_CONTROL, targetPosition=self.headRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, head, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, head)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]

    def compute17(self, Posture_data_current17, Posture_data_next17, interval):
        origin_neck = Posture_data_current17
        new_neck = Posture_data_next17
        self.neckRot = p.getQuaternionSlerp(origin_neck, new_neck, interval)
    def Position_control17(self, kp):
        p.setJointMotorControlMultiDof(self.id, neck, p.POSITION_CONTROL, targetPosition=self.neckRot,
                                       positionGain=kp)
        p.enableJointForceTorqueSensor(self.id, neck, 1)  # 添加关节传感器
        JointForceTorque = p.getJointState(self.id, neck)  # 计算关节力矩

        p.stepSimulation()
        return JointForceTorque[2]  # 返回力和力矩 [Fx, Fy, Fz, Mx, My, Mz]