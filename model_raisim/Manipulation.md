

## Dribbling based on the MPC method

- [Dribbling based on the MPC method](#dribbling-based-on-the-mpc-method)
  - [球的单自由度运动](#球的单自由度运动)
    - [球的运动方程](#球的运动方程)
    - [仿真实验过程](#仿真实验过程)
    - [方法验证仿真](#方法验证仿真)
      - [方法一：基于MPC构造完整的动力学模型（运用近似分段动力学）](#方法一基于mpc构造完整的动力学模型运用近似分段动力学)
        - [实验组一：完整分段动力学](#实验组一完整分段动力学)
        - [实验组二：更改控制速度（拍球频率）](#实验组二更改控制速度拍球频率)
        - [实验组三：更改目标函数权重系数](#实验组三更改目标函数权重系数)
        - [实验组四：更改输入边界条件](#实验组四更改输入边界条件)
      - [方法二：基于MPC的力参考轨迹](#方法二基于mpc的力参考轨迹)
        - [实验组一：虚拟弹簧和常量力构造力参考轨迹](#实验组一虚拟弹簧和常量力构造力参考轨迹)
      - [方法三：基于动力学控制的力参考轨迹](#方法三基于动力学控制的力参考轨迹)
        - [实验组一：虚拟弹簧和常量力构造力参考轨迹](#实验组一虚拟弹簧和常量力构造力参考轨迹-1)
        - [实验组二：更改运球高度验证控制器](#实验组二更改运球高度验证控制器)
        - [实验组三：更改运球期望速度验证控制器](#实验组三更改运球期望速度验证控制器)
        - [实验组四：更改运球期望速度和高度验证控制器](#实验组四更改运球期望速度和高度验证控制器)
    - [Raisim 仿真](#raisim-仿真)
  - [<font color = 'yellow'>篮球运动三维运动控制</font>](#font-color--yellow篮球运动三维运动控制font)
    - [基于力反馈的运动控制方法](#基于力反馈的运动控制方法)
      - [实验一：球的二维平面控制](#实验一球的二维平面控制)
      - [实验二：球的三维三角运动控制](#实验二球的三维三角运动控制)
      - [实验三：球的三维定点运动控制](#实验三球的三维定点运动控制)
    - [基于MPC和轨迹优化的球的的运动控制方法](#基于mpc和轨迹优化的球的的运动控制方法)
      - [基于MPC和仅给定落脚点的球的运动控制](#基于mpc和仅给定落脚点的球的运动控制)
      - [基于三次函数的参考轨迹的MPC球运动控制](#基于三次函数的参考轨迹的mpc球运动控制)
      - [基于五次样条曲线轨迹优化的mpc球运动控制](#基于五次样条曲线轨迹优化的mpc球运动控制)
  - [<font color='yellow'>臂球系统的三维运动控制</font>](#font-coloryellow臂球系统的三维运动控制font)
      - [基于MPC和轨迹优化的臂球系统二维V型的运动控制。](#基于mpc和轨迹优化的臂球系统二维v型的运动控制)
  - [<font color='yellow'>臂球系统的高速稳定运动控制</font>](#font-coloryellow臂球系统的高速稳定运动控制font)
    - [基于Z向轨迹和PD姿态控制的运动](#基于z向轨迹和pd姿态控制的运动)


### 球的单自由度运动

<font color = 'yellow'>只考虑球的运动，机械臂只作为力源，不写入状态量中</font>

#### 球的运动方程



#### 仿真实验过程

实验组一

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: ``x_ref = -0.1        v_ref = -3 m/s​``
   - 目标函数使得球达到给定参考速度:  `` J = (dx_b - v_ref) ** 2 + r * u ** 2 ``
   - 边界条件:  ``x = [-10, 10],   u = [-50,  50]``<font color = 'coral'>(实际运动中，x > -0.1，u <= 0)</font>
   - 初始条件:  ``x0 = -0.1，v0 =  2 m/s``
   - 控制条件:  不考虑机械臂与球间断性接触

2. <font color = 'yellow'>仿真结果</font>

   <img src="https://i.loli.net/2021/05/28/lvouiyexAWLRpBg.png" alt="Figure_1" style="zoom: 80%;" />

3. <font color = 'yellow'>结果分析</font>

   从图中可以发现，MPC控制i虽然能将球的速度控制到预期的v_ref，但是所需的控制时间太长，由结果可以换分析，大概在0.8s左右速度达到期望水平，但是此时已经远远超过x = -0.1 m的实际运动限制，即该控制器的控制速率太慢



实验组二

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: ``x_ref = -0.1        v_ref = -3 m/s​``
   - 目标函数使得球达到给定参考速度:  `` J = (dx_b - v_ref) ** 2 + r * u ** 2 ``
   - 边界条件:  ``x = [-10, 10],   u = [-50,  50]``<font color = 'coral'>(实际运动中，x > - 0.1，u <= 0)</font>
   - 初始条件:  ``x0 = -0.1，v0 =  2 m/s``
   - 控制条件:  考虑机械臂与球间断性接触（在x = -0.1处分离），即在x = - 0.1之后，将计算得到的球反弹回的速度作为新的初始量给控制器

2. <font color = 'yellow'>仿真结果</font>

   <img src="https://i.loli.net/2021/05/28/sMqdR6pK1iljcVo.png" alt="Figure_2"  />

3. <font color = 'yellow'>结果分析</font>

   从实验结果可以 分析出控制器的确能实验周期运动，但是速度无法达到期望值就开始了新的周期，这即是实验组一出现的问题，使得无法在周期运动时满足要求

4. <font color = 'yellow'>问题解决方法</font>

   - 一种可能是动力学模型中缺少球自由运动的过程，使得估计时动力学错误，因此可以加上自由运动的动力学尝试一下结果
   - 另一种给一个x的参考轨迹，并在优化函数中加上参考轨迹项。



实验组三

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: ``x_ref = -0.1        v_ref = -3 m/s​``
   - 目标函数使得球达到给定参考速度:  `` J = (dx_b - v_ref) ** 2 + r * u ** 2 ``
   - 边界条件:  ``x = [-0.5, 0.5],   u = [-50,  50]``<font color = 'coral'>(实际运动中，x > - 0.1，u <= 0)</font>
   - 初始条件:  ``x0 = -0.1，v0 =  2 m/s``
   - 控制条件:  考虑机械臂与球间断性接触（在x = -0.1处分离），即在x = - 0.1之后，将计算得到的球反弹回的速度作为新的初始量给控制器

2. <font color = 'yellow'>仿真结果</font>

   <img src="https://i.loli.net/2021/05/28/ptamHMgFfiujQqJ.png" alt="Figure_3" style="zoom:80%;" />

3. <font color = 'yellow'>结果分析</font>

   从实验组二的结果看到球的实际运动范围远小于设置边界条件，因此本实验组便将位移边界缩小，但是出现了一些问题，首先球的运动范围变大、运动频率减小、运动速度有趋于0的趋势、力作用情况差异较大，<font color = 'coral'>为何只改变位移边界范围导致结果差异这么大？</font>



实验组四

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: ``x_ref = -0.1        v_ref = -3 m/s​``
     - 目标函数使得球达到给定参考速度:  `` J = (dx_b - v_ref) ** 2 + r * u ** 2 ``
   - 边界条件:  ``x = [-10.5, 10.5],   u = [-50,  0]``<font color = 'coral'>(实际运动中，x > - 0.1，u <= 0)</font>
   - 初始条件:  ``x0 = -0.1，v0 =  2 m/s``
   - 控制条件:  考虑机械臂与球间断性接触（在x = -0.1处分离），即在x = - 0.1之后，将计算得到的球反弹回的速度作为新的初始量给控制器

2. <font color = 'yellow'>仿真结果</font>

   ![Figure_4](https://i.loli.net/2021/05/28/FvRXhW1HMumjt68.png)

3. <font color = 'yellow'>结果分析</font>

   从实验组二的结果看到球的实际运动所需外力远小于设置边界条件，因此本实验组便将输入力边界缩小，但是出现了一些问题，结果与实验三部分现象相同：球的运动范围变大、运动频率减小、外力完全不作用整个运动过程全靠重力作用，<font color = 'coral'>为何只结果差异这么大？</font>



#### 方法验证仿真

##### 方法一：基于MPC构造完整的动力学模型（运用近似分段动力学）

###### 实验组一：完整分段动力学

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: ``x_ref = -0.1        v_ref = -3 m/s​``

   - 目标函数使得球达到给定参考速度:  `` J = (dx_b - v_ref) ** 2 + r * u ** 2 ``

   - 边界条件:  ``x = [-10.5, 10.5],   u = [-500,  0]``<font color = 'coral'>(实际运动中，x > - 0.1，u <= 0)</font>

   - 初始条件:  ``x0 = -0.1，v0 =  2 m/s``

   - 控制条件:  构造球与机械手接触、球的自由飞行、球的地面反弹（用弹簧模拟），并通过logistic函数近似阶跃函数来通过一个动力学表达式限制不同阶段的动力学方程

     ```python
     def logistic(x):
         return 0.01*np.exp(x*500)/(1+0.01*(np.exp(500*x)-1))
     
     dx_b_next = vertcat(
         -g + u / m * logistic(x_b-x_ref) + (k_con * (-x_b + x_reb)) * logistic(x_reb-x_b)
     )
     model.set_rhs('dx_b', dx_b_next)
     ```

   - MPC控制器参数

     ```python
     setup_mpc = {
         'n_horizon': 50,
         't_step': 0.001,
         'n_robust': 1,
         'store_full_solution': True,
     }
     ```

     

   - 其他控制参数: logistic指数： r = 500 

     ​                       反弹的参考位置： x_reb = -0.47

     ​                       反弹近似弹簧刚度：k_c = 5000<font color = 'coral'>(实际运动中，考虑能量损失可与加上阻尼项c_con)</font>

2. <font color = 'yellow'>仿真结果</font>

   <img src="https://i.loli.net/2021/05/28/y1E4LT9PCi27AOs.png" alt="Figure_5" style="zoom: 50%;" />

3. <font color = 'yellow'>结果分析</font>

   从实验结果可以看到，整个球的运动方程近似合理、但是球离机械臂时的速度仍然没由达到期望速度v_ref，同时，球运动范围超过了-0.5 m，这可能可以通过增大弹簧刚度来近似解决<font color='coral'>（该方法经验证可以解决位移越界问题，k_c大约在10000左右）</font>。



###### 实验组二：更改控制速度（拍球频率）

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: ``x_ref = -0.1        v_ref = -5 m/s​``

   - 目标函数使得球达到给定参考速度:  `` J = (dx_b - v_ref) ** 2 + r * u ** 2 ``

   - 边界条件:  ``x = [-10.5, 10.5],   u = [-500,  0]``<font color = 'coral'>(实际运动中，x > - 0.1，u <= 0)</font>

   - 初始条件:  ``x0 = -0.1，v0 =  2 m/s``

   - 控制条件:  构造球与机械手接触、球的自由飞行、球的地面反弹（用弹簧模拟），并通过logistic函数近似阶跃函数来通过一个动力学表达式限制不同阶段的动力学方程

     ```python
     def logistic(x):
         return 0.01*np.exp(x*500)/(1+0.01*(np.exp(500*x)-1))
     
     dx_b_next = vertcat(
         -g + u / m * logistic(x_b-x_ref) + (k_con * (-x_b + x_reb)) * logistic(x_reb-x_b)
     )
     model.set_rhs('dx_b', dx_b_next)
     ```

   - MPC控制器参数

     ```python
     setup_mpc = {
         'n_horizon': 50,
         't_step': 0.001,
         'n_robust': 1,
         'store_full_solution': True,
     }
     
     q1 = 1;
     q2 = 1000
     r = 1
     mterm = q1 * (x_b - x_ref) ** 2 + q2 * (dx_b - v_ref) ** 2
     lterm = q1 * (x_b - x_ref) ** 2 + q2 * (dx_b - v_ref) ** 2 + r * u ** 2
     ```

     

   - 其他控制参数: logistic指数： r = 500 

     ​                       反弹的参考位置： x_reb = -0.47

     ​                       反弹近似弹簧刚度：k_c = 10000<font color = 'coral'>(实际运动中，考虑能量损失可与加上阻尼项c_con)</font>

2. <font color = 'yellow'>仿真结果：位移速度图、相空间图</font>

   <img src="https://i.loli.net/2021/06/07/DJS1kWFxAvrngHf.png" alt="Figure_v5_r1" style="zoom: 33%;" />

   <img src="https://i.loli.net/2021/06/07/hFncLmzry3AG5Ud.png" alt="phase-v5-r1-c1" style="zoom:50%;" />

3. <font color = 'yellow'>结果分析</font>

   从实验结果可以看到，更改控球速度确实可以更改排球频率，但是速度仍然没有达到期望速度<font color = 'coral'>该基于完整分段动力学的MPC控制器方法验证可行</font>
   
   

###### 实验组三：更改目标函数权重系数

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = -0.1        v_ref = -5 m/s``</font>

   - 目标函数使得球达到给定参考速度:  `` J = (dx_b - v_ref) ** 2 + r * u ** 2 ``

   - 边界条件:   <font color = 'yellow'>``x = [-0.5, 0.5],   u = [-500,  0]``</font><font color = 'coral'>(实际运动中，x > - 0.1，u <= 0)</font>

   - 初始条件:  ``x0 = -0.1，v0 =  2 m/s`` 

   - 控制条件:  构造球与机械手接触、球的自由飞行、球的地面反弹（用弹簧模拟），并通过logistic函数近似阶跃函数来通过一个动力学表达式限制不同阶段的动力学方程

     ```python
     def logistic(x):
         return 0.01*np.exp(x*500)/(1+0.01*(np.exp(500*x)-1))
     
     dx_b_next = vertcat(
         -g + u / m * logistic(x_b-x_ref) + (k_con * (-x_b + x_reb)) * logistic(x_reb-x_b)
     )
     model.set_rhs('dx_b', dx_b_next)
     ```

   - MPC控制器参数

     ```python
     setup_mpc = {
         'n_horizon': 50,
         't_step': 0.001,
         'n_robust': 1,
         'store_full_solution': True,
     }
     
     q1 = 100
     q2 = 100000
     r = 0.1
     mterm = q1 * (x_b - x_ref) ** 2 + q2 * (dx_b - v_ref) ** 2
     lterm = q1 * (x_b - x_ref) ** 2 + q2 * (dx_b - v_ref) ** 2 + r * u ** 2
     ```

     

   - 其他控制参数: logistic指数： r = 500 

     ​                       反弹的参考位置： x_reb = -0.47

     ​                       反弹近似弹簧刚度：k_c = 10000<font color = 'coral'>(实际运动中，考虑能量损失可与加上阻尼项c_con)</font>

2. <font color = 'yellow'>仿真结果：位移速度图、相空间图</font>

   <img src="https://i.loli.net/2021/06/07/h3sw9zNPYLkIZRG.png" alt="Figure_v5" style="zoom: 33%;" />

   <img src="https://i.loli.net/2021/06/07/W9dmKOVDghIuJsr.png" alt="phase-v5-r0.01-r1" style="zoom:50%;" />

3. <font color = 'yellow'>结果分析</font>

   从实验结果可以看到，更改控球速度确实可以更改排球频率，但是速度4.7仍然没有达到期望速度5<font color = 'coral'>该基于完整分段动力学的MPC控制器方法验证可行</font>



###### 实验组四：更改输入边界条件

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = -0.1        v_ref = -5 m/s``</font>

   - 目标函数使得球达到给定参考速度:  `` J = (dx_b - v_ref) ** 2 + r * u ** 2 ``

   - 边界条件:   <font color = 'yellow'>``x = [-0.5, 0.5],   u = [-50,  0]``</font><font color = 'coral'>(实际运动中，x > - 0.1，u <= 0)</font>

   - 初始条件:  ``x0 = -0.1，v0 =  2 m/s`` 

   - 控制条件:  构造球与机械手接触、球的自由飞行、球的地面反弹（用弹簧模拟），并通过logistic函数近似阶跃函数来通过一个动力学表达式限制不同阶段的动力学方程

     ```python
     def logistic(x):
         return 0.01*np.exp(x*500)/(1+0.01*(np.exp(500*x)-1))
     
     dx_b_next = vertcat(
         -g + u / m * logistic(x_b-x_ref) + (k_con * (-x_b + x_reb)) * logistic(x_reb-x_b)
     )
     model.set_rhs('dx_b', dx_b_next)
     ```

   - MPC控制器参数

     ```python
     setup_mpc = {
         'n_horizon': 50,
         't_step': 0.001,
         'n_robust': 1,
         'store_full_solution': True,
     }
     
     q1 = 100
     q2 = 100000
     r = 0.1
     mterm = q1 * (x_b - x_ref) ** 2 + q2 * (dx_b - v_ref) ** 2
     lterm = q1 * (x_b - x_ref) ** 2 + q2 * (dx_b - v_ref) ** 2 + r * u ** 2
     ```

     

   - 其他控制参数: logistic指数： r = 500 

     ​                       反弹的参考位置： x_reb = -0.47

     ​                       反弹近似弹簧刚度：k_c = 10000<font color = 'coral'>(实际运动中，考虑能量损失可与加上阻尼项c_con)</font>

2. <font color = 'yellow'>仿真结果：位移速度图、相空间图</font>

   <img src="https://i.loli.net/2021/06/07/CMOiI68HYNtQXnD.png" alt="Figure_v5_r1-50" style="zoom: 50%;" />

   <img src="https://i.loli.net/2021/06/07/xwJCXu6QDl7TaIR.png" alt="phase-v5-r0.01-u50" style="zoom:50%;" />

3. <font color = 'yellow'>结果分析</font>

   从实验结果可以看到，减小输出力到实际的力控范围仍然可以保证球的运动，但是在接触时力一直保持最大值，这与只管的人运球过程中用力不太符合，可以尝试修改目标函数的权重系数和目标量、初始值等参数。



##### 方法二：基于MPC的力参考轨迹

###### 实验组一：虚拟弹簧和常量力构造力参考轨迹

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = -0.1        v_ref = -15 m/s​``</font>

   - 目标函数使得球达到给定参考速度:  `` J = (dx_b - v_ref) ** 2 + r * u ** 2 ``

   - 边界条件:  ``x = [-0.5, 0.5],   u = [-500,  0]``<font color = 'coral'>(实际运动中，x > - 0.1，u <= 0)</font>

   - 初始条件:  <font color = 'yellow'>``x0 = -0.1，v0 =  10 m/s``</font>

   - 控制条件:  构造球与机械手接触、球的自由飞行、球的地面反弹（用弹簧模拟），并通过tanh函数近似阶跃函数来通过一个动力学表达式限制不同阶段的动力学方程

     ```python
     def tanh_sig(x):
         return 0.5 + 0.5 * np.tanh(1000 * x)
     
     dx_b_next = vertcat(
         -g + u / m * tanh_sig(x_b-x_ref) + (k_con * (-x_b + x_reb)) * tanh_sig(x_reb-x_b)
     )
     model.set_rhs('dx_b', dx_b_next)
     
     # ==============
     # 力参考轨迹
     k_vir = 560
     f1 = 25
     f2 = 150
     def F_TRA(x, v):
         u1 = - k_vir * (x - x_ref) - f1    # 上升阶段力轨迹
         u2 = - k_vir * (x - x_ref) - f2    # 下降阶段力轨迹
         
         u_ref = (u1 * tanh_sig(v) + u2 * tanh_sig(-v))
     
         return u_ref
     
     ```

   - MPC控制器参数

     ```python
     setup_mpc = {
         'n_horizon': 50,
         't_step': 0.001,
         'n_robust': 1,
         'store_full_solution': True,
     }
     
     # ===============
     # 目标函数
     q1 = 100
     q2 = 1000
     r = 10
     u_tra = F_TRA(model.x['x_b'], model.x['dx_b'])
     
     mterm = q2 * (model.x['dx_b'] - v_ref) ** 2
     lterm = r * (model.u['u'] - u_tra) ** 2
     ```

     

   - 其他控制参数: tanh指数： r = 1000 

     ​                       反弹的参考位置： x_reb = -0.47

     ​                       反弹近似弹簧刚度：k_c = 5000<font color = 'coral'>(实际运动中，考虑能量损失可与加上阻尼项c_con)</font>

2. <font color = 'yellow'>仿真结果</font>

<img src="https://i.loli.net/2021/06/13/lYvnO62HaBPdwCT.png" alt="Figure_1-f-tra" style="zoom: 67%;" />

3. <font color = 'yellow'>结果分析</font>

- 从实验结果可以看到，在MPC里面对于给控制变量u一个参考轨迹或者分段参考轨迹得到的结果存在较大的问题，原因是MPC方法本身问题还是设计的轨迹有问题还有待研究，<font color='coral'>可以通过在简单的弹簧质量模型中给u一个简单的参考轨迹的方式来验证</font>
- 而针对与用虚拟弹簧和一个常力的方式给出力的参考轨迹，在基于单自由度的运球控制简单情况下，是可以求出其状态的解析解的，因此可以先用直接的动力学控制来得到一个可用的控制器



##### 方法三：基于动力学控制的力参考轨迹

###### 实验组一：虚拟弹簧和常量力构造力参考轨迹

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = -0.1        v_ref = -15 m/s​``</font>

   - 初始条件:  <font color = 'yellow'>``x0 = -0.1，v0 =  10 m/s``</font>

   - 控制方法:  球与机械手接触、球的自由飞行、球的地面反弹（用弹簧模拟）三个阶段，其中球与机械手接触的部分用<font color='yellow'>一个虚拟弹簧和常力来构造力作用轨迹</font>，原理如下图

     <img src="https://i.loli.net/2021/06/21/QPlEBie3RvMpLwF.png" alt="原理" style="zoom:50%;" />

   - 动力学方程：

     1. 在动力学方程中，<font color='coral'>力的轨迹方程可以表示为</font>

     $$
     F = 
     \begin{cases}
     -K_{vir}(x_{B} - x_{ref}) - f_{up},  & {if \:ball \:is \:uplifting} \\
     -K_{vir}(x_{B} - x_{ref}) - f_{down},  & {if \:ball \:is\: downwarding} \tag{1}
     \end{cases}
     $$

     ​       其中，$K_{vir}$，$f_{up}, f_{down}$可以通过上升和下降过程中的能量守恒来获得
     $$
     {上升阶段：} \quad \frac{1}{2}K_{vir}\Delta{x}^2 + mg\Delta{x} + f_{up}\Delta{x} = \frac{1}{2}mV_{init}^2 \tag{2}  \\
     $$

     $$
    {下降阶段：} \quad \frac{1}{2}K_{vir}\Delta{x}^2 + mg\Delta{x} + f_{down}\Delta{x} = \frac{1}{2}mV_{ref}^2 \tag{3} \\
     $$

     ​       上式中（3）-（2）式便可以通过<font color='coral'>选择合适的$\Delta{x}$和$f_{up}$，便可以通过式（4）得到$f_{down}$</font>：
     $$
     \left(f_{down}-f_{up}\right) \Delta{x} = \frac{1}{2}m\left(V_{ref}^2-V_{init}^2 \right) \tag{4} \\
     $$
     ​       之后带入到式（3）或者（2）中便可以得到$K_{vir}$

     2. 通过分析球的运动，其运动可以分为三个阶段：球与机械手接触、球的自由飞行、球的地面反弹（用弹簧模拟），他们的方程可以统一写成公式（5），只不过公式右边的项中除去重力外的外力项不同而已，<font color='coral'>接触时为力轨迹的方程、自由飞行时为0，地面反弹时而高刚度弹簧力</font>
        $$
        \ddot{x} = -g +\frac{\left(-K_{vir}(x - x_{ref}) - f_{up}\right)}{m}	 \tag{5} \\
        $$
        上述方程为二阶线性微分方程，可以通过通解和特解的方式求得<font color='coral'>运动解析解</font>
        $$
        \begin{aligned}
        {位移方程：} \quad x &= C_1 cos(\beta t) + C_2 sin(\beta t) + a	\\
        {速度方程：} \quad v &= - \beta C_1 sin(\beta t) + \beta C_2 cos(\beta t)	\\
        {输出力方程：} \quad u &= -K_{vir}(x - x_{ref}) - f_{up}
        \end{aligned}	\tag{6}
        $$
        其中各式中的<font color='coral'>参数可以通过初始条件得到</font>：
        $$
        \begin{aligned}
        \beta &= \sqrt{2K_{vir}} \\
        a &=\frac{-g}{2K_{vir}} +  \frac{\left(K_{vir}x_{ref} - f_{up}\right)}{2K_{vir}m} \\
        C_1 &= x_{init} - a \\
        C_2 &= \frac{v_{init}}{\beta}
        \end{aligned}\tag{7}
        $$

   

   - 其他控制参数: 手与球分离参考位置：x_ref = -0.1

     ​                       反弹的参考位置： x_reb = -0.47

     ​                       反弹近似弹簧刚度：k_c = 5000<font color = 'coral'>(实际运动中，考虑能量损失可与加上阻尼项c_con)</font>

2. <font color = 'yellow'>仿真结果</font>

   球运动过程中位置、速度、力变化图

<img src="https://i.loli.net/2021/06/21/qKcsJbdarm5CzL1.png" alt="f_v0-10_vref-15" style="zoom:40%;" />

​     球运动的速度与位置的相空间图

<img src="https://i.loli.net/2021/06/21/wDbzNSgLM9yoXnp.png" alt="phase_f_v0-10_vref-15" style="zoom:50%;" />

​        球运动过程中三个阶段的时间和一个周期的时间：可以得到该工况下一个周期时间为$T=0.128s$

​        ![fT_v0-10_vref-15](https://i.loli.net/2021/06/21/XunJ9cfV4xzHBIW.png)

3. <font color = 'yellow'>结果分析</font>

- 从实验结果可以看到，球的运动位置和速度非常符合预期轨迹，力的轨迹也可以很好的跟随，可以改变球的初始速度、期望速度、运球高度来验证控制器
- 但是该控制起由于方程推导原因，<font color = 'coral'>需要球的初始位置和球与手分离的位置固定，这也是后期需要进行不修改完善的部分</font>



###### 实验组二：更改运球高度验证控制器

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = -0.1        v_ref = -15 m/s​``</font>

   - 初始条件:  <font color = 'yellow'>``x0 = -0.1，v0 =  10 m/s, ``${\color{Yellow}\Delta x=0.4}$</font>

   - 控制方法:  球与机械手接触、球的自由飞行、球的地面反弹（用弹簧模拟）三个阶段，其中球与机械手接触的部分用<font color='yellow'>一个虚拟弹簧和常力来构造力作用轨迹</font>，原理如下图

     <img src="https://i.loli.net/2021/06/21/QPlEBie3RvMpLwF.png" alt="原理" style="zoom:50%;" />

2. <font color = 'yellow'>仿真结果</font>

   球运动过程中位置、速度、力变化图

   <img src="https://i.loli.net/2021/06/21/VNCyRK8L5ptYuhm.png" alt="f_v0-10_vref-15_x-0.4" style="zoom:40%;" />

​       球运动的速度与位置的相空间

​                                                                   <img src="https://i.loli.net/2021/06/21/ZlWt7nSOAUr8mXR.png" alt="phase_f_v0-10_vref-15_x-0.4" style="zoom: 40%;" />

​       球运动过程中三个阶段的时间和一个周期的时间：可以得到该工况下一个周期时间为$T=0.175s$

![fT_v0-10_vref-15_x-0.4](https://i.loli.net/2021/06/21/Vlmaewhcf7tPqGE.png)       

3. <font color = 'yellow'>结果分析</font>

- 从实验结果可以看到，改变运球高度可以增加一个周期的时间，但是由于速度过快效果并不明显



###### 实验组三：更改运球期望速度验证控制器

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = -0.1        v_ref = -8 m/s​``</font>

   - 初始条件:  <font color = 'yellow'>``x0 = -0.1，v0 =  10 m/s, ``${\color{Yellow}\Delta x=0.25}$</font>

   - 控制方法:  球与机械手接触、球的自由飞行、球的地面反弹（用弹簧模拟）三个阶段，其中球与机械手接触的部分用<font color='yellow'>一个虚拟弹簧和常力来构造力作用轨迹</font>，原理如下图

     <img src="https://i.loli.net/2021/06/21/QPlEBie3RvMpLwF.png" alt="原理" style="zoom:50%;" />

2. <font color = 'yellow'>仿真结果</font>

   球运动过程中位置、速度、力变化图

   <img src="https://i.loli.net/2021/06/21/A7LCKWkMbaiRPTB.png" alt="f_v0-10_vref-8" style="zoom:40%;" />

   球运动的速度与位置的相空间

   <img src="https://i.loli.net/2021/06/21/Eveylr3CxLOWYRo.png" alt="phase_f_v0_10-vref_8-dx_0.25" style="zoom:40%;" />

   球运动过程中三个阶段的时间和一个周期的时间：可以得到该工况下一个周期时间为$T=0.187s$

   ![fT_v0-10_vref-8](https://i.loli.net/2021/06/21/SvH8gyfF2eE9wCD.png)

3. <font color = 'yellow'>结果分析</font>

- 从实验结果可以看到，改变运球期望速度可以增加一个周期的时间，效果明显





###### 实验组四：更改运球期望速度和高度验证控制器

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = -0.1        v_ref = -8 m/s​``</font>

   - 初始条件:  <font color = 'yellow'>``x0 = -0.1，v0 =  9 m/s, ``${\color{Yellow}\Delta x=0.45}$</font>

   - 控制方法:  球与机械手接触、球的自由飞行、球的地面反弹（用弹簧模拟）三个阶段，其中球与机械手接触的部分用<font color='yellow'>一个虚拟弹簧和常力来构造力作用轨迹</font>，原理如下图

     <img src="https://i.loli.net/2021/06/21/QPlEBie3RvMpLwF.png" alt="原理" style="zoom:50%;" />

   - 动力学方程：

   - 其他控制参数: 手与球分离参考位置：x_ref = -0.1

     ​                       反弹的参考位置： x_reb = -0.47

     ​                       反弹近似弹簧刚度：k_c = 5000<font color = 'coral'>(实际运动中，考虑能量损失可与加上阻尼项c_con)</font>

2. <font color = 'yellow'>仿真结果</font>

   球运动过程中位置、速度、力变化图

   <img src="https://i.loli.net/2021/06/21/7KkWPsbfymGXvar.png" alt="f_v0-9_vref-8_x-0.45" style="zoom:40%;" />

   球运动的速度与位置的相空间

   <img src="https://i.loli.net/2021/06/21/bwvATCpcVP53Y2I.png" alt="phase_f_v0-9_vref-8_x-0.45" style="zoom:50%;" />

   球运动过程中三个阶段的时间和一个周期的时间：可以得到该工况下一个周期时间为$T=0.287s$

   ![fT_v0-9_vref-8_x-0.45](https://i.loli.net/2021/06/21/VXDu38oHcUBp97K.png)

3. <font color = 'yellow'>结果分析</font>

- 从实验结果可以看到，同时改变运球速度和运球高度可以有效的改变运球周期，效果最为明显



#### Raisim 仿真

#####　方法一：基于力轨迹

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = 0.35 v_ref = -8 m/s​``</font>

   - 初始条件:  <font color = 'yellow'>``x0 = 0.3 v0 = 6 m/s, ``${\color{Yellow} x_{top}=0.6}$</font>

   - 控制方法:  球与机械手接触阶段用<font color='yellow'>一个虚拟弹簧和常力来构造力作用轨迹</font>，并在接触时刻根据已知参数计算完成的力轨迹，原理如下图

     <img src="https://i.loli.net/2021/06/21/QPlEBie3RvMpLwF.png" alt="原理" style="zoom:50%;" />

   - 动力学方程：

   - 其他控制参数: 手与球分离参考位置：x_ref = 0.35

     ​                                                            F_up = 15

     ​															Kp = 40,  Kd = 0.8

     ​															K_virx = 10000

2. <font color = 'yellow'>仿真结果</font>

   球运动过程中位置、速度、力变化图

   <img src="https://i.loli.net/2021/07/19/skUaB5PML7zXQ9h.png" alt="vel8" style="zoom:50%;" />

   机械臂关节力矩和关节速度相空间图

   <img src="https://i.loli.net/2021/07/19/m8uxvY5c4MGldtH.png" alt="vel8-phase1" style="zoom:50%;" /><img src="https://i.loli.net/2021/07/19/lcfmPaAyh9zVgx7.png" alt="vel8-phase2" style="zoom:50%;" />

   

3. <font color = 'yellow'>结果分析</font>

- 从实验结果可以看到，基于理论计算得到的力轨迹的方法在Raisim仿真中真实的接触力总是小于计算得到的接触力，这就导致球脱手时的速度总是无法达到期望速度，同时基于力轨迹相当于提前计算好力，除了并不是力反馈控制而是基于位置反馈的控制
- 该方法可以很好的降低关节所需力矩，将其控制在20N.m以内，符合实际电机能力
- 之后可以考虑采用力反馈的控制方式，将控制刚度系数确定，当速度小于期望速度时，不断施加作用力，直到达到期望速度，这样便不需要控制脱手位置和作用高度dx



#####　方法二：基于反馈的力控方法

1. <font color = 'yellow'>实验条件：</font>

   - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = 0.35 v_ref = -8 m/s​``</font>

   - 初始条件:  <font color = 'yellow'>``x0 = 0.3 v0 = 6 m/s, ``${\color{Yellow} x_{top}=0.6}$</font>

   - 控制方法: 球与机械手接触的部分用<font color='yellow'>一个虚拟弹簧和常力来构造力作用轨迹</font>，设置弹簧刚度为常数，期望速度与实际速度存在差值时一直作用力，直到达到期望速度原理如下图

     <img src="https://i.loli.net/2021/06/21/QPlEBie3RvMpLwF.png" alt="原理" style="zoom:50%;" />

   - 动力学方程：

     1. 在动力学方程中，<font color='coral'>力的轨迹方程可以表示为</font>

     $$
     F = 
     \begin{cases}
     -K_{vir}(x_{B} - x_{ref}) - f_{up},  & {if\: ball\: is\: uplifting} \\
     -K_{vir}(x_{B} - x_{ref}) - f_{down},  & {if\: ball\: is\: downwarding} \tag{1}
     \end{cases}
     $$

     2. 其中F_up，F_down可以用实际与理论的速度差乘以一个系数表示
        $$
        \begin{cases}
        F_{up} = K_{errvup}(v_{B} - v_{ref}),  & {if\: ball\: is \:uplifting} \\
        F_{up} = K_{errvdown}(v_{B} - v_{ref}),  & {if\: ball\: is\: downwarding} \tag{2}
        \end{cases}
        $$

     3. 雅克比矩阵求解
        1. 如下图，其中末端的位置可以表示为
        $$x = x0 - l_1sin(\theta_1) - l_2sin(\theta_1+\theta_2) \\
        z = z0 - l_1cos(\theta_1) - l_2cos(\theta_1+\theta_2) \tag{3}$$
        2. 对两边求导，可以得到
        $$\begin{bmatrix}
        \dot{x} \\
        \dot{z}
        \end{bmatrix}
         = 
        \begin{bmatrix}
        -l_1cos(\theta_1) - l_2cos(\theta_1+\theta_2) &  - l_2cos(\theta_1+\theta_2) \\
        l_1sin(\theta_1) + l_2sin(\theta_1+\theta_2) & l_2sin(\theta_1+\theta_2)
        \end{bmatrix}
        \begin{bmatrix}
        \dot{\theta_1} \\
        \dot{\theta_2}
        \end{bmatrix} \tag{4}$$
        4. 所以关节角度到末端位置正运动学可以通过雅克比矩阵表示，其中雅克比矩阵J可以表示为
        $$
        \dot{X} = J\dot{\theta}\\
        J =
        \begin{bmatrix}
        -l_1cos(\theta_1) - l_2cos(\theta_1+\theta_2) &  - l_2cos(\theta_1+\theta_2) \\
        l_1sin(\theta_1) + l_2sin(\theta_1+\theta_2) & l_2sin(\theta_1+\theta_2)
        \end{bmatrix}
        $$
        因此<font color='coral'>末端力到关节力矩的映射可以表示为(用能量守恒原理）</font>：
        $$
        \begin{aligned}
        F\delta x + \tau \delta \theta &= 0\\
        两边同除以dt，可以得到　F \dot{X} + \tau \dot{\theta} &= 0 \\
        \begin{bmatrix}
        \tau_1 \\
        \tau_2
        \end{bmatrix}
         &= 
        -J^T
        \begin{bmatrix}
        F_x \\
        F_z
        \end{bmatrix}
        \end{aligned} \tag{4}
        $$
        

   - 其他控制参数: 

     ```python
      Pgain: 40.0
      Dgain: 0.8
      K_virx: 10000.0
      K_virz: 500.0
      K_errv_up: 5.0
      K_errv_down: 20.0
     ```

     

1. <font color = 'yellow'>仿真结果</font>

   球运动过程中位置、速度、力变化图

   <img src="https://i.loli.net/2021/07/19/TlQ47jzihXLqkdu.png" alt="varyspeed2" style="zoom: 50%;" />

   机械臂关节力矩和关节速度相空间图

   <img src="/home/stylite-y/Documents/Nutstore/Manipulator Arm/ppt/fig/hiptorque_vary.png" alt="hiptorque_vary" style="zoom:40%;" /><img src="https://i.loli.net/2021/07/19/XdHfYOCeMvLac23.png" alt="kneetorque_vary" style="zoom:40%;" />

   

2. <font color = 'yellow'>结果分析</font>

- 从实验结果可以看到，基于理论计算得到的力轨迹的方法在Raisim仿真中可以很好的达到期望速度，同事在仿真中可以实时改变控球速度
- 然而，在特定的一组K_vir，Kerrv_up，K_errv_down系数下，控制器能跟随的速度时有上限和下限的，速度过高一方面刚度相对低导致运动幅度很大，超过限制，另一方面由于参数设置在该速度下不存在稳定的周期解，导致速度无法跟随，速度低时相对刚度较大，机械臂几乎没有运动
- 因此，之后一部分，<font color='coral'>首先可以考虑加入机械臂的惯量和刚度，计算系统的最大固有频率（即排球的频率上限），其次可以考虑控制参数对控球频率的影响，对于给定参数，当速度达到多大时不存在稳定的周期解</font>



### <font color = 'yellow'>篮球运动三维运动控制</font>

#### 基于力反馈的运动控制方法

##### 实验一：球的二维平面控制

- 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = 0.5m vz_ref = -6 m/s vx_ref = 6 m/s``</font>

- 初始条件:  <font color = 'yellow'>``x0 = 0.5 vz_init = -5 m/s,vx_init = 2 m/s ``${\color{Yellow} x_{top}=0.6}$</font>

- 控制方法: 假设球高于0.5m时开始施加外力（球与机械手接触），此时z方向用<font color='yellow'>一个虚拟弹簧和常力来构造力作用轨迹</font>，设置弹簧刚度为常数，期望速度与实际速度存在差值时一直作用力，直到达到期望速度原理如下图；水平方向用同样的方法，但是为了保持在上升过程中力方向维持不变，修改了上升过程中力的映射函数。

  ​                                  <img src="https://i.loli.net/2021/06/21/QPlEBie3RvMpLwF.png" alt="原理" style="zoom:50%;" /><img src="https://i.loli.net/2021/08/31/dT7fZqIBkx94X3y.png" alt="水平方向模型" style="zoom: 80%;" />

- 动力学方程：

  ![二维运动模型](https://i.loli.net/2021/08/31/4hUw2Y3KOMqif9u.png)

  1. 垂直方向，在动力学方程中，<font color='coral'>力的轨迹方程可以表示为</font>

  $$
  F = 
  \begin{cases}
  -K_{vir}(x_{B} - x_{ref}) - f_{up},  & {if ball is uplifting} \\
  -K_{vir}(x_{B} - x_{ref}) - f_{down},  & {if ball is downwarding} \tag{1}
  \end{cases}
  $$

  2. 其中F_up，F_down可以用实际与理论的速度差乘以一个系数表示
     $$
     \begin{cases}
     F_{up} = K_{errvup}(v_{B} - v_{ref}),  & {if ball is uplifting} \\
     F_{up} = K_{errvdown}(v_{B} - v_{ref}),  & {if ball is downwarding} \tag{2}
     \end{cases}
     $$

  3. 水平方向(1)中F_up表示为
     $$
     V_{ratio} = \frac{V_x}{V_z} \\
     \begin{cases}
     F_{up} = V_{ratio} * F_z,  & {if ball is uplifting} \\
     F_{up} = K_{errvdown}(v_{B} - v_{ref}),  & {if ball is downwarding} \tag{3}
     \end{cases}
     $$

  4. 其他控制参数: 

     ```python
     K_xd = 500
     K_zd = 300
     K_zvup = 5
     K_zvdown = 15
     k_xvdown = 20
     ```

- 实验结果

  <img src="https://i.loli.net/2021/08/31/27XE6Ge1AgMypoF.png" alt="3" style="zoom: 50%;" />

  <img src="https://i.loli.net/2021/08/31/fOS87tY9J1WyxHi.png" alt="1D_tra" style="zoom: 37%;" /><img src="https://i.loli.net/2021/08/31/lxgCz3U4WceRh9j.png" alt="pos-force" style="zoom: 35%;" />

  - 结果分析

    在基于力反馈的二维的运动控制中，球能够很好的跟随指令运动，与期望速度的误差很小，结果很理想，可以尝试将其拓展到三维空间的运动控制中

  

##### 实验二：球的三维三角运动控制

  - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = 0.5m vz_ref = -6 m/s vx_ref = 6 m/s vy_ref = 6 m/s``</font>
  
  - 初始条件:  <font color = 'yellow'>``x0 = 0.5 vz_init = -5 m/s, vx_init = 2 m/s   ``${\color{Yellow} x_{top}=0.6}$</font>
  
  - 控制方法: 假设球高于0.5m时开始施加外力（球与机械手接触），此时z方向用<font color='yellow'>一个虚拟弹簧和常力来构造力作用轨迹</font>，设置弹簧刚度为常数，期望速度与实际速度存在差值时一直作用力，直到达到期望速度原理如下图；水平方向用同样的方法，但是为了保持在上升过程中力方向维持不变，修改了上升过程中力的映射函数。
  
    ​                                  <img src="https://i.loli.net/2021/06/21/QPlEBie3RvMpLwF.png" alt="原理" style="zoom:50%;" /><img src="https://i.loli.net/2021/08/31/dT7fZqIBkx94X3y.png" alt="水平方向模型" style="zoom: 75%;" />
  
- 动力学方程:

  ​            ![三角运动模型](https://i.loli.net/2021/08/31/TPB5kMyEFj8X41J.png)<img src="https://i.loli.net/2021/08/31/HOYRT6FnQEVwSyL.png" alt="三角运动模型" style="zoom:90%;" />

  1. 垂直方向，在动力学方程中，<font color='coral'>力的轨迹方程可以表示为</font>

  $$
  F = 
  \begin{cases}
  -K_{vir}(x_{B} - x_{ref}) - f_{up},  & {if ball is uplifting} \\
  -K_{vir}(x_{B} - x_{ref}) - f_{down},  & {if ball is downwarding} \tag{1}
  \end{cases}
  $$

  2. 其中F_up，F_down可以用实际与理论的速度差乘以一个系数表示
     $$
     \begin{cases}
     F_{up} = K_{errvup}(v_{B} - v_{ref}),  & {if ball is uplifting} \\
     F_{up} = K_{errvdown}(v_{B} - v_{ref}),  & {if ball is downwarding} \tag{2}
     \end{cases}
     $$

  3. x水平方向(1)中F_up表示为
     $$
     V_{ratio} = \frac{V_x}{V_z} \\
     \begin{cases}
     F_{up} = V_{ratio} * F_z,  & {if ball is uplifting} \\
     F_{up} = K_{errvdown}(v_{B} - v_{ref}),  & {if ball is downwarding} \tag{3}
     \end{cases}
     $$

  4. 水平y方向与x方向的力控方法相同

  5. 其他控制参数: 

     ```python
     K_xd = 500
     K_zd = 300
     K_zvup = 5
     K_zvdown = 15
     k_xvdown = 50
     k_yvdown = 100
     ```

  - 实验结果

    <img src="https://i.loli.net/2021/08/31/PAmKl1x7kuJg8qd.png" alt="3dparams" style="zoom: 50%;" />

    ​                 <img src="https://i.loli.net/2021/08/31/oN72eCLfjwbycSt.png" alt="xy plane" style="zoom: 40%;" /><img src="https://i.loli.net/2021/08/31/RAojnxkZvWNXPU9.png" alt="xz plane" style="zoom: 42%;" />

- 结果分析

  1. 从结果可以发现，在xy平面, 三角运动平面会逐渐偏移, 原因是因为在上升阶段我们用的时V_ratio去获得y轴的方向, 然而当上升到最高点附近时,<font color='coral'> z方向的速度极小, 就会导致V_ratio极大</font>, 会瞬间在y方向给一个极大的力(第一张图的y方向力的尖峰), <font color='coral'>导致球在y方向先获得一个很大的速度</font>, 在下降阶段在调整到期望速度, <font color='yellow'>这时候其运动轨迹就会呈现上左图中右下角的弧线轨迹</font>, 进而每个就会都会产生偏移, 导致偏差越来越大

  2. <font color='yellow'>解决方法</font>: 水平方向运动时, 球在上升过程中y方向的力始终为零即可,便可以得到下面的运动结果

     <img src="https://i.loli.net/2021/08/31/wJFazQ4LdkOfC9b.png" alt="3D_state" style="zoom: 50%;" />

     <img src="https://i.loli.net/2021/08/31/EKDjT9HlchWMx7s.png" alt="3D_xy" style="zoom: 40%;" /><img src="https://i.loli.net/2021/08/31/2uE8GXWdHN17C5Q.png" alt="3D_xz" style="zoom:40%;" />

- 运动实验拓展: 三维定点分析

  1. 对于定点运动的篮球控制, 对于理想的运动轨迹如左图, 在实际过程中由于速度不可能完美的达到期望速度, 总会有偏差, <font color='coral'>这些偏差随着时间会积累的越来越大, 导致球落脚点离期望落脚点越来越大.</font>
  2. <font color='coral'>原因分析</font>: 这种离散是偏差导致的, 但是任何控制器都会有偏差, <font color='coral'>因为在这个过程中只考虑下一步落脚点的位置, 导致为了满足这个落足点, 控制器会使得速度非常异常从而导致后面的轨迹离理想轨迹偏差越来越大,导致控制器也无法达到期望落足点</font>, 相反,<font color='yellow'>如果通过MPC的方法, 先计算多步, 从而为了满足目标函数和约束条件的情况下舍弃部分下一步落脚点的精度,从而能够在满足之后运动落脚点的同时达到期望运动轨迹, 因此之后可以采用MPC的控制方法去设计篮球的三维定点运动.</font>


  ​                   <img src="https://i.loli.net/2021/08/31/oHUfeJmb4zqrQwC.png" alt="定点运动模型" style="zoom:80%;" /><img src="https://i.loli.net/2021/08/31/IeTqa8MZwQzOnNl.png" alt="误差模型" style="zoom:95%;" />

  

##### 实验三：球的三维定点运动控制

  - 期望球离手时的位置和速度: <font color = 'yellow'>``x_ref = 0.5m vz_ref = -6 m/s vx_ref = 6 m/s vy_ref = 6 m/s``</font>

  - 初始条件:  <font color = 'yellow'>``x0 = 0.5 vz_init = -5 m/s, vx_init = 2 m/s   ``${\color{Yellow} x_{top}=0.6}$</font>

  - 控制方法: 假设球高于0.5m时开始施加外力（球与机械手接触），此时z方向用<font color='yellow'>一个虚拟弹簧和常力来构造力作用轨迹</font>，设置弹簧刚度为常数，期望速度与实际速度存在差值时一直作用力，直到达到期望速度原理如下图；水平方向用同样的方法，但是为了保持在上升过程中力方向维持不变，修改了上升过程中力的映射函数。

    ​                                  <img src="https://i.loli.net/2021/06/21/QPlEBie3RvMpLwF.png" alt="原理" style="zoom:50%;" /><img src="https://i.loli.net/2021/08/31/dT7fZqIBkx94X3y.png" alt="水平方向模型" style="zoom: 75%;" />

- 动力学方程:

  方程同实验二:

  ​      ​<img src="https://i.loli.net/2021/08/31/oHUfeJmb4zqrQwC.png" alt="定点运动模型" style="zoom:80%;" />     <img src="https://i.loli.net/2021/08/31/IeTqa8MZwQzOnNl.png" alt="误差模型" style="zoom:95%;" />
  
- 实验结果

  <img src="https://i.loli.net/2021/09/23/tN6wLOzrg3X2FkK.png" alt="f_参数" style="zoom: 50%;" />
  
  ​               <img src="https://i.loli.net/2021/09/23/NhPOqAMrJBLpx5y.png" alt="f_xy" style="zoom:44%;" /> <img src="https://i.loli.net/2021/09/23/lPqMAcBNz64htoU.png" alt="f_xz" style="zoom:40%;" />
  
- 结果分析
  对于三维定点运动，由于PD控制中始终存在稳态误差，因此在多个周期的运动之后，<font color='coral'>误差会逐渐放大</font>，直至发散，<font color='coral'>但是PD控制对于单自由度或者二维运动而言仍然是有望的和可行的，可以作为后面臂球系统控制方法的选择</font>
- <font color='yellow'>解决方法</font>：
  PD控制只根据误差计算当前步所需要的力，即是舍弃了整个周期内的精度换取一定的当前精度，但这样对于后面的运动就会造成一定的误差，因此可以<font color='coral'>通过MPC控制的方法进行预测控制</font>，提高控制精度。
  



#### 基于MPC和轨迹优化的球的的运动控制方法
##### 基于MPC和仅给定落脚点的球的运动控制
- 基于MPC方法的落脚点设置： x = -0.15, y = 0.0

- MPC的标准控制方程：
  
   <img src="https://s2.loli.net/2021/12/07/FRX9t3vbGDhNipj.png" alt="mpc" style="zoom: 60%;" />

- 基于MPC的方法，首先我希望MPC的动力学中包含完整的接触、自由飞行和反弹的所有动力学，预测一个周期来进行操作，因此系统的动力学方程为下图，反弹过程用一个高刚度弹簧模拟，同事采用tanh函数来模拟分段接触过程
  
  <img src="https://s2.loli.net/2021/12/07/lhOL4MxfeoqbUFw.png" alt="状态方程" style="zoom:67%;" />

  <img src="https://s2.loli.net/2021/12/07/WwzLNEB8CVoMm16.png" alt="状态方程图" style="zoom:67%;" />

- 仿真结果
  

通过采用现有的do_mpc现有的库进行仿真，设置mpc的每步时长t_step = 0.01， 预测步数为30步，得到如下结果:

<img src="https://s2.loli.net/2021/12/07/FSeoUBxuXHVy5c7.png" alt="落脚点结果" style="zoom:67%;" />

- 结果分析：
  1. 由结果可以看到，当希望mpc仅仅根据落脚点推断整个轨迹，由于给的动力学过于复杂，而且mpc为了使得目标函数最小，<font color='coral'>就会使得球的运动尽可能接近x， y， z的边界使得使用的力最小</font>，同时由于接触过程完全交给mpc，因此其<font color='coral'>控球时间完全不可控，因此无法改变其运球频率。</font>
  2. 解决方法：<font color='coral'>可以预先给出更加详细的参考轨迹来作为mpc的参考轨迹，这样其时间也可以控制，而参考轨迹可以通过约束条件通过轨迹优化得到</font>。

##### 基于三次函数的参考轨迹的MPC球运动控制
- MPC的动力学方程：
  

因为我们只需要给出接触过程中的参考轨迹，因此动力学中也不需要加入额外的球自由飞行和反弹的动力学方程，因此简化的动力学方程为：

<img src="https://s2.loli.net/2021/12/07/7gINwv16GZKUojX.png" alt="状态方程参考轨迹" style="zoom:70%;" />

- 三次样条参考轨迹：
  

当仅仅考虑目标位置和初始位置的位置和速度时有12个方程，因此我们可以考虑一个三次函数（含有12个参数）这样便可以先简化轨迹优化为一个唯一解的参考轨迹求解，不需要优化，来尝试该方法的可行性，具体的参考轨迹形式和约束条件为：

<img src="https://s2.loli.net/2021/12/07/4Y5ZD3N6oAB2SrQ.png" alt="三次参考轨迹" style="zoom:67%;" />

- 实验测试1：
  
  首先进行了二维面内的V型运动轨迹测试，mpc设置的参数如下
  
  ```
  mpc t_step = 0.0005
  mpc horizons = 30
  simulation time = 10s
  x_ref = -0.2
  y_ref = 0.0
  ```

- 实验1仿真结果：

  <img src="https://s2.loli.net/2021/12/07/JbVX2yuklrseSEQ.png" alt="image-20211207203419847" style="zoom: 80%;" />

  <img src="https://s2.loli.net/2021/12/07/ve7EVpqGNcUrnO2.png" alt="三次轨迹结果1" style="zoom:80%;" />

- 实验1结果分析：

    从途中可以看出，无论时力还是速度，位置的跟随结果都重合的很好，同事mpc得到的作用结果和给出的参考轨迹几乎完全重合，因此该方法时可行的，且能通过控制参考轨迹参数时间t来控制控球频率。

- 实验2仿真测试：

    ```
    MPC仿真t_step：0.0005
    MPC预测区间：20 step
    仿真时间(2s)：102s
    x_b,ref=[−0.2,−0.5, 0.0]
    y_b,ref=[0.0, 0.6, 0.6]
    ```

    <img src="https://s2.loli.net/2021/12/07/m54gIJfolwavxER.png" alt="运动图" style="zoom:80%;" />

- 实验二仿真结果：

    <img src="https://s2.loli.net/2021/12/07/Q1gC6eRuWXKiaB4.png" alt="image-20211207204419659" style="zoom: 67%;" />

    <img src="https://s2.loli.net/2021/12/07/pwSX526QNkKERau.png" alt="三次轨迹结果" style="zoom:67%;" />

- 实验二仿真结果分析：

    从结果可以看到，无论时二维运动，还是三维定点运动控制，都有很好的控制结果，落脚点位置重合很好，进本符合要求并在可行空间内。

- 下一步仿真方法：

    由于当前的参考轨迹仅仅考虑初始和目标位置速度月约束条件，而<font color='coral'>实际运动控制中还含有力、位置的约束范围，因此需要更加复杂的参考轨迹形式来基于所有的约束条件来进行参考轨迹优化。</font>

    

##### 基于五次样条曲线轨迹优化的mpc球运动控制

- 由于本控制工况自由度并不高，不必采用分段多项式优化的复杂形式，可以采用更加普遍的五次函数的目标优化函数形式来进行轨迹优化，其函数形式由约束条件如下图所示，我们可以通过对于<font color='coral'>设置的采样时间[0, t]内合理的采样来消除时间t影响</font>， 将优化问题转化为标准的二次型优化问题来进行求解：

  <img src="https://s2.loli.net/2021/12/07/zu7Er3hdtfP6xjc.png" alt="五次轨迹" style="zoom:67%;" />

  <img src="https://s2.loli.net/2021/12/07/MGFc7PSZk6QeCnp.png" alt="轨迹优化简化" style="zoom:67%;" />

- 仿真测试：

  仿真中采用的参数如下：

  ```
  MPC仿真t_step：0.0005
  MPC预测区间：20 step
  轨迹优化采样值N：10
  轨迹时间设置：0.2s
  x_b,ref=[−0.2,−0.5, 0.0]
  y_b,ref=[0.0, 0.6, 0.6]
  ```

- 仿真结果

  <img src="https://s2.loli.net/2021/12/07/Ik5rMKdshFiP28z.png" alt="traj-res-force" style="zoom:67%;" />

  <img src="https://s2.loli.net/2021/12/07/u2w6P5D4CbhjKN8.png" alt="trj-res-xyz" style="zoom:67%;" />

  <img src="https://s2.loli.net/2021/12/07/DPjdOa7MrFHYomZ.png" alt="traj-拟合结果" style="zoom:67%;" />

  <img src="https://s2.loli.net/2021/12/07/b2V3c7PiIxdpHKa.png" alt="traj-拟合结果2" style="zoom:67%;" />

- 结果分析：

  从结果可以看到，无论时二维运动，还是三维定点运动控制，都有很好的控制结果，落脚点位置重合很好，由于给定的约束条件与三次函数的结果很相似，且得到的结果四次项和五次项的系数很小，起到的作用不大，因此得到的结果与三次函数参考轨迹的形式很像，但是该过程<font color='coral'>涉及到力的大小和方向始终是变化的，这为之后的力的施加带来了极大的困难</font>.

- 下一步的计划：

  1. 至此基于MPC和轨迹优化的球的运动控制已经在仿真理论上可以完美实现，因此后面<font color='coral'>可以添加机械臂来对整个臂球系统进行控制、稳定性分析和鲁棒控制以及机械臂设计参数的参考。</font>

  2. 对于如何施加变方向的力，一方面可以使用下图方法，另一方面的一些灵感可以参考onenote笔记。

     <img src="https://s2.loli.net/2021/12/07/92c31eJUnDjTtOG.png" alt="力变的实现方法" style="zoom: 67%;" />

     

### <font color='yellow'>臂球系统的三维运动控制</font>

##### 基于MPC和轨迹优化的臂球系统二维V型的运动控制。
- 设计力方向不变的五次轨迹函数，基于现有的机械臂的运动空间约束和力的需求约束来给出约束条件，函数形式和约束条件如下图所示：

  

- 仿真参数设置：
  地方 


### <font color='yellow'>臂球系统的高速稳定运动控制</font>
#### 基于Z向轨迹和PD姿态控制的运动
- 实验目的：
  本章实验仿真目的是实现高频的运动控制，并通过参数优化使得需求力矩在现有的机械腿电机的力矩限制之内。
- 实验方法：
  本次实验采用DLR的运球方法，竖直方向通过给出运动轨迹进行位置控制，其他方向通过机械手的俯仰和滚转角的PD控制使得机械臂能在在期望点附近稳定运动。
- 实验原理：
  1. 竖直方向通过文章中的参数和机械腿的参数，给出z向的运动轨迹，如下形式：
     $$
     z(t)=
     \begin{cases}
     Asin(\frac{5\pi}{4T}t)+z_0,  & for \: t \in \left[0, \frac{4}{5}T \right] \\
     -\frac{1}{4}Asin(\frac{5\pi}{T}t)+z_0,  & for \: t \in \left[\frac{4}{5}T, T \right] \tag{1}
     \end{cases}
     $$
  2. 水平xy方向通过末端执行器的滚转和俯仰角的PD控制进行稳定控制，每次接触通过提前计算得到的xy方向的速度和位置信息通过PD方法得到需要偏转的角度$\beta, \gamma$,来逐渐使得球收敛到期望点。
     - 首先在球每次碰撞地面时通过其速度大小和方向以及位置信息，通过与期望点的对比通过PD控制得到下一次接触末端执行器需要偏转的角度：
     $$
     \begin{cases}
     \beta_{des} = -K_{p\beta}(x_{des} - x) - K_{d\beta}(0.0 - \dot{x}),  \\
     \gamma_{des} = K_{p\gamma}(y_{des} - y) + K_{d\gamma}(0.0 - \dot{y}),\tag{2}
     \end{cases}
     $$
     - 由于raisim旋转雅克比矩阵的问题，为了机械臂可以跟随篮球的位置，本文将末端位置控制和手的姿态控制解耦，然后通过PD阻抗控制计算出需要的末端力和姿态需要的关节转矩：
     $$
     \begin{cases}
     F_{x} = K_{p,x}(x_{eff,des} - x_{eff}) - K_{d,x}(\dot{e}_{eff,x}),  \\
     F_{y} = K_{p,y}(y_{eff,des} - y_{eff}) - K_{d,y}(\dot{e}_{eff,y}), \\
     F_{z} = K_{p,z}(z_{eff,des} - z_{eff}) - K_{d,z}(\dot{e}_{eff,z}),  \tag{3}
     \end{cases}
     $$
     $$
     \begin{cases}
     Tor_{\beta} = K_{p,\beta}(\beta_{des} - \beta_{eff}) - K_{d,\beta}(\dot{e}_{eff,\beta}),  \\
     Tor_{\gamma} = K_{p,\gamma}(\gamma_{des} - \gamma_{eff}) - K_{d,\gamma}(\dot{e}_{eff,\gamma}),   \tag{4}
     \end{cases}
     $$
     - 然后通过速度雅克比矩阵将末端力转换到前三个关节的关节力矩上，而得到的姿态力矩直接作用的后后两个关节上。
- 实验参数设置
  ```python
  # z方向参考轨迹的参数
  T_Period = 0.33
  A = 0.16
  z0 = 0.48

  # ref point
  x_Ball_des = 0.4
  y_Ball_des = 0.18

  # PID params for desired angle cal
  K_Bdes_p = 2
  K_Bdes_d = 0.2
  K_Gdes_p = 1
  K_Gdes_d = 0.2

  # PD control
  Kx_F = np.diag([1500, 1500, 3000])
  Kd_F = np.diag([100, 200, 100])
  Kx_t = np.diag([100, 20])
  Kd_t = np.diag([0.1, 0.1])

  # 地面参数和初始状态
  world.setMaterialPairProp("rub", "rub", 1.0, 0.9, 0.0001)
  jointNominalConfig = np.array([0.4, 0.18, 0.3, 1.0, 0.0, 0.0, 0.0])
  jointVelocityTarget = np.array([0.0, 0.0, -3.0, 0.0, 0.0, 0.0])
  ```
- 仿真实验结果
  1. 2.5Hz
   <img src="https://s2.loli.net/2021/12/21/9VhBnAxeovy1XIQ.png" alt="fd_state-2.5hz">
   <img src="https://s2.loli.net/2021/12/21/Dkgrjs7MOcWJPza.png" alt="fd_tor-2.5hz">
  2. 3Hz
  <img src="https://s2.loli.net/2021/12/21/FvSnfx1Xd2uc7gb.png" alt="fd_state">
  <img src="https://s2.loli.net/2021/12/21/hDY9lgiUzXMECsB.png" alt="fd_state-3hz">
  3. 5Hz
   <img src="https://s2.loli.net/2021/12/21/pPoLVhvZmJ5WB3k.png" alt="fd_state-5hz">
  <img src="https://s2.loli.net/2021/12/21/1hSJXHWxuwk2lzK.png" alt="fd_tor-5hz">
  4. 5Hz下由于关节力矩只有瞬间达到峰值，因此可以采用力矩范围限制的方法进行力矩约束，测试结果发现仍然可以稳定运动
   <img src="https://s2.loli.net/2021/12/21/iBR2oZWvmgFxVSe.png" alt="fd_state-5hz-clip">
   <img src="https://s2.loli.net/2021/12/21/HRY1TIspoiUarvf.png" alt="fd_tor-5hz-clip">
  

- 实验结果分析
  该方法可以实现球的稳定的稳定控制，但是在每次球接触时都会产生一个极大的冲击量，不确定对硬件实验是否有影响。
- 后续试验计划
  1. 该方法z方向为位置控制，刚度很大，因此会产生很大的冲击量，该问题一方面可以通过末端的位置控制修改为弹簧模拟的力控，通过硬件是末端执行器换成弹性板。
  2. 从控制上解决，可以考虑柔顺控制。
  
