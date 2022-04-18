

## 三级倒立摆模型

![Two_LInks_DualArm .drawio](https://s2.loli.net/2022/03/28/n5QY7xasghDrq8V.png)

运动学关系
$$
\left\{
\begin{aligned}
&x_{b}= l_{b}S_1\\
&y_{b}= l_{b}C_1\\
&x_{t}= L_bS_1+l_{t}S_{1,2}\\
&y_{t}= L_bC_1+l_{t}C_{1,2}\\
&x_{h}= L_bS_1+L_tS_{1,2} + l_{s}S_{1,2, 3}\\
&y_{h}= L_bC_1+L_tC_{1,2} + l_{s}C_{1,2, 3}\\
\end{aligned}
\right.

​\Rightarrow​
\left\{\begin{aligned}
&\dot{x}_{b}= l_{b}C_1\dot{\theta}_1\\
&\dot{y}_{b}= -l_{b}S_1\dot{\theta}_1\\
&\dot{x}_{t}= L_bC_1\dot{\theta}_1+l_{t}C_{1,2}\dot{\theta}_{1,2}\\
&\dot{y}_{t}= -L_bS_1\dot{\theta}_1-l_{t}S_{1,2}\dot{\theta}_{1,2}\\
&\dot{x}_{h}= L_bC_1\dot{\theta}_1+L_{t}C_{1,2}\dot{\theta}_{1,2} + l_{s}C_{1,2, 3}\dot{\theta}_{1,2,3}\\
&\dot{y}_{h}= -L_bS_1\dot{\theta}_1-L_{t}S_{1,2}\dot{\theta}_{1,2} - l_{s}S_{1,2, 3}\dot{\theta}_{1,2,3}\\
\end{aligned}
\right.
$$

动能
$$
\begin{aligned}
T=&\frac{1}{2}m_b\dot{x}_b^2+\frac{1}{2}m_b\dot{y}_b^2+\frac{1}{2}I_{b}\dot{\theta}^2_b\\
+&\frac{1}{2}m_t\dot{x}^2_{t}+\frac{1}{2}m_t\dot{y}^2_{t}+\frac{1}{2}I_{t}\dot{\theta}^2_{1,2}\\
+&\frac{1}{2}m_h\dot{x}^2_{h}+\frac{1}{2}m_h\dot{y}^2_{h}+\frac{1}{2}I_{h}\dot{\theta}^2_{1,2,3}\\
\end{aligned}
$$

势能
$$
V=m_bgy_b+m_tgy_{t}+m_hgy_{h}
$$

### 动力学方程

$$
\begin{bmatrix}
M_{11} &M_{12} &M_{13}\\
M_{21} &M_{22} &M_{23}\\
M_{31} &M_{32} &M_{33}\\
\end{bmatrix}
\begin{bmatrix}
\ddot{\theta}_1\\
\ddot{\theta}_2\\
\ddot{\theta}_3\\
\end{bmatrix}
+
\begin{bmatrix}
C_1\\
C_2\\
C_3
\end{bmatrix}
+
\begin{bmatrix}
G_1\\
G_2\\
G_3\\
\end{bmatrix}
=
\begin{bmatrix}
\tau_1\\
\tau_2\\
\tau_3\\
\end{bmatrix}
$$

$$
\left\{
\begin{aligned}
M_{11} &=
I_1+I_2+I_b+L_b^2m_1+L_b^2m_2+L_t^2m_2+l_b^2m_b+l_s^2m_2+l_t^2m_1+2L_bL_tm_2C_2+2L_bl_sm_2C_{2,3}+2L_bl_tm_1C_2+2L_tl_sm_2C_3\\
M_{12} &=
I_1+I_2+L_t^2m_2+l_s^2m_2+l_t^2m_1+L_bL_tm_2C_2+L_bl_sm_2C_{2,3}+L_bl_tm_1C_2+2L_tl_sm_2C_3\\
M_{13} &=
I_2+l_s^2m_2 + L_bl_sm_2C_{2,3}+L_tl_sm_2C_3\\
M_{21} &=
I_1+I_2+L_t^2m_2+l_s^2m_2+l_t^2m_1+L_bL_tm_2C_2+L_bl_sm_2C_{2,3}+L_bl_tm_1C_2+2L_tl_sm_2C_3\\
&=M_{12}\\
M_{22} &=
I_1+I_2+L_t^2m_2+l_s^2m_2+l_t^2m_1+2L_tl_sm_2C_3\\
M_{23} &=
I_2+l_s^2m_2+L_tl_sm_2C_3\\
M_{31} &=
I_2+l_s^2m_2 + L_bl_sm_2C_{2,3}+L_tl_sm_2C_3\\
&=M_{13}\\
M_{32} &=
I_2+l_s^2m_2+L_tl_sm_2C_3\\
&=M_{23}\\
M_{33} &=
I_2+l_s^2m_2\\
\end{aligned}
\right.
$$

$$
\left\{
\begin{aligned}
G_1 &=
-\left(
L_bm_1S_1+L_bm_2S_1 + L_tm_2S_{1,2}+l_bm_bS_1+l_tm_1S_{1,2}+l_sm_2S_{1,2,3}
\right)g\\
G_2 &=
-\left(
L_tm_2S_{1,2}+l_tm_1S_{1,2}+l_sm_2S_{1,2,3}
\right)g\\
G_3 &=
-\left(
l_sm_2S_{1,2,3}
\right)g
\end{aligned}
\right.
$$

$$
\left\{
\begin{aligned}
C_1 =&
-2L_b(L_tm_2S_2+l_sm_2S_{2,3}+l_tm_1S_2)\dot{\theta}_1\dot{\theta}_2\\
&-2l_sm_2(L_bS_{2,3}+L_tS_3)\dot{\theta}_1\dot{\theta}_3\\
&-L_b(L_tm_2S_2+l_sm_2S_{2,3}+l_tm_1S_2)\dot{\theta}_2^2\\
&-2l_sm_2(L_bS_{2,3}+L_tS_3)\dot{\theta}_2\dot{\theta}_3\\
&-l_sm_2(L_bS_{2,3}+L_tS_3)\dot{\theta}_3^2\\
C_2 =&
L_b(L_tm_2S_2+l_sm_2S_{2,3}+l_tm_1S_2)\dot{\theta}_1^2\\
&-2L_tl_sm_2S_3\dot{\theta}_1\dot{\theta}_3\\
&-2L_tl_sm_2S_3\dot{\theta}_2\dot{\theta}_3\\
&-L_tl_sm_2S_3\dot{\theta}_3^2\\
C_3 =&
l_sm_2(L_bS_{2,3}+L_tS_3)\dot{\theta}_1^2\\
&+2L_tl_sm_2S_3\dot{\theta}_1\dot{\theta}_2\\
&+L_tl_sm_2S_3\dot{\theta}_2^2\\
\end{aligned}
\right.
$$

### 方程线性化

动力学方程线性化：当$\theta_1和\theta_2$都很小时，可以近似认为是零，因此在平衡点附近$\theta_1=0, \phi_2(\phi_2=\pi+\theta_2)=0,\theta_3=0,\dot{\theta}_1=0,\dot{\theta}_2=0,\dot{\theta}_3=0$，在进行线性化时所以由于角度变化很小，因此所有的二阶小量都被舍去（$\theta\dot{\theta}$或者$\dot{\theta}^2$），因此离心力和科氏力项中由于简化后存在二阶小量，该矩阵各项都会为0.

其中：$sin\theta\approx\theta,cos\theta\approx1,sin(\theta_2+\theta_3)\approx\theta_2+\theta_3,cos(\theta_2+\theta_3)\approx1$


$$
\left\{
\begin{aligned}
M_{11} &=
I_1+I_2+I_b+L_b^2m_1+L_b^2m_2+L_t^2m_2+l_b^2m_b+l_s^2m_2+l_t^2m_1+2L_bL_tm_2(-1)+2L_bl_sm_2(-1)+2L_bl_tm_1(-1)+2L_tl_sm_2\\
M_{12} &=
I_1+I_2+L_t^2m_2+l_s^2m_2+l_t^2m_1+L_bL_tm_2(-1)+L_bl_sm_2(-1)+L_bl_tm_1(-1)+2L_tl_sm_2\\
M_{13} &=
I_2+l_s^2m_2 + L_bl_sm_2(-1)+L_tl_sm_2\\
M_{21} &=
I_1+I_2+L_t^2m_2+l_s^2m_2+l_t^2m_1+L_bL_tm_2(-1)+L_bl_sm_2(-1)+L_bl_tm_1(-1)+2L_tl_sm_2\\
&=M_{12}\\
M_{22} &=
I_1+I_2+L_t^2m_2+l_s^2m_2+l_t^wm_1+2L_tl_sm_2\\
M_{23} &=
I_2+l_s^2m_2+L_tl_sm_2\\
M_{31} &=
I_2+l_s^2m_2 + L_bl_sm_2(-1)+L_tl_sm_2\\
&=M_{13}\\
M_{32} &=
I_2+l_s^2m_2+L_tl_sm_2\\
&=M_{23}\\
M_{33} &=
I_2+l_s^2m_2\\
\end{aligned}
\right.
$$
重力项线性化
$$
\left\{
\begin{aligned}
G_1 &=
-\left(
L_bm_1sin(\theta_1)+L_bm_2sin(\theta_1) - L_tm_2sin(\theta_1+\phi_2)+l_bm_bsin(\theta_1)-l_tm_1sin(\theta_1+\phi_2)-l_sm_2sin(\theta_1+\phi_2+\theta_3)
\right)g\\
G_2 &=
\left(
L_tm_2sin(\theta_1+\phi_2)+l_tm_1sin(\theta_1+\phi_2)+l_sm_2sin(\theta_1+\phi_2+\theta_3)
\right)g\\
G_3 &=
\left(
l_sm_2sin(\theta_1+\phi_2+\theta_3)
\right)g
\end{aligned}
\right.
$$

$$
\left\{
\begin{aligned}
G_1 &=
-\left(L_bm_1\theta_1+L_bm_2\theta_1- L_tm_2(\theta_1+\phi_2)+l_bm_b\theta_1-l_tm_1(\theta_1+\phi_2)-l_sm_2(\theta_1+\phi_2+\theta_3)\right)g\\
G_2 &=
\left(L_tm_2(\theta_1+\phi_2)+l_tm_1(\theta_1+\phi_2)+l_sm_2(\theta_1+\phi_2+\theta_3)\right)g\\
G_3 &=
\left(l_sm_2(\theta_1+\phi_2+\theta_3)\right)g
\end{aligned}
\right.
$$
离心力项线性化
$$
\left\{
\begin{aligned}
C_1 =&
-2L_b(L_tm_2S_2+l_sm_2S_{2,3}+l_tm_1S_2)\dot{\theta}_1\dot{\theta}_2\\
&-2l_sm_2(L_bS_{2,3}+L_tS_3)\dot{\theta}_1\dot{\theta}_3\\
&-L_b(L_tm_2S_2+l_sm_2S_{2,3}+l_tm_1S_2)\dot{\theta}_2^2\\
&-2l_sm_2(L_bS_{2,3}+L_tS_3)\dot{\theta}_2\dot{\theta}_3\\
&-l_sm_2(L_bS_{2,3}+L_tS_3)\dot{\theta}_3^2\\
&=0\\
C_2 =&
L_b(L_tm_2S_2+l_sm_2S_{2,3}+l_tm_1S_2)\dot{\theta}_1^2\\
&-2L_tl_sm_2S_3\dot{\theta}_1\dot{\theta}_3\\
&-2L_tl_sm_2S_3\dot{\theta}_2\dot{\theta}_3\\
&-L_tl_sm_2S_3\dot{\theta}_3^2\\&=0\\
C_3 =&
l_sm_2(L_bS_{2,3}+L_tS_3)\dot{\theta}_1^2\\
&+2L_tl_sm_2S_3\dot{\theta}_1\dot{\theta}_2\\
&+L_tl_sm_2S_3\dot{\theta}_2^2\\
&=0\\
\end{aligned}
\right.
$$



## 轨迹优化

$$
\begin{aligned}
&\min_{q_i,\tau_i}{\sum^n_{i=0}(\lvert \tau_i \rvert+\lvert q_i - q_{tar}\rvert)+\lvert q_n-q_{tar} \rvert}\\
&\quad\quad s.t.\quad
\begin{aligned}
&\color{Crimson}{\left\{
    \begin{aligned}
    &q_{i+1}-q_i-\frac{dt}{2}(\dot{q}_{i+1}+\dot{q}_{i})=0\\
    &M(q_i)(\dot{q}_{i+1}-\dot{q}_i)+C(q_i,\dot{q}_{i+1})dt+G(q_i)dt-\tau_i dt=0\\
    \end{aligned}
\right.} &\color{Crimson}{Dynamics}\\
&\color{DodgerBlue}{\left\{
    \begin{aligned}
    &\underline{q}\le q \le \bar{q}\\
    &\underline{\dot{q}} \le \dot{q} \le \bar{\dot{q}}\\
    &\underline{\tau}(\dot{q})\le \tau \le \bar{\tau}(\dot{q})\\
    \end{aligned}
\right.} &\color{DodgerBlue}{Boundary}\\
\end{aligned}
\end{aligned}
$$

<img src="https://s2.loli.net/2022/04/06/IoAUBLMbX246i3x.png" alt="traj-mpc-0.002" style="zoom:72%;" />

<img src="https://s2.loli.net/2022/04/10/UAjKxbz4SMdvLR7.png" alt="traj-mpc-0.01" style="zoom:72%;" />



## MPC控制

###　MPC_Raisim测试

- mpc 控制方程

$$
\begin{aligned}
&\min_{q_i,\tau_i}{\sum^n_{i=0}(\lvert \tau_i \rvert+\lvert q_i - q_{tar}\rvert)+\lvert q_n-q_{tar} \rvert}\\
&\quad\quad s.t.\quad
\begin{aligned}
&\color{Crimson}{\left\{
    \begin{aligned}
    &q_{i+1}-q_i-\frac{dt}{2}(\dot{q}_{i+1}+\dot{q}_{i})=0\\
    &M(q_i)(\dot{q}_{i+1}-\dot{q}_i)+C(q_i,\dot{q}_{i+1})dt+G(q_i)dt-\tau_i dt=0\\
    \end{aligned}
\right.} &\color{Crimson}{Dynamics}\\
&\color{DodgerBlue}{\left\{
    \begin{aligned}
    &\underline{q}\le q \le \bar{q}\\
    &\underline{\dot{q}} \le \dot{q} \le \bar{\dot{q}}\\
    &\underline{\tau}(\dot{q})\le \tau \le \bar{\tau}(\dot{q})\\
    \end{aligned}
\right.} &\color{DodgerBlue}{Boundary}\\
\end{aligned}
\end{aligned}
$$



- 一组好的测试参数和结果

  测试结果文件：X-2022-04-14-18-22-51-MPC-Pos_100-Tor_5-Vel_30-dt_0.01-T_4-Tp_0.8-ML_0.5k文件夹

  测试结果：基于当前的质量和惯量分布，预测时间Tp必须大于0.6s，但是测试的惯量分布三级杆很近似

  <img src="https://s2.loli.net/2022/04/15/Db4tcZM9IxzaNJF.png" alt="2022-04-14-18-22-51-MPC-Pos_100-Tor_5-Vel_30-dt_0.01-T_4-Tp_0.8-ML_0.5k" style="zoom:72%;" />

- 工作计划：

  对于质量和惯量构建一定的无量纲化参数，来减少可调的参数量

  测试最大可调节身体偏移量，增加力矩限制范围