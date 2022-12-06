# Manipulator 研究方向文献综述

## 碰撞检测

## 运动规划与轨迹优化

- 运动规划算法和轨迹优化问题通常需要求解避开障碍物的轨迹

### 几何碰撞检查算法

多边形碰撞检查方法(Polygonal collision checking)： Flexible Collision Library (FCL), LIBCCD [1], [2] and their base algorithms, Gilbert–Johnson–Keerthi (GJK), and Expanding Polytope Algorithm (EPA) 

## 正逆运动学求解

 _Fast and Robust Inverse Kinematics of Serial Robots Using Halley’s Method_. [Steffan Lloyd](https://ieeexplore.ieee.org/document/9787063)

Open source library: Orocos Kinematics and Dynamics Library (KDL)

-  heuristic method: Cyclic Coordinate Descent (CCD)
- optimization-based
- Jacobian-based method: first-order Jacobian transpose techniques &  second-order methods
  -  Jacobian pseudoinverse methods  and variations thereof (Newton-Raphson pseudoinverse solver.)
  - Damped Least-Squares (DLS) method (also known as the Levenberg–Marquardt algorithm) 
  - Selectively Damped Least-Squares method 
  - The Broyden–Fletcher–Goldfarb–Shannon (BFGS) algorithm

