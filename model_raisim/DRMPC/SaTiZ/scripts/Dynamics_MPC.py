import numpy as np
from numpy import sin  as s
from numpy import cos as c
from casadi import *
from casadi.tools import *
import do_mpc
import os
import yaml
from scipy import signal
import matplotlib.pyplot as plt
from sympy import Symbol as sym
import sympy

class RobotProperty():
    def __init__(self, ParamData):
        self.I = ParamData['Robot']['Mass']['inertia']
        self.m = ParamData['Robot']['Mass']['mass']
        self.Mc = ParamData['Robot']['Mass']['massCenter']
        self.L = [ParamData['Robot']['Geometry']['L_body'],
                ParamData['Robot']['Geometry']['L_thigh'],
                ParamData['Robot']['Geometry']['L_shank']]
        
        self.g = ParamData['Environment']['Gravity']

    def getMassMatrix(self, O1, O2, O3):
        M11 = self.I[0] + self.I[1] + self.I[2] + (self.L[0]**2 + self.L[1]**2 + self.Mc[2]**2)*self.m[2]+\
            (self.L[0]**2 + self.Mc[1]**2) * self.m[1] + self.Mc[0]**2*self.m[0] + \
            2*self.L[0]*self.m[2]*(self.L[1]*c(O2) + self.Mc[2]*c(O2+O3)) + \
            2*self.L[0]*self.Mc[1]*self.m[1]*c(O2) + 2*self.L[1]*self.Mc[2]*self.m[2]*c(O3)

        M12 = self.I[1] + self.I[2] + (self.L[1]**2 + self.Mc[2]**2)*self.m[2] + self.Mc[1]**2*self.m[1] + \
            self.L[0]*self.L[1]*self.m[2]*c(O2) + self.L[0]*self.m[2]*self.Mc[2]*c(O2+O3) + \
            self.L[0]*self.Mc[1]*self.m[1]*c(O2) + 2*self.L[1]*self.Mc[2]*self.m[2]*c(O3)

        M13 = self.I[2] + self.Mc[2]**2*self.m[2] + self.L[0]*self.Mc[2]*self.m[2]*c(O2+O3) + self.L[1]*self.Mc[2]*self.m[2]*c(O3)
        
        M21 = M12
        M22 = self.I[1] + self.I[2] + (self.L[1]**2 + self.Mc[2]**2)*self.m[2] + self.Mc[1]**2*self.m[1] + \
            2*self.L[1]*self.Mc[2]*self.m[2]*c(O3)
        M23 = self.I[2] + self.Mc[2]**2*self.m[2] + self.L[1]*self.Mc[2]*self.m[2]*c(O3)

        M31 = M13
        M32 = M23
        M33 = self.I[2] + self.Mc[2]**2*self.m[2]

        return [[M11, M12, M13],
                [M21, M22, M23],
                [M31, M32, M33]]
    
    def getGravity(self, O1, O2, O3):
        G1 = -(self.L[0]*self.m[1]*s(O1) + self.L[0]*self.m[2]*s(O1) + self.L[1]*self.m[2]*s(O1+O2) + \
            self.Mc[0]*self.m[0]*s(O1) + self.Mc[1]*self.m[1]*s(O1+O2) + self.Mc[2]*self.m[2]*s(O1+O2+O3))
        
        G2 = -(self.L[1]*self.m[2]*s(O1+O2) + self.Mc[1]*self.m[1]*s(O1+O2) + self.Mc[2]*self.m[2]*s(O1+O2+O3))

        G3 = -self.Mc[2]*self.m[2]*s(O1+O2+O3)

        return [G1*self.g, G2*self.g, G3*self.g]

    def getcCoriolis(self, O1, O2, O3, dO1, dO2, dO3):
        C1 = -2*self.L[0]*(self.L[1]*self.m[2]*s(O2) + self.Mc[2]*self.m[2]*s(O2+O3) + self.Mc[1]*self.m[1]*s(O2)) * dO2*dO3 \
            - 2*self.Mc[2]*self.m[2]*(self.L[0]*s(O2+O3) + self.L[1]*s(+O3))* dO1*dO3 \
            - self.L[0]*(self.L[1]*self.m[2]*s(O2) + self.Mc[2]*self.m[2]*s(O2+O3) + self.Mc[1]*self.m[1]*s(O2)) * dO2*dO2 \
            - 2*self.Mc[2]*self.m[2]*(self.L[0]*s(O2+O3) + self.L[1]*s(+O3))* dO2*dO3 \
            - self.Mc[2]*self.m[2]*(self.L[0]*s(O2+O3) + self.L[1]*s(+O3))* dO3*dO3 
        
        C2 = self.L[0]*(self.L[1]*self.m[2]*s(O2) + self.Mc[2]*self.m[2]*s(O2+O3) + self.Mc[1]*self.m[1]*s(O2)) * dO1*dO1 \
            - 2*self.L[1]*self.Mc[2]*self.m[2]*s(O3) * dO1*dO3 \
            - 2*self.L[1]*self.Mc[2]*self.m[2]*s(O3) * dO2*dO3 \
            - self.L[1]*self.Mc[2]*self.m[2]*s(O3) * dO3*dO3

        C3 = self.Mc[2]*self.m[2]*(self.L[0]*s(O2+O3) + self.L[1]*s(+O3))* dO1*dO1 \
            + 2*self.L[1]*self.Mc[2]*self.m[2]*s(O3) * dO1*dO2 \
            + self.L[1]*self.Mc[2]*self.m[2]*s(O3) * dO2*dO2 \

        return [C1, C2, C3]

    def getGravityLinearMatrix(self):
        G11_coef = -(self.L[0]*self.m[1] + self.L[0]*self.m[2] - self.L[1]*self.m[2] + \
                     self.Mc[0]*self.m[0] - self.Mc[1]*self.m[1] - self.Mc[2]*self.m[2]) *self.g
        G12_coef = (self.L[1]*self.m[2]  + self.Mc[1]*self.m[1] + self.Mc[2]*self.m[2]) *self.g
        G13_coef = (self.Mc[2]*self.m[2]) *self.g
        G21_coef = (self.L[1]*self.m[2]  + self.Mc[1]*self.m[1] + self.Mc[2]*self.m[2]) *self.g
        G22_coef = (self.L[1]*self.m[2]  + self.Mc[1]*self.m[1] + self.Mc[2]*self.m[2]) *self.g
        G23_coef = (self.Mc[2]*self.m[2]) *self.g
        G31_coef = (self.Mc[2]*self.m[2]) *self.g
        G32_coef = (self.Mc[2]*self.m[2]) *self.g
        G33_coef = (self.Mc[2]*self.m[2]) *self.g

        G_coef = np.array([[G11_coef, G12_coef, G13_coef],
                          [G21_coef, G22_coef, G23_coef],
                          [G31_coef, G32_coef, G33_coef]])

        return G_coef

    def getMassLinearMatrix(self):
        M11 = self.I[0] + self.I[1] + self.I[2] + (self.L[0]**2 + self.L[1]**2 + self.Mc[2]**2)*self.m[2]+\
            (self.L[0]**2 + self.Mc[1]**2) * self.m[1] + self.Mc[0]**2*self.m[0] \
            - 2*self.L[0]*self.m[2]*(self.L[1] + self.Mc[2]) \
            - 2*self.L[0]*self.Mc[1]*self.m[1] + 2*self.L[1]*self.Mc[2]*self.m[2]

        M12 = self.I[1] + self.I[2] + (self.L[1]**2 + self.Mc[2]**2)*self.m[2] + self.Mc[2]**2*self.m[1] - \
            self.L[0]*self.L[1]*self.m[2] - self.L[0]*self.m[2]*self.Mc[2] - \
            self.L[0]*self.Mc[1]*self.m[1] + 2*self.L[1]*self.Mc[2]*self.m[2]

        M13 = self.I[2] + self.Mc[2]**2*self.m[2] - self.L[0]*self.Mc[2]*self.m[2] + self.L[1]*self.Mc[2]*self.m[2]
        
        M21 = M12
        M22 = self.I[1] + self.I[2] + (self.L[1]**2 + self.Mc[2]**2)*self.m[2] + self.Mc[2]**2*self.m[1] + \
            2*self.L[1]*self.Mc[2]*self.m[2]
        M23 = self.I[2] + self.Mc[2]**2*self.m[2] + self.L[1]*self.Mc[2]*self.m[2]

        M31 = M13
        M32 = M23
        M33 = self.I[2] + self.Mc[2]**2*self.m[2]

        return [[M11, M12, M13],
                [M21, M22, M23],
                [M31, M32, M33]]

    def getGravityLinearMatrix2(self):
        G11_coef = -(self.L[0]*self.m[1] + self.L[0]*self.m[2] + self.L[1]*self.m[2] + \
                     self.Mc[0]*self.m[0] + self.Mc[1]*self.m[1] + self.Mc[2]*self.m[2]) *self.g
        G12_coef = -(self.L[1]*self.m[2]  + self.Mc[1]*self.m[1] + self.Mc[2]*self.m[2]) *self.g
        G13_coef = -(self.Mc[2]*self.m[2]) *self.g
        G21_coef = -(self.L[1]*self.m[2]  + self.Mc[1]*self.m[1] + self.Mc[2]*self.m[2]) *self.g
        G22_coef = -(self.L[1]*self.m[2]  + self.Mc[1]*self.m[1] + self.Mc[2]*self.m[2]) *self.g
        G23_coef = -(self.Mc[2]*self.m[2]) *self.g
        G31_coef = -(self.Mc[2]*self.m[2]) *self.g
        G32_coef = -(self.Mc[2]*self.m[2]) *self.g
        G33_coef = -(self.Mc[2]*self.m[2]) *self.g

        G_coef = np.array([[G11_coef, G12_coef, G13_coef],
                          [G21_coef, G22_coef, G23_coef],
                          [G31_coef, G32_coef, G33_coef]])

        return G_coef

    def getMassLinearMatrix2(self):
        M11 = self.I[0] + self.I[1] + self.I[2] + (self.L[0]**2 + self.L[1]**2 + self.Mc[2]**2)*self.m[2]+\
            (self.L[0]**2 + self.Mc[1]**2) * self.m[1] + self.Mc[0]**2*self.m[0] \
            + 2*self.L[0]*self.m[2]*(self.L[1] + self.Mc[2]) \
            + 2*self.L[0]*self.Mc[1]*self.m[1] + 2*self.L[1]*self.Mc[2]*self.m[2]

        M12 = self.I[1] + self.I[2] + (self.L[1]**2 + self.Mc[2]**2)*self.m[2] + self.Mc[1]**2*self.m[1] + \
            self.L[0]*self.L[1]*self.m[2] + self.L[0]*self.m[2]*self.Mc[2] + \
            self.L[0]*self.Mc[1]*self.m[1] + 2*self.L[1]*self.Mc[2]*self.m[2]

        M13 = self.I[2] + self.Mc[2]**2*self.m[2] + self.L[0]*self.Mc[2]*self.m[2] + self.L[1]*self.Mc[2]*self.m[2]
        
        M21 = M12
        M22 = self.I[1] + self.I[2] + (self.L[1]**2 + self.Mc[2]**2)*self.m[2] + self.Mc[1]**2*self.m[1] + \
            2*self.L[1]*self.Mc[2]*self.m[2]
        M23 = self.I[2] + self.Mc[2]**2*self.m[2] + self.L[1]*self.Mc[2]*self.m[2]

        M31 = M13
        M32 = M23
        M33 = self.I[2] + self.Mc[2]**2*self.m[2]

        return [[M11, M12, M13],
                [M21, M22, M23],
                [M31, M32, M33]]

def Dribble_model(ParamData,symvar_type='SX'):
    """
    dynamics model:
    x1_dot = x2
    x2_dot = -g - F/m
    """

    I = ParamData['Robot']['Mass']['inertia']
    m = ParamData['Robot']['Mass']['mass']
    Mc = ParamData['Robot']['Mass']['massCenter']
    L = [ParamData['Robot']['Geometry']['L_body'],
            ParamData['Robot']['Geometry']['L_thigh'],
            ParamData['Robot']['Geometry']['L_shank']]
    
    g = ParamData['Environment']['Gravity']

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # set variable of the dynamics system
    theta1 = model.set_variable(var_type='_x', var_name='theta1', shape=(1, 1))
    theta2 = model.set_variable(var_type='_x', var_name='theta2', shape=(1, 1))
    theta3 = model.set_variable(var_type='_x', var_name='theta3', shape=(1, 1))

    # x2 = model.set_variable(var_type='_x', var_name='theta_dot', shape=(1, 1))
    dtheta1 = model.set_variable(var_type='_x', var_name='dtheta1', shape=(1, 1))
    dtheta2 = model.set_variable(var_type='_x', var_name='dtheta2', shape=(1, 1))
    dtheta3 = model.set_variable(var_type='_x', var_name='dtheta3', shape=(1, 1))
    ddtheta1 = model.set_variable('_z', 'ddtheta1', (1,1))
    ddtheta2 = model.set_variable('_z', 'ddtheta2', (1,1))
    ddtheta3 = model.set_variable('_z', 'ddtheta3', (1,1))
    t1 = model.set_variable(var_type='_u', var_name='t1', shape=(1, 1))
    t2 = model.set_variable(var_type='_u', var_name='t2', shape=(1, 1))

    # rhs
    model.set_rhs('theta1', dtheta1)
    model.set_rhs('theta2', dtheta2)
    model.set_rhs('theta3', dtheta3)
    model.set_rhs('dtheta1', ddtheta1)
    model.set_rhs('dtheta2', ddtheta2)
    model.set_rhs('dtheta3', ddtheta3)

    # get robot dynamics property
    Robot = RobotProperty(ParamData)
    Mass = Robot.getMassMatrix(theta1, theta2, theta3)
    Gravity = Robot.getGravity(theta1, theta2, theta3)
    Coriolis = Robot.getcCoriolis(theta1, theta2, theta3, dtheta1, dtheta2, dtheta3)
    # print(Mass)
    # print(Gravity)
    # print(Coriolis)

    euler_lagrange = vertcat(
        Mass[0][0]*ddtheta1 + Mass[0][1]*ddtheta2 + Mass[0][2]*ddtheta3 + Gravity[0] + Coriolis[0],
        Mass[1][0]*ddtheta1 + Mass[1][1]*ddtheta2 + Mass[1][2]*ddtheta3 + Gravity[1] + Coriolis[1] - t1,
        Mass[2][0]*ddtheta1 + Mass[2][1]*ddtheta2 + Mass[2][2]*ddtheta3 + Gravity[2] + Coriolis[2] - t2,
    
    )
    model.set_alg('euler_lagrange', euler_lagrange)

    yb = Mc[0]*c(theta1)
    yt = L[0]*c(theta1) + Mc[1]*c(theta1 + theta2)
    ys = L[0]*c(theta1) + L[1]*c(theta1 + theta2)+  Mc[2]*c(theta1 + theta2 + theta3)

    dxb = Mc[0]*c(theta1)*dtheta1
    dyb = -Mc[0]*s(theta1)*dtheta1
    dxt = L[0]*c(theta1)*dtheta1 + Mc[1]*c(theta1 + theta2)*(dtheta1 + dtheta2)
    dyt = -L[0]*s(theta1)*dtheta1 - Mc[1]*s(theta1 + theta2)*(dtheta1 + dtheta2)
    dxs = L[0]*c(theta1)*dtheta1 + L[1]*c(theta1 + theta2)*(dtheta1 + dtheta2) + Mc[2]*c(theta1 + theta2 + theta3)*(dtheta1 + dtheta2 + dtheta3)
    dys = -L[0]*s(theta1)*dtheta1 - L[1]*s(theta1 + theta2)*(dtheta1 + dtheta2) - Mc[2]*s(theta1 + theta2 + theta3)*(dtheta1 + dtheta2 + dtheta3)
    
    E_b = 0.5 * m[0] * (dxb**2 + dyb**2) + 0.5*I[0]*dtheta1**2
    E_t = 0.5 * m[1] * (dxt**2 + dyt**2) + 0.5*I[1]*(dtheta1 + dtheta2)**2
    E_s = 0.5 * m[2] * (dxs**2 + dys**2) + 0.5*I[2]*(dtheta1 + dtheta2 + dtheta3)**2

    E = E_b + E_s + E_t

    V = m[0]*g*yb + m[1]*g*yt + m[2]*g*ys
    V1 = m[0]*g*yb

    model.set_expression('E', E)
    model.set_expression('V1', V1)
    model.set_expression('V', V)

    model.setup()

    return model

def Dribble_mpc(model, ParamData):

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 150,
        't_step': 0.005,
        'n_robust': 1,
        'store_full_solution': True,  
        # 'open_loop': True,  
        # 'state_discretization': 'collocation',
        # 'collocation_type': 'radau',
        # 'collocation_deg': 4,
        # 'collocation_ni': 2,
    }
    mpc.set_param(**setup_mpc)

    # mterm = (model.x['theta1'])**2
    # lterm = (model.x['theta1'])**2
    mterm = model.aux['E'] - model.aux['V1'] + (model.x['theta1'])**2
    lterm = model.aux['E'] - model.aux['V1'] + (model.x['theta1'])**2
    
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(t1=1e-4, t2=1e-4)

    mpc.bounds['lower', '_x', 'theta1'] = ParamData['Controller']['Boundary']['theta1'][0]
    mpc.bounds['upper', '_x', 'theta1'] = ParamData['Controller']['Boundary']['theta1'][1]

    mpc.bounds['lower', '_x', 'theta2'] = ParamData['Controller']['Boundary']['theta2'][0]
    mpc.bounds['upper', '_x', 'theta2'] = ParamData['Controller']['Boundary']['theta2'][1]

    mpc.bounds['lower', '_x', 'theta3'] = ParamData['Controller']['Boundary']['theta3'][0]
    mpc.bounds['upper', '_x', 'theta3'] = ParamData['Controller']['Boundary']['theta3'][1]

    mpc.bounds['lower', '_u', 't1'] = -500.0
    mpc.bounds['upper', '_u', 't1'] = 500.0

    mpc.bounds['lower', '_u', 't2'] = -500.0
    mpc.bounds['upper', '_u', 't2'] = 500.0

            
    # mpc.set_tvp_fun(tvp_fun)

    mpc.setup()
    # setuptime = datetime.datetime.now()
    # print("setup time: ", setuptime -setbound_etime)

    return mpc

def Dribble_simulator(model):
    """
    --------------------------------------------------------------------------
    template_simulator: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        # Note: cvode doesn't support DAE systems.
        'integration_tool': 'idas',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 0.005
    }

    simulator.set_param(**params_simulator)

    tvp_template = simulator.get_tvp_template()

    def tvp_fun(t_ind):
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator

def StepResponse(cfg):
    Robot = RobotProperty(cfg)
    MassMatrix = Robot.getMassMatrix(0, np.pi, 0)
    Gravity = Robot.getGravityLinearMatrix()
    s = sym('s')
    for i in range(3):
        for j in range(3):
            Mass = MassMatrix[i][j] * s * s
    K = Mass + Gravity
    K = sympy.Matrix(K)
    print(K)
    K_inv = K.inv()
    row1 = K_inv.row(0)
    col2 = row1.col(0)
    col2 = sympy.simplify(col2)
    print(col2)

    up1 = [0.000088, 0, 0.063, 0, 6.95, 0, 144.28]
    low1 = [0.00045, 0, 0.32, 0, 35.45, 0, 736]
    sys1 = signal.TransferFunction(up1, low1)
    t1, y1 = signal.step(sys1)
    # plt.figure()
    # plt.plot(t1, y1)
    # plt.show()
    pass

if __name__ =="__main__":
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/Dual.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    cfg = yaml.load(ParamFile, Loader=yaml.FullLoader)

    StepResponse(cfg)
    pass

# if __name__ =="__main__":
#     ## get params config data
#     ## /..../Simulation/model_raisim/DRMPC/SaTiZ_3D/scripts
#     # FilePath1 = os.path.dirname(os.path.abspath(__file__))
#     ## os.path.abspath(__file__): /.../DRMPC/SaTiZ_3D/scripts/Dynamics_MPC.py
#     ## os.path.dirname(__file__): /.../DRMPC/SaTiZ_3D/scripts

#     # /..../Simulation/model_raisim/DRMPC/SaTiZ_3D
#     FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     ParamFilePath = FilePath + "/config/Dual.yaml"
#     ParamFile = open(ParamFilePath, "r", encoding="utf-8")
#     ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

#     # Dribble_model(ParamData)
#     model = Dribble_model(ParamData)
#     mpc = Dribble_mpc(model, ParamData)
#     simulator = Dribble_simulator(model)
#     estimator = do_mpc.estimator.StateFeedback(model)

#     """
#     Set initial state
#     """
#     # x0 = np.pi*np.array([-0.05, 1.05, 0.5, 0, 0, 0]).reshape(-1,1)
#     simulator.x0['theta1'] = -0.05*np.pi
#     simulator.x0['theta2'] = 1.05*np.pi
#     simulator.x0['theta3'] = 0.5*np.pi

#     x0 = simulator.x0.cat.full()
#     mpc.x0 = x0
#     simulator.x0 = x0
#     estimator.x0 = x0

#     mpc.set_initial_guess()

#     """
#     Setup graphic:
#     """

#     fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
#     plt.ion()

#     """
#     Run MPC main loop:
#     """
#     show_animation = True
#     mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
#     sim_graphics = do_mpc.graphics.Graphics(simulator.data)
#     for k in range(500):
#         u0 = mpc.make_step(x0)
#         y_next = simulator.make_step(u0)
#         x0 = estimator.make_step(y_next)

#         # if show_animation:
#         #     graphics.plot_results(t_ind=k)
#         #     graphics.plot_predictions(t_ind=k)
#         #     graphics.reset_axes()
#         #     plt.show()
#         #     plt.pause(0.01)

#     # mpc_graphics.plot_predictions(t_ind=0)
#     # Plot results until current time
#     mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
#     sim_graphics = do_mpc.graphics.Graphics(simulator.data)
#     sim_graphics.plot_results()
#     sim_graphics.reset_axes()
#     plt.show()

#     SavePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     SavePath = SavePath + '/data/2022-03-21/'
#     store_results = True
#     if store_results:
#         # do_mpc.data.save_results([mpc, simulator], SavePath)
#         do_mpc.data.save_results([mpc, simulator])

