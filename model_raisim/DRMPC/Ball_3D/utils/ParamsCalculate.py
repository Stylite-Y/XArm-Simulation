import os
import numpy as np

class ControlParamCal(object):
    def __init__(self, mass = 0.5):
        self.mass = mass

    def ParamsCal(self, x0, v0, CtlParam, g):
        dx_up = CtlParam["x_top"] - x0
        dx_down = CtlParam["x_top"] - CtlParam["x_ref"]

        k_vir = (self.mass * v0 ** 2 - 2 * self.mass * g * dx_up - 2 * CtlParam["f_up"] * dx_up) \
                / ((CtlParam["x_top"] - CtlParam["x_ref"]) ** 2 - (x0 - CtlParam["x_ref"]) ** 2)
        f_down = (self.mass * (CtlParam["v_ref"]) ** 2 - 2 * self.mass * g * dx_down - k_vir * dx_down ** 2) / (2 * dx_down)

        print("dx_up and dx_down is ", dx_up, dx_down)
        if k_vir < 0:
            raise ValueError('invalid value: k_vir is negative, can not sqrt:')
        return k_vir, f_down