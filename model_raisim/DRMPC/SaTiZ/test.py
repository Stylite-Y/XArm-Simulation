from casadi.casadi import linspace
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

class A():
    def __init__(self, x):
        self.opti = ca.Opti()
        self.a = x[0]
        self.b = x[1]
        self.NN = [self.a, self.b]
        self.q = []
        self.q.append([self.opti.variable(2) for _ in range(self.NN[0])])
        self.q.append([self.opti.variable(2) for _ in range(self.NN[1])])

    def sumtest(self, a):
        sumres = a + a
        return sumres

    def multi(self, b):
        res = b * b
        return res
    
if __name__ == "__main__":
    x = np.array([1, 2])
    res = A(x)
    print(res.q)
    pass

