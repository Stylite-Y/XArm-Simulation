import numpy as np
from scipy.spatial.transform import Rotation

Quad = np.array([-0.0085,  0.5497, -0.0016,  0.8353])
P_ho = np.array([[0.1277,  0.0327, -0.0173]])
P_lo = np.array([0.0828, 0.1211, 0.2202, 1])
RotMatrix = Rotation.from_quat(Quad)
rotation_m = RotMatrix.as_matrix()
P = -rotation_m.T @ P_ho.T
Trans_m2 = np.concatenate((rotation_m, P_ho.T), axis = 1)
Trans_m2 = np.concatenate((Trans_m2, np.array([[0, 0, 0, 1]])), axis = 0)
print(np.linalg.inv(Trans_m2))
Trans_m = np.concatenate((rotation_m.T, P), axis = 1)
Trans_m = np.concatenate((Trans_m, np.array([[0, 0, 0, 1]])), axis = 0)
# P_lo = np.array([[Data[n][(i-1)*16], Data[n][(i-1)*16+1], Data[n][(i-1)*16+2], 1]])
P_lh = Trans_m @ P_lo.T
print(Trans_m, P_lh)