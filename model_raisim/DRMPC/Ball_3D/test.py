import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

theta = np.linspace(0, 2*math.pi, 200)
x = np.sin(theta)
y = np.cos(theta)
z = 0.15 + 0.6 * np.abs(np.cos(1.5 * theta))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z)
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Reference Trajectory')

plt.show()