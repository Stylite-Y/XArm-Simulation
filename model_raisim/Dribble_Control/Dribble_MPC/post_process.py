import numpy as np
import math
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from numpy.core.fromnumeric import _all_dispatcher

matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['lines.linewidth'] = 2
# matplotlib.rcParams['axes.grid'] = True

f = open('./results/BALL_robust_MPC_5-r0.1-u50.pkl','rb')
data = pickle.load(f)

x = data['mpc']['_x','x_b']
dx = data['mpc']['_x','dx_b']

x = np.array(x.flatten())
dx = np.array(dx.flatten())
print(dx.shape)
print(x.shape)


# dydx = x
dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
print(dydx.shape)
points = np.array([x, dx]).T.reshape(-1, 1, 2)
temp = np.array([x, dx])
print("temp: ", temp)

segments = np.concatenate([points[:-1], points[1:]], axis=1)
print("point: ",points)
print("segment: ", segments)
print(segments.shape)
fig, axs = plt.subplots(1, 1)
norm = plt.Normalize(dydx.min(), dydx.max())  
print(type(norm))
lc = LineCollection(segments, cmap='viridis', norm=norm)
# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs.add_collection(lc)
# fig.colorbar(line, ax=axs)

# axs.set_facecolor('blue')

# axs.broken_barh([(-0.6, 0.13), (-0.47, 0.37), (-0.1, 0.2)], (-8, 16),
#                facecolors=( '#f4c7ab', '#deedf0', '#fff5eb'))

axs.broken_barh([(-0.6, 0.13), (-0.47, 0.37), (-0.1, 0.2)], (-8, 16),
               facecolors=( '#fffbdf', '#c6ffc1', '#34656d'))

axs.set_xlim(-0.6, 0.05)
axs.set_ylim(-8, 8)
plt.xlabel('x_b')
plt.ylabel('dx_b')
plt.show()