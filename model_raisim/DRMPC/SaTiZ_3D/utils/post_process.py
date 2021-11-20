import numpy as np
import math
import os
import yaml
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from numpy.core.fromnumeric import _all_dispatcher

matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['lines.linewidth'] = 2
# matplotlib.rcParams['axes.grid'] = True

# f = open('../data/2021-08-09/2021-08-09-x_ref_0.35-v0_6-vref_-8-K_500.0-Kd_up_5.0-Kd_down_20.0.pkl','rb')
# data = pickle.load(f)

FilePath = os.path.abspath(os.getcwd())
ParamFilePath = FilePath + "/data/2021-08-09/2021-08-09-x_ref_0.35-v0_6-vref_-8-K_500.0-Kd_up_5.0-Kd_down_20.0.pkl"
ParamFile = open(ParamFilePath, "rb")
data = pickle.load(ParamFile)

x = data['BallState'][1:, 0]
dx = data['BallState'][1:, 1]

x = np.array(x.flatten())
dx = np.array(dx.flatten())
print(dx.shape)
print(x.shape)


# dydx = x
dydx = np.sin(0.5 * (x[:-1] + x[1:]))  # first derivative
print(dydx.shape)
points = np.array([x, dx]).T.reshape(-1, 1, 2)
temp = np.array([x, dx])
print("temp: ", temp)

segments = np.concatenate([points[:-1], points[1:]], axis=1)
print("point: ",points)
print("segment: ", segments)
print(segments.shape)
fig, axs = plt.subplots(1, 1)
fig.suptitle('Phase space of Dribbling Ball', fontsize = 20)
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

axs.broken_barh([(0, 0.1), (0.1, 0.4), (0.37, 1)], (-20, 40),
               facecolors=( '#fffbdf', '#c6ffc1', '#34656d'))

axs.set_xlim(0, 0.6)
axs.set_ylim(-8, 8)
t = np.linspace(0, 1, 50)
maxv = - np.ones([len(t)]) * 6
plt.plot(t, maxv, '#FFFF00')
plt.xlabel('Height of ball (m)')
plt.ylabel('Velocity of ball (m/s)')
# plt.show()

footstate = data['BallState']

footpos = footstate[:, 0]
footvel = footstate[:, 1]

time = data['time']
num = 0
sample = np.array([0.0])
sample_t = np.array([0.0])
for i in range(20000):
    if footvel[i] < 0 and footvel[i + 1] > 0:
        print(time[i])
        num = num + 1
        sample_t = np.concatenate([sample_t, [time[i]]])

sample_t = sample_t[1:]
print(num)
print(sample_t)

for i in range(len(sample_t)):
    sample = np.concatenate([sample, footstate[sample_t[i] + 0.55]])

print(sample)