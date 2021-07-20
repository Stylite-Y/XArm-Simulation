import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.collections import LineCollection

dates = pd.date_range("2017-01-01", "2017-06-20", freq="7D" )
y = np.cumsum(np.random.normal(size=len(dates)))

s = pd.Series(y, index=dates)

fig, ax = plt.subplots()

#convert dates to numbers first
inxval = mdates.date2num(s.index.to_pydatetime())
points = np.array([inxval, s.values]).T.reshape(-1,1,2)
segments = np.concatenate([points[:-1],points[1:]], axis=1)
print("point: ",points)
print(points.shape)
print("segment: ", segments)
print(segments.shape)

lc = LineCollection(segments, cmap="plasma", linewidth=3)
# set color to date values
lc.set_array(inxval)
# note that you could also set the colors according to y values
# lc.set_array(s.values)
# add collection to axes
ax.add_collection(lc)


ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator())
monthFmt = mdates.DateFormatter("%b")
ax.xaxis.set_major_formatter(monthFmt)
ax.autoscale_view()
plt.show()