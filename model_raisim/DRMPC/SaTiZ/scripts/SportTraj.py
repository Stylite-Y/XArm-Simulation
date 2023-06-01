import os
import sympy as syt 
import numpy as np
import matplotlib.pyplot as plt
import datetime


def Badminton():
    Path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DataFile_e = Path + '/data/Sport_Traj/badminton_elbow.txt'
    DataFile_s = Path + '/data/Sport_Traj/badminton_shoulder.txt'

    Data_e = np.loadtxt(DataFile_e)
    Data_s = np.loadtxt(DataFile_s)

    # arm len
    l = 0.3

    ang_e = Data_e[0:93,1]
    ang_s = Data_s[0:93,1]

    ang_s = ang_s*np.pi/180
    ang_e = ang_e*np.pi/180


    x1 = -l*np.cos(ang_s)
    y1 = l*np.sin(ang_s)
    x2 = -l*np.cos(ang_s) - l*np.cos(ang_e+ang_s-np.pi)
    y2 = l*np.sin(ang_s) + l*np.sin(ang_e+ang_s-np.pi)

    # print(int(np.linspace(0,92,6)))

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'legend.fontsize': 30,
        'axes.labelsize': 35,
        'lines.linewidth': 3,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 8,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    fig2, axs = plt.subplots(1, 1, figsize=(12, 12))
    ax1 = axs

    num_frame = 6
    for i in np.linspace(0, 92, num_frame):
        id = np.int(i)
        ax1.plot([0, x1[id]], [0, y1[id]], 'o-', color = 'C0', ms=1, alpha=i/92*0.8+0.2)
        ax1.plot([x1[id], x2[id]], [y1[id], y2[id]], 'o-', color = 'C0', ms=1, alpha=i/92*0.8+0.2)

    line1, = ax1.plot(x2,y2, label = 'Badminton')
    ax1.scatter(x2,y2)
    ax1.set_xlabel('X(m)')
    ax1.set_ylabel('Y(m)')
    plt.legend()

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    savename1 =  save_dir + "Badminton traj.jpg"
    fig2.savefig(savename1, dpi=500)
    
    
    plt.show()
    pass

def Tennis():
    Path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DataFile_s = Path + '/data/Sport_Traj/tennis_s_fle2.txt'
    DataFile_e = Path + '/data/Sport_Traj/tennis_elbow.txt'

    Data_e = np.loadtxt(DataFile_e)
    Data_s = np.loadtxt(DataFile_s)

    # arm len
    l = 0.3

    ang_e = Data_e[:,1]
    ang_s = Data_s[:,1]

    ang_s = ang_s*np.pi/180
    ang_e = ang_e*np.pi/180


    x1 = l*np.cos(ang_s)
    y1 = l*np.sin(ang_s)
    x2 = l*np.cos(ang_s) - l*np.cos(ang_e-ang_s)
    y2 = l*np.sin(ang_s) + l*np.sin(ang_e-ang_s)

    # print(int(np.linspace(0,92,6)))

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'lines.linewidth': 4,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 8,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    fig2, axs = plt.subplots(1, 1, figsize=(12, 12))
    ax1 = axs

    num_frame = 6
    for i in np.linspace(0, 51, num_frame):
        id = np.int(i)
        ax1.plot([0, x1[id]], [0, y1[id]], 'o-', color = 'C0', ms=1, alpha=i/52*0.8+0.2)
        ax1.plot([x1[id], x2[id]], [y1[id], y2[id]], 'o-', color = 'C0', ms=1, alpha=i/52*0.8+0.2)

    ax1.plot(x2,y2)
    ax1.scatter(x2,y2)
    ax1.set_xlabel('X(m)')
    ax1.set_ylabel('Z(m)')
    
    plt.show()
    pass

def TableTennis():
    Path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DataFile = Path + '/data/Sport_Traj/Tabletennis.txt'

    Data = np.loadtxt(DataFile)
    x2 = Data[:,0]
    y2 = Data[:,1]+0.2

    # arm len
    l = 0.3

    # inverse kinematic
    q0 = np.arctan2(y2, x2)
    a0 = np.sqrt(x2**2+y2**2)
    tmp = (x2**2+y2**2)/(2*a0*l)
    q1 = -np.arccos(tmp)+q0
    q2 = np.arccos(tmp)+q0-q1
    print(q1)
    print(q2)

    # ang_e = Data_e[0:93,1]

    # ang_s = ang_s*np.pi/180
    # ang_e = ang_e*np.pi/180


    x1 = l*np.cos(q1)
    y1 = l*np.sin(q1)
    x2 = l*np.cos(q1) + l*np.cos(q1+q2)
    y2 = l*np.sin(q1) + l*np.sin(q1+q2)

    # print(int(np.linspace(0,92,6)))

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'legend.fontsize': 30,
        'axes.labelsize': 35,
        'lines.linewidth': 3,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 8,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    fig2, axs = plt.subplots(1, 1, figsize=(12, 12))
    ax1 = axs

    num_frame = 5
    for i in np.linspace(0, 59, num_frame):
        id = np.int(i)
        ax1.plot([0, x1[id]], [0, y1[id]], 'o-', color = 'C0', ms=1, alpha=1.2-(i/59*0.8+0.2))
        ax1.plot([x1[id], x2[id]], [y1[id], y2[id]], 'o-', color = 'C0', ms=1, alpha=1.2-(i/59*0.8+0.2))

    line1, = ax1.plot(x2,y2, label = 'Table tennis')
    # add_arrow_to_line2D(ax1, line1, arrow_locs=[0.2, 0.4, 0.6, 0.8], arrowstyle='-|>')
    ax1.scatter(x2,y2)
    ax1.set_xlabel('X(m)')
    ax1.set_ylabel('Y(m)')
    ax1.set_xlim([-0.3,0.4])
    plt.legend()

    StorePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todaytime=datetime.date.today()
    save_dir = StorePath + "/data/" + str(todaytime) + "/"
    savename1 =  save_dir + "table tennis traj.jpg"
    fig2.savefig(savename1, dpi=500)
    
    plt.show()
    pass

def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle='-|>', arrowsize=2, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows



if __name__ == "__main__":
    # Badminton()
    TableTennis()
    # Tennis()
    pass