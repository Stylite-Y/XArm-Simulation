import os
import sympy as sy
import raisimpy as raisim
import yaml
import time
from cv2 import ellipse2Poly
import numpy as np
from numpy import sin as s
from numpy import cos as c
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import proj3d
from scipy.optimize import fsolve

def EigAndSVD():
    l1 = 1
    l2 = 1
    theta1 = np.pi/3
    theta2 = np.pi/6

    Ot = np.linspace(0, 2*np.pi, 50)
    x = c(Ot)
    y = s(Ot)

    Jacobian = np.array([[-l1*s(theta1)-l2*s(theta1+theta2), -l2*s(theta1+theta2)],
                [l1*c(theta1)+l2*c(theta1+theta2), l2*c(theta1+theta2)]])

    eig, fv  = np.linalg.eig(Jacobian)
    fv_eig1 = np.dot(Jacobian, fv[:,0])
    fv_eig2 = np.dot(Jacobian, fv[:,1])
    U, Sigma, V  = np.linalg.svd(Jacobian)
    V_eig1 = np.dot(Jacobian, V[:,0])
    V_eig2 = np.dot(Jacobian, V[:,1])

    nx, ny = [], []
    for i in range(len(x)):
        pos = np.array([[x[i]],[y[i]]])
        npos = np.dot(Jacobian, pos)
        nx.append(npos[0])
        ny.append(npos[1])

    # region: print
    print("="*50)
    print("Eigenvalue")
    print(eig)
    print("="*50)
    print("Eigenvector")
    print(fv)
    print("="*50)
    print("Sigular vector")
    print(V)
    print("="*50)
    print("Sigular Value")
    print(Sigma)
    print("="*50)
    print("Sigular unit")
    print(U)
    # endregion

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 6.0,
        'axes.labelpad': 15.0,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax1 = ax[0]
    ax2 = ax[1]

    # region: eig and fv arrow plot
    mc = 20
    arrow1 = mpatches.FancyArrowPatch((0, 0), (fv[0][0], fv[1][0]), mutation_scale=mc, color = 'C0', alpha = 0.5)
    arrow2 = mpatches.FancyArrowPatch((0, 0), (fv_eig1[0], fv_eig1[1]), mutation_scale=mc, color = 'C0')
    arrow3 = mpatches.FancyArrowPatch((0, 0), (fv[0][1], fv[1][1]), mutation_scale=mc, color = 'C1', alpha = 0.5)
    arrow4 = mpatches.FancyArrowPatch((0, 0), (fv_eig2[0], fv_eig2[1]), mutation_scale=mc, color = 'C1')
    ax1.add_patch(arrow1)
    ax1.add_patch(arrow2)
    ax1.add_patch(arrow3)
    ax1.add_patch(arrow4)

    ax1.scatter(x, y, s=30, marker='o', c='none',edgecolors='black', alpha = 0.4, label = "x")
    # ax.scatter(x, y, s=40, marker='*', color = 'black', alpha = 0.5)
    ax1.scatter(nx, ny, s=40, color = 'black', alpha = 0.6, label = "Ax")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Eig')
    ax1.axis('equal')
    ax1.legend()
    # endregion

    # region: SVD arrow plot
    mc = 20
    arrow1 = mpatches.FancyArrowPatch((0, 0), (V[0][0], V[1][0]), mutation_scale=mc, color = 'C0', alpha = 0.5)
    arrow2 = mpatches.FancyArrowPatch((0, 0), (V_eig1[0], V_eig1[1]), mutation_scale=mc, color = 'C0')
    arrow3 = mpatches.FancyArrowPatch((0, 0), (V[0][1], V[1][1]), mutation_scale=mc, color = 'C1', alpha = 0.5)
    arrow4 = mpatches.FancyArrowPatch((0, 0), (V_eig2[0], V_eig2[1]), mutation_scale=mc, color = 'C1')
    ax2.add_patch(arrow1)
    ax2.add_patch(arrow2)
    ax2.add_patch(arrow3)
    ax2.add_patch(arrow4)

    ax2.scatter(x, y, s=30, marker='o', c='none',edgecolors='black', alpha = 0.4, label = "x")
    # ax.scatter(x, y, s=40, marker='*', color = 'black', alpha = 0.5)
    ax2.scatter(nx, ny, s=40, color = 'black', alpha = 0.6, label = "Ax")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('SVD')
    ax2.axis('equal')
    ax2.legend()
    # endregion
    plt.show()  
    pass

def EigAndSVD2():
    l1 = 1
    l2 = 1
    l3 = 1
    m1 = 4
    m2 = 2
    m2 = 1
    theta1 = np.pi/8
    theta2 = np.pi/8
    theta3 = np.pi/8

    Ot = np.linspace(0, 2*np.pi, 200)
    at = np.linspace(0, 2*np.pi, 200)
    x = s(np.pi/6)*c(Ot)
    y = s(np.pi/6)*s(Ot)
    z = c(np.pi/6)

    lx1 = l1*c(theta1)
    ly1 = l1*s(theta1)
    lx2 = l1*c(theta1)+l2*c(theta1+theta2)
    ly2 = l1*s(theta1)+l2*s(theta1+theta2)
    lx3 = l1*c(theta1)+l2*c(theta1+theta2)+l3*c(theta1+theta2+theta3)
    ly3 = l1*s(theta1)+l2*s(theta1+theta2)+l3*s(theta1+theta2+theta3)

    Jacobian = np.array([[-l1*s(theta1)-l2*s(theta1+theta2)-l3*s(theta1+theta2+theta3), -l2*s(theta1+theta2)-l3*s(theta1+theta2+theta3), -l3*s(theta1+theta2+theta3)],
                [l1*c(theta1)+l2*c(theta1+theta2)+l3*c(theta1+theta2+theta3), l2*c(theta1+theta2)+l3*c(theta1+theta2+theta3), l3*c(theta1+theta2+theta3)]])

    Maxtrx_m = np.array([[m2*l1**2+m1*l1**2/3+m2*l2**2/3+m2*l1*l2*c(theta2), m2*l2**2/3+m2*l1*l2*c(theta2)],
                        [m2*l2**2/3+m2*l1*l2*c(theta2), m2*l2**2/3]])

    # M_inv = np.linalg.inv(Maxtrx_m)
    JacobianM = np.dot(Jacobian, Jacobian.T)
    # eig, fv = np.linalg.eig(Jacobian)
    eig, fv = 1, 1
    detvalue = np.linalg.det(JacobianM)
    Um, Sigma_m, Vm  = np.linalg.svd(Jacobian)
    # Vm_eig1 = np.dot(Jacobian, Vm[:,0])
    # Vm_eig2 = np.dot(Jacobian, Vm[:,1])
    Vm_eig1 = Sigma_m[0] * Um[:,0]
    Vm_eig2 = Sigma_m[1] * Um[:,1]
    print(detvalue)

    nx, ny = [], []
    for i in range(len(x)):
        pos = np.array([[x[i]],[y[i]], [z]])
        npos = np.dot(Jacobian, pos)
        nx.append(npos[0])
        ny.append(npos[1])

    # ellipse plot
    # nx = lx3+nx
    # ny = ly3+ny
    longaxs = 2*np.sqrt(Vm_eig1[0]**2 + Vm_eig1[1]**2)
    shortaxs = 2*np.sqrt(Vm_eig2[0]**2 + Vm_eig2[1]**2)
    ang = 180* np.arctan(Um[1][0]/Um[0][0]) / np.pi
    if Sigma_m[0] ==0:
        longaxs = 0.0
    if Sigma_m[1] ==0:
        shortaxs = 0.0

    # region: print
    print("="*50)
    print("JM det")
    print(detvalue)
    print("="*50)
    print("Eigenvalue")
    print(eig)
    print("="*50)
    print("Eigenvector")
    print(fv)
    print("="*50)
    print("Sigular vector")
    print(Vm)
    print("="*50)
    print("Sigular Value")
    print(Sigma_m)
    print("="*50)
    print("Sigular unit")
    print(Um)
    print("="*50)
    print("JM V")
    print(Vm_eig1, Vm_eig2)
    print("="*50)
    print("angle")
    print(ang)
    # endregion

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 20.0,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    Ot2 = np.linspace(0, 2*np.pi, 50)
    x = c(Ot2)
    y = s(Ot2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax2 = ax

    # region: SVD arrow plot
    mc = 20
    arrow2 = mpatches.FancyArrowPatch((0, 0), (Vm_eig1[0]+0, Vm_eig1[1]+0), mutation_scale=mc, color = 'C3', alpha = 0.3)
    arrow4 = mpatches.FancyArrowPatch((0, 0), (Vm_eig2[0]+0, Vm_eig2[1]+0), mutation_scale=mc, color = 'C4', alpha = 0.3)
    ax2.add_patch(arrow2)
    ax2.add_patch(arrow4)

    # ax2.plot([0, lx1], [0, ly1], 'o-', ms=20, lw = 8, c= 'C0')
    # ax2.plot([lx1, lx2], [ly1, ly2], 'o-', ms=20, lw = 8, c= 'C1')
    # ax2.plot([lx2, lx3], [ly2, ly3], 'o-', ms=20, lw = 8, c= 'C1')
    # ax2.scatter(nx, ny, s=20, color = 'black', alpha = 0.2, label = "Ax")
    ell1 = mpatches.Ellipse(xy=(0, 0), width=longaxs, height=shortaxs, angle=ang, 
                                    facecolor='none', edgecolor='black', lw = 2, alpha = 0.5)
    ax2.add_artist(ell1)

    ax2.set_xlabel(r'$\dot X$')
    ax2.set_ylabel(r'$\dot Y$')
    ax2.set_xlim(-3, 3.0)
    ax2.set_ylim(-3.0, 3.0)
    ax2.set_title('SVD')
    # ax2.axis('equal')
    
    # ax1 = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(0.6, 0.6, 0.4, 0.4),
    #                bbox_transform=ax.transAxes, loc=2, borderpad=0)
    # ax1 = inset_axes(ax2, width="40%", height="40%",bbox_to_anchor=(-1.0, -1.2, 3.5, 3.5),
    #                bbox_transform=ax.transData, loc=2)    # 奇异值工况

    fig2, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={"projection": "3d"})
    ax1 = ax
    mc = 20
    arrow1 = Arrow3D((0, Vm[0][0]), (0, Vm[1][0]), (0, Vm[2][0]), mutation_scale=mc, color = 'C3', alpha = 0.5)
    arrow2 = Arrow3D((0, Vm[0][1]), (0, Vm[1][1]), (0, Vm[2][1]), mutation_scale=mc, color = 'C4', alpha = 0.5)
    arrow3 = Arrow3D((0, Vm[0][2]), (0, Vm[1][2]), (0, Vm[2][2]), mutation_scale=mc, color = 'C5', alpha = 0.5)

    ax1.add_artist(arrow1)
    ax1.add_artist(arrow2)
    ax1.add_artist(arrow3)


    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax1.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha = 0.3)
    ax1.set_xlabel(r'$\dot{\theta}_1$')
    ax1.set_ylabel(r'$\dot{\theta}_2$')
    ax1.set_zlabel(r'$\dot{\theta}_3$')
    ax1.set_title('Joint Space')
    # ax1.legend()
    ax1.axis('equal')
    # endregion

    plt.show()  
    pass


## two links manipulability
def TwoLinksSVD():
    l1 = 1
    l2 = 1
    m1 = 4
    m2 = 2
    theta1 = np.pi/4
    theta2 = np.pi/6

    Ot = np.linspace(0, 2*np.pi, 200)
    x = c(Ot)
    y = s(Ot)

    lx1 = l1*c(theta1)
    ly1 = l1*s(theta1)
    lx2 = l1*c(theta1)+l2*c(theta1+theta2)
    ly2 = l1*s(theta1)+l2*s(theta1+theta2)

    Jacobian = np.array([[-l1*s(theta1)-l2*s(theta1+theta2), -l2*s(theta1+theta2)],
                [l1*c(theta1)+l2*c(theta1+theta2), l2*c(theta1+theta2)]])

    # region: J
    # JacobianSVD = np.dot(Jacobian.T, Jacobian)
    detvalue = np.linalg.det(Jacobian)
    eig, fv  = np.linalg.eig(Jacobian)
    U, Sigma, V  = np.linalg.svd(Jacobian)
    V_eig1 = np.dot(Jacobian, V[:,0])
    V_eig2 = np.dot(Jacobian, V[:,1])

    nx, ny = [], []
    for i in range(len(x)):
        pos = np.array([[x[i]],[y[i]]])
        npos = np.dot(Jacobian, pos)
        nx.append(npos[0])
        ny.append(npos[1])

    # ellipse plot
    nx = lx2+nx
    ny = ly2+ny
    longaxs = 2*np.sqrt(V_eig1[0]**2 + V_eig1[1]**2)
    shortaxs = 2*np.sqrt(V_eig2[0]**2 + V_eig2[1]**2)
    ang = 180* np.arctan(U[1][0]/U[0][0]) / np.pi
    if Sigma[0] ==0:
        longaxs = 0.0
    if Sigma[1] ==0:
        shortaxs = 0.0
    # endregion

    # region: print
    print("="*50)
    print("Jacobian")
    print(Jacobian)
    print("="*50)
    print("Jacobian det")
    print(detvalue)
    print("="*50)
    print("Eigenvalue")
    print(eig)
    print("="*50)
    print("Eigenvector")
    print(fv)
    print("="*50)
    print("Sigular vector")
    print(V)
    print("="*50)
    print("Sigular Value")
    print(Sigma)
    print("="*50)
    print("Sigular unit")
    print(U)
    print("="*50)
    print("angle")
    print(ang)
    # endregion

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    Ot2 = np.linspace(0, 2*np.pi, 50)
    x = c(Ot2)
    y = s(Ot2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax2 = ax

    # region: SVD arrow plot
    mc = 20
    arrow2 = mpatches.FancyArrowPatch((lx2, ly2), (V_eig1[0]+lx2, V_eig1[1]+ly2), mutation_scale=mc, color = 'C3', alpha = 0.3)
    arrow4 = mpatches.FancyArrowPatch((lx2, ly2), (V_eig2[0]+lx2, V_eig2[1]+ly2), mutation_scale=mc, color = 'C4', alpha = 0.3)
    ax2.add_patch(arrow2)
    ax2.add_patch(arrow4)

    ax2.plot([0, lx1], [0, ly1], 'o-', ms=20, lw = 8, c= 'C0')
    ax2.plot([lx1, lx2], [ly1, ly2], 'o-', ms=20, lw = 8, c= 'C1')
    ax2.scatter(nx, ny, s=20, color = 'black', alpha = 0.2, label = "Ax")
    ell1 = mpatches.Ellipse(xy=(lx2, ly2), width=longaxs, height=shortaxs, angle=ang, 
                                    facecolor='none', edgecolor='black', lw = 2, alpha = 0.5)
    ax2.add_artist(ell1)

    ax2.set_xlabel('$\dot X$')
    ax2.set_ylabel('$\dot Y$')
    ax2.set_title('SVD')
    ax2.axis('equal')
    # ax2.set_xlim([-1.5, 5])
    # ax2.legend()
    # endregion
    # region: SVD arrow plot
    # fig2, ax = plt.subplots(1, 1, figsize=(12, 12))
    # ax1 = ax
    ax1 = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(0.65, 0.6, 0.35, 0.35),
                   bbox_transform=ax.transAxes, loc=2, borderpad=0)
    # ax1 = inset_axes(ax2, width="40%", height="40%",bbox_to_anchor=(-1.0, -1.2, 3.5, 3.5),
    #                bbox_transform=ax.transData, loc=2)    # 奇异值工况

    
    mc = 20
    arrow1 = mpatches.FancyArrowPatch((0, 0), (V[0][0], V[1][0]), mutation_scale=mc, color = 'C3', alpha = 0.5)
    arrow3 = mpatches.FancyArrowPatch((0, 0), (V[0][1], V[1][1]), mutation_scale=mc, color = 'C4', alpha = 0.5)
    ax1.add_patch(arrow1)
    ax1.add_patch(arrow3)

    ax1.scatter(x, y, s=20, color = 'black', alpha = 0.2, label = "Joint space")
    ell2 = mpatches.Ellipse(xy=(0, 0), width=2, height=2, angle=0, 
                                    facecolor='none', edgecolor='black', lw = 2, alpha = 0.5)
    ax1.add_artist(ell2)

    ax1.set_xlabel(r'$\dot{\theta}_1$')
    ax1.set_ylabel(r'$\dot{\theta}_2$')
    ax1.set_title('Joint Space')
    # ax1.legend()
    ax1.axis('equal')
    # endregion

    plt.show()  
    pass

## two links inertia manipulability
def TwoLinksInetSVD():
    l1 = 1
    l2 = 1
    m1 = 4
    m2 = 2
    theta1 = np.pi/4
    theta2 = np.pi/6

    Ot = np.linspace(0, 2*np.pi, 200)
    x = c(Ot)
    y = s(Ot)

    lx1 = l1*c(theta1)
    ly1 = l1*s(theta1)
    lx2 = l1*c(theta1)+l2*c(theta1+theta2)
    ly2 = l1*s(theta1)+l2*s(theta1+theta2)

    Jacobian = np.array([[-l1*s(theta1)-l2*s(theta1+theta2), -l2*s(theta1+theta2)],
                [l1*c(theta1)+l2*c(theta1+theta2), l2*c(theta1+theta2)]])

    Maxtrx_m = np.array([[m2*l1**2+m1*l1**2/3+m2*l2**2/3+m2*l1*l2*c(theta2), m2*l2**2/3+m2*l1*l2*c(theta2)],
                        [m2*l2**2/3+m2*l1*l2*c(theta2), m2*l2**2/3]])

    M_inv = np.linalg.inv(Maxtrx_m)
    JacobianM = np.dot(Jacobian, M_inv)
    eig, fv = np.linalg.eig(JacobianM)
    JMDet = np.abs(np.linalg.det(JacobianM))
    # JMDet2 = np.abs(np.linalg.det(Jacobian)/np.linalg.det(Maxtrx_m))
    Um, Sigma_m, Vm  = np.linalg.svd(JacobianM)
    # Vm_eig1 = np.dot(JacobianM, Vm[:,0])
    # Vm_eig2 = np.dot(JacobianM, Vm[:,1])
    Vm_eig1 = Sigma_m[0] * Um[:,0]
    Vm_eig2 = Sigma_m[1] * Um[:,1]

    nx, ny = [], []
    for i in range(len(x)):
        pos = np.array([[x[i]],[y[i]]])
        npos = np.dot(JacobianM, pos)
        nx.append(npos[0])
        ny.append(npos[1])

    # ellipse plot
    nx = lx2+nx
    ny = ly2+ny
    longaxs = 2*np.sqrt(Vm_eig1[0]**2 + Vm_eig1[1]**2)
    shortaxs = 2*np.sqrt(Vm_eig2[0]**2 + Vm_eig2[1]**2)
    ang = 180* np.arctan(Um[1][0]/Um[0][0]) / np.pi
    if Sigma_m[0] ==0:
        longaxs = 0.0
    if Sigma_m[1] ==0:
        shortaxs = 0.0

    # region: print
    print("="*50)
    print("JM det")
    print(JMDet)
    print("="*50)
    print("Eigenvalue")
    print(eig)
    print("="*50)
    print("Eigenvector")
    print(fv)
    print("="*50)
    print("Sigular vector")
    print(Vm)
    print("="*50)
    print("Sigular Value")
    print(Sigma_m)
    print("="*50)
    print("Sigular unit")
    print(Um)
    print("="*50)
    print("JM V")
    print(Vm_eig1, Vm_eig2)
    print("="*50)
    print("angle")
    print(ang)
    # endregion

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    Ot2 = np.linspace(0, 2*np.pi, 50)
    x = c(Ot2)
    y = s(Ot2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax2 = ax

    # region: SVD arrow plot
    mc = 20
    arrow2 = mpatches.FancyArrowPatch((lx2, ly2), (Vm_eig1[0]+lx2, Vm_eig1[1]+ly2), mutation_scale=mc, color = 'C3', alpha = 0.3)
    arrow4 = mpatches.FancyArrowPatch((lx2, ly2), (Vm_eig2[0]+lx2, Vm_eig2[1]+ly2), mutation_scale=mc, color = 'C4', alpha = 0.3)
    ax2.add_patch(arrow2)
    ax2.add_patch(arrow4)

    ax2.plot([0, lx1], [0, ly1], 'o-', ms=20, lw = 8, c= 'C0')
    ax2.plot([lx1, lx2], [ly1, ly2], 'o-', ms=20, lw = 8, c= 'C1')
    ax2.scatter(nx, ny, s=20, color = 'black', alpha = 0.2, label = "Ax")
    ell1 = mpatches.Ellipse(xy=(lx2, ly2), width=longaxs, height=shortaxs, angle=ang, 
                                    facecolor='none', edgecolor='black', lw = 2, alpha = 0.5)
    ax2.add_artist(ell1)

    ax2.set_xlabel(r'$\ddot{\theta}_1$')
    ax2.set_ylabel(r'$\ddot{\theta}_2$')
    ax2.set_title('SVD')
    ax2.axis('equal')
    
    ax1 = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(0.55, 0.1, 0.4, 0.4),
                   bbox_transform=ax.transAxes, loc=2, borderpad=0)
    # ax1 = inset_axes(ax2, width="40%", height="40%",bbox_to_anchor=(-1.0, -1.2, 3.5, 3.5),
    #                bbox_transform=ax.transData, loc=2)    # 奇异值工况

    
    mc = 20
    arrow1 = mpatches.FancyArrowPatch((0, 0), (Vm[0][0], Vm[1][0]), mutation_scale=mc, color = 'C3', alpha = 0.5)
    arrow3 = mpatches.FancyArrowPatch((0, 0), (Vm[0][1], Vm[1][1]), mutation_scale=mc, color = 'C4', alpha = 0.5)
    ax1.add_patch(arrow1)
    ax1.add_patch(arrow3)

    ax1.scatter(x, y, s=20, color = 'black', alpha = 0.2, label = "Joint space")
    ell2 = mpatches.Ellipse(xy=(0, 0), width=2, height=2, angle=0, 
                                    facecolor='none', edgecolor='black', lw = 2, alpha = 0.5)
    ax1.add_artist(ell2)

    ax1.set_xlabel(r'$\tau_1$')
    ax1.set_ylabel(r'$\tau_2$')
    ax1.set_title('Joint Space')
    # ax1.legend()
    ax1.axis('equal')
    # endregion

    plt.show()  
    pass


## 3D arrow of FancyArrowPatch function extension
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


## Three links manipulability
def ThreeLinksInetSVD():
    l1 = 1
    l2 = 1
    l3 = 1
    m1 = 4
    m2 = 2
    m2 = 1
    theta1 = np.pi/8
    theta2 = np.pi/8
    theta3 = np.pi/8

    Ot = np.linspace(0, 2*np.pi, 200)
    at = np.linspace(0, 2*np.pi, 200)
    x = s(np.pi/6)*c(Ot)
    y = s(np.pi/6)*s(Ot)
    z = c(np.pi/6)

    lx1 = l1*c(theta1)
    ly1 = l1*s(theta1)
    lx2 = l1*c(theta1)+l2*c(theta1+theta2)
    ly2 = l1*s(theta1)+l2*s(theta1+theta2)
    lx3 = l1*c(theta1)+l2*c(theta1+theta2)+l3*c(theta1+theta2+theta3)
    ly3 = l1*s(theta1)+l2*s(theta1+theta2)+l3*s(theta1+theta2+theta3)

    Jacobian = np.array([[-l1*s(theta1)-l2*s(theta1+theta2)-l3*s(theta1+theta2+theta3), -l2*s(theta1+theta2)-l3*s(theta1+theta2+theta3), -l3*s(theta1+theta2+theta3)],
                [l1*c(theta1)+l2*c(theta1+theta2)+l3*c(theta1+theta2+theta3), l2*c(theta1+theta2)+l3*c(theta1+theta2+theta3), l3*c(theta1+theta2+theta3)]])

    Maxtrx_m = np.array([[m2*l1**2+m1*l1**2/3+m2*l2**2/3+m2*l1*l2*c(theta2), m2*l2**2/3+m2*l1*l2*c(theta2)],
                        [m2*l2**2/3+m2*l1*l2*c(theta2), m2*l2**2/3]])

    # M_inv = np.linalg.inv(Maxtrx_m)
    JacobianM = np.dot(Jacobian, Jacobian.T)
    # eig, fv = np.linalg.eig(Jacobian)
    eig, fv = 1, 1
    detvalue = np.linalg.det(JacobianM)
    Um, Sigma_m, Vm  = np.linalg.svd(Jacobian)
    # Vm_eig1 = np.dot(Jacobian, Vm[:,0])
    # Vm_eig2 = np.dot(Jacobian, Vm[:,1])
    Vm_eig1 = Sigma_m[0] * Um[:,0]
    Vm_eig2 = Sigma_m[1] * Um[:,1]
    print(detvalue)

    nx, ny = [], []
    for i in range(len(x)):
        pos = np.array([[x[i]],[y[i]], [z]])
        npos = np.dot(Jacobian, pos)
        nx.append(npos[0])
        ny.append(npos[1])

    # ellipse plot
    nx = lx3+nx
    ny = ly3+ny
    longaxs = 2*np.sqrt(Vm_eig1[0]**2 + Vm_eig1[1]**2)
    shortaxs = 2*np.sqrt(Vm_eig2[0]**2 + Vm_eig2[1]**2)
    ang = 180* np.arctan(Um[1][0]/Um[0][0]) / np.pi
    if Sigma_m[0] ==0:
        longaxs = 0.0
    if Sigma_m[1] ==0:
        shortaxs = 0.0

    # region: print
    print("="*50)
    print("JM det")
    print(detvalue)
    print("="*50)
    print("Eigenvalue")
    print(eig)
    print("="*50)
    print("Eigenvector")
    print(fv)
    print("="*50)
    print("Sigular vector")
    print(Vm)
    print("="*50)
    print("Sigular Value")
    print(Sigma_m)
    print("="*50)
    print("Sigular unit")
    print(Um)
    print("="*50)
    print("JM V")
    print(Vm_eig1, Vm_eig2)
    print("="*50)
    print("angle")
    print(ang)
    # endregion

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 20.0,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    Ot2 = np.linspace(0, 2*np.pi, 50)
    x = c(Ot2)
    y = s(Ot2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax2 = ax

    # region: SVD arrow plot
    mc = 20
    arrow2 = mpatches.FancyArrowPatch((lx3, ly3), (Vm_eig1[0]+lx3, Vm_eig1[1]+ly3), mutation_scale=mc, color = 'C3', alpha = 0.3)
    arrow4 = mpatches.FancyArrowPatch((lx3, ly3), (Vm_eig2[0]+lx3, Vm_eig2[1]+ly3), mutation_scale=mc, color = 'C4', alpha = 0.3)
    ax2.add_patch(arrow2)
    ax2.add_patch(arrow4)

    ax2.plot([0, lx1], [0, ly1], 'o-', ms=20, lw = 8, c= 'C0')
    ax2.plot([lx1, lx2], [ly1, ly2], 'o-', ms=20, lw = 8, c= 'C1')
    ax2.plot([lx2, lx3], [ly2, ly3], 'o-', ms=20, lw = 8, c= 'C1')
    # ax2.scatter(nx, ny, s=20, color = 'black', alpha = 0.2, label = "Ax")
    ell1 = mpatches.Ellipse(xy=(lx3, ly3), width=longaxs, height=shortaxs, angle=ang, 
                                    facecolor='none', edgecolor='black', lw = 2, alpha = 0.5)
    ax2.add_artist(ell1)

    ax2.set_xlabel(r'$\dot X$')
    ax2.set_ylabel(r'$\dot Y$')
    ax2.set_xlim(-1.5, 6.0)
    ax2.set_ylim(-1.0, 5.0)
    ax2.set_title('SVD')
    # ax2.axis('equal')
    
    # ax1 = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(0.6, 0.6, 0.4, 0.4),
    #                bbox_transform=ax.transAxes, loc=2, borderpad=0)
    # ax1 = inset_axes(ax2, width="40%", height="40%",bbox_to_anchor=(-1.0, -1.2, 3.5, 3.5),
    #                bbox_transform=ax.transData, loc=2)    # 奇异值工况

    fig2, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={"projection": "3d"})
    ax1 = ax
    mc = 20
    arrow1 = Arrow3D((0, Vm[0][0]), (0, Vm[1][0]), (0, Vm[2][0]), mutation_scale=mc, color = 'C3', alpha = 0.5)
    arrow2 = Arrow3D((0, Vm[0][1]), (0, Vm[1][1]), (0, Vm[2][1]), mutation_scale=mc, color = 'C4', alpha = 0.5)
    arrow3 = Arrow3D((0, Vm[0][2]), (0, Vm[1][2]), (0, Vm[2][2]), mutation_scale=mc, color = 'C5', alpha = 0.5)

    ax1.add_artist(arrow1)
    ax1.add_artist(arrow2)
    ax1.add_artist(arrow3)


    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax1.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha = 0.3)
    ax1.set_xlabel(r'$\dot{\theta}_1$')
    ax1.set_ylabel(r'$\dot{\theta}_2$')
    ax1.set_zlabel(r'$\dot{\theta}_3$')
    ax1.set_title('Joint Space')
    # ax1.legend()
    ax1.axis('equal')
    # endregion

    plt.show()  
    pass

## three links Dexterity
def ThreeLinksDexterity():
    l1 = 1
    l2 = 1
    l3 = 1
    m1 = 4
    m2 = 2
    m2 = 1
    theta1 = np.pi/8
    theta2 = np.pi/8
    theta3 = np.pi/8

    Ot = np.linspace(0, 2*np.pi, 200)
    at = np.linspace(0, 2*np.pi, 200)
    x = s(np.pi/6)*c(Ot)
    y = s(np.pi/6)*s(Ot)
    z = c(np.pi/6)

    lx1 = l1*c(theta1)
    ly1 = l1*s(theta1)
    lx2 = l1*c(theta1)+l2*c(theta1+theta2)
    ly2 = l1*s(theta1)+l2*s(theta1+theta2)
    lx3 = l1*c(theta1)+l2*c(theta1+theta2)+l3*c(theta1+theta2+theta3)
    ly3 = l1*s(theta1)+l2*s(theta1+theta2)+l3*s(theta1+theta2+theta3)
    print(lx3, ly3)

    Jacobian = np.array([[-l1*s(theta1)-l2*s(theta1+theta2)-l3*s(theta1+theta2+theta3), -l2*s(theta1+theta2)-l3*s(theta1+theta2+theta3), -l3*s(theta1+theta2+theta3)],
                [l1*c(theta1)+l2*c(theta1+theta2)+l3*c(theta1+theta2+theta3), l2*c(theta1+theta2)+l3*c(theta1+theta2+theta3), l3*c(theta1+theta2+theta3)]])

    # eig, fv = np.linalg.eig(Jacobian)
    eig, fv = 1, 1
    Um, Sigma_m, Vm  = np.linalg.svd(Jacobian)
    # Vm_eig1 = np.dot(Jacobian, Vm[:,0])
    # Vm_eig2 = np.dot(Jacobian, Vm[:,1])
    Vm_eig1 = Sigma_m[0] * Um[:,0]
    Vm_eig2 = Sigma_m[1] * Um[:,1]

    # region: inverse postion cal
    x = lx3
    y = ly3
    q1 = sympy.Symbol('q1')
    q2 = sympy.Symbol('q2')
    fx1 = l1*sympy.cos(q1)+l2*sympy.cos(q2+q1)
    fy1 = l1*sympy.sin(q1)+l2*sympy.sin(q1+q2)
    theta_f = np.linspace(0, np.pi, 8)
    for i in range(len(theta_f)):
        xx = x - l3*c(theta_f[i])
        yy = y - l3*s(theta_f[i])
        res = sympy.solve([fx1-xx, fy1-yy], [q1, q2])
        print(res)

    # endregion 

    # ellipse plot
    nx = lx3+nx
    ny = ly3+ny
    longaxs = 2*np.sqrt(Vm_eig1[0]**2 + Vm_eig1[1]**2)
    shortaxs = 2*np.sqrt(Vm_eig2[0]**2 + Vm_eig2[1]**2)
    ang = 180* np.arctan(Um[1][0]/Um[0][0]) / np.pi
    if Sigma_m[0] ==0:
        longaxs = 0.0
    if Sigma_m[1] ==0:
        shortaxs = 0.0

    # region: print
    print("="*50)
    print("JM det")
    print(Jacobian)
    print("="*50)
    print("Eigenvalue")
    print(eig)
    print("="*50)
    print("Eigenvector")
    print(fv)
    print("="*50)
    print("Sigular vector")
    print(Vm)
    print("="*50)
    print("Sigular Value")
    print(Sigma_m)
    print("="*50)
    print("Sigular unit")
    print(Um)
    print("="*50)
    print("JM V")
    print(Vm_eig1, Vm_eig2)
    print("="*50)
    print("angle")
    print(ang)
    # endregion

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    Ot2 = np.linspace(0, 2*np.pi, 50)
    x = c(Ot2)
    y = s(Ot2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax2 = ax

    # region: SVD arrow plot
    mc = 20
    arrow2 = mpatches.FancyArrowPatch((lx3, ly3), (Vm_eig1[0]+lx3, Vm_eig1[1]+ly3), mutation_scale=mc, color = 'C3', alpha = 0.3)
    arrow4 = mpatches.FancyArrowPatch((lx3, ly3), (Vm_eig2[0]+lx3, Vm_eig2[1]+ly3), mutation_scale=mc, color = 'C4', alpha = 0.3)
    ax2.add_patch(arrow2)
    ax2.add_patch(arrow4)

    ax2.plot([0, lx1], [0, ly1], 'o-', ms=20, lw = 8, c= 'C0')
    ax2.plot([lx1, lx2], [ly1, ly2], 'o-', ms=20, lw = 8, c= 'C1')
    ax2.plot([lx2, lx3], [ly2, ly3], 'o-', ms=20, lw = 8, c= 'C1')
    # ax2.scatter(nx, ny, s=20, color = 'black', alpha = 0.2, label = "Ax")
    ell1 = mpatches.Ellipse(xy=(lx3, ly3), width=longaxs, height=shortaxs, angle=ang, 
                                    facecolor='none', edgecolor='black', lw = 2, alpha = 0.5)
    ax2.add_artist(ell1)

    ax2.set_xlabel(r'$\dot X$')
    ax2.set_ylabel(r'$\dot Y$')
    ax2.set_xlim(-1.5, 6.0)
    ax2.set_ylim(-5.0, 5.0)
    ax2.set_title('SVD')
    # ax2.axis('equal')
    
    # ax1 = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(0.6, 0.6, 0.4, 0.4),
    #                bbox_transform=ax.transAxes, loc=2, borderpad=0)
    # ax1 = inset_axes(ax2, width="40%", height="40%",bbox_to_anchor=(-1.0, -1.2, 3.5, 3.5),
    #                bbox_transform=ax.transData, loc=2)    # 奇异值工况

    fig2, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={"projection": "3d"})
    ax1 = ax
    mc = 20
    arrow1 = Arrow3D((0, Vm[0][0]), (0, Vm[1][0]), (0, Vm[2][0]), mutation_scale=mc, color = 'C1', alpha = 0.5)
    arrow2 = Arrow3D((0, Vm[0][1]), (0, Vm[1][1]), (0, Vm[2][1]), mutation_scale=mc, color = 'C2', alpha = 0.5)
    arrow3 = Arrow3D((0, Vm[0][2]), (0, Vm[1][2]), (0, Vm[2][2]), mutation_scale=mc, color = 'C3', alpha = 0.5)

    ax1.add_artist(arrow1)
    ax1.add_artist(arrow2)
    ax1.add_artist(arrow3)

    # ax1.scatter(x, y, s=20, color = 'black', alpha = 0.2, label = "Joint space")
    # ell2 = mpatches.Ellipse(xy=(0, 0), width=2, height=2, angle=0, 
    #                                 facecolor='none', edgecolor='black', lw = 2, alpha = 0.5)
    # ax1.add_artist(ell2)

    ax1.set_xlabel(r'$\dot{\theta}_1$')
    ax1.set_ylabel(r'$\dot{\theta}_2$')
    ax1.set_title('Joint Space')
    # ax1.legend()
    ax1.axis('equal')
    # endregion

    # plt.show()  
    pass

def DynamicsIndex():
    # get params config data
    FilePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ParamFilePath = FilePath + "/config/default_cfg.yaml"
    ParamFile = open(ParamFilePath, "r", encoding="utf-8")
    ParamData = yaml.load(ParamFile, Loader=yaml.FullLoader)

    # load activation file and urdf file
    # raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/activation.raisim")
    Human_urdf_file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/urdf/DynamicIndex.urdf"
    bag_urdf_file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/urdf/SandBag.urdf"
    
    # raisim world config setting
    world = raisim.World()

    # set simulation step
    ground = world.addGround(0)

    gravity = world.getGravity()
    print(gravity)
    Human = world.addArticulatedSystem(Human_urdf_file)
    Human.setName("Human")
    bag = world.addArticulatedSystem(bag_urdf_file)
    bag.setName("bag")
    print(Human.getDOF())

    world.setGravity([0, 0, 0])
    world.setTimeStep(0.001)
    # gravity1 = world.getGravity() 

    ## ====================
    # ball control initial pos and vel setting
    jointNominalConfig = np.array([1.45, -3.0, 0.0])
    jointVelocityTarget = np.array([0.0, 0.0, 0.0])
    Human.setGeneralizedCoordinate(jointNominalConfig)
    Human.setGeneralizedVelocity(jointVelocityTarget)

    
    # raisim world server setting
    server = raisim.RaisimServer(world)
    server.launchServer(8080)

    # server.startRecordingVideo("sy.mp4")
    # time.sleep(0.5)
    l1 = 0.3
    l2 = 0.3
    m1 = 1.0
    m2 = 1.0
    vx = 15.0
    cont_flag = 0
    A = 0.6
    w = 4*np.pi
    err = 0.0
    err_last = 0.0
    err_sum = 0.0
    for i in range(2000):
        time.sleep(0.02)
        JointPos, JointVel = Human.getState()
        
        # print(JointPos)
        theta1 = JointPos[0]
        theta2 = JointPos[1]
        Jacobian = np.array([[-l1*np.sin(theta1)-l2*np.sin(theta1+theta2), -l2*np.sin(theta1+theta2)],
                    [l1*np.cos(theta1)+l2*np.cos(theta1+theta2), l2*np.cos(theta1+theta2)]])
        Maxtrx_m = np.array([[m2*l1**2+m1*l1**2/3+m2*l2**2/3+m2*l1*l2*np.cos(theta2), m2*l2**2/3+m2*l1*l2*np.cos(theta2)],
                        [m2*l2**2/3+m2*l1*l2*np.cos(theta2), m2*l2**2/3]])


        # contact detect
        ContactPoint = Human.getContacts()
        contact_flag = False
        # print(Human.getFrameIdxByName("toe_fr_joint"))
        for c in ContactPoint:
            # contact_flag = c.getlocalBodyIndex() == Human.getBodyIdx("toe_fr")
            contact_flag = c.getlocalBodyIndex() == Human.getFrameIdxByName("toe_fr_joint")-5
            # print(c.getlocalBodyIndex())
            if contact_flag:
                impluse = c.getImpulse()
                impluse_w = np.dot(c.getContactFrame().T, impluse)
                if cont_flag ==0 and contact_flag:
                    print(impluse, impluse_w)
                    print(i)
                    print(v)
                    M_inv = np.linalg.inv(Maxtrx_m)
                    temp1 = np.dot(Jacobian, M_inv)
                    temp2 = np.dot(temp1, Jacobian.T)
                    Mc = np.linalg.inv(temp2)
                    Lambda = np.dot(Mc, -v)
                    print(Lambda)
                    v = np.dot(Jacobian, [JointVel[0], JointVel[1]])
                    print(v)
            if cont_flag == 0:
                cont_flag=1
            if(contact_flag):
                break
            pass

        # traj ref cal
        t = i * 0.001
        if cont_flag==0:
            t = i * 0.001
        pos_traj = np.array([A*(1-np.cos(w*t))+0.09, 0])
        # vel_f = np.array([A*w*np.sin(w*t), 0])
        theta1_ref = np.arccos(pos_traj[0]/(2*l1))
        theta_f = np.array([theta1_ref, - 2*theta1_ref])
        err = theta_f - JointPos
        derr = err - err_last
        err_sum += err * 0.001

        # pd tor cal
        Kp = np.array([400, 300])
        KI = np.array([50, 50])
        Kd = np.array([0.01, 0.01])
        # tor = Kp*err + KI * err_sum + Kd*derr/0.001
        if cont_flag==0:
            v = np.dot(Jacobian, [JointVel[0], JointVel[1]])
            tor = Kp*err + KI * err_sum + Kd*derr/0.001
            print(v)
            print("JointPos: ", JointPos)
            print("pos ref: ", theta_f)
            print("tor", tor)
            print("="*50)
        # tor = np.array([-5.0, 9.0, 0.0]*2)
        err_last = err
        Human.setGeneralizedForce(tor)
        server.integrateWorldThreadSafe()
    # server.stopRecordingVideo()

    server.killServer()
    pass

def test():
    l1 = 0.3
    l2 = 0.3
    m1 = 1.5
    m2 = 1.5
    q1 = np.pi*0.3
    q2 = np.pi*0.3
    g = 9.8
    # q1= 0
    # q2=0
    gamma = 3
    M_motor = 0.3

    # jacobian matrix
    Jq = np.array([[-l1*s(q1)-l2*s(q1+q2), -l2*s(q1+q2)],
                [l1*c(q1)+l2*c(q1+q2), l2*c(q1+q2)]])

    Mq = np.array([[m2*l1**2+m1*l1**2/3+m2*l2**2/3+m2*l1*l2*c(q2), m2*l2**2/3+m2*l1*l2*c(q2)],
                        [m2*l2**2/3+m2*l1*l2*c(q2), m2*l2**2/3]])
    M_inv = np.linalg.inv(Mq)
    tmp = Jq@M_inv@Jq.T
    Mc = np.linalg.inv(tmp)
    print(Jq)
    print(Mq)
    print(tmp)
    print(Mc)

def DIPImpactModel():
    l1 = 0.3
    l2 = 0.3
    m1 = 1.5
    m2 = 1.5
    q1 = np.pi*0.3
    q2 = np.pi*0.2
    g = 9.8
    gamma = 3
    M_motor = 500e-6

    # jacobian matrix
    Jq = np.array([[-l1*s(q1)-l2*s(q1+q2), -l2*s(q1+q2)],
                [l1*c(q1)+l2*c(q1+q2), l2*c(q1+q2)]])

    Mq = np.array([[M_motor*gamma**2 + m2*l1**2+m1*l1**2/3+m2*l2**2/3+m2*l1*l2*c(q2), m2*l2**2/3+m2*l1*l2*c(q2)],
                        [m2*l2**2/3+m2*l1*l2*c(q2), m2*l2**2/3+M_motor*gamma**2]])

    ## 动量守恒和能量守恒
    # mm = 0.005
    # 物来v回v：乒乓球、羽毛球、。、高尔夫、网球、棒球
    mm = [0.0027, 0.005, 0.045, 0.057, 0.145, 70]
    mball = [0.0027, 0.005, 0.045, 0.057, 0.145, 0.27, 0.4]
    dtball = [1.4, 4.0, 0.42, 5.2, 0.8, 12.7, 9.3]
    # 物来0回v：排球、足球、篮球、
    # mm = [0.27, 0.4, 0.67]
    # 高龄球、铅球
    # mm = [0.67, 2.0, 5.0, 7.2]
    m_v1 = 0.0
    w = 4.0
    m_v2 = 0.001
    a = 0.98
    dt = 0.01
    dt = [0.0014, 0.004, 0.00042, 0.0052, 0.0008]
    Lracket = [0.08, 0.45, 1.0, 0.5, 0.75, 0]

    # Mc calculate
    Mq_inv = np.linalg.inv(Mq)
    print("="*50)
    print("Mq")
    print(Mq)
    print("="*50)
    print("Jq")
    print(Jq)
    J_inv = np.linalg.inv(Jq)
    tmp = Jq @ Mq_inv @ Jq.T
    Mc = np.linalg.inv(tmp)
    print("="*50)
    print("Jq_inv")
    print(J_inv)
    print("="*50)
    print("Mq_inv")
    print(Mq_inv)
    print("="*50)
    print("J.M-1.J.T")
    print(Jq@Mq_inv@Jq.T)
    # print(np.linalg.eig(Mc))
    # print(np.linalg.eig(tmp))

    torque1 = []
    torque2 = []
    vel = []
    # M_v = -(m_v1+m_v2)*mm[1]/(Mc[0][0]) * 0.65 / (0.65+Lracket[1]) / (1-a)
    M_v = -(m_v1+m_v2)*mm[-1]/(Mc[0][0]) * 0.65 / (0.65+Lracket[-1])
    V_end = np.array([[M_v], [0.0]])
    print("V")
    print(V_end)
    print("Mc")
    print(Mc)
    E_kent = 0.5 * V_end.T @ Mc @ V_end

    Pm = 300

    dt_avg = E_kent / Pm
    print(dt_avg)
    # print(Mc@V_end)

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'lines.linewidth': 3,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    ax1 = axs[0]
    ax2 = axs[1]
    # plt.show()

    
    pass

def ContTimePlot():
    mball = [0.0027, 0.005, 0.045, 0.057, 0.145, 0.27, 0.4]
    dtball = np.array([1.4, 4.0, 0.42, 5.2, 0.8, 12.7, 9.3])
    dtball = 0.001*dtball

    mbox = [60]
    dtbox = [0.02]

    mkent = [0.5, 2.0]
    dtkent = [3.0, 10.0]

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'lines.linewidth': 3,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 15,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    ax1 = axs

    ax1.loglog(dtball, mball, 's', c='C0')
    ax1.loglog(dtkent, mkent, 'o', c='C1')
    ax1.loglog(dtbox, mbox, '^', c='C2')
    ax1.set_xlabel('Contact Time (s)')
    ax1.set_ylabel('Mass (kg)')
    # ax1.set_title('SVD')
    ax1.axis('equal')
    plt.show()

    pass

def SwingTimePlot():
    mball = [0.0027, 0.005, 0.045, 0.057, 0.145, 0.27, 0.4]
    dtball = np.array([150, 80, 250, 140, 200, 260, 150])

    mbox = [60]
    dtbox = [230]

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'lines.linewidth': 3,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 15,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    ax1 = axs

    ax1.semilogy(dtball, mball, 's', c='C0')
    ax1.semilogy(dtbox, mbox, '^', c='C2')
    # ax1.set_ylim(0.001, 100)
    ax1.set_xlabel('Swing Time (ms)')
    ax1.set_ylabel('Mass (kg)')
    # ax1.set_title('SVD')
    # ax1.axis('equal')
    plt.show()

    pass


# 减速比-冲量
def ImpactBioFit():
    # gamma = 4
    c = 0.02
    f = 0.04
    g = 1.0
    I_m = 5e-4
    I_l = 2.0*0.6**2/12
    tm = 10
    qm = 100*np.pi

    # y = sy.symbols('y')
    # Lambda = qm*(y**2*I_m+I_l)*(1-sy.exp(1-f*c*y**g*y*tm/(y**2*I_m+I_l)))

    # d_Lambda = sy.diff(Lambda, y)

    # sol = sy.solve(d_Lambda, y)
    # sol = sy.nsolve(d_Lambda, 10)

    # print(d_Lambda)
    # print(sol)

    gamma = np.linspace(1,20,20)
    k = c * gamma*tm/(gamma**2*I_m+I_l)
    t = f*gamma**g
    Lambda = qm*(gamma**2*I_m+I_l)*(1-np.exp(-f*c*gamma**g*gamma*tm/(gamma**2*I_m+I_l)))/gamma
    # Lambda = qm*(1-np.exp(-f*c*gamma**g*gamma*tm/(gamma**2*I_m+I_l)))/gamma
    x = f*c*gamma**g*gamma*tm/(gamma**2*I_m+I_l)
    tmp = np.exp(-x)
    # print(x)
    # print(tmp)

    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'lines.linewidth': 3,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 15,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    ax1 = axs[0]
    ax2 = axs[1]
    ax22 = ax2.twinx()

    ax1.plot(gamma, Lambda,'o-')
    ax2.plot(gamma, k,'o-', label='k')
    ax22.plot(gamma, t,'o-', label='t',c='C1')
    ax1.set_xlabel(r'Reduction ratio $\gamma$')
    ax1.set_ylabel(r'Impact $\Lambda (kg.m.s^{-1})$')
    ax2.legend()
    ax22.legend(loc="lower right")
    # ax1.set_title('SVD')
    # ax1.axis('equal')
    plt.show()

# link mass-impace
def ImpactBioFit2():
    # gamma = 4
    c = 0.02
    f = 0.04
    g = 0.8
    I_m = 5e-4
    I_l = 2.0*0.6**2/12
    tm = 10
    qm = 100*np.pi
    gamma = 5

    I_l = np.linspace(0.5,8,50)*0.6**2/12
    k = c * gamma*tm/(gamma**2*I_m+I_l)
    t = f*I_l**g
    # Lambda = qm*(y**2*I_m+I_l)*(1-np.exp(1-f*c*y**g*y*tm/(y**2*I_m+I_l)))
    Lambda = qm*(1-np.exp(-f*c*I_l**g*gamma*tm/(gamma**2*I_m+I_l)))
    x = f*c*gamma**g*gamma*tm/(gamma**2*I_m+I_l)
    tmp = np.exp(-x)
 
    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'lines.linewidth': 3,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 15,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    ax1 = axs[0]
    ax2 = axs[1]
    ax22 = ax2.twinx()

    ax1.plot(I_l, Lambda,'o-')
    ax2.plot(I_l, k,'o-', label='k')
    ax22.plot(I_l, t,'o-', label='t',c='C1')
    ax1.set_xlabel(r'Link Inertia $I_l (kg.m^2)$')
    ax1.set_ylabel(r'Impact $\Lambda (kg.m.s^{-1})$')
    ax22.legend(loc="lower right")
    ax2.legend()
    # ax1.set_title('SVD')
    # ax1.axis('equal')
    plt.show()


def ImpactBioFit3():
    U0 = 24.0
    R = 0.127
    Kv = 0.6
    Kt = 0.075
    I_m = 5e-4

    # maxon ec60 48v, 400w
    U0 = 48.0
    R = 0.844
    Kv = 0.231
    Kt = 0.231
    I_m = 1.e-3

    t = 0.2

    I_l = 4.0*0.7**2/3
    gamma = np.linspace(1,15,50)
    # gamma = np.linspace(1,200,200)
    kd = 0.1
    
    # I_l = np.linspace(0.5,8,50)*0.7**2/3
    # gamma = 5
    f = kd*gamma
    a = (gamma**2*I_m+I_l)
    b = gamma**2*Kt*Kv/R
    c = gamma*Kt*U0/R
    
    dq = c*(1-np.exp(-b*t/a))/b
    Lambda = (gamma**2*I_m+I_l) * dq
    print(gamma)
 
    plt.style.use("science")
    params = {
        'text.usetex': True,
        'font.size': 20,
        'axes.labelsize': 22,
        'lines.linewidth': 3,
        'axes.titlesize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlepad': 3.0,
        'axes.labelpad': 5.0,
        'lines.markersize': 15,
        'figure.subplot.wspace': 0.4,
        'figure.subplot.hspace': 0.5,
    }

    plt.rcParams.update(params)
    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    ax1 = axs
    print(dq*gamma)

    # ax1.plot(I_l, Lambda,'o-')
    ax1.plot(gamma, Lambda,'o-')
    ax1.set_xlabel(r'Reduction ratio $\gamma$')
    ax1.set_ylabel(r'Impact $\Lambda (kg.m.s^{-1})$')
    # ax1.set_title('SVD')
    # ax1.axis('equal')
    plt.show()


def GammaOpti():
    y = sy.symbols('gamma')
    U0 = sy.symbols('U0')
    kv = sy.symbols('kv')
    kt = sy.symbols('kt')
    R = sy.symbols('R')
    Im = sy.symbols('Im')
    Il = sy.symbols('Il')
    t = sy.symbols('t')


    f = (y*y*Im+Il)*U0*(1-sy.exp(-y*y*kv*kt*t/(R*(y*y*Im+Il))))/(y*kv)
    df = f.diff(y)
    print(df)

    res = sy.nonlinsolve(df, [y])
    print(df)
    print(res)
   

if __name__ == "__main__":
    # EigAndSVD()
    # EigAndSVD2()
    # TwoLinksSVD()
    # TwoLinksInetSVD()
    # ThreeLinksInetSVD()
    # ThreeLinksDexterity()
    # DynamicsIndex()
    # DIPImpactModel()
    # ballPlot()
    # test()
    # ImpactBioFit()
    # ImpactBioFit2()
    ImpactBioFit3()
    # SwingTimePlot()
    # GammaOpti()
    pass