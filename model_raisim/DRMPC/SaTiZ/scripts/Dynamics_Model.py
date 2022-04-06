'''
This script is used to derive dynamics equation using Lagrange method
Author: hyyuan
CopyRight @ Xmech, ZJU, 2022.03.18
'''
import sympy
from sympy import sin
from sympy import cos
from sympy import symbols
from sympy import Symbol as sym
from sympy import Function as Fun
from sympy import init_printing, pprint
import os


def DynamicsModel():
    t = sym('t')

    # ================= variable and parameter defination ==================
    # define state variable
    theta1 = Fun('theta1', real=True)(t)
    theta2 = Fun('theta2', real=True)(t)
    theta3 = Fun('theta3', real=True)(t)
    
    theta1_d = theta1.diff(t)
    theta2_d = theta2.diff(t)
    theta3_d = theta3.diff(t)

    # define geometry and mass parameter
    mb = sym('mb')
    Ib = sym('Ib')
    m = [sym('m'+str(1+i)) for i in range(2)]
    I = [sym('I'+str(1+i)) for i in range(2)]
    g = sym('g')
    Lb = sym('Lb')  
    Ls = sym('Ls')
    Lt = sym('Lt')
    lt = sym('lt')  # mass center of thigh
    ls = sym('ls')  # mass center of shank
    lb = sym('lb')  # mass center of body

    # ==================== geometry constraint ===================
    # position relationship
    theta1_hat = theta1
    theta2_hat = theta1 + theta2
    theta3_hat = theta1 + theta2 + theta3

    xb = lb * sin(theta1)       # body link x mass center position
    yb = lb * cos(theta1)       # body link y mass center position
    x1 = Lb * sin(theta1) + lt * sin(theta1 + theta2)       # link 1 x mcp
    y1 = Lb * cos(theta1) + lt * cos(theta1 + theta2)       # link 1 y mcp
    x2 = Lb * sin(theta1) + Lt * sin(theta1 + theta2) + ls * sin(theta1 + theta2 + theta3)       # link 2 x mcp
    y2 = Lb * cos(theta1) + Lt * cos(theta1 + theta2) + ls * cos(theta1 + theta2 + theta3)       # link 2 y mcp

    # velocity relationship
    theta1_hat_d = theta1_d
    theta2_hat_d = theta1_d + theta2_d
    theta3_hat_d = theta1_d + theta2_d + theta3_d

    xb_d = lb * cos(theta1) * theta1_d       # body link x mass center vel
    yb_d = - lb * sin(theta1) * theta1_d      # body link y mass center vel
    x1_d = Lb * cos(theta1) * theta1_d + lt * cos(theta1 + theta2) * (theta1_d + theta2_d)      # link 1 x mcv
    y1_d = - Lb * sin(theta1) * theta1_d  - lt * sin(theta1 + theta2) * (theta1_d + theta2_d)       # link 1 y mcv
    x2_d = Lb * cos(theta1) * theta1_d + Lt * cos(theta1 + theta2) * (theta1_d + theta2_d) + \
        ls * cos(theta1 + theta2 + theta3) * (theta1_d + theta2_d + theta3_d)       # link 2 x mcv
    y2_d = - Lb * sin(theta1) * theta1_d - Lt * sin(theta1 + theta2) * (theta1_d + theta2_d) - \
        ls * sin(theta1 + theta2 + theta3) * (theta1_d + theta2_d + theta3_d)       # link 2 y mcv

    # ==================== kinematic and potential energy ===================
    # 动能计算： 刚体动能 = 刚体质心平移动能 + 绕质心转动动能 = 1 / 2 * m * vc ** 2 + 1 / 2 * Ic * dtheta ** 2
    Tb = 0.5 * mb * xb_d ** 2 + 0.5 * mb * yb_d ** 2 + 0.5 * Ib * theta1_hat_d ** 2    
    T1 = 0.5 * m[0] * x1_d ** 2 + 0.5 * m[0] * y1_d ** 2 + 0.5 * I[0] * theta2_hat_d ** 2
    T2 = 0.5 * m[1] * x2_d ** 2 + 0.5 * m[1] * y2_d ** 2 + 0.5 * I[1] * theta3_hat_d ** 2

    T = Tb + T1 + T2

    # potential energy
    V = mb * g * yb + m[0] * g * y1 + m[1] * g * y2

    # Lagrange function
    L = T - V

    os.system('cls' if os.name == 'nt' else 'clear')
    init_printing()

    eq1 = L.diff(theta1_d).diff(t) - L.diff(theta1)
    eq2 = L.diff(theta2_d).diff(t) - L.diff(theta2)
    eq3 = L.diff(theta3_d).diff(t) - L.diff(theta3)
    # print(eq1)

    eq = [eq1, eq2, eq3]
    

    def get_inertia_term(f):
        print("Inertia term")

        def try_print(expre, s):
            temp = sympy.collect(expre.expand(), s, evaluate=False)
            print("-"*50)

            try:
                pprint(sympy.trigsimp(temp[s]), use_unicode=True)
                # pprint(temp[s], use_unicode=True)
            except:
                print("")
            pass

        for i in range(3):
            print("="*50)
            print("Inertia term wrt. joint ", i+1)
            try_print(f[i], theta1.diff(t).diff(t))
            try_print(f[i], theta2.diff(t).diff(t))
            try_print(f[i], theta3.diff(t).diff(t))
        pass

    def get_gravity_term(f):
        print("Gravity term")

        def try_print(expre, s):
            temp = sympy.collect(expre.expand(), s, evaluate=False)
            print("-"*50)
            try:
                pprint(temp[s], use_unicode=True)
            except:
                print("")
            pass

        for i in range(3):
            print("="*50)
            print("Gravity term wrt. joint ", i+1)
            try_print(f[i], g)
        pass

    def get_coriolis_term(f):
        print("Coriolis term")
        s = [theta1_d, theta2_d, theta3_d]
        ss = [sym('O1\''), sym('O2\''), sym('O3\'')]

        for i in range(3):
            f = f.replace(s[i], ss[i])
            pass
        # print(f)
        sss = []
        for i in range(3):
            for j in range(i, 3):
                sss.append(ss[i]*ss[j])
                pass
            pass
        print(sss)
        # s = [Xb.diff(t), Yb.diff(t), Ob.diff(t), O1[0].diff(t),
        #      O2[0].diff(t), O3[0].diff(t)]
        # temp = sympy.collect(
        #     f.expand(), sss, evaluate=False)
        # pprint(temp)
        cor = None
        for i in range(3):
            for j in range(i, 3):
                print("-"*50)
                temp= sympy.collect(f.expand(), ss[i]*ss[j],  evaluate=False)
                print(i, j)
                
                try:
                    tttt = temp[ss[i]*ss[j]]*s[i]*s[j]
                    # cor = cor + tttt if cor else tttt
                    # print(cor)
                    # print(tttt)
                    cor = sympy.simplify(tttt)
                    pprint(cor, use_unicode=True)

                except:
                    pass
                pass
            # print("-"*50)
            pass
        print("-"*50)

        # print(cor)
        # cor = sympy.simplify(cor)
        # # cor = sympy.factor(cor)
        # pprint(cor, use_unicode=True)
        pass

    get_inertia_term(eq)
    # print("\n"*5)
    get_gravity_term(eq)
    # print("\n"*5)
    # get_coriolis_term(eq[2])

def TwoLinkDynamics():
    t = sym('t')

    # ================= variable and parameter defination ==================
    # define state variable
    theta1 = Fun('theta1', real=True)(t)
    theta2 = Fun('theta2', real=True)(t)
    
    theta1_d = theta1.diff(t)
    theta2_d = theta2.diff(t)

    # define geometry and mass parameter
    mb = sym('mb')
    Ib = sym('Ib')
    m = [sym('m'+str(1+i)) for i in range(1)]
    I = [sym('I'+str(1+i)) for i in range(1)]
    g = sym('g')
    Lb = sym('Lb')  
    Ls = sym('Ls')
    Lt = sym('Lt')
    lt = sym('lt')  # mass center of thigh
    ls = sym('ls')  # mass center of shank
    lb = sym('lb')  # mass center of body

    # ==================== geometry constraint ===================
    # position relationship
    theta1_hat = theta1
    theta2_hat = theta1 + theta2

    xb = lb * sin(theta1)       # body link x mass center position
    yb = lb * cos(theta1)       # body link y mass center position
    x1 = Lb * sin(theta1) + lt * sin(theta1 + theta2)       # link 1 x mcp
    y1 = Lb * cos(theta1) + lt * cos(theta1 + theta2)       # link 1 y mcp

    # velocity relationship
    theta1_hat_d = theta1_d
    theta2_hat_d = theta1_d + theta2_d

    xb_d = lb * cos(theta1) * theta1_d       # body link x mass center vel
    yb_d = - lb * sin(theta1) * theta1_d      # body link y mass center vel
    x1_d = Lb * cos(theta1) * theta1_d + lt * cos(theta1 + theta2) * (theta1_d + theta2_d)      # link 1 x mcv
    y1_d = - Lb * sin(theta1) * theta1_d  - lt * sin(theta1 + theta2) * (theta1_d + theta2_d)       # link 1 y mcv
    
    # ==================== kinematic and potential energy ===================
    # 动能计算： 刚体动能 = 刚体质心平移动能 + 绕质心转动动能 = 1 / 2 * m * vc ** 2 + 1 / 2 * Ic * dtheta ** 2
    Tb = 0.5 * mb * xb_d ** 2 + 0.5 * mb * yb_d ** 2 + 0.5 * Ib * theta1_hat_d ** 2    
    T1 = 0.5 * m[0] * x1_d ** 2 + 0.5 * m[0] * y1_d ** 2 + 0.5 * I[0] * theta2_hat_d ** 2

    T = Tb + T1

    # potential energy
    V = mb * g * yb + m[0] * g * y1

    # Lagrange function
    L = T - V

    os.system('cls' if os.name == 'nt' else 'clear')
    init_printing()

    eq1 = L.diff(theta1_d).diff(t) - L.diff(theta1)
    eq2 = L.diff(theta2_d).diff(t) - L.diff(theta2)
    # print(eq1)

    eq = [eq1, eq2]
    

    def get_inertia_term(f):
        print("Inertia term")

        def try_print(expre, s):
            temp = sympy.collect(expre.expand(), s, evaluate=False)
            print("-"*50)

            try:
                pprint(sympy.trigsimp(temp[s]), use_unicode=True)
                # pprint(temp[s], use_unicode=True)
            except:
                print("")
            pass

        for i in range(2):
            print("="*50)
            print("Inertia term wrt. joint ", i+1)
            try_print(f[i], theta1.diff(t).diff(t))
            try_print(f[i], theta2.diff(t).diff(t))
        pass

    def get_gravity_term(f):
        print("Gravity term")

        def try_print(expre, s):
            temp = sympy.collect(expre.expand(), s, evaluate=False)
            print("-"*50)
            try:
                pprint(temp[s], use_unicode=True)
            except:
                print("")
            pass

        for i in range(2):
            print("="*50)
            print("Gravity term wrt. joint ", i+1)
            try_print(f[i], g)
        pass

    def get_coriolis_term(f):
        print("Coriolis term")
        s = [theta1_d, theta2_d]
        ss = [sym('O1\''), sym('O2\'')]

        for i in range(3):
            f = f.replace(s[i], ss[i])
            pass
        # print(f)
        sss = []
        for i in range(2):
            for j in range(i, 2):
                sss.append(ss[i]*ss[j])
                pass
            pass
        print(sss)
        # s = [Xb.diff(t), Yb.diff(t), Ob.diff(t), O1[0].diff(t),
        #      O2[0].diff(t), O3[0].diff(t)]
        # temp = sympy.collect(
        #     f.expand(), sss, evaluate=False)
        # pprint(temp)
        cor = None
        for i in range(2):
            for j in range(i, 2):
                print("-"*50)
                temp= sympy.collect(f.expand(), ss[i]*ss[j],  evaluate=False)
                print(i, j)
                
                try:
                    tttt = temp[ss[i]*ss[j]]*s[i]*s[j]
                    # cor = cor + tttt if cor else tttt
                    # print(cor)
                    # print(tttt)
                    cor = sympy.simplify(tttt)
                    pprint(cor, use_unicode=True)

                except:
                    pass
                pass
            # print("-"*50)
            pass
        print("-"*50)

        # print(cor)
        # cor = sympy.simplify(cor)
        # # cor = sympy.factor(cor)
        # pprint(cor, use_unicode=True)
        pass

    get_inertia_term(eq)
    # print("\n"*5)
    get_gravity_term(eq)
    # print("\n"*5)
    # get_coriolis_term(eq[2])
    pass
 
def TwoLink2ndDown():
    t = sym('t')

    # ================= variable and parameter defination ==================
    # define state variable
    theta1 = Fun('theta1', real=True)(t)
    theta2 = Fun('theta2', real=True)(t)
    
    theta1_d = theta1.diff(t)
    theta2_d = theta2.diff(t)

    # define geometry and mass parameter
    # mb = sym('mb')
    # Ib = sym('Ib')
    m = [sym('m'+str(1+i)) for i in range(2)]
    I = [sym('I'+str(1+i)) for i in range(2)]
    g = sym('g')
    L1 = sym('L1')  
    Ls = sym('Ls')
    L2 = sym('L2')
    l2 = sym('l2')  # mass center of thigh
    ls = sym('ls')  # mass center of shank
    l1 = sym('l1')  # mass center of body

    # ==================== geometry constraint ===================
    # position relationship
    theta1_hat = theta1
    theta2_hat = theta1 + theta2

    x1 = l1 * sin(theta1)       # body link x mass center position
    y1 = l1 * cos(theta1)       # body link y mass center position
    x2 = L1 * sin(theta1) - l2 * sin(theta1 + theta2)       # link 1 x mcp
    y2 = L1 * cos(theta1) - l2 * cos(theta1 + theta2)       # link 1 y mcp

    # velocity relationship
    theta1_hat_d = theta1_d
    theta2_hat_d = theta1_d + theta2_d

    x1_d = l1 * cos(theta1) * theta1_d       # body link x mass center vel
    y1_d = - l1 * sin(theta1) * theta1_d      # body link y mass center vel
    x2_d = L1 * cos(theta1) * theta1_d - l2 * cos(theta1 + theta2) * (theta1_d + theta2_d)      # link 1 x mcv
    y2_d = - L1 * sin(theta1) * theta1_d  + l2 * sin(theta1 + theta2) * (theta1_d + theta2_d)       # link 1 y mcv
    
    # ==================== kinematic and potential energy ===================
    # 动能计算： 刚体动能 = 刚体质心平移动能 + 绕质心转动动能 = 1 / 2 * m * vc ** 2 + 1 / 2 * Ic * dtheta ** 2
    Tb = 0.5 * m[0] * x1_d ** 2 + 0.5 * m[0] * y1_d ** 2 + 0.5 * I[0] * theta1_hat_d ** 2    
    T1 = 0.5 * m[1] * x2_d ** 2 + 0.5 * m[1] * y2_d ** 2 + 0.5 * I[1] * theta2_hat_d ** 2

    T = Tb + T1

    # potential energy
    V = m[0] * g * y1 + m[1] * g * y2

    # Lagrange function
    L = T - V

    os.system('cls' if os.name == 'nt' else 'clear')
    init_printing()

    eq1 = L.diff(theta1_d).diff(t) - L.diff(theta1)
    eq2 = L.diff(theta2_d).diff(t) - L.diff(theta2)
    # print(eq1)

    eq = [eq1, eq2]
    

    def get_inertia_term(f):
        print("Inertia term")

        def try_print(expre, s):
            temp = sympy.collect(expre.expand(), s, evaluate=False)
            print("-"*50)

            try:
                pprint(sympy.trigsimp(temp[s]), use_unicode=True)
                # pprint(temp[s], use_unicode=True)
            except:
                print("")
            pass

        for i in range(2):
            print("="*50)
            print("Inertia term wrt. joint ", i+1)
            try_print(f[i], theta1.diff(t).diff(t))
            try_print(f[i], theta2.diff(t).diff(t))
        pass

    def get_gravity_term(f):
        print("Gravity term")

        def try_print(expre, s):
            temp = sympy.collect(expre.expand(), s, evaluate=False)
            print("-"*50)
            try:
                pprint(temp[s], use_unicode=True)
            except:
                print("")
            pass

        for i in range(2):
            print("="*50)
            print("Gravity term wrt. joint ", i+1)
            try_print(f[i], g)
        pass

    def get_coriolis_term(f):
        print("Coriolis term")
        s = [theta1_d, theta2_d]
        ss = [sym('O1\''), sym('O2\'')]

        for i in range(3):
            f = f.replace(s[i], ss[i])
            pass
        # print(f)
        sss = []
        for i in range(2):
            for j in range(i, 2):
                sss.append(ss[i]*ss[j])
                pass
            pass
        print(sss)
        # s = [Xb.diff(t), Yb.diff(t), Ob.diff(t), O1[0].diff(t),
        #      O2[0].diff(t), O3[0].diff(t)]
        # temp = sympy.collect(
        #     f.expand(), sss, evaluate=False)
        # pprint(temp)
        cor = None
        for i in range(2):
            for j in range(i, 2):
                print("-"*50)
                temp= sympy.collect(f.expand(), ss[i]*ss[j],  evaluate=False)
                print(i, j)
                
                try:
                    tttt = temp[ss[i]*ss[j]]*s[i]*s[j]
                    # cor = cor + tttt if cor else tttt
                    # print(cor)
                    # print(tttt)
                    cor = sympy.simplify(tttt)
                    pprint(cor, use_unicode=True)

                except:
                    pass
                pass
            # print("-"*50)
            pass
        print("-"*50)

        # print(cor)
        # cor = sympy.simplify(cor)
        # # cor = sympy.factor(cor)
        # pprint(cor, use_unicode=True)
        pass

    get_inertia_term(eq)
    # print("\n"*5)
    get_gravity_term(eq)
    # print("\n"*5)
    # get_coriolis_term(eq[2])
    pass
    
def test():
    t = sym('t') 
    theta1 = Fun('theta1', real=True)(t)
    theta2 = Fun('theta2', real=True)(t)
    theta1_d =  theta1.diff(t)
    theta2_d =  theta2.diff(t)
    L = 0.5 * theta1_d ** 2 + 0.5 * theta2_d ** 2 + 0.5 * theta1 + theta1_d * theta2_d + theta1_d** 2 * sin(theta1 + theta2) ** 2 + theta2_d** 2 * cos(theta2) ** 2
    T = L.diff(theta1_d).diff(t)
    temp = sympy.collect(T.expand(), theta2_d, evaluate=False)
    print(temp)
    print(temp[theta2_d])
    # eq = T.diff(theta1_d).diff(t)
    s= [theta1_d, theta2_d]
    ss = [sym('O1\''), sym('O2\'')]
    sss = []
    for i in range(2):
        T = T.replace(s[i], ss[i])
        pass
    print(T)
    for i in range(2):
        for j in range(i, 2):
            sss.append(ss[i]*ss[j])
            pass
        pass
    # temp={}
    # for i in range(2):
    #     a = sympy.collect(
    #             T.expand(), sss[i],  evaluate=False)
    # print(temp)
    cor = None
    for i in range(2):
        for j in range(i, 2):
            temp= sympy.collect(
                T.expand(), ss[i]*ss[j],  evaluate=False)
            print(i, j)
            print(temp)
            try:
                tttt = temp[ss[i]*ss[j]]*s[i]*s[j]
                cor = cor + tttt if cor else tttt
                print( cor)
            except:
                pass
            pass
        pass
    print(sss)
    
    cor = sympy.simplify(cor)
    # cor = sympy.factor(cor)
    pprint(cor, use_unicode=True)
    # pprint(sympy.trigsimp(temp[theta1.diff(t).diff(t)]), use_unicode=True)

    a = []
    a.append([[1, 2] for _ in range(3)])
    a.append([[2, 2] for _ in range(3)])

    b = []
    b.extend([a[0][1][k] for k in range(2)])

    c = [[1, 2] for _ in range(3)]

    print(a)
    print(b)
    print(c)

def test2():
    g = sym('g')
    M = sym('M')
    m = sym('m')
    l = sym('l')
    I = sym('I')
    M = sympy.Matrix([[M+m, m*l], [m*l, I+m*l**2]])
    G = sympy.Matrix([[0, 0], [0, m*g*l]])
    temp = sympy.simplify(M.inv()*G)
    # print(M.inv())
    # print(temp)

    m0 = 1
    m1 = 0.5
    m2 = 0.5
    L1 = 0.4
    L2 = 0.4
    g=9.81
    theta1 = sym('theta1')
    theta2 = sym('theta2')
    M1 = sympy.Matrix([[m0+m1+m2, (m1/2+m2)*L1*cos(theta1), m2*L2/2*cos(theta2)],
                      [(m1/2+m2)*L1*cos(theta1), (m1/3+m2)*L1**2, m2*L1*L2/2*cos(theta1-theta2)],
                      [m2*L2/2*cos(theta2), m2*L1*L2/2*cos(theta1-theta2), m2*L2**2/3]])
    G1 = sympy.Matrix([[0],
                      [-(m1/2+m2)*L1*g*sin(theta1)],
                      [-m2*L2*g/2*sin(theta2)]])
    temp2 = sympy.simplify(M1.inv()*G1)
    print(M1.inv())
    print(temp2)

if __name__ == "__main__":
    # test()
    test2()
    # DynamicsModel()
    # TwoLinkDynamics()
    # TwoLink2ndDown()