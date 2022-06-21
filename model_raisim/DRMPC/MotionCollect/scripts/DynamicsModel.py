import sympy
from sympy import sin
from sympy import cos
from sympy import symbols
from sympy import Symbol as sym
from sympy import Function as Fun
from sympy import init_printing, pprint
import os

def HumanModel():
    t = sym('t')

    # ================= variable and parameter defination ==================
    # define state variable
    x0 = Fun('x0', real=True)(t)
    z0 = Fun('z0', real=True)(t)
    theta1 = Fun('theta1', real=True)(t)
    theta2 = Fun('theta2', real=True)(t)
    theta3 = Fun('theta3', real=True)(t)
    theta4 = Fun('theta4', real=True)(t)
    theta5 = Fun('theta5', real=True)(t)
    
    dx0 = x0.diff(t)
    dz0 = z0.diff(t)
    dtheta1 = theta1.diff(t)
    dtheta2 = theta2.diff(t)
    dtheta3 = theta3.diff(t)
    dtheta4 = theta4.diff(t)
    dtheta5 = theta5.diff(t)
    
    # define geometry and mass parameter
    m = [sym('m'+str(1+i)) for i in range(5)]
    I = [sym('I'+str(1+i)) for i in range(5)]
    L = [sym('L'+str(1+i)) for i in range(5)]
    l = [sym('l'+str(1+i)) for i in range(5)]
    g = sym('g')

    # ==================== geometry constraint ===================
    # position relationship
    htheta1 = theta1
    htheta2 = theta1 + theta2
    htheta3 = theta1 + theta3
    htheta4 = theta1 + theta4
    htheta5 = theta1 + theta5

    x1 = x0       # body link x mass center position
    z1 = z0       # body link y mass center position
    x2 = x0 - l[0]* sin(theta1) + l[1] * sin(theta1 + theta2)       # link 2 x mcp
    z2 = z0 - l[0]* cos(theta1) + l[1] * cos(theta1 + theta2)       # link 2 y mcp
    x3 = x0 - l[0]* sin(theta1) + l[2] * sin(theta1 + theta3)       # link 2 x mcp
    z3 = z0 - l[0]* cos(theta1) + l[2] * cos(theta1 + theta3)       # link 2 y mcp
    x4 = x0 + l[0]* sin(theta1) + l[3] * sin(theta1 + theta4)       # link 2 x mcp
    z4 = z0 + l[0]* cos(theta1) + l[3] * cos(theta1 + theta4)       # link 2 y mcp
    x5 = x0 + l[0]* sin(theta1) + l[4] * sin(theta1 + theta5)       # link 2 x mcp
    z5 = z0 + l[0]* cos(theta1) + l[4] * cos(theta1 + theta5)       # link 2 y mcp

    # velocity relationship
    dhtheta1 = dtheta1
    dhtheta2 = dtheta1 + dtheta2
    dhtheta3 = dtheta1 + dtheta3
    dhtheta4 = dtheta1 + dtheta4
    dhtheta5 = dtheta1 + dtheta5

    dx1 = dx0      # body link x mass center position
    dz1 = dz0       # body link y mass center position
    dx2 = dx0 - l[0]* cos(theta1) * dtheta1 + l[1] * cos(theta1 + theta2) * (dtheta1 + dtheta2)      # link 2 x mcp
    dz2 = dz0 + l[0]* sin(theta1) * dtheta1 - l[1] * sin(theta1 + theta2) * (dtheta1 + dtheta2)       # link 2 y mcp
    dx3 = dx0 - l[0]* cos(theta1) * dtheta1 + l[2] * cos(theta1 + theta3) * (dtheta1 + dtheta3)      # link 2 x mcp
    dz3 = dz0 + l[0]* sin(theta1) * dtheta1 - l[2] * sin(theta1 + theta3) * (dtheta1 + dtheta3)       # link 2 y mcp
    dx4 = dx0 + l[0]* cos(theta1) * dtheta1 + l[3] * cos(theta1 + theta4) * (dtheta1 + dtheta4)       # link 2 x mcp
    dz4 = dz0 - l[0]* sin(theta1) * dtheta1 - l[3] * sin(theta1 + theta4) * (dtheta1 + dtheta4)       # link 2 y mcp
    dx5 = dx0 + l[0]* cos(theta1) * dtheta1 + l[4] * cos(theta1 + theta5) * (dtheta1 + dtheta5)      # link 2 x mcp
    dz5 = dz0 - l[0]* sin(theta1) * dtheta1 - l[4] * sin(theta1 + theta5) * (dtheta1 + dtheta5)       # link 2 y mcp
    
    # ==================== kinematic and potential energy ===================
    # 动能计算： 刚体动能 = 刚体质心平移动能 + 绕质心转动动能 = 1 / 2 * m * vc ** 2 + 1 / 2 * Ic * dtheta ** 2
    T1 = 0.5 * m[0] * (dx1**2 + dz1**2) + 0.5 * I[0] * (dhtheta1**2)
    T2 = 0.5 * m[1] * (dx2**2 + dz2**2) + 0.5 * I[1] * (dhtheta2**2)
    T3 = 0.5 * m[2] * (dx3**2 + dz3**2) + 0.5 * I[2] * (dhtheta3**2)
    T4 = 0.5 * m[3] * (dx4**2 + dz4**2) + 0.5 * I[3] * (dhtheta4**2)
    T5 = 0.5 * m[4] * (dx5**2 + dz5**2) + 0.5 * I[4] * (dhtheta5**2)

    T = T1 + T2 + T3 + T4 + T5

    # potential energy
    V = m[0] * g * z1 + m[1] * g * z2 + m[2] * g * z3 + m[3] * g * z4 + m[4] * g * z5

    # Lagrange function
    L = T - V

    os.system('cls' if os.name == 'nt' else 'clear')
    init_printing()

    eq1 = L.diff(dx0).diff(t) - L.diff(x0)
    eq2 = L.diff(dz0).diff(t) - L.diff(z0)
    eq3 = L.diff(dtheta1).diff(t) - L.diff(theta1)
    eq4 = L.diff(dtheta2).diff(t) - L.diff(theta2)
    eq5 = L.diff(dtheta3).diff(t) - L.diff(theta3)
    eq6 = L.diff(dtheta4).diff(t) - L.diff(theta4)
    eq7 = L.diff(dtheta5).diff(t) - L.diff(theta5)
    print(eq7)

    eq = [eq1, eq2, eq3, eq4, eq5, eq6, eq7]
    

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

        for i in range(len(f)):
            print("="*50)
            if i < 2:
                print("Ineria term wrt. disp", i)
            else:
                print("Inertia term wrt. joint ", i-1)
            try_print(f[i], x0.diff(t).diff(t))
            try_print(f[i], z0.diff(t).diff(t))
            try_print(f[i], theta1.diff(t).diff(t))
            try_print(f[i], theta2.diff(t).diff(t))
            try_print(f[i], theta3.diff(t).diff(t))
            try_print(f[i], theta4.diff(t).diff(t))
            try_print(f[i], theta5.diff(t).diff(t))
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

        for i in range(len(f)):
            print("="*50)
            if i < 2:
                print("Ineria term wrt. disp", i)
            else:
                print("Inertia term wrt. joint ", i-1)
            try_print(f[i], g)
        pass

    def get_coriolis_term(f):
        print("Coriolis term")
        s = [dx0, dz0, dtheta1, dtheta2, dtheta3, dtheta4, dtheta5]
        ss = [sym('x0\''), sym('z0\''), sym('O1\''), sym('O2\''), sym('O3\'') , sym('O4\''), sym('O5\'')]

        for i in range(len[s]):
            f = f.replace(s[i], ss[i])
            pass
        # print(f)
        sss = []
        for i in range(len(s)):
            for j in range(i, len(s)):
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
        for i in range(len(s)):
            for j in range(i, len(s)):
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

    # get_inertia_term(eq)
    # print("\n"*5)
    get_gravity_term(eq)
    # print("\n"*5)
    # get_coriolis_term(eq[2])
    pass

if __name__ == "__main__":
    # DynamicsModel()
    # TwoLinkDynamics()
    HumanModel()
