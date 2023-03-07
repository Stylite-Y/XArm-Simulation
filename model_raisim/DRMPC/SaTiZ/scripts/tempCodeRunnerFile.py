(walker.N-1):
            AccF = walker.SupportForce(walker.q[i], walker.dq[i], walker.ddq[i])
            Fx = AccF[0]
            Fy = AccF[1]
            ceq.extend([Fy >= 0])
            ceq.extend([Fy <= 4000])
            ceq.extend([Fx <= 4000])
            ceq.extend([Fx >= -4000])
            ceq.extend([Fy*walker.mu - Fx >= 0])  # 摩擦域条件
            ceq.extend([Fy