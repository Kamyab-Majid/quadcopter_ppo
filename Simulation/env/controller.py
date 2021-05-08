import sympy as sp
import numpy as np
import math


class Controller:
    def __init__(self):
        self.constants = (
            self.K_mu,
            self.mass,
            self.rho,
            self.Rmr,
            self.CL0,
            self.Rtr,
            self.CLa,
            self.CLatr,
            self.CD0,
            self.xfus,
            self.yfus,
            self.zfus,
            self.zcg,
            self.Sxfus,
            self.Syfus,
            self.Szfus,
            self.fQ0,
            self.Kbeta,
            self.Alon,
            self.Blat,
            self.taufb,
            self.N,
            self.ntr,
            self.Kt,
            self.c_constant,
            self.OMEGA,
            self.ctr,
        ) = (
            1.0,
            11.0,
            1.1073,
            1.03,
            0.0077,
            0.15,
            5.496,
            5,
            0.03,
            -1.22,
            0.0,
            -0.09,
            -0.32,
            0.1019,
            0.8256,
            0.505749,
            1.5,
            254.5,
            0.999,
            0.9875,
            0.04,
            2,
            6,
            0.0,
            0.082,
            115.19,
            0.03,
        )
        self.uwind, self.vwind, self.wwind = 0.0, 0.0, 0.0
        self.vh = sp.sqrt((self.mass) * 9.8 / (2 * self.rho * 3.1415 * self.Rmr ** 2))
        self.vhtr = -sp.sqrt(self.fQ0 / (2 * self.rho * 3.1415 * self.Rtr ** 2))
        self.OMEGAtr = self.ntr * self.OMEGA
        self.sigmamr = self.N * self.c_constant / (3.1415 * self.Rmr)
        self.sigmatr = self.N * self.ctr / (3.1415 * self.Rtr)
        self.A = np.array([0, 2 / 9, 1 / 3, 3 / 4, 1, 5 / 6])
        self.B = np.array(
            [
                [0, 0, 0, 0, 0],
                [2 / 9, 0, 0, 0, 0],
                [1 / 12, 1 / 4, 0, 0, 0],
                [69 / 128, -243 / 128, 135 / 64, 0, 0],
                [-17 / 12, 27 / 4, -27 / 5, 16 / 15, 0],
                [65 / 432, -5 / 16, 13 / 16, 4 / 27, 5 / 144],
            ]
        )
        self.C = np.array([1 / 9, 0, 2 / 20, 16 / 45, 1 / 12])
        self.CH = np.array([47 / 450, 0, 12 / 25, 32 / 225, 1 / 30, 6 / 25])
        self.CT = np.array([-1 / 150, 0, 3 / 100, -16 / 75, -1 / 20, 6 / 25])
        self.CD0n, self.Kmu, self.thetach = 0.0055, 0.999, 0.0974
        self.I_moment = np.array([[0.297831, 0, 0], [0, 1.5658, 0], [0, 0, 2]], dtype=float)
        self.inverse_I_moment = np.linalg.inv(self.I_moment)
        self.Delta = np.array([[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]], dtype=float)
        self.Is = np.array([[(0.19 + 0.0004) / 2, 0, 0], [0, (0.19 + 0.0004) / 2, 0], [0, 0, 0.19]], dtype=float)
        self.I_moment_Is_inverse = np.linalg.inv(self.I_moment + self.Is)
        self.action = [1, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1]

    def Controller_model(self, current_states, time_input, action=[1, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1]):
        self.action = action
        [
            u_velocity,
            v_velocity,
            w_velocity,
            p_angle,
            q_angle,
            r_angle,
            fi_angle,
            theta_angle,
            si_angle,
            xI,
            yI,
            zI,
        ] = current_states[0:12]
        THETA = np.array([fi_angle, theta_angle, si_angle], dtype=float)
        omega = np.array(([p_angle], [q_angle], [r_angle]), dtype=float)
        Velocity = np.array(([u_velocity], [v_velocity], [w_velocity]), dtype=float)
        eta = np.array(([action[9]], [action[10]], [action[11]], [action[12]]), dtype=float)
        F = np.array(([action[1]], [action[2]], [action[3]], [action[4]]), dtype=float)
        D = np.array([[0, -action[0], 0], [action[0], 0, 0], [0, 0, 0]], dtype=float)
        # P = np.array([xI, yI, zI], dtype=float).reshape(3, 1)
        wind_velocity = self.RbI_control(THETA).T.dot(
            np.array(([self.uwind], [self.vwind], [self.wwind]), dtype=float)
        )  # Contr
        Uf = wind_velocity - Velocity
        Va_induced = float(Uf[2] / self.vh)
        mumr = math.sqrt(
            np.power((u_velocity - wind_velocity[0]), 2) + np.power((v_velocity - wind_velocity[1]), 2)
        ) / (self.OMEGA * self.Rmr)
        Vatr = (v_velocity - self.vwind + self.xfus * r_angle) / self.vhtr
        mu = (np.power((self.uwind - v_velocity), 2) + np.power((self.vwind - v_velocity), 2)) / self.vh ** 2
        mu_tr = ((self.uwind - v_velocity) ** 2 + (self.wwind - w_velocity) ** 2) / self.vh ** 2
        rOMEGA = r_angle - self.OMEGA
        qOMEGA = q_angle + self.OMEGAtr
        va_induced = 16 / (sp.pi * (((Va_induced * 1.5 + 1.9)) ** 2 + 0.9)) + 0.01
        va_inducedtr = 16 / (sp.pi * (((Vatr * 1.5 + 1.9)) ** 2 + 0.9)) + 0.01

        Vi = va_induced * self.vh / math.sqrt(1 + mu)
        Vit = va_inducedtr * self.vhtr / math.sqrt(1 + mu_tr)
        Vzi = w_velocity - Vi - self.wwind
        Vyi = v_velocity - Vit - self.vwind
        Vxq = v_velocity + q_angle * self.zcg - self.uwind
        Vyp = v_velocity - p_angle * self.zcg - self.vwind
        Ku = 2 * self.Kmu * (4 * self.thetach / 3 - Vi / (self.OMEGA * self.Rmr))
        Kv = -Ku
        Kw = (
            16
            * self.Kmu
            * mumr ** 2
            * np.sign(mumr)
            / ((1 - mumr ** 2 / 2) * (8 * np.sign(mumr) + self.CLa * self.sigmamr))
        )
        Vfus = math.sqrt(
            (u_velocity - self.uwind) ** 2 + (v_velocity - self.vwind) ** 2 + (w_velocity - self.wwind - Vi) ** 2
        )
        Xfus = -0.5 * self.rho * self.Sxfus * Vfus * (u_velocity - self.uwind)
        Yfus = -0.5 * self.rho * self.Syfus * Vfus * (v_velocity - self.vwind)
        Zfus = -0.5 * self.rho * self.Szfus * Vfus * (w_velocity - self.wwind - Vi)
        Fdrag = np.array([Xfus, Yfus, Zfus], dtype=float).reshape(3, 1)
        Qdot = self.thetabi_controller(THETA).dot(omega)
        fidot = float(Qdot[0])
        thetadot = float(Qdot[1])
        fs = (
            (1 / math.cos(theta_angle)) * np.array([float(0), math.sin(fi_angle), math.cos(fi_angle)], dtype=float)
        ).reshape(1, 3)
        fq = float(fs.dot(np.array([0, thetadot * math.tan(theta_angle), fidot], dtype=float)))
        fr = float(fs.dot(np.array([0, -fidot, thetadot * math.tan(theta_angle)], dtype=float)))
        mux = (u_velocity - self.uwind) / (self.OMEGA * self.Rmr)
        muz = (w_velocity - self.wwind) / (self.OMEGA * self.Rmr)
        muy = (v_velocity - self.vwind) / (self.OMEGA * self.Rmr)
        av = 1 / self.taufb * (Ku * mux + Kw * muz)
        bv = 1 / self.taufb * Kv * muy
        fTmr = (
            1
            / 4
            * self.rho
            * 3.1415
            * self.Rmr ** 4
            * self.OMEGA ** 2
            * self.sigmamr
            * (self.CL0 * (2 / 3 + mux ** 2 + muy ** 2) + self.CLa * (muz - Vi / (self.OMEGA * self.Rmr)))
        )
        fLmr = self.taufb * (bv - p_angle) * (self.Kbeta - fTmr * self.zcg)
        fMmr = self.taufb * (av - q_angle) * (self.Kbeta - fTmr * self.zcg)
        fNmr = (
            0.25
            * 1
            / 12
            * self.rho
            * self.Rmr ** 2
            * self.sigmamr
            * 3.1415
            * self.CLa
            * self.Rmr
            * (
                6 * self.CD0n * (Vxq ** 2 + Vyp ** 2 + self.Rmr ** 2 * rOMEGA ** 2)
                - 3 * self.Rmr ** 2 * (p_angle ** 2 + q_angle ** 2)
                - 12 * Vzi ** 2
            )
        )
        fTtr = (
            1.5
            * 1
            / 12
            * self.rho
            * self.Rtr ** 2
            * self.sigmatr
            * 3.1415
            * self.CLatr
            * self.Rtr
            * (
                (3 * q_angle + 2 * self.OMEGAtr) * (p_angle * self.zfus - r_angle * self.xfus)
                - 2 * qOMEGA * Vyi
                + (u_velocity - self.uwind) * p_angle
                + (w_velocity - self.wwind - self.Kt * Vi) * r_angle
            )
        )
        fXmr = self.taufb * fTmr * (q_angle - av)
        fYmr = self.taufb * fTmr * (bv - p_angle)
        bTtr = (
            1
            / 12
            * self.rho
            * self.Rtr ** 2
            * self.sigmatr
            * 3.1415
            * self.CLatr
            * np.array(
                [
                    0,
                    0,
                    0,
                    3 * ((u_velocity - self.uwind) + q_angle * self.zfus - r_angle * self.yfus) ** 2
                    + 3 * ((w_velocity - self.wwind - self.Kt * Vi) + p_angle * self.yfus - q_angle * self.xfus) ** 2
                    + 2 * self.Rtr ** 2 * (q_angle + self.OMEGAtr) ** 2,
                ],
                dtype=float,
            ).T
        )
        bTmr = (
            1
            / 12
            * self.rho
            * self.Rmr ** 2
            * self.sigmamr
            * 3.1415
            * self.CLa
            * np.array([0, 0, 2 * self.Rmr ** 2 * rOMEGA ** 2 + 3 * Vxq ** 2 + 3 * Vyp ** 2, 0])
        )
        bLmr = np.array(
            [0, self.Blat * (self.Kbeta - fTmr * self.zcg), (p_angle - bv) * self.taufb * self.zcg * bTmr[2], 0],
            dtype=float,
        ).T
        bMmr = np.array(
            [self.Alon * (self.Kbeta - fTmr * self.zcg), 0, (q_angle - av) * self.taufb * self.zcg * bTmr[2], 0],
            dtype=float,
        )
        bNmr = (
            1
            / 12
            * self.rho
            * self.Rmr ** 2
            * self.sigmamr
            * 3.1415
            * self.CLa
            * np.array(
                [
                    -0.25 * self.Rmr * (6 * Vxq * Vzi - 3 * self.Rmr ** 2 * p_angle * rOMEGA),
                    0.25 * self.Rmr * (6 * Vyp * Vzi - 3 * self.Rmr ** 2 * q_angle * rOMEGA),
                    -self.Rmr ** 2
                    * (p_angle * (u_velocity - self.uwind) + q_angle * (v_velocity - self.vwind) - 2 * rOMEGA * Vzi),
                    0,
                ],
                dtype=float,
            )
        )
        bXmr = np.array([-self.Alon * fTmr, 0, self.taufb * bTmr[2] * (q_angle - av), 0], dtype=float)
        bYmr = np.array([0, self.Blat * fTmr, self.taufb * bTmr[2] * (bv - p_angle), 0], dtype=float)
        # Forces and moments
        F0 = np.array(([fXmr], [fTtr + fYmr], [-fTmr]), dtype=float)
        FU = np.array([bXmr, bTtr + bYmr, -bTmr], dtype=float)
        M0 = np.array([fLmr - fTtr * self.zfus, fMmr, fNmr + fTtr * self.xfus], dtype=float).reshape(3, 1)
        MU = np.array([bLmr - bTtr * self.zfus, bMmr, bNmr + bTtr * self.xfus], dtype=float)
        b1 = self.RbI_control(THETA).dot((1 / self.mass * FU + D.dot(self.inverse_I_moment.dot(MU))))
        # y formulas
        f1 = self.RbI_control(THETA).dot(
            1 / self.mass * (F0 + Fdrag)
            + D.dot(self.inverse_I_moment).dot(M0 - (np.cross(omega.T, (self.I_moment.dot(omega)).T)).T)
        ) + np.array([0, 0, 9.8]).reshape(3, 1)

        f2 = (
            float(
                fs.dot(
                    self.I_moment_Is_inverse.dot(M0)
                    - self.inverse_I_moment.dot(np.cross(omega.T, (self.I_moment.dot(omega)).T).T)
                )
            )
            + fq * q_angle
            + fr * r_angle
        )
        f = np.append(f1, f2).reshape(4, 1)
        b2 = np.array([fs.dot(self.I_moment_Is_inverse.dot(MU))], dtype=float)
        b_Control = np.append(b1, b2).reshape(4, 4)
        # control section
        s, sdotr = self.reward_sliding_mode(current_states, time_input, action)
        K = np.linalg.inv(np.eye(4) - self.Delta).dot(F + self.Delta.dot(abs(-f + sdotr)) + eta)
        Kmod = np.zeros((4, 1))
        for i in range(4):
            Kmod[i] = K[i] * math.tanh(s[i])
        ctrl = np.linalg.inv(b_Control).dot(-f + sdotr - Kmod)
        deltacol, deltalat, deltalon, deltaped = ctrl[2], ctrl[1], ctrl[0], ctrl[3]
        ctrl = [float(deltacol), float(deltalat), float(deltalon), float(deltaped)]
        return ctrl

    def Desired_Trajectory(self, t_input, time=0):
        cp_action = self.action[0]
        pi = 3.1415
        Ts = 250
        Ti1 = 80
        Ti2 = 200
        Tf = 250
        if t_input < Ts:
            # 4.72E-03 - 1.06E-04 - 3.35E-04 - 1.13E-01
            # 2.16E-05 - 7.60E-03 - 1.76E-03 - 8.40E-02
            # 5.88E-03 - 1.48E-05
            # 5.27E-05 - 2.67E-05
            # 7.11E-04

            x_des, x_ddes, x_dddes = 0, 0, 0
            y_des, y_ddes, y_dddes = 0, 0, 0
            z_des, z_ddes, z_dddes = -cp_action, 0, 0
            psi_des, psi_ddes, psi_dddes = 0, 0, 0
        elif Ts <= t_input < Ti1:
            t = t_input - Ts
            tf = Ti1 - Ts
            R = 40
            tau = 120
            omega = 2 * pi / tau
            A1 = np.array(
                [
                    [tf ** 3, tf ** 4, tf ** 5],
                    [3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                    [6 * tf, 12 * tf ** 2, 20 * tf ** 3],
                ]
            )
            bx = np.array([40, 2 * pi / 3, 0]).reshape(3, 1)
            x1 = np.linalg.solve(A1, bx)
            a3, a4, a5 = x1[0], x1[1], x1[2]
            by = np.array([0, 0, R * omega ** 2]).reshape(3, 1)
            y1 = np.linalg.solve(A1, by)
            b3, b4, b5 = y1[0], y1[1], y1[2]
            x_des = a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3
            x_ddes = 5 * a5 * t ** 4 + 4 * a4 * t ** 3 + 3 * a3 * t ** 2
            x_dddes = 20 * a5 * t ** 3 + 12 * a4 * t ** 2 + 6 * a3 * t
            y_des = b5 * t ** 5 + b4 * t ** 4 + b3 * t ** 3
            y_ddes = 5 * b5 * t ** 4 + 4 * b4 * t ** 3 + 3 * b3 * t ** 2
            y_dddes = 20 * b5 * t ** 3 + 12 * b4 * t ** 2 + 6 * b3 * t
            z_des, z_ddes, z_dddes = -cp_action, 0, 0
            psi_des, psi_ddes, psi_dddes = 0, 0, 0
        elif Ti1 <= t_input < Ti2:
            t = t_input - Ti1
            R = 40
            tau = 120
            omega = 2 * pi / tau
            x_des = R * math.sin(omega * t) + 40
            x_ddes = R * omega * math.cos(omega * t)
            x_dddes = -R * omega ** 2 * math.sin(omega * t)
            y_des = -R * math.cos(omega * t) + 40
            y_ddes = R * omega * math.sin(omega * t)
            y_dddes = R * omega ** 2 * math.cos(omega * t)
            z_des, z_ddes, z_dddes = -cp_action, 0, 0
            psi_des, psi_ddes, psi_dddes = omega * t, omega, 0
        elif Ti2 <= t_input < Tf:
            t = t_input - Ti2
            tf = Tf - Ti2
            R = 40
            tau = 120
            omega = 2 * pi / tau
            A2 = [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3],
            ]
            b2 = np.array([40, 2 * pi / 3, 0, 80, 0, 0]).reshape(6, 1)
            x2 = np.linalg.solve(A2, b2)
            x_des = x2[5] * t ** 5 + x2[4] * t ** 4 + x2[3] * t ** 3 + x2[2] * t ** 2 + x2[1] * t + x2[0]
            x_ddes = 5 * x2[5] * t ** 4 + 4 * x2[4] * t ** 3 + 3 * x2[3] * t ** 2 + 2 * x2[2] * t + x2[1]
            x_dddes = 20 * x2[5] * t ** 3 + 12 * x2[4] * t ** 2 + 6 * x2[3] * t + 2 * x2[2]

            c2 = np.array([0, 0, R * omega ** 2, 0, 0, 0]).reshape(6, 1)
            y2 = np.linalg.solve(A2, c2)
            y_des = y2[5] * t ** 5 + y2[4] * t ** 4 + y2[3] * t ** 3 + y2[2] * t ** 2 + y2[1] * t + y2[0]
            y_ddes = 5 * y2[5] * t ** 4 + 4 * y2[4] * t ** 3 + 3 * y2[3] * t ** 2 + 2 * y2[2] * t + y2[1]
            y_dddes = 20 * y2[5] * t ** 3 + 12 * y2[4] * t ** 2 + 6 * y2[3] * t + 2 * y2[2]

            z_des, z_ddes, z_dddes = -cp_action, 0, 0
            psi_des, psi_ddes, psi_dddes = 2 * pi, 0, 0

        else:
            x_des, x_ddes, x_dddes = 80, 0, 0
            y_des, y_ddes, y_dddes = 0, 0, 0
            z_des, z_ddes, z_dddes = -cp_action, 0, 0
            psi_des, psi_ddes, psi_dddes = 2 * pi, 0, 0
        Y_des = np.array([x_des, y_des, z_des, psi_des], dtype=float).reshape(4, 1)
        Y_ddes = np.array([x_ddes, y_ddes, z_ddes, psi_ddes], dtype=float).reshape(4, 1)
        Y_dddes = np.array([x_dddes, y_dddes, z_dddes, psi_dddes], dtype=float).reshape(4, 1)
        return Y_des, Y_ddes, Y_dddes

    def RbI_control(self, THETA):
        A = np.array(
            [
                [
                    math.cos(THETA[2]) * math.cos(THETA[1]),
                    math.cos(THETA[2]) * math.sin(THETA[1]) * math.sin(THETA[0])
                    - math.sin(THETA[2]) * math.cos(THETA[0]),
                    math.cos(THETA[2]) * math.sin(THETA[1]) * math.cos(THETA[0])
                    + math.sin(THETA[2]) * math.sin(THETA[0]),
                ],
                [
                    math.sin(THETA[2]) * math.cos(THETA[1]),
                    math.sin(THETA[2]) * math.sin(THETA[1]) * math.sin(THETA[0])
                    + math.cos(THETA[2]) * math.cos(THETA[0]),
                    math.sin(THETA[2]) * math.sin(THETA[1]) * math.cos(THETA[0])
                    - math.cos(THETA[2]) * math.sin(THETA[0]),
                ],
                [-math.sin(THETA[1]), math.cos(THETA[1]) * math.sin(THETA[0]), math.cos(THETA[1]) * math.cos(THETA[0])],
            ]
        )
        return A

    def thetabi_controller(self, THETA):
        A = np.array(
            [
                [1, math.sin(THETA[0]) * math.tan(THETA[1]), math.cos(THETA[0]) * math.tan(THETA[1])],
                [0, math.cos(THETA[0]), -math.sin(THETA[0])],
                [0, math.sin(THETA[0]) / math.cos(THETA[1]), math.cos(THETA[0]) / math.cos(THETA[1])],
            ]
        )
        return A

    def reward_sliding_mode(self, current_states, time_input, action) -> float:  # produce s and sdotr
        [
            u_velocity,
            v_velocity,
            w_velocity,
            p_angle,
            q_angle,
            r_angle,
            fi_angle,
            theta_angle,
            si_angle,
            xI,
            yI,
            zI,
        ] = current_states[0:12]
        THETA = np.array([fi_angle, theta_angle, si_angle], dtype=float)
        position = np.array([xI, yI, zI], dtype=float).reshape(3, 1)  # Yposition
        d_b = np.array([0, 0, -action[0]], dtype=float).reshape(3, 1)
        lambda1 = np.array(
            [[action[5], 0, 0, 0], [0, action[6], 0, 0], [0, 0, action[7], 0], [0, 0, 0, action[8]]], dtype=float
        )
        Xcp = position + self.RbI_control(THETA).dot(d_b)
        omega = np.array([p_angle, q_angle, r_angle], dtype=float).reshape(3, 1)
        Velocity = np.array([u_velocity, v_velocity, w_velocity], dtype=float).reshape(3, 1)
        Xcpdot = self.RbI_control(THETA).dot(Velocity + (np.cross(omega.T, d_b.T)).T)
        THETAdot = self.thetabi_controller(THETA).dot(omega)
        Y = np.append(Xcp, si_angle).reshape(4, 1)
        Ydot = (np.append(Xcpdot, THETAdot[2])).reshape(4, 1)
        Yd, Ydotd, Ydotdotd = self.Desired_Trajectory(time_input, 1)
        ytilde = Y - Yd
        ydottilde = Ydot - Ydotd
        s = ydottilde + lambda1.dot(ytilde)
        sdotr = (Ydotdotd - lambda1.dot(ydottilde)).reshape(4, 1)  # sr = Ydotd - lambda1 * ytilde
        return s, sdotr

    def Yposition(self, t, x_state):
        u_velocity, v_velocity, w_velocity, p_angle, q_angle, r_angle, fi_angle, theta_angle, si_angle, xI, yI, zI = (
            x_state[0],
            x_state[1],
            x_state[2],
            x_state[3],
            x_state[4],
            x_state[5],
            x_state[6],
            x_state[7],
            x_state[8],
            x_state[9],
            x_state[10],
            x_state[11],
        )
        THETA = np.array([fi_angle, theta_angle, si_angle], dtype=float)
        omega = np.array([p_angle, q_angle, r_angle], dtype=float).reshape(3, 1)
        Velocity = np.array([u_velocity, v_velocity, w_velocity], dtype=float).reshape(3, 1)
        Yd, Ydotd, Ydotdotd = self.Desired_Trajectory(t, 0)
        P = np.array([xI, yI, zI], dtype=float).reshape(3, 1)  # Yposition
        dB = np.array([0, 0, -self.action[0]], dtype=float).reshape(3, 1)
        Xcp = P + self.RbI_control(THETA).dot(dB)
        Xcpdot = self.RbI_control(THETA).dot(Velocity + (np.cross(omega.T, dB.T)).T)
        THETAdot = self.thetabi_controller(THETA).dot(omega)
        Y = np.append(Xcp, si_angle).reshape(4, 1)
        Ydot = (np.append(Xcpdot, THETAdot[2])).reshape(4, 1)
        return Yd, Ydotd, Ydotdotd, Y, Ydot
