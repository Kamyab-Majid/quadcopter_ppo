import sympy as sp
import numpy as np


class Helicopter:
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

    def RK45(self, x0, y0, ydot, h, u_input, trunc_error=False) -> np.array:
        k1 = h * np.array(ydot(y0, x0 + self.A[0] * h, *u_input), dtype=float)
        k2 = h * np.array(ydot(y0 + self.B[1, 0] * k1, x0 + self.A[1] * h, *u_input), dtype=float)
        k3 = h * np.array(ydot(y0 + self.B[2, 0] * k1 + self.B[2, 1] * k2, x0 + self.A[2] * h, *u_input), dtype=float)
        k4 = h * np.array(
            ydot(y0 + self.B[3, 0] * k1 + self.B[3, 1] * k2 + self.B[3, 2] * k3, x0 + self.A[3] * h, *u_input),
            dtype=float,
        )
        k5 = h * np.array(
            ydot(
                y0 + self.B[4, 0] * k1 + self.B[4, 1] * k2 + self.B[4, 2] * k3 + self.B[4, 3] * k4,
                x0 + self.A[4] * h,
                *u_input
            ),
            dtype=float,
        )
        k6 = h * np.array(
            ydot(
                y0 + self.B[5, 0] * k1 + self.B[5, 1] * k2 + self.B[5, 2] * k3 + self.B[5, 3] * k4 + self.B[5, 4] * k5,
                x0 + self.A[5] * h,
                *u_input
            ),
            dtype=float,
        )
        y_new = (
            y0
            + k1 * self.CH[0]
            + k2 * self.CH[1]
            + k3 * self.CH[2]
            + k4 * self.CH[3]
            + k5 * self.CH[4]
            + k6 * self.CH[5]
        )
        if trunc_error:
            trunc_error = (
                k1 * self.CT[0]
                + k2 * self.CT[1]
                + k3 * self.CT[2]
                + k4 * self.CT[3]
                + k5 * self.CT[4]
                + k6 * self.CT[5]
            )
        return y_new

    def RbI(self, THETA):
        A = sp.Matrix(
            [
                [
                    sp.cos(THETA[2]) * sp.cos(THETA[1]),
                    sp.cos(THETA[2]) * sp.sin(THETA[1]) * sp.sin(THETA[0]) - sp.sin(THETA[2]) * sp.cos(THETA[0]),
                    sp.cos(THETA[2]) * sp.sin(THETA[1]) * sp.cos(THETA[0]) + sp.sin(THETA[2]) * sp.sin(THETA[0]),
                ],
                [
                    sp.sin(THETA[2]) * sp.cos(THETA[1]),
                    sp.sin(THETA[2]) * sp.sin(THETA[1]) * sp.sin(THETA[0]) + sp.cos(THETA[2]) * sp.cos(THETA[0]),
                    sp.sin(THETA[2]) * sp.sin(THETA[1]) * sp.cos(THETA[0]) - sp.cos(THETA[2]) * sp.sin(THETA[0]),
                ],
                [-sp.sin(THETA[1]), sp.cos(THETA[1]) * sp.sin(THETA[0]), sp.cos(THETA[1]) * sp.cos(THETA[0])],
            ]
        )
        return A

    def thetabi(self, THETA):
        A = sp.Matrix(
            [
                [1, sp.sin(THETA[0]) * sp.tan(THETA[1]), sp.cos(THETA[0]) * sp.tan(THETA[1])],
                [0, sp.cos(THETA[0]), -sp.sin(THETA[0])],
                [0, sp.sin(THETA[0]) / sp.cos(THETA[1]), sp.cos(THETA[0]) / sp.cos(THETA[1])],
            ]
        )
        return A

    def lambd_eq_maker(self, t, x_state, U_input):  # for_ode_int
        My_helicopter = Helicopter()
        symp_eq = My_helicopter.Helicopter_model(t, x_state, U_input)
        jacobian = ((sp.Matrix(symp_eq)).jacobian(x_state)).replace(
            sp.DiracDelta(sp.sqrt(x_state[0] ** 2 + x_state[1] ** 2)), 0
        )

        J_symb_math = sp.lambdify((x_state, t) + U_input, jacobian, modules=["numpy"])
        symb_math = sp.lambdify((x_state, t) + U_input, symp_eq, modules=["numpy"])
        return symb_math, J_symb_math

    def Helicopter_model(self, t, x_state, U_input):
        (
            u_velocity,
            v_velocity,
            w_velocity,
            p_angle,
            q_angle,
            r_angle,
            fi_angle,
            theta_angle,
            si_angle,
            _,
            _,
            _,
            a_flapping,
            b_flapping,
            c_flapping,
            d_flapping,
        ) = (
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
            x_state[12],
            x_state[13],
            x_state[14],
            x_state[15],
        )
        A_b, B_a, taus, Dlat, Kc, Kd, Clon = 0.1, 0.1, 0.20008, 0, 0.3058, 0.3058, 0
        I_moment = sp.Matrix([[0.297831, 0, 0], [0, 1.5658, 0], [0, 0, 2]])
        inverse_I_moment = I_moment ** (-1)
        THETA = sp.Matrix([fi_angle, theta_angle, si_angle])
        omega = sp.Matrix([p_angle, q_angle, r_angle]).reshape(3, 1)
        wind_velocity = (self.RbI(THETA) * (sp.Matrix([self.uwind, self.vwind, self.wwind]))).reshape(3, 1)
        Velocity = sp.Matrix([u_velocity, v_velocity, w_velocity]).reshape(3, 1)
        Uf = wind_velocity - Velocity
        Uftr = Velocity - wind_velocity
        Va_induced = Uf[2] / self.vh
        Va_induced_t = Uftr[1] / self.vhtr
        mu = ((Uf.norm()) / self.vh) ** 2 - Va_induced ** 2
        mu_tr = ((Uftr.norm()) / self.vhtr) ** 2 - Va_induced_t ** 2
        romega = r_angle - self.OMEGA
        qomega = q_angle + self.OMEGAtr
        mumr = sp.sqrt((u_velocity - self.uwind) ** 2 + (v_velocity - self.vwind) ** 2) / (self.OMEGA * self.Rmr)
        main_induced_v = 16 / (sp.pi * (((Va_induced * 1.5 + 1.9)) ** 2 + 0.9)) + 0.01
        tail_induced_v = 16 / (sp.pi * (((Va_induced_t * 1.5 + 1.9)) ** 2 + 0.9)) + 0.01
        Vi = main_induced_v * self.vh / sp.sqrt(1 + mu)
        Vi_t = tail_induced_v * self.vhtr / sp.sqrt(1 + mu_tr)
        Vyi, Vzi = v_velocity - Vi_t - self.vwind, w_velocity - Vi - self.wwind
        Vxq = u_velocity + q_angle * self.zcg - self.uwind
        Vyp = v_velocity - p_angle * self.zcg - self.vwind
        Ku = 2 * self.K_mu * (4 * U_input[0] / 3 - Vi / (self.OMEGA * self.Rmr))
        Kv = -Ku
        Kw = (
            16
            * self.K_mu
            * mumr ** 2
            * sp.sign(mumr)
            / ((1 - mumr ** 2 / 2) * (8 * sp.sign(mumr) + self.CLa * self.sigmamr))
        )
        Vfus = sp.sqrt(
            (u_velocity - self.uwind) ** 2 + (v_velocity - self.vwind) ** 2 + (w_velocity - self.wwind - Vi) ** 2
        )
        Xfus, Yfus, Zfus = (
            -0.5 * self.rho * self.Sxfus * Vfus * (u_velocity - self.uwind),
            -0.5 * self.rho * self.Syfus * Vfus * (v_velocity - self.vwind),
            -0.5 * self.rho * self.Szfus * Vfus * (w_velocity - self.wwind - Vi),
        )
        Fdrag = sp.Matrix([Xfus, Yfus, Zfus]).T
        mux, muy, muz = (
            -(Uf[0]) / (self.OMEGA * self.Rmr),
            -(Uf[1]) / (self.OMEGA * self.Rmr),
            -(Uf[2]) / (self.OMEGA * self.Rmr),
        )
        lambda0 = Vi / (self.OMEGA * self.Rmr)
        fTmr = (
            1
            / 4
            * self.rho
            * 3.1415
            * self.Rmr ** 4
            * self.OMEGA ** 2
            * self.sigmamr
            * (self.CL0 * (2 / 3 + mux ** 2 + muy ** 2) + self.CLa * (muz - lambda0))
        )
        bTmr = (
            1
            / 4
            * self.rho
            * 3.1415
            * self.Rmr ** 4
            * self.OMEGA ** 2
            * self.sigmamr
            * self.CLa
            * sp.Matrix([mux ** 2 + muy ** 2 + 2 / 3, -muy, mux, 0])
        )
        fQmr = (
            1
            / 8
            * self.rho
            * 3.1415
            * self.Rmr ** 5
            * self.OMEGA ** 2
            * self.sigmamr
            * self.CLa
            * (self.CD0 / self.CLa * (1 + mux ** 2 + muy ** 2) - 2 * (muz - lambda0) ** 2)
        )
        bQmr = (
            1
            / 12
            * self.rho
            * self.Rmr ** 2
            * self.sigmamr
            * 3.1415
            * self.CLa
            * sp.Matrix(
                [
                    -self.Rmr ** 2
                    * (p_angle * (u_velocity - self.uwind) + q_angle * (v_velocity - self.vwind) - 2 * romega * Vzi),
                    0.25 * self.Rmr * (6 * Vyp * Vzi - 3 * self.Rmr ** 2 * q_angle * romega),
                    -0.25 * self.Rmr * (6 * Vxq * Vzi - 3 * self.Rmr ** 2 * p_angle * romega),
                    0,
                ]
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
                - 2 * qomega * Vyi
                + (u_velocity - self.uwind) * p_angle
                + (w_velocity - self.wwind - self.Kt * Vi) * r_angle
            )
        )
        bTtr = (
            1
            / 12
            * self.rho
            * self.Rtr ** 2
            * self.sigmatr
            * 3.1415
            * self.CLatr
            * sp.Matrix(
                [
                    0,
                    0,
                    0,
                    3 * ((u_velocity - self.uwind) + q_angle * self.zfus - r_angle * self.yfus) ** 2
                    + 3 * ((w_velocity - self.wwind - self.Kt * Vi) + p_angle * self.yfus - q_angle * self.xfus) ** 2
                    + 2 * self.Rtr ** 2 * (q_angle + self.OMEGAtr) ** 2,
                ]
            )
        )
        Tmr = fTmr + bTmr.dot(U_input)
        Ttr = fTtr + bTtr.dot(U_input)
        forces = sp.Matrix([-a_flapping * Tmr, b_flapping * Tmr + Ttr, -Tmr]).T
        F = (forces + Fdrag).reshape(3, 1)
        F_gravity = (sp.Matrix([0, 0, self.mass * 9.8])).reshape(3, 1)
        F_total = F + (self.RbI(THETA)) ** (-1) * F_gravity
        Q = fQmr + bQmr.dot(U_input)
        Mroll = (self.Kbeta - Tmr * self.zcg) * b_flapping
        Mpitch = (self.Kbeta - Tmr * self.zcg) * a_flapping
        Myaw = Q + Ttr * self.xfus
        M = sp.Matrix([Mroll, Mpitch, Myaw]).reshape(3, 1)
        x_dot1_3 = F_total / self.mass - omega.cross(Velocity)
        x_dot4_6 = (inverse_I_moment * M) - inverse_I_moment * omega.cross(I_moment * omega)
        x_dot7_9 = self.thetabi(THETA) * omega
        x_dot10_12 = self.RbI(THETA) * Velocity
        x_dot13 = (
            -q_angle
            - a_flapping / self.taufb
            + 1
            / (self.taufb * self.OMEGA * self.Rmr)
            * (Ku * (u_velocity - self.uwind) + Kw * (w_velocity - self.wwind))
            + self.Alon / self.taufb * (U_input[2] + Kc * c_flapping)
            - A_b * b_flapping / self.taufb
        )
        x_dot14 = (
            -p_angle
            - b_flapping / self.taufb
            + 1 / (self.taufb * self.OMEGA * self.Rmr) * Kv * (v_velocity - self.vwind)
            + self.Blat / self.taufb * (U_input[1] + Kd * d_flapping)
            + B_a * a_flapping / self.taufb
        )
        x_dot15 = -q_angle - c_flapping / taus + Clon / taus * U_input[2]
        x_dot16 = -p_angle - d_flapping / taus + Dlat / taus * U_input[1]
        return [
            x_dot1_3[0],
            x_dot1_3[1],
            x_dot1_3[2],
            x_dot4_6[0],
            x_dot4_6[1],
            x_dot4_6[2],
            x_dot7_9[0],
            x_dot7_9[1],
            x_dot7_9[2],
            x_dot10_12[0],
            x_dot10_12[1],
            x_dot10_12[2],
            x_dot13,
            x_dot14,
            x_dot15,
            x_dot16,
        ]
