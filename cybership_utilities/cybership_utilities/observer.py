"""
Kalman filter for dynamic positioning (Fossen 2021, Ch. 13.4.6; TTK4190 L8), with an optional
wave filter (WF):

    xi_dot  = A_w xi + E_w w_w                  WF motion, per DOF a 2nd-order resonator
                                                [xi1; xi2]' = [0 I; -Omega^2 -2 Lambda Omega] [xi1; xi2] + [0; K_w] w
    eta_dot = R(psi) nu                         LF motion, eta = [N, E, psi]
    b_dot   = -T_b^-1 b + w_b                   bias in NED: slowly varying forces the model lacks
                                                (current, wave drift, model errors), T_b^-1 = 0: random walk
    M nu_dot = -D nu + R(psi)^T b + tau + w_nu  nu = [u, v, r], linear low-speed DP model (L3b p. 45)

    y_gnss    = [N, E] + xi2_NE + eps     position of CO (GNSS antenna lever arm removed beforehand)
    y_compass = psi + xi2_psi + eps       heading (true north)

The controller uses only the LF estimates (eta, nu): the WF motion is filtered out, so the thrusters
do not try to follow every wave (Ch. 13.2, wave filtering).

Optionally the linear damping -D nu is replaced by the vessel model's full damping d(nu) (quadratic
surge resistance, cross-flow drag; the same model the controller uses), linearized for the
covariance: an extended Kalman filter (Ch. 13.3). Fossen's linear DP model is meant for low speed;
at transit speed (~1 m/s) the quadratic resistance dominates and the linear model lags.

Linear time-varying through R(psi), with psi taken from the estimate. Discretized with Euler's
method as in Fossen's case study (A_d = I + h A, Q_d = h E Q E'). Asynchronous measurements
(Ch. 13.4.4): predict() at the filter rate, correct() only when a measurement has arrived;
otherwise the corrector equals the predictor. The heading innovation uses the smallest signed
angle (Ch. 13.4.3), and the covariance the Joseph form (as MSS exKalmanWavefilterAutopilot.m).

State x = [xi (6, with the wave filter), eta (3), b (3), nu (3)].
"""

import math

import numpy as np


def ssa(angle):
    """Smallest signed angle, in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def Rz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class DPKalmanFilter:
    def __init__(self, M, D, q_bias, q_nu, P0=None, x0=None, T_b=None, damping=None, wave_filter=None):
        """
        M, D:    3x3 (surge, sway, yaw) mass incl. added mass and linear damping
        q_bias:  3 process noise intensities of the bias [N^2/s, N^2/s, (N m)^2/s]
        q_nu:    3 process noise intensities on the velocity equation (unmodelled forces) [same units]
        T_b:     3 bias time constants [s] (None: random walk)
        damping: optional function nu (3) -> damping force D(nu) nu (3, positive = opposing), e.g. from
                 MarineCraft.damping; None: the linear D nu
        wave_filter: optional dict(omega0 [rad/s], zeta (Lambda, ~0.1), q (3 intensities of w_w)):
                 the peak (encounter) frequency and damping of the WF model, Fossen 13.4.6
        """
        self.damping = damping
        self.M = np.asarray(M, dtype=float)
        self.Minv = np.linalg.inv(self.M)
        self.D = np.asarray(D, dtype=float)
        self.Tb_inv = np.zeros(3) if T_b is None else 1.0 / np.asarray(T_b, dtype=float)
        self.nw = 6 if wave_filter else 0
        self.n = self.nw + 9
        w = self.nw
        self.i_eta, self.i_b, self.i_nu = slice(w, w + 3), slice(w + 3, w + 6), slice(w + 6, w + 9)
        q_w = []
        if wave_filter:
            w0 = float(wave_filter["omega0"])
            lam = float(wave_filter.get("zeta", 0.1))
            self.A_w = np.block([[np.zeros((3, 3)), np.eye(3)],
                                 [-w0**2 * np.eye(3), -2 * lam * w0 * np.eye(3)]])
            q_w = list(wave_filter.get("q", [0.1, 0.1, 0.001]))
        self.Qc = np.diag(np.r_[q_w, q_bias, q_nu])
        self.x = np.zeros(self.n) if x0 is None else np.asarray(x0, dtype=float).copy()
        p0 = [1.0, 1.0, 0.1, 100.0, 100.0, 10.0, 0.1, 0.1, 0.01]
        self.P = np.diag([0.01] * w + p0) if P0 is None else P0.copy()

    # --- model ---------------------------------------------------------------------------------
    def damping_force(self, nu):
        return self.D @ nu if self.damping is None else np.asarray(self.damping(nu), dtype=float)

    def damping_jacobian(self, nu, eps=1e-4):
        if self.damping is None:
            return self.D
        J = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3)
            d[i] = eps
            J[:, i] = (self.damping_force(nu + d) - self.damping_force(nu - d)) / (2 * eps)
        return J

    def A(self, psi, nu=None):
        """System matrix, linearized about nu for nonlinear damping."""
        R = Rz(psi)
        A = np.zeros((self.n, self.n))
        e, b, v = self.i_eta, self.i_b, self.i_nu
        if self.nw:
            A[0:6, 0:6] = self.A_w
        A[e, v] = R
        A[b, b] = -np.diag(self.Tb_inv)
        A[v, b] = self.Minv @ R.T
        A[v, v] = -self.Minv @ self.damping_jacobian(self.x[v] if nu is None else nu)
        return A

    def E(self):
        """Process noise input: [w_w (3, on xi2), w_b (3), w_nu (3)]."""
        E = np.zeros((self.n, self.nw // 2 + 6))
        k = self.nw // 2
        if self.nw:
            E[3:6, 0:3] = np.eye(3)
        E[self.i_b, k:k + 3] = np.eye(3)
        E[self.i_nu, k + 3:k + 6] = self.Minv
        return E

    # --- Kalman filter ----------------------------------------------------------------------------
    def predict(self, tau, h):
        """State and covariance predictor over h seconds with control force tau = [X, Y, N] (body)."""
        A = self.A(self.x[self.i_eta][2])
        Ad = np.eye(self.n) + h * A
        Ed = self.E()
        # state: Euler step of the (nonlinear) model
        eta, b, nu = self.x[self.i_eta], self.x[self.i_b], self.x[self.i_nu]
        R = Rz(eta[2])
        nu_dot = self.Minv @ (-self.damping_force(nu) + R.T @ b + np.asarray(tau, dtype=float))
        xi_dot = self.A_w @ self.x[0:6] if self.nw else np.zeros(0)
        self.x = self.x + h * np.r_[xi_dot, R @ nu, -self.Tb_inv * b, nu_dot]
        self._wrap()
        self.P = Ad @ self.P @ Ad.T + h * Ed @ self.Qc @ Ed.T
        self.P = 0.5 * (self.P + self.P.T)

    def correct(self, y, C, Rd, angle_rows=()):
        """Corrector with measurement y = C x + eps, eps ~ N(0, Rd); angle rows use ssa innovations."""
        C = np.atleast_2d(C)
        innovation = np.asarray(y, dtype=float) - C @ self.x
        for i in angle_rows:
            innovation[i] = ssa(innovation[i])
        S = C @ self.P @ C.T + Rd
        K = self.P @ C.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self._wrap()
        IKC = np.eye(self.n) - K @ C
        self.P = IKC @ self.P @ IKC.T + K @ Rd @ K.T  # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return innovation

    def _wrap(self):
        k = self.nw + 2
        self.x[k] = ssa(self.x[k])

    def correct_gnss(self, north, east, sigma):
        """y = [N, E] + xi2_NE (WF position) + eps."""
        C = np.zeros((2, self.n))
        C[0, self.nw + 0] = C[1, self.nw + 1] = 1.0
        if self.nw:
            C[0, 3] = C[1, 4] = 1.0
        return self.correct([north, east], C, np.eye(2) * sigma**2)

    def correct_compass(self, psi, sigma):
        """y = psi + xi2_psi (WF heading) + eps."""
        C = np.zeros((1, self.n))
        C[0, self.nw + 2] = 1.0
        if self.nw:
            C[0, 5] = 1.0
        return self.correct([psi], C, np.array([[sigma**2]]), angle_rows=(0,))

    # --- outputs ----------------------------------------------------------------------------------
    @property
    def eta(self):
        """LF position and heading."""
        return self.x[self.i_eta]

    @property
    def bias(self):
        """Bias force in NED [N, N, N m]."""
        return self.x[self.i_b]

    @property
    def nu(self):
        """LF velocity (BODY)."""
        return self.x[self.i_nu]

    @property
    def eta_wf(self):
        """WF position and heading (zeros without the wave filter)."""
        return self.x[3:6] if self.nw else np.zeros(3)


def remove_lever_arm(p_antenna, lever_arm, phi, theta, psi):
    """CO position from the GNSS antenna position: p = p_a - R(phi, theta, psi) r_a (NED)."""
    from cybership_utilities.marine_craft.kinematics import Rzyx
    return np.asarray(p_antenna, dtype=float) - Rzyx(phi, theta, psi) @ np.asarray(lever_arm, dtype=float)
