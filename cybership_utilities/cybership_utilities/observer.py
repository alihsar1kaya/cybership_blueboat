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

Ocean current (optional, Fossen 10.3): the current is irrotational and constant in NED, so the model
is written in the velocity relative to the water, nu_r, and the current only enters the kinematics:

    eta_dot    = R(psi) nu_r + [v_c; 0]         v_c = [V_N, V_E] current velocity in NED, v_c_dot = w_c
    M nu_r_dot = -d(nu_r) + R(psi)^T b + tau    the damping acts on nu_r, so the current force follows
                                                the hull when the boat turns

A constant force bias cannot do that: it is fixed in NED, while the drag from a current depends on how
the hull lies in it. The bias is kept for what is a force (wind, wave drift, thrust errors). At rest
on a fixed heading the two cannot be told apart; they separate when the heading or speed changes.

State x = [xi (6, with the wave filter), eta (3), b (3), nu (3), v_c (2, with the current)].
"""

import math
from collections import deque

import numpy as np


def ssa(angle):
    """Smallest signed angle, in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def Rz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class DPKalmanFilter:
    def __init__(self, M, D, q_bias, q_nu, P0=None, x0=None, T_b=None, damping=None, wave_filter=None,
                 current=None):
        """
        M, D:    3x3 (surge, sway, yaw) mass incl. added mass and linear damping
        q_bias:  3 process noise intensities of the bias [N^2/s, N^2/s, (N m)^2/s]
        q_nu:    3 process noise intensities on the velocity equation (unmodelled forces) [same units]
        T_b:     3 bias time constants [s] (None: random walk)
        damping: optional function nu (3) -> velocity-dependent forces D(nu) nu (+ C(nu) nu) (3, positive =
                 opposing), e.g. vessel_model.velocity_forces_3dof; None: the linear D nu
        wave_filter: optional dict(omega0 [rad/s], zeta (Lambda, ~0.1), q (3 intensities of w_w)):
                 the peak (encounter) frequency and damping of the WF model, Fossen 13.4.6
        current: optional dict(q (2 intensities of w_c [(m/s)^2/s]), p0 (initial variance [(m/s)^2])):
                 estimate the current velocity; the velocity state is then relative to the water
        """
        self.damping = damping
        self.M = np.asarray(M, dtype=float)
        self.Minv = np.linalg.inv(self.M)
        self.D = np.asarray(D, dtype=float)
        self.Tb_inv = np.zeros(3) if T_b is None else 1.0 / np.asarray(T_b, dtype=float)
        self.nw = 6 if wave_filter else 0
        self.nc = 2 if current else 0
        self.n = self.nw + 9 + self.nc
        w = self.nw
        self.i_eta, self.i_b, self.i_nu = slice(w, w + 3), slice(w + 3, w + 6), slice(w + 6, w + 9)
        self.i_c = slice(w + 9, w + 9 + self.nc)
        q_w = []
        if wave_filter:
            w0 = float(wave_filter["omega0"])
            lam = float(wave_filter.get("zeta", 0.1))
            self.A_w = np.block([[np.zeros((3, 3)), np.eye(3)],
                                 [-w0**2 * np.eye(3), -2 * lam * w0 * np.eye(3)]])
            q_w = list(wave_filter.get("q", [0.1, 0.1, 0.001]))
        q_c = list(np.broadcast_to(current.get("q", 1e-4), 2)) if current else []
        self.Qc = np.diag(np.r_[q_w, q_bias, q_nu, q_c])
        self.x = np.zeros(self.n) if x0 is None else np.asarray(x0, dtype=float).copy()
        p0 = [1.0, 1.0, 0.1, 100.0, 100.0, 10.0, 0.1, 0.1, 0.01]
        p0 += [float(current.get("p0", 0.04))] * self.nc if current else []
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
        if self.nc:
            A[self.nw:self.nw + 2, self.i_c] = np.eye(2)
        return A

    def E(self):
        """Process noise input: [w_w (3, on xi2), w_b (3), w_nu (3), w_c (2)]."""
        E = np.zeros((self.n, self.nw // 2 + 6 + self.nc))
        k = self.nw // 2
        if self.nw:
            E[3:6, 0:3] = np.eye(3)
        E[self.i_b, k:k + 3] = np.eye(3)
        E[self.i_nu, k + 3:k + 6] = self.Minv
        if self.nc:
            E[self.i_c, k + 6:k + 8] = np.eye(2)
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
        eta_dot = R @ nu
        if self.nc:
            eta_dot[0:2] += self.x[self.i_c]        # nu is relative to the water when the current is estimated
        self.x = self.x + h * np.r_[xi_dot, eta_dot, -self.Tb_inv * b, nu_dot, np.zeros(self.nc)]
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
        """LF velocity over ground (BODY)."""
        if not self.nc:
            return self.x[self.i_nu]
        return self.x[self.i_nu] + Rz(self.x[self.nw + 2]).T @ np.r_[self.x[self.i_c], 0.0]

    @property
    def nu_r(self):
        """LF velocity relative to the water (BODY); equals nu without the current estimate."""
        return self.x[self.i_nu]

    @property
    def current(self):
        """Current velocity in NED [V_N, V_E] (zeros without the current estimate)."""
        return self.x[self.i_c] if self.nc else np.zeros(2)

    @property
    def eta_wf(self):
        """WF position and heading (zeros without the wave filter)."""
        return self.x[3:6] if self.nw else np.zeros(3)


def remove_lever_arm(p_antenna, lever_arm, phi, theta, psi):
    """CO position from the GNSS antenna position: p = p_a - R(phi, theta, psi) r_a (NED)."""
    from cybership_utilities.marine_craft.kinematics import Rzyx
    return np.asarray(p_antenna, dtype=float) - Rzyx(phi, theta, psi) @ np.asarray(lever_arm, dtype=float)


class AsyncObserver:
    """
    Runs a DPKalmanFilter the way the observer node does, without ROS (the node and the offline tests
    share it): predict at a fixed rate, correct with every heading, and apply a delayed GNSS fix at
    the time it was taken. For a delayed fix the filter goes back to the stored state at that time,
    corrects, and predicts forward again with the stored inputs and headings.

    All times are in seconds on one clock (the node uses ROS time).
    """

    def __init__(self, kf, h, sigma_psi, delay_compensation=True, history=1.0):
        self.kf, self.h, self.sigma_psi = kf, float(h), float(sigma_psi)
        self.delay_compensation = delay_compensation
        # states before each prediction and the headings in between
        self.history = deque(maxlen=int(history / self.h))
        self.headings = deque(maxlen=int(2.0 * history / self.h))
        self.initialized = {"gnss": False, "psi": False}

    @property
    def ready(self):
        return all(self.initialized.values())

    def heading(self, t, psi):
        """Compass heading psi [rad] received at time t."""
        if not self.initialized["psi"]:
            self.kf.x[self.kf.nw + 2] = psi
            self.initialized["psi"] = True
        self.kf.correct_compass(psi, self.sigma_psi)
        self.headings.append((t, psi))

    def gnss(self, t, t_meas, north, east, sigma):
        """Position of CO [m, NED] taken at t_meas and received at t."""
        if not self.initialized["gnss"]:
            self.kf.x[self.kf.nw:self.kf.nw + 2] = north, east
            self.initialized["gnss"] = True
        hist = list(self.history)
        k = None
        if self.delay_compensation and t - t_meas > 0.5 * self.h:
            k = next((i for i in range(len(hist) - 1, -1, -1) if hist[i][0] <= t_meas), None)
        if k is None:                       # not delayed, or older than the history: use it as new
            self.kf.correct_gnss(north, east, sigma)
            return
        self.kf.x, self.kf.P = hist[k][1].copy(), hist[k][2].copy()
        self.kf.correct_gnss(north, east, sigma)
        times = [h[0] for h in hist[k:]] + [t]
        for j, (t_j, _, _, tau_j) in enumerate(hist[k:]):
            self.kf.predict(tau_j, self.h)
            for t_psi, psi in self.headings:
                if t_j < t_psi <= times[j + 1]:
                    self.kf.correct_compass(psi, self.sigma_psi)

    def step(self, t, tau):
        """One filter step at time t with the input tau = [X, Y, N]: store the state, then predict."""
        if not self.ready:
            return
        tau = np.asarray(tau, dtype=float)
        self.history.append((t, self.kf.x.copy(), self.kf.P.copy(), tau.copy()))
        self.kf.predict(tau, self.h)
