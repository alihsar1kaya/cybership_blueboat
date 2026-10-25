"""
Dynamics (Fossen 2021, Ch. 3-7), structured like Fossen's Otter USV model (PythonVehicleSimulator):

    M nu_r_dot + C(nu_r) nu_r + D(nu_r) nu_r + g(eta) = tau + tau_ext        (eq. 10.141)

    M      = M_RB + M_A                         rigid body + added mass (constant)
    C      = C_RB(nu_r) + C_A(nu_r)             Coriolis/centripetal: C_RB independent of linear
                                                velocity (eq. 3.63, Property 10.1), C_A via m2c
    D(nu_r) nu_r = D_L nu_r                     linear damping, in surge faded out with speed:
                                                X_u e^(-k|u_r|) u_r (k = surge_linear_decay)
                 + X_uu |u_r| u_r               quadratic surge resistance
                 + N_rr |r| r                   nonlinear yaw damping
                 - tau_crossflow(nu_r)          cross-flow drag (strip theory, Hoerner)
    g(eta) = G eta                              linear restoring (heave, roll, pitch)

nu_r = nu - nu_c is the velocity relative to a current nu_c (zero until an environment sets it).
"""

import numpy as np

from cybership_utilities.marine_craft.kinematics import Smtrx, Rzyx, J

# Hoerner's 2D cross-flow drag coefficient, digitised as in Fossen's MSS / PythonVehicleSimulator
_HOERNER_B_2T = np.array([
    0.0109, 0.1766, 0.3530, 0.4519, 0.4728, 0.4929, 0.4933, 0.5585, 0.6464, 0.8336,
    0.9880, 1.3081, 1.6392, 1.8600, 2.3129, 2.6000, 3.0088, 3.4508, 3.7379, 4.0031])
_HOERNER_CD = np.array([
    1.9661, 1.9657, 1.8976, 1.7872, 1.5837, 1.2786, 1.2108, 1.0836, 0.9986, 0.8796,
    0.8284, 0.7599, 0.6914, 0.6571, 0.6307, 0.5962, 0.5868, 0.5859, 0.5599, 0.5593])


def m2c(M, nu):
    """Coriolis/centripetal matrix C(nu) from a 6x6 mass matrix M (Fossen 2021, Ch. 3)."""
    M = 0.5 * (M + M.T)
    nu1, nu2 = nu[0:3], nu[3:6]
    a = M[0:3, 0:3] @ nu1 + M[0:3, 3:6] @ nu2
    b = M[3:6, 0:3] @ nu1 + M[3:6, 3:6] @ nu2
    C = np.zeros((6, 6))
    C[0:3, 3:6] = -Smtrx(a)
    C[3:6, 0:3] = -Smtrx(a)
    C[3:6, 3:6] = -Smtrx(b)
    return C


def crb_linear_velocity_independent(M_RB, nu):
    """
    Rigid-body Coriolis/centripetal matrix parametrized independent of linear velocity
    (Fossen 2021, eq. 3.63), CG in CO:

        C_RB(nu) = [ m S(w)      0     ]
                   [   0     -S(I_g w) ]

    Gives the same C_RB(nu) nu as m2c(M_RB, nu), but only this form satisfies Property 10.1,
    M_RB nu_dot + C_RB(nu) nu = M_RB nu_r_dot + C_RB(nu_r) nu_r, which the equations of relative
    motion in a current (eq. 10.141) need.
    """
    m = M_RB[0, 0]
    w = nu[3:6]
    C = np.zeros((6, 6))
    C[0:3, 0:3] = m * Smtrx(w)
    C[3:6, 3:6] = -Smtrx(M_RB[3:6, 3:6] @ w)
    return C


def hoerner(B, T):
    """2D cross-flow drag coefficient as a function of B/(2T) (Hoerner's curve)."""
    return float(np.interp(B / (2.0 * T), _HOERNER_B_2T, _HOERNER_CD))


def cross_flow_drag(L, B, T, nu_r, rho=1000.0, n=20):
    """
    Cross-flow drag in sway and yaw by strip theory (Fossen 2021, Ch. 6):

        Y = -1/2 rho int T Cd |v + x r| (v + x r) dx,   N = -1/2 rho int x T Cd |v + x r| (v + x r) dx
    """
    dx = L / n
    cd = hoerner(B, T)
    x = -L / 2 + dx / 2 + dx * np.arange(n)
    vx = nu_r[1] + x * nu_r[5]
    f = -0.5 * rho * T * cd * np.abs(vx) * vx * dx
    return np.array([0.0, f.sum(), 0.0, 0.0, 0.0, (x * f).sum()])


class MarineCraft:
    """
    6DOF marine craft. Drop-in replacement for shoeboxpy.model6dof.Shoebox in the simulator:
    attributes eta, nu, step(tau, dt) (RK4) and get_states().
    """

    # Munk moments in C_A(nu_r) nu_r, as (row, column) pairs; the transposed pair keeps C skew-symmetric:
    #   roll  K = (A33 - A22) v w,  pitch M = (A11 - A33) u w,  yaw N = (A22 - A11) u v
    _MUNK = {"roll": [(3, 1), (3, 2)], "pitch": [(4, 0), (4, 2)], "yaw": [(5, 0), (5, 1)]}

    def __init__(self, M_RB, M_A, D_L, G, *, surge_quadratic=0.0, yaw_quadratic=0.0,
                 cross_flow=None, cancel_munk=(), rho=1000.0, eta0=None, nu0=None,
                 surge_linear_decay=0.0):
        self.M_RB = np.asarray(M_RB, dtype=float)
        self.M_A = np.asarray(M_A, dtype=float)
        self.M = self.M_RB + self.M_A
        self.Minv = np.linalg.inv(self.M)
        self.D_L = np.asarray(D_L, dtype=float)
        self.G = np.asarray(G, dtype=float)
        self.surge_quadratic = float(surge_quadratic)
        # Blending as in Fossen's Otter model (MSS otter.m, 2026-10-14): the linear surge damping
        # dominates at low (DP) speed and fades out as the quadratic resistance takes over, so the
        # two are not counted twice at high speed. 0 = linear damping at all speeds.
        self.surge_linear_decay = float(surge_linear_decay)
        self.yaw_quadratic = float(yaw_quadratic)
        self.cross_flow = cross_flow  # dict(length, beam, draft) or None
        self.cancel_munk = tuple(cancel_munk)  # e.g. ("yaw",) as in Fossen's Otter model
        self.rho = rho
        self.current_ned = np.zeros(3)  # [m/s] current velocity in NED (set by an environment model)
        self.eta = np.zeros(6) if eta0 is None else eta0
        self.nu = np.zeros(6) if nu0 is None else nu0

    # The simulator assigns column vectors (6, 1) on reset; always store flat arrays.
    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = np.asarray(value, dtype=float).reshape(6).copy()

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        self._nu = np.asarray(value, dtype=float).reshape(6).copy()

    def get_states(self):
        return self.eta.copy(), self.nu.copy()

    def coriolis(self, nu, nu_r):
        C_A = m2c(self.M_A, nu_r)
        # The Munk moments are destabilising. A real hull is stabilised by lift on the keel/skeg,
        # which this model does not include, so they can be cancelled (Fossen does so for yaw in Otter).
        for dof in self.cancel_munk:
            for i, j in self._MUNK[dof]:
                C_A[i, j] = C_A[j, i] = 0.0
        return crb_linear_velocity_independent(self.M_RB, nu) + C_A

    def damping(self, nu_r):
        """Damping force D(nu_r) nu_r with the sign of a force on the vessel (i.e. negative)."""
        tau = -self.D_L @ nu_r
        if self.surge_linear_decay:
            tau[0] += self.D_L[0, 0] * (1.0 - np.exp(-self.surge_linear_decay * abs(nu_r[0]))) * nu_r[0]
        tau[0] -= self.surge_quadratic * abs(nu_r[0]) * nu_r[0]
        tau[5] -= self.yaw_quadratic * abs(nu_r[5]) * nu_r[5]
        if self.cross_flow:
            cf = self.cross_flow
            tau += cross_flow_drag(cf["length"], cf["beam"], cf["draft"], nu_r, self.rho)
        return tau

    def derivatives(self, eta, nu, tau, tau_ext):
        # Current: constant in NED, expressed in BODY; its time derivative from the body rotation
        R = Rzyx(eta[3], eta[4], eta[5])
        nu_c = np.concatenate([R.T @ self.current_ned, np.zeros(3)])
        nu_c_dot = np.concatenate([-Smtrx(nu[3:6]) @ nu_c[0:3], np.zeros(3)])
        nu_r = nu - nu_c

        sum_tau = tau + tau_ext + self.damping(nu_r) - self.coriolis(nu, nu_r) @ nu_r - self.G @ eta
        return J(eta) @ nu, nu_c_dot + self.Minv @ sum_tau

    def step(self, tau=None, dt=0.01, tau_ext=None):
        """Advance (eta, nu) one time step with 4th-order Runge-Kutta."""
        tau = np.zeros(6) if tau is None else np.asarray(tau, dtype=float).reshape(6)
        tau_ext = np.zeros(6) if tau_ext is None else np.asarray(tau_ext, dtype=float).reshape(6)
        eta, nu = self.eta, self.nu
        k1e, k1n = self.derivatives(eta, nu, tau, tau_ext)
        k2e, k2n = self.derivatives(eta + 0.5 * dt * k1e, nu + 0.5 * dt * k1n, tau, tau_ext)
        k3e, k3n = self.derivatives(eta + 0.5 * dt * k2e, nu + 0.5 * dt * k2n, tau, tau_ext)
        k4e, k4n = self.derivatives(eta + dt * k3e, nu + dt * k3n, tau, tau_ext)
        self.eta = eta + dt / 6.0 * (k1e + 2 * k2e + 2 * k3e + k4e)
        self.nu = nu + dt / 6.0 * (k1n + 2 * k2n + 2 * k3n + k4n)
