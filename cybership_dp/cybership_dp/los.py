"""
LOS guidance for straight-line path following (Fossen 2021 Ch. 12.3-12.5, TTK4190 L6 slides 41, 59-61).

Path from waypoint p_i to p_i+1 (NED, metres):

    pi_h = atan2(y_i+1 - y_i, x_i+1 - x_i)                     path-tangential angle
    [x_e, y_e] = R(pi_h)^T ([x, y] - [x_i, y_i])                along-track and cross-track error

2-D adaptive LOS (ALOS, Fossen 2023, IEEE TCST 31(6)), heading autopilot:

    psi_d       = pi_h - beta_hat - atan(y_e / Delta)
    beta_hat_dot = gamma * Delta / sqrt(Delta^2 + y_e^2) * y_e

beta_hat estimates the crab angle beta_c = atan(v / u) (chi = psi + beta_c) from currents, wind and waves,
which plain proportional LOS (gamma = 0) leaves as a steady cross-track error. USGES (Fossen 2023).

Integral LOS (ILOS, Borhaug et al. 2008, Fossen 2021 Ch. 12.5.2), method='ilos':

    psi_d       = pi_h - atan(K_p y_e + K_i y_int),   K_p = 1/Delta, K_i = kappa K_p
    y_int_dot   = Delta y_e / (Delta^2 + (y_e + kappa y_int)^2)
"""

import math


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ALOS:
    def __init__(self, lookahead, gamma=0.0, max_crab_deg=30.0, method="alos", kappa=0.2):
        """lookahead Delta [m] (K_p = 1/Delta), gamma [rad/(m s)] adaptation gain (0 = proportional LOS);
        method 'alos' (or 'los': gamma forced to 0) or 'ilos' with integral gain kappa."""
        self.delta = float(lookahead)
        self.method = method
        self.gamma = 0.0 if method == "los" else float(gamma)
        self.kappa = float(kappa)
        self.y_int = 0.0
        self.max_crab = math.radians(max_crab_deg)
        self.beta_hat = 0.0
        self.p0 = self.p1 = (0.0, 0.0)
        self.pi_h = 0.0
        self.length = 0.0

    def set_path(self, p0, p1, keep_crab=False):
        """New straight line from p0 to p1; the crab estimate starts at 0 unless keep_crab."""
        self.p0, self.p1 = (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))
        self.pi_h = math.atan2(self.p1[1] - self.p0[1], self.p1[0] - self.p0[0])
        self.length = math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])
        if not keep_crab:
            self.beta_hat = 0.0

    def errors(self, x, y):
        """(x_e, y_e): along-track distance from p0 and cross-track error (positive to the right)."""
        dx, dy = x - self.p0[0], y - self.p0[1]
        c, s = math.cos(self.pi_h), math.sin(self.pi_h)
        return c * dx + s * dy, -s * dx + c * dy

    def remaining(self, x, y):
        """Along-track distance left to p1."""
        return self.length - self.errors(x, y)[0]

    def update(self, x, y, h):
        """Desired heading psi_d for position (x, y); integrates beta_hat over h seconds (Euler)."""
        return self.heading(self.pi_h, self.errors(x, y)[1], h)

    def heading(self, pi_h, ye, h):
        """ALOS law for any path (also curved, Fossen 2021 Ch. 12.5): path-tangential angle pi_h at the
        closest point and cross-track error ye there."""
        if self.method == "ilos":
            kp = 1.0 / self.delta
            psi_d = wrap(pi_h - math.atan(kp * ye + self.kappa * kp * self.y_int))
            self.y_int += h * self.delta * ye / (self.delta**2 + (ye + self.kappa * self.y_int)**2)
            return psi_d
        psi_d = wrap(pi_h - self.beta_hat - math.atan(ye / self.delta))
        rate = self.gamma * self.delta / math.sqrt(self.delta**2 + ye**2) * ye
        self.beta_hat = max(-self.max_crab, min(self.max_crab, self.beta_hat + h * rate))
        return psi_d
