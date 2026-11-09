"""
Environment (Fossen 2021, Ch. 10): disturbances acting on the marine craft.

Implemented so far:
    OceanCurrent   2-D irrotational current (Sec. 10.3), enters the dynamics through the
                   relative velocity nu_r = nu - nu_c (see MarineCraft.current_ned).

    v_c^n = [V_c cos(beta_c), V_c sin(beta_c), 0]      current velocity in NED, beta_c = direction
                                                       the current flows TOWARDS (0 = north, 90 = east)
    u_c = V_c cos(beta_c - psi),  v_c = V_c sin(beta_c - psi)     (in BODY, done by MarineCraft)

The speed varies as a bounded first-order Gauss-Markov process (Fossen: V_c_dot + mu V_c = w,
V_min <= V_c <= V_max). Here it is mean-reverting to the set speed, so the set speed is the
average current:

    V_c_dot = -mu (V_c - V_mean) + w,    w ~ N(0, sigma^2)    (Euler-Maruyama)

mu = 0 and sigma = 0 give a constant current. The direction can drift the same way.

    Wind           wind loads (Sec. 10.1, TTK4190 L7 slides 5-10), Blendermann (1994) coefficients,
                   added to the dynamics as tau_ext (MarineCraft.tau_ext).

Waves (Sec. 10.2) are in waves.py.
"""

import math

import numpy as np


class OceanCurrent:
    def __init__(self, speed=0.0, direction_deg=0.0, mu=0.0, sigma=0.0, speed_min=0.0,
                 speed_max=np.inf, direction_mu=0.0, direction_sigma_deg=0.0, seed=None):
        self.mean_speed = float(speed)
        self.mean_direction = math.radians(direction_deg)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.direction_mu = float(direction_mu)
        self.direction_sigma = math.radians(direction_sigma_deg)
        self.rng = np.random.default_rng(seed)
        self.speed = self.mean_speed
        self.direction = self.mean_direction

    def set(self, speed=None, direction_deg=None):
        """Change the mean speed/direction (e.g. from a ROS parameter); jumps there directly."""
        if speed is not None:
            self.mean_speed = self.speed = float(speed)
        if direction_deg is not None:
            self.mean_direction = self.direction = math.radians(direction_deg)

    def step(self, dt):
        """Advance the Gauss-Markov processes one time step."""
        if self.mu or self.sigma:
            dw = self.sigma * math.sqrt(dt) * self.rng.standard_normal()
            self.speed += -self.mu * (self.speed - self.mean_speed) * dt + dw
            self.speed = float(np.clip(self.speed, self.speed_min, self.speed_max))
        if self.direction_mu or self.direction_sigma:
            dw = self.direction_sigma * math.sqrt(dt) * self.rng.standard_normal()
            self.direction += -self.direction_mu * (self.direction - self.mean_direction) * dt + dw

    def velocity_ned(self):
        """Current velocity v_c^n = [north, east, down] [m/s]."""
        return np.array([self.speed * math.cos(self.direction),
                         self.speed * math.sin(self.direction), 0.0])


# Blendermann (1994) wind load parameters per vessel type (Fossen 2021 Table 10.2, L7 slide 10):
# CD_t, CD_lAF(0) (wind from ahead), CD_lAF(pi) (from astern), delta, kappa
BLENDERMANN = {
    "car_carrier": (0.95, 0.55, 0.60, 0.80, 1.2),
    "cargo_loaded": (0.85, 0.65, 0.55, 0.40, 1.7),
    "cargo_container_on_deck": (0.85, 0.55, 0.50, 0.40, 1.4),
    "container_ship_loaded": (0.90, 0.55, 0.55, 0.40, 1.4),
    "destroyer": (0.85, 0.60, 0.65, 0.65, 1.1),
    "diving_support": (0.90, 0.60, 0.80, 0.55, 1.7),
    "drilling_vessel": (1.00, 0.85, 0.925, 0.10, 1.7),     # table gives ranges 0.70-1.00 / 0.75-1.10
    "ferry": (0.90, 0.45, 0.50, 0.80, 1.1),
    "fishing_vessel": (0.95, 0.70, 0.70, 0.40, 1.1),
    "lng_tanker": (0.70, 0.60, 0.65, 0.50, 1.1),
    "offshore_supply": (0.90, 0.55, 0.80, 0.55, 1.2),
    "passenger_liner": (0.90, 0.40, 0.40, 0.80, 1.2),
    "research_vessel": (0.85, 0.55, 0.65, 0.60, 1.4),
    "speed_boat": (0.90, 0.55, 0.60, 0.60, 1.1),
    "tanker_loaded": (0.70, 0.90, 0.55, 0.40, 3.1),
    "tanker_ballast": (0.70, 0.75, 0.55, 0.40, 2.2),
    "tender": (0.85, 0.55, 0.55, 0.65, 1.1),
}


def blendermann(gamma, A_Fw, A_Lw, s_L, L_oa, CD_t, CD_lAF_bow, CD_lAF_stern, delta, kappa):
    """
    Wind coefficients (C_X, C_Y, C_K, C_N) for relative wind angle gamma [rad] (L7 slide 10):

        C_X = -CD_lAF cos(g) / den,   C_Y = CD_t sin(g) / den,   C_K = kappa C_Y,
        C_N = [s_L / L_oa - 0.18 (g - pi/2)] C_Y,
        den = 1 - delta/2 (1 - CD_l / CD_t) sin^2(2g),   CD_l = CD_lAF A_Fw / A_Lw

    gamma = 0: wind from ahead, +pi/2: from port (pushes to starboard). The formulas are for
    0 <= g <= pi; port/starboard symmetry gives the other side (C_Y, C_K, C_N change sign).
    """
    g = abs(math.atan2(math.sin(gamma), math.cos(gamma)))
    sign = 1.0 if math.sin(gamma) >= 0 else -1.0
    CD_lAF = CD_lAF_bow if g <= math.pi / 2 else CD_lAF_stern
    CD_l = CD_lAF * A_Fw / A_Lw
    den = 1.0 - 0.5 * delta * (1.0 - CD_l / CD_t) * math.sin(2 * g) ** 2
    CX = -CD_lAF * math.cos(g) / den
    CY = CD_t * math.sin(g) / den
    CN = (s_L / L_oa - 0.18 * (g - math.pi / 2)) * CY
    return CX, sign * CY, sign * kappa * CY, sign * CN


class Wind:
    """
    Wind loads on a moving craft (Fossen 2021 Sec. 10.1, L7 slide 9):

        u_w = V_w cos(beta_w - psi),  v_w = V_w sin(beta_w - psi)     wind velocity in BODY
        u_rw = u - u_w,  v_rw = v - v_w,  V_rw = sqrt(u_rw^2 + v_rw^2)
        gamma_rw = -atan2(v_rw, u_rw)                                  (MSS sign: 0 = head wind)
        tau_wind = 1/2 rho_a V_rw^2 [C_X A_Fw, C_Y A_Lw, 0, C_K A_Lw H_Lw, 0, C_N A_Lw L_oa]

    beta_w is the direction the wind blows TOWARDS (0 = north, 90 = east), like the current and the
    waves in this simulator (a 'north wind' that comes from the north has beta_w = 180 deg).
    The speed may vary as a mean-reverting Gauss-Markov process (gusts, as OceanCurrent).
    """

    def __init__(self, areas, coefficients="speed_boat", rho_a=1.225, speed=0.0, direction_deg=0.0,
                 mu=0.0, sigma=0.0, seed=None):
        self.A_Fw, self.A_Lw = float(areas["A_Fw"]), float(areas["A_Lw"])
        self.H_Lw, self.s_L, self.L_oa = float(areas["H_Lw"]), float(areas["s_L"]), float(areas["L_oa"])
        self.coef = BLENDERMANN[coefficients] if isinstance(coefficients, str) else tuple(coefficients)
        self.rho_a = float(rho_a)
        self.gusts = OceanCurrent(speed, direction_deg, mu=mu, sigma=sigma, seed=seed)

    @property
    def speed(self):
        return self.gusts.speed

    def set(self, speed=None, direction_deg=None):
        self.gusts.set(speed, direction_deg)

    def step(self, dt):
        self.gusts.step(dt)

    def velocity_ned(self):
        return self.gusts.velocity_ned()

    def loads(self, eta, nu):
        """tau_wind (6, BODY) for the craft state eta = [x y z phi theta psi], nu = [u v w p q r]."""
        V, beta, psi = self.gusts.speed, self.gusts.direction, eta[5]
        u_rw = nu[0] - V * math.cos(beta - psi)
        v_rw = nu[1] - V * math.sin(beta - psi)
        V_rw2 = u_rw**2 + v_rw**2
        if V_rw2 < 1e-12:
            return np.zeros(6)
        CX, CY, CK, CN = blendermann(-math.atan2(v_rw, u_rw), self.A_Fw, self.A_Lw, self.s_L, self.L_oa,
                                     *self.coef)
        q = 0.5 * self.rho_a * V_rw2
        return q * np.array([CX * self.A_Fw, CY * self.A_Lw, 0.0, CK * self.A_Lw * self.H_Lw, 0.0,
                             CN * self.A_Lw * self.L_oa])
