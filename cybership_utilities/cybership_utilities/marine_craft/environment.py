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

Wind and waves (Sec. 10.1-10.2) are not modelled: they need wind coefficients and wave RAOs
that do not exist for this vessel yet.
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
