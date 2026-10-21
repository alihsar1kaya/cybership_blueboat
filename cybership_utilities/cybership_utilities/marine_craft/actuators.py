"""
Actuators (thruster dynamics): turn the commanded thrust from the allocation into the thrust the
thrusters actually produce. In Fossen's block diagram this sits between control allocation
(f_d, alpha_d) and the marine craft (tau = T(alpha) f).

    FixedThruster:    force along a fixed direction
    AzimuthThruster:  force along an angle alpha that turns with a limited rate

Both have:
  - saturation:   -reverse_ratio * max_force <= f <= max_force (propellers push less in reverse)
  - deadband:     commands below 'deadband' give no thrust
  - first-order thrust dynamics:  T_f f_dot = f_cmd - f  (time_constant T_f, 0 = instant)

The azimuth follows the commanded direction alpha_d = atan2(f_y, f_x) by the shortest way at most
max_rate, and pushes along the angle it actually has while turning (Fossen 2021, eq. 11.24-11.25).

Defaults (no limits, instant) reproduce the old simulator, so vessels without actuator data behave
as before.
"""

import math

import numpy as np


def _first_order(f, target, time_constant, dt):
    if time_constant <= 0.0:
        return target
    return target + (f - target) * math.exp(-dt / time_constant)


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class FixedThruster:
    def __init__(self, max_force=np.inf, reverse_ratio=1.0, deadband=0.0, time_constant=0.0):
        self.max_force = float(max_force)
        self.reverse_ratio = float(reverse_ratio)
        self.deadband = float(deadband)
        self.time_constant = float(time_constant)
        self.force = 0.0      # actual thrust [N]
        self.command = 0.0    # last command [N]

    def saturate(self, f):
        f = float(np.clip(f, -self.reverse_ratio * self.max_force, self.max_force))
        return 0.0 if abs(f) < self.deadband else f

    def set_command(self, fx, fy=0.0):
        self.command = self.saturate(fx)
        return self.command, 0.0

    def step(self, dt):
        self.force = _first_order(self.force, self.command, self.time_constant, dt)

    def output(self):
        """Actual thrust in the thruster frame (fx, fy) and angle."""
        return self.force, 0.0, 0.0


class AzimuthThruster:
    def __init__(self, max_force=np.inf, reverse_ratio=1.0, deadband=0.0, time_constant=0.0,
                 max_rate=np.inf, angle=0.0):
        self.max_force = float(max_force)
        self.reverse_ratio = float(reverse_ratio)  # kept for completeness; alpha covers all directions
        self.deadband = float(deadband)
        self.time_constant = float(time_constant)
        self.max_rate = float(max_rate)            # [rad/s]
        self.angle = float(angle)                  # actual angle alpha [rad]
        self.angle_rate = 0.0
        self.force = 0.0                           # actual thrust magnitude [N]
        self.command = (0.0, 0.0)                  # (F_d, alpha_d)

    def set_command(self, fx, fy):
        """Extended thrust (fx, fy) -> magnitude and angle (Fossen eq. 11.24); magnitude saturated."""
        magnitude = math.hypot(fx, fy)
        if magnitude < self.deadband:
            self.command = (0.0, self.command[1])  # stop, keep the last direction
            return 0.0, 0.0
        alpha_d = math.atan2(fy, fx)
        magnitude = min(magnitude, self.max_force)
        self.command = (magnitude, alpha_d)
        return magnitude * math.cos(alpha_d), magnitude * math.sin(alpha_d)

    def step(self, dt):
        f_d, alpha_d = self.command
        turn = _wrap(alpha_d - self.angle)
        if math.isfinite(self.max_rate):
            turn = float(np.clip(turn, -self.max_rate * dt, self.max_rate * dt))
        self.angle = _wrap(self.angle + turn)
        self.angle_rate = turn / dt if dt > 0 else 0.0
        self.force = _first_order(self.force, f_d, self.time_constant, dt)

    def output(self):
        """Actual thrust in the thruster frame, along the ACTUAL angle: (fx, fy, alpha)."""
        return self.force * math.cos(self.angle), self.force * math.sin(self.angle), self.angle


def make_thruster(t):
    """Build an actuator from one vessel.yaml thruster entry."""
    common = dict(
        max_force=float(t.get("max_force", np.inf)),
        reverse_ratio=float(t.get("reverse_ratio", 1.0)),
        deadband=float(t.get("deadband", 0.0)),
        time_constant=float(t.get("time_constant", 0.0)),
    )
    if t["type"] == "fixed":
        return FixedThruster(**common)
    rate = t.get("max_rate_deg_s")
    return AzimuthThruster(max_rate=math.radians(rate) if rate is not None else np.inf, **common)
