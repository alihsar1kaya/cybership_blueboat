"""
Actuators (thruster dynamics): turn the commanded thrust from the allocation into the thrust the
thrusters actually produce. In Fossen's block diagram this sits between control allocation
(f_d, alpha_d) and the marine craft (tau = T(alpha) f).

    FixedThruster:    force along a fixed direction
    AzimuthThruster:  force along an angle alpha that turns with a limited rate

Both have:
  - saturation:   -reverse_ratio * max_force <= f <= max_force (propellers push less in reverse)
  - deadband:     commands below 'deadband' give no thrust (simple force model), or
  - ESC/PWM:      PwmThrusterCurve (e.g. Blue Robotics T200 + ESC): the driver turns the force into a
                  PWM pulse in whole microseconds and skips the ESC deadzone, the ESC/propeller turn
                  the pulse back into thrust T = K |n| n (Fossen 2021, Ch. 9.1)
  - first-order thrust dynamics:  T_f f_dot = f_cmd - f  (time_constant T_f, 0 = instant)

The azimuth follows the commanded direction alpha_d = atan2(f_y, f_x) by the shortest way at most
max_rate, and pushes along the angle it actually has while turning (Fossen 2021, eq. 11.24-11.25).

Defaults (no limits, instant) reproduce the old simulator, so vessels without actuator data behave
as before.
"""

import math

import numpy as np


class PwmThrusterCurve:
    """
    Thrust from an ESC pulse width, as for a Blue Robotics T200 with Basic ESC (1100-1900 us, neutral
    1500 us, deadzone +-25 us). Outside the deadzone the shaft speed is roughly proportional to the
    pulse, so the thrust T = K |n| n is quadratic in it (Fossen 2021, eq. 9.7, J_a ~ 0 for DP):

        f = max_force (s / s_max)^2                     s = pwm - (neutral + deadzone) > 0
        f = -reverse_ratio max_force (s / s_max)^2      s = (neutral - deadzone) - pwm > 0

    to_pwm() is the thruster driver: the inverse, which skips the deadzone (so small forces are not
    lost) and rounds to whole microseconds like a real PWM output.
    """

    def __init__(self, max_force, reverse_ratio=1.0, neutral=1500, deadzone=25, half_range=400):
        self.max_force = float(max_force)
        self.reverse_ratio = float(reverse_ratio)
        self.neutral = int(neutral)
        self.deadzone = int(deadzone)
        self.s_max = float(half_range - deadzone)

    def force(self, pwm):
        fwd = pwm - (self.neutral + self.deadzone)
        rev = (self.neutral - self.deadzone) - pwm
        if fwd > 0:
            return self.max_force * min(fwd / self.s_max, 1.0) ** 2
        if rev > 0:
            return -self.reverse_ratio * self.max_force * min(rev / self.s_max, 1.0) ** 2
        return 0.0

    def to_pwm(self, f):
        if f > 0:
            s = self.s_max * math.sqrt(min(f / self.max_force, 1.0))
            return self.neutral + self.deadzone + int(round(s))
        if f < 0:
            s = self.s_max * math.sqrt(min(-f / (self.reverse_ratio * self.max_force), 1.0))
            return self.neutral - self.deadzone - int(round(s))
        return self.neutral


def _first_order(f, target, time_constant, dt):
    if time_constant <= 0.0:
        return target
    return target + (f - target) * math.exp(-dt / time_constant)


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class FixedThruster:
    def __init__(self, max_force=np.inf, reverse_ratio=1.0, deadband=0.0, time_constant=0.0, pwm=None):
        self.max_force = float(max_force)
        self.reverse_ratio = float(reverse_ratio)
        self.deadband = float(deadband)
        self.time_constant = float(time_constant)
        self.pwm_curve = pwm   # PwmThrusterCurve or None
        self.pwm = None        # last pulse width [us] (with a PWM curve)
        self.force = 0.0       # actual thrust [N]
        self.command = 0.0     # last command [N]

    def saturate(self, f):
        f = float(np.clip(f, -self.reverse_ratio * self.max_force, self.max_force))
        if self.pwm_curve is not None:
            self.pwm = self.pwm_curve.to_pwm(f)
            return self.pwm_curve.force(self.pwm)
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
                 max_rate=np.inf, angle=0.0, pwm=None):
        self.max_force = float(max_force)
        self.reverse_ratio = float(reverse_ratio)  # kept for completeness; alpha covers all directions
        self.deadband = float(deadband)
        self.time_constant = float(time_constant)
        self.max_rate = float(max_rate)            # [rad/s]
        self.angle = float(angle)                  # actual angle alpha [rad]
        self.angle_rate = 0.0
        self.force = 0.0                           # actual thrust magnitude [N]
        self.command = (0.0, 0.0)                  # (F_d, alpha_d)
        self.pwm_curve = pwm                       # PwmThrusterCurve or None (thrust magnitude)
        self.pwm = None

    def set_command(self, fx, fy):
        """
        Extended thrust (fx, fy) -> magnitude and angle (Fossen eq. 11.24); magnitude saturated.
        The direction of (fx, fy) is the angle set-point even when the magnitude is below the
        deadband (the steering servo is independent of the throttle), so an allocation can turn
        the azimuth without thrust by sending a tiny vector.
        """
        magnitude = math.hypot(fx, fy)
        alpha_d = math.atan2(fy, fx) if magnitude > 0.0 else self.command[1]
        if self.pwm_curve is not None:
            self.pwm = self.pwm_curve.to_pwm(min(magnitude, self.max_force))
            magnitude = self.pwm_curve.force(self.pwm)
        elif magnitude < self.deadband:
            self.command = (0.0, alpha_d)
            return 0.0, 0.0
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
    if "pwm" in t:  # ESC model: {neutral, deadzone, half_range} [us]
        common["pwm"] = PwmThrusterCurve(common["max_force"], common["reverse_ratio"], **t["pwm"])
    if t["type"] == "fixed":
        return FixedThruster(**common)
    rate = t.get("max_rate_deg_s")
    return AzimuthThruster(max_rate=math.radians(rate) if rate is not None else np.inf, **common)
