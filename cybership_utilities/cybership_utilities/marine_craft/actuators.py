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
Its propeller turns both ways (T200): a force can be given at alpha_d forward or at alpha_d + pi in
reverse, and the azimuth takes the one closer to where it points (no half turn when the demand changes
sign, which made it push the wrong way for ~2 s at 90 deg/s and overshoot the heading).

Operating conditions (set by the simulator every step, ThrusterConditions):
  - battery voltage: the ESC passes the battery voltage to the motor, so the same pulse gives less
    thrust on a low battery. Full thrust follows the manufacturer's thrust-voltage table (T200: 3.71,
    5.25, 6.7 kgf at 12, 16, 20 V); the driver still maps force -> pulse with the rated curve, so a low
    battery is a model error the controller has to absorb, as on the real boat.
  - inflow speed u_a along the thrust axis (Fossen 2021, Ch. 9.1): with a linear K_T(J),
        T = T_0 - sqrt(|T_0| T_max) u_a / u_0
    where T_0 is the thrust at rest (bollard) and u_0 the inflow speed at which full throttle gives no
    thrust. Moving forward reduces forward thrust; u_0 = None leaves it out (thrust independent of speed,
    as Fossen's Otter model).

Defaults (no limits, instant, rated voltage, no speed effect) reproduce the old simulator, so vessels
without actuator data behave as before.
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


def voltage_factor(voltage, table, rated_voltage):
    """Full-throttle thrust at 'voltage' relative to 'rated_voltage', from a thrust-voltage table
    {volts: [...], thrust: [...]} (linear interpolation, linear extrapolation at the ends)."""
    v, f = np.asarray(table["volts"], dtype=float), np.asarray(table["thrust"], dtype=float)

    def at(x):
        if x <= v[0]:
            return f[0] + (x - v[0]) * (f[1] - f[0]) / (v[1] - v[0])
        if x >= v[-1]:
            return f[-1] + (x - v[-1]) * (f[-1] - f[-2]) / (v[-1] - v[-2])
        return float(np.interp(x, v, f))
    return max(at(float(voltage)), 0.0) / at(float(rated_voltage))


class ThrusterConditions:
    """Battery voltage and inflow: turn the thrust of the rated curve into the thrust at these conditions."""

    def _init_conditions(self, zero_thrust_speed=None):
        self.voltage_factor = 1.0          # full thrust now / full thrust at the rated voltage
        self.inflow = 0.0                  # water speed into the propeller along the thrust axis [m/s]
        self.zero_thrust_speed = None if zero_thrust_speed is None else float(zero_thrust_speed)

    def actual(self, f):
        """Thrust delivered for the rated-curve thrust f [N]."""
        f = f * self.voltage_factor
        u0 = self.zero_thrust_speed
        if u0 is None or f == 0.0:
            return f
        f_max = self.voltage_factor * (self.max_force if f > 0 else self.reverse_ratio * self.max_force)
        u0 = u0 * math.sqrt(self.voltage_factor)   # shaft speed ~ sqrt(thrust)
        sign = 1.0 if f > 0 else -1.0
        loss = math.sqrt(abs(f) * f_max) * self.inflow * sign / u0
        return sign * float(np.clip(abs(f) - loss, 0.0, 1.3 * abs(f)))


def _first_order(f, target, time_constant, dt):
    if time_constant <= 0.0:
        return target
    return target + (f - target) * math.exp(-dt / time_constant)


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class FixedThruster(ThrusterConditions):
    def __init__(self, max_force=np.inf, reverse_ratio=1.0, deadband=0.0, time_constant=0.0, pwm=None,
                 zero_thrust_speed=None):
        self._init_conditions(zero_thrust_speed)
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
        return self.actual(self.force), 0.0, 0.0


class AzimuthThruster(ThrusterConditions):
    REVERSE_HYSTERESIS = math.radians(15)

    def __init__(self, max_force=np.inf, reverse_ratio=1.0, deadband=0.0, time_constant=0.0,
                 max_rate=np.inf, angle=0.0, pwm=None, zero_thrust_speed=None):
        self._init_conditions(zero_thrust_speed)
        self.max_force = float(max_force)
        self.reverse_ratio = float(reverse_ratio)  # reverse thrust / forward thrust (0: forward only)
        self.deadband = float(deadband)
        self.time_constant = float(time_constant)
        self.max_rate = float(max_rate)            # [rad/s]
        self.angle = float(angle)                  # actual angle alpha [rad]
        self.angle_rate = 0.0
        self.force = 0.0                           # actual thrust along the angle [N], < 0 in reverse
        self.command = (0.0, 0.0)                  # (signed F_d, alpha_d)
        self.pwm_curve = pwm                       # PwmThrusterCurve or None
        self._reverse = False                      # pushing in reverse along the angle
        self.pwm = None

    def set_command(self, fx, fy):
        """
        Extended thrust (fx, fy) -> signed thrust and angle (Fossen eq. 11.24), saturated.
        The force is given forward at alpha_d = atan2(fy, fx) or in reverse at alpha_d + pi, whichever
        angle is closer to the actual one (with REVERSE_HYSTERESIS so it does not toggle at 90 deg).
        The direction is the angle set-point even when the magnitude is below the deadband (the
        steering servo is independent of the throttle), so an allocation can turn the azimuth without
        thrust by sending a tiny vector.
        """
        magnitude = math.hypot(fx, fy)
        if magnitude > 0.0:
            alpha_fwd = math.atan2(fy, fx)
            alpha_rev = _wrap(alpha_fwd + math.pi)
            fwd_err = abs(_wrap(alpha_fwd - self.angle))
            rev_err = abs(_wrap(alpha_rev - self.angle))
            # switch mode only when the other one is clearly closer (hysteresis)
            if self.reverse_ratio > 0.0:
                if self._reverse and fwd_err < rev_err - self.REVERSE_HYSTERESIS:
                    self._reverse = False
                elif not self._reverse and rev_err < fwd_err - self.REVERSE_HYSTERESIS:
                    self._reverse = True
            if self._reverse:
                alpha_d, signed = alpha_rev, -magnitude
            else:
                alpha_d, signed = alpha_fwd, magnitude
        else:
            alpha_d, signed = self.command[1], 0.0
        signed = float(np.clip(signed, -self.reverse_ratio * self.max_force, self.max_force))
        if self.pwm_curve is not None:
            self.pwm = self.pwm_curve.to_pwm(signed)
            signed = self.pwm_curve.force(self.pwm)
        elif abs(signed) < self.deadband:
            signed = 0.0
        self.command = (signed, alpha_d)
        return signed * math.cos(alpha_d), signed * math.sin(alpha_d)

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
        f = self.actual(self.force)
        return f * math.cos(self.angle), f * math.sin(self.angle), self.angle


def make_thruster(t):
    """Build an actuator from one vessel.yaml thruster entry."""
    common = dict(
        max_force=float(t.get("max_force", np.inf)),
        reverse_ratio=float(t.get("reverse_ratio", 1.0)),
        deadband=float(t.get("deadband", 0.0)),
        time_constant=float(t.get("time_constant", 0.0)),
        zero_thrust_speed=t.get("zero_thrust_speed"),
    )
    if "pwm" in t:  # ESC model: {neutral, deadzone, half_range} [us]
        common["pwm"] = PwmThrusterCurve(common["max_force"], common["reverse_ratio"], **t["pwm"])
    if t["type"] == "fixed":
        return FixedThruster(**common)
    rate = t.get("max_rate_deg_s")
    return AzimuthThruster(max_rate=math.radians(rate) if rate is not None else np.inf, **common)
