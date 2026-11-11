"""
System identification for the vessel.yaml parameters from tests on the real boat (Fossen 2021, Ch. 6-7
and TTK4190 L3b; the same quantities MSS otter.m is calibrated with). Each function takes plain arrays,
so it can be checked against the simulator ("virtual tank test", test_identification.py) before it is
used on logged data (tools/identify.py reads the rosbags).

    bollard_curve    thrust vs. ESC pulse (bollard pull)       -> max_force, reverse_ratio, deadzone
    surge_speeds     steady speed at several thrust levels     -> X_u, X_|u|u  (max_speed for a given thrust)
    surge_response   speed after thrust steps (time series)    -> M_11 (incl. added mass), X_u, X_|u|u
    roll_decay       free roll after pushing one side down     -> natural frequency, zeta, kappa
    latency          sensor header stamp vs. arrival time      -> latency.<sensor>

Units: SI (N, m, s, rad), pulse widths in microseconds.
"""

import math

import numpy as np
from scipy.optimize import least_squares


# --- thrusters --------------------------------------------------------------------------------------

def bollard_curve(pwm, thrust, neutral=1500, half_range=400):
    """
    Fit the ESC/propeller curve of actuators.PwmThrusterCurve to bollard-pull data:

        f = F_max (s / s_max)^2,            s = pwm - (neutral + deadzone) > 0
        f = -r F_max (s / s_max)^2,         s = (neutral - deadzone) - pwm > 0,   s_max = half_range - deadzone

    Returns dict(max_force, reverse_ratio, deadzone) [N, -, us].
    """
    pwm, thrust = np.asarray(pwm, dtype=float), np.asarray(thrust, dtype=float)

    def model(p, x):
        fmax, r, dz = p
        s_max = half_range - dz
        fwd = np.clip(x - (neutral + dz), 0, s_max) / s_max
        rev = np.clip((neutral - dz) - x, 0, s_max) / s_max
        return fmax * fwd**2 - r * fmax * rev**2

    p0 = [max(np.max(np.abs(thrust)), 1.0), 0.8, 25.0]
    sol = least_squares(lambda p: model(p, pwm) - thrust, p0, bounds=([0, 0, 0], [np.inf, 2.0, 150.0]))
    fmax, r, dz = sol.x
    return {"max_force": float(fmax), "reverse_ratio": float(r), "deadzone": float(dz)}


# --- surge --------------------------------------------------------------------------------------------

def surge_damping(u, Xu, Xuu, decay=3.0):
    """Surge damping force magnitude as in vessel.yaml (MSS otter.m): X_u e^(-decay|u|) u + X_|u|u |u| u."""
    u = np.asarray(u, dtype=float)
    return Xu * np.exp(-decay * np.abs(u)) * u + Xuu * np.abs(u) * u


def surge_speeds(thrust, speed, decay=3.0):
    """
    Steady state: thrust = X_u e^(-decay u) u + X_|u|u |u| u at each steady speed (linear least squares).
    Use the thrust the propellers give at that speed (bollard curve, thrust drop with speed).
    Returns dict(X_u, X_uu). The time constant needs the mass as well, see surge_response.
    """
    u, F = np.asarray(speed, dtype=float), np.asarray(thrust, dtype=float)
    A = np.column_stack([np.exp(-decay * np.abs(u)) * u, np.abs(u) * u])
    (Xu, Xuu), *_ = np.linalg.lstsq(A, F, rcond=None)
    return {"X_u": float(Xu), "X_uu": float(Xuu)}


def surge_response(t, speed, thrust, decay=3.0, x0=None):
    """
    Fit M_11 u_dot = F(t) - X_u e^(-decay|u|) u - X_|u|u |u| u to a logged speed after thrust changes
    (simulation-error method: the model is integrated with the logged thrust and compared with the log).
    Several step sizes (small ones for X_u, large for X_|u|u) make all three identifiable.
    Returns dict(M_11, X_u, X_uu, surge_time_constant = M_11 / X_u).
    """
    t, u_log, F = (np.asarray(a, dtype=float) for a in (t, speed, thrust))

    def simulate(p):
        m, Xu, Xuu = p
        u = np.empty_like(u_log)
        u[0] = u_log[0]
        for k in range(len(t) - 1):
            h = t[k + 1] - t[k]
            f = lambda v: (F[k] - surge_damping(v, Xu, Xuu, decay)) / m
            k1 = f(u[k]); k2 = f(u[k] + 0.5 * h * k1); k3 = f(u[k] + 0.5 * h * k2); k4 = f(u[k] + h * k3)
            u[k + 1] = u[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return u

    p0 = x0 if x0 is not None else [30.0, 5.0, 5.0]
    sol = least_squares(lambda p: simulate(p) - u_log, p0, bounds=([1e-3, 0, 0], [np.inf, np.inf, np.inf]),
                        x_scale="jac")
    m, Xu, Xuu = sol.x
    return {"M_11": float(m), "X_u": float(Xu), "X_uu": float(Xuu), "surge_time_constant": float(m / Xu)}


# --- roll decay -----------------------------------------------------------------------------------------

def roll_decay(t, phi):
    """
    Free roll decay (push one side down, let go). From the peaks of |phi|:

        damped period T_d = 2 x mean time between successive peaks of the same sign
        equivalent damping per half cycle: zeta_eq = delta / sqrt(pi^2 + delta^2),  delta = ln(A_i / A_i+1)
        with K = D44 (1 + kappa |p|) p, zeta_eq(A) = zeta (1 + kappa 8/(3 pi) omega_n A)   (equivalent
        linearization of the quadratic term), so a line through zeta_eq vs. amplitude gives zeta and kappa.

    Returns dict(omega_n [rad/s], period [s], zeta, kappa).
    """
    t, phi = np.asarray(t, dtype=float), np.asarray(phi, dtype=float)
    phi = phi - np.mean(phi[-max(len(phi) // 10, 1):])           # settle level
    a = np.abs(phi)
    peaks = [i for i in range(1, len(a) - 1) if a[i] >= a[i - 1] and a[i] > a[i + 1] and a[i] > 0.02 * a.max()]
    # one peak per half cycle (sign changes between them)
    half = [peaks[0]]
    for i in peaks[1:]:
        if np.sign(phi[i]) != np.sign(phi[half[-1]]):
            half.append(i)
        elif a[i] > a[half[-1]]:
            half[-1] = i
    if len(half) < 3:
        raise ValueError("need at least three roll peaks")
    tp, A = t[half], a[half]
    T_d = 2.0 * float(np.mean(np.diff(tp)))
    delta = np.log(A[:-1] / A[1:])
    zeta_eq = delta / np.sqrt(math.pi**2 + delta**2)
    amp = 0.5 * (A[:-1] + A[1:])
    slope, zeta = np.polyfit(amp, zeta_eq, 1) if len(amp) > 1 else (0.0, float(zeta_eq[0]))
    zeta = float(max(zeta, 1e-6))
    omega_n = 2 * math.pi / T_d / math.sqrt(1 - zeta**2)
    kappa = float(slope / (zeta * 8 / (3 * math.pi) * omega_n)) if len(amp) > 1 else 0.0
    return {"omega_n": omega_n, "period": 2 * math.pi / omega_n, "zeta": zeta, "kappa": kappa}


# --- latency --------------------------------------------------------------------------------------------

def latency(stamp, received):
    """Delay between the measurement time (header.stamp) and the arrival time [s]: mean, std, max."""
    d = np.asarray(received, dtype=float) - np.asarray(stamp, dtype=float)
    return {"mean": float(np.mean(d)), "std": float(np.std(d)), "max": float(np.max(d)), "n": int(d.size)}
