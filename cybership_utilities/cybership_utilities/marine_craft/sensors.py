"""
Sensors (Fossen 2021, Ch. 13.1 and 14.1-14.2; TTK4190 L8 and the IMU example): what the navigation
system measures, at the sensor's own rate, with noise and slowly varying bias.

    Gnss        position of the ANTENNA:  p_a = p + R(Theta) r_a  (lever arm r_a in BODY), as
                latitude/longitude (flat Earth, MSS flat2llh.m) and local NED, rate ~5 Hz
    Compass     heading psi (true north; the declination psi = psi_m + delta is applied by the
                autopilot / AHRS, e.g. ArduPilot COMPASS_DEC), rate ~50 Hz
    Gyro        yaw rate  r_m = r + b + w,  b_dot = w_b  (Fossen eq. 14.6-14.7), rate ~50 Hz

Each error is   e = white noise N(0, sigma^2)  +  bias b,
                b_dot = -b / tau + w_b   (first-order Gauss-Markov, stationary std sigma_b),
which covers both a constant bias (tau = inf) and the slow wander of a non-RTK GNSS (tau ~ minutes).

update(t, eta, nu) returns a measurement when one is due at the sensor rate (asynchronous
sensors, L8 13.4.4), else None.
"""

import math

import numpy as np

from cybership_utilities.marine_craft.kinematics import Rzyx

# WGS-84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = 2 * _F - _F**2


def flat2llh(x_n, y_n, z_n, lon0, lat0, h0=0.0):
    """Local NED position -> (lon, lat, h), flat-Earth approximation (MSS flat2llh.m), radians."""
    Rn = _A / math.sqrt(1 - _E2 * math.sin(lat0) ** 2)
    Rm = Rn * (1 - _E2) / (1 - _E2 * math.sin(lat0) ** 2)
    lat = lat0 + x_n / (Rm + h0)
    lon = lon0 + y_n / ((Rn + h0) * math.cos(lat0))
    lon = (lon + math.pi) % (2 * math.pi) - math.pi
    return lon, lat, h0 - z_n


def llh2flat(lon, lat, h, lon0, lat0, h0=0.0):
    """Inverse of flat2llh (MSS llh2flat.m)."""
    Rn = _A / math.sqrt(1 - _E2 * math.sin(lat0) ** 2)
    Rm = Rn * (1 - _E2) / (1 - _E2 * math.sin(lat0) ** 2)
    dlon = (lon - lon0 + math.pi) % (2 * math.pi) - math.pi
    return (lat - lat0) * (Rm + h0), dlon * (Rn + h0) * math.cos(lat0), h0 - h


class _Error:
    """White noise + first-order Gauss-Markov bias, for n channels."""

    def __init__(self, sigma, bias_sigma=0.0, bias_tau=math.inf, bias0=None, rng=None, n=1):
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (n,)).copy()
        self.bias_sigma = np.broadcast_to(np.asarray(bias_sigma, dtype=float), (n,)).copy()
        self.bias_tau = float(bias_tau)
        self.rng = rng if rng is not None else np.random.default_rng()
        if bias0 is not None:
            self.bias = np.broadcast_to(np.asarray(bias0, dtype=float), (n,)).copy()
        else:
            self.bias = self.bias_sigma * self.rng.standard_normal(n)

    def sample(self, dt):
        if math.isfinite(self.bias_tau) and dt > 0:
            # exact discretization of b_dot = -b/tau + w, stationary std bias_sigma
            a = math.exp(-dt / self.bias_tau)
            self.bias = a * self.bias + self.bias_sigma * math.sqrt(1 - a * a) * self.rng.standard_normal(len(self.bias))
        return self.bias + self.sigma * self.rng.standard_normal(len(self.sigma))


class _Sensor:
    def __init__(self, rate):
        self.period = 1.0 / float(rate)
        self.t_last = None

    def due(self, t):
        if self.t_last is None or t - self.t_last >= self.period - 1e-9:
            dt = 0.0 if self.t_last is None else t - self.t_last
            self.t_last = t
            return True, dt
        return False, 0.0


class Gnss(_Sensor):
    def __init__(self, rate=5.0, sigma=0.17, bias_sigma=0.0, bias_tau=math.inf, lever_arm=(0, 0, 0),
                 origin_lat_deg=63.4305, origin_lon_deg=10.3951, rng=None):
        super().__init__(rate)
        self.lever_arm = np.asarray(lever_arm, dtype=float)
        self.lat0, self.lon0 = math.radians(origin_lat_deg), math.radians(origin_lon_deg)
        self.error = _Error(sigma, bias_sigma, bias_tau, rng=rng, n=2)

    def antenna_ned(self, eta):
        """Antenna position p + R(Theta) r_a (no noise)."""
        return eta[0:3] + Rzyx(*eta[3:6]) @ self.lever_arm

    def update(self, t, eta, nu=None):
        ok, dt = self.due(t)
        if not ok:
            return None
        p = self.antenna_ned(eta)
        p[0:2] += self.error.sample(dt)
        lon, lat, h = flat2llh(p[0], p[1], p[2], self.lon0, self.lat0)
        return {"ned": p, "lat": math.degrees(lat), "lon": math.degrees(lon), "alt": h,
                "sigma": float(np.hypot(self.error.sigma[0], self.error.bias_sigma[0]))}


class Compass(_Sensor):
    def __init__(self, rate=50.0, sigma_deg=0.5, bias_deg=0.0, rng=None):
        super().__init__(rate)
        self.error = _Error(math.radians(sigma_deg), bias0=math.radians(bias_deg), rng=rng)

    def update(self, t, eta, nu=None):
        ok, dt = self.due(t)
        if not ok:
            return None
        psi = eta[5] + self.error.sample(dt)[0]
        return {"psi": (psi + math.pi) % (2 * math.pi) - math.pi}


class Gyro(_Sensor):
    def __init__(self, rate=50.0, sigma_deg_s=0.1, bias_sigma_deg_s=0.0, bias_tau=math.inf,
                 bias_deg_s=None, rng=None):
        super().__init__(rate)
        b0 = None if bias_deg_s is None else math.radians(bias_deg_s)
        self.error = _Error(math.radians(sigma_deg_s), math.radians(bias_sigma_deg_s), bias_tau,
                            bias0=b0, rng=rng)

    def update(self, t, eta, nu):
        ok, dt = self.due(t)
        if not ok:
            return None
        return {"r": nu[5] + self.error.sample(dt)[0], "p": nu[3], "q": nu[4]}


def gnss_config(config, profile="standard"):
    """GNSS section of vessel.yaml: 'gnss' (standard) or 'gnss_<profile>' (e.g. gnss_rtk)."""
    s = config.get("sensors") or {}
    key = "gnss" if profile in ("", "standard") else f"gnss_{profile}"
    if key not in s:
        raise KeyError(f"vessel.yaml has no sensors.{key}")
    return s[key]


def make_sensors(config, seed=None, gnss_profile="standard"):
    """Sensors from a vessel.yaml 'sensors' section; missing sensors are left out."""
    rng = np.random.default_rng(seed)
    s = config.get("sensors") or {}
    out = {}
    if "gnss" in s:
        out["gnss"] = Gnss(rng=rng, **gnss_config(config, gnss_profile))
    if "compass" in s:
        out["compass"] = Compass(rng=rng, **s["compass"])
    if "gyro" in s:
        out["gyro"] = Gyro(rng=rng, **s["gyro"])
    return out
