"""
Unit tests for the sensor models (Fossen 2021, Ch. 13.1/14.1), no ROS needed.
"""

import math

import numpy as np
import pytest

from cybership_utilities.marine_craft.sensors import Compass, Gnss, Gyro, flat2llh, llh2flat, make_sensors


def run(sensor, eta, nu, seconds, dt=0.01):
    out = []
    for k in range(int(round(seconds / dt))):
        m = sensor.update(k * dt, eta, nu)
        if m is not None:
            out.append(m)
    return out


def test_flat_earth_round_trip_trondheim():
    lon0, lat0 = math.radians(10.3951), math.radians(63.4305)
    lon, lat, h = flat2llh(2000.0, 5000.0, 50.0, lon0, lat0)
    assert llh2flat(lon, lat, h, lon0, lat0) == pytest.approx((2000.0, 5000.0, 50.0), abs=1e-6)
    # 1 m north ~ 1/111 km of latitude
    _, lat1, _ = flat2llh(1.0, 0.0, 0.0, lon0, lat0)
    assert math.degrees(lat1 - lat0) == pytest.approx(1 / 111_400, rel=0.01)


def test_sensor_rates():
    eta, nu = np.zeros(6), np.zeros(6)
    assert len(run(Gnss(rate=5.0), eta, nu, 10.0)) == 50
    assert len(run(Compass(rate=50.0), eta, nu, 10.0)) == 500
    assert len(run(Gyro(rate=50.0), eta, nu, 10.0)) == 500


def test_gnss_noise_and_lever_arm():
    """Antenna 0.3 m above CO: rolling 10 deg starboard down leans it 0.3 sin(10 deg) to starboard (+y)."""
    rng = np.random.default_rng(1)
    g = Gnss(rate=5.0, sigma=0.17, lever_arm=(0, 0, -0.3), rng=rng)
    eta = np.array([10.0, -5.0, 0.0, math.radians(10), 0.0, 0.0])
    p = np.array([m["ned"][0:2] for m in run(g, eta, np.zeros(6), 2000.0, dt=0.2)])
    assert p.mean(axis=0) == pytest.approx([10.0, -5.0 + 0.3 * math.sin(math.radians(10))], abs=0.02)
    assert p.std(axis=0) == pytest.approx([0.17, 0.17], rel=0.05)


def test_gnss_bias_wanders_slowly():
    """Gauss-Markov bias: stationary std bias_sigma, correlated over bias_tau (non-RTK GNSS wander)."""
    g = Gnss(rate=5.0, sigma=0.0, bias_sigma=0.5, bias_tau=60.0, rng=np.random.default_rng(2))
    p = np.array([m["ned"][0] for m in run(g, np.zeros(6), np.zeros(6), 20000.0, dt=0.2)])
    assert p.std() == pytest.approx(0.5, rel=0.15)
    lag = 5  # 1 s: almost the same
    assert np.corrcoef(p[:-lag], p[lag:])[0, 1] > 0.95


def test_compass_and_gyro():
    rng = np.random.default_rng(3)
    eta = np.array([0, 0, 0, 0, 0, math.radians(179.9)])
    nu = np.array([0, 0, 0, 0, 0, 0.1])
    psi = np.array([m["psi"] for m in run(Compass(sigma_deg=0.5, rng=rng), eta, nu, 100.0)])
    assert np.all(np.abs(psi) <= math.pi)                       # wrapped to [-pi, pi)
    err = (psi - eta[5] + math.pi) % (2 * math.pi) - math.pi     # smallest signed angle
    assert np.degrees(err.std()) == pytest.approx(0.5, rel=0.1)
    r = np.array([m["r"] for m in run(Gyro(sigma_deg_s=0.1, bias_deg_s=0.5, rng=rng), eta, nu, 100.0)])
    assert np.degrees(r.mean() - 0.1) == pytest.approx(0.5, abs=0.02)   # constant bias


def test_make_sensors_from_config():
    s = make_sensors({"sensors": {"gnss": {"rate": 5.0, "lever_arm": [0, 0, -0.3]},
                                  "compass": {"rate": 50.0}}}, seed=0)
    assert set(s) == {"gnss", "compass"}
    assert make_sensors({}) == {}
