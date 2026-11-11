"""
Virtual tank tests: the identification functions must find the simulator's own parameters again from
simulated test data, before they are trusted on logs from the real boat.

    python3 -m pytest -q cybership_utilities/test/test_identification.py
"""
import math
import os

import numpy as np
import pytest

from cybership_utilities import identification as ident
from cybership_utilities import vessel_model
from cybership_utilities.marine_craft.actuators import PwmThrusterCurve

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


@pytest.fixture(scope="module")
def params():
    path = os.path.join(REPO, "cybership_config", "config", "microampere", "vessel.yaml")
    return vessel_model.model_parameters(vessel_model.load_vessel_config(path=path)[0])


def craft(params, **kw):
    return vessel_model.make_marine_craft(params, **kw)


def test_bollard_curve_from_noisy_pull_data():
    curve = PwmThrusterCurve(67.0, 0.8, neutral=1500, deadzone=25, half_range=400)
    pwm = np.arange(1100, 1901, 10)
    thrust = np.array([curve.force(p) for p in pwm]) + np.random.default_rng(0).normal(0, 0.5, pwm.size)
    r = ident.bollard_curve(pwm, thrust)
    assert r["max_force"] == pytest.approx(67.0, rel=0.02)
    assert r["reverse_ratio"] == pytest.approx(0.8, abs=0.02)
    assert r["deadzone"] == pytest.approx(25, abs=3)


def test_surge_damping_from_steady_speeds(params):
    speeds, thrusts = [], []
    for F in (1, 3, 6, 10, 20, 40, 60):
        m = craft(params)
        for _ in range(6000):                       # 60 s to steady state
            m.step(np.array([F, 0, 0, 0, 0, 0]), 0.01)
        speeds.append(m.nu[0])
        thrusts.append(F)
    r = ident.surge_speeds(thrusts, speeds)
    assert r["X_u"] == pytest.approx(craft(params).D_L[0, 0], rel=0.01)
    assert r["X_uu"] == pytest.approx(params["nonlinear"]["surge_quadratic"], rel=0.01)


def test_surge_mass_and_time_constant_from_thrust_steps(params):
    m = craft(params)
    steps = [(0, 2.0), (40, 8.0), (80, 40.0), (120, 0.0)]       # small steps for X_u, large for X_|u|u
    t, u, F = [], [], []
    for k in range(16000):
        tk = 0.01 * k
        Fk = [f for t0, f in steps if tk >= t0][-1]
        if k % 10 == 0:
            t.append(tk), u.append(m.nu[0]), F.append(Fk)
        m.step(np.array([Fk, 0, 0, 0, 0, 0]), 0.01)
    r = ident.surge_response(t, u, F)
    truth = craft(params)
    assert r["M_11"] == pytest.approx(truth.M[0, 0], rel=0.02)
    assert r["X_u"] == pytest.approx(truth.D_L[0, 0], rel=0.02)
    assert r["surge_time_constant"] == pytest.approx(truth.M[0, 0] / truth.D_L[0, 0], rel=0.03)


def test_roll_damping_and_kappa_from_free_decay(params):
    m = craft(params, eta0=np.radians([0, 0, 0, 8, 0, 0]))
    t, phi = [], []
    for k in range(3000):
        if k % 2 == 0:                                # IMU at 50 Hz
            t.append(0.01 * k), phi.append(m.eta[3])
        m.step(np.zeros(6), 0.01)
    r = ident.roll_decay(t, phi)
    truth = craft(params)
    assert r["omega_n"] == pytest.approx(math.sqrt(truth.G[3, 3] / truth.M[3, 3]), rel=0.03)
    assert r["zeta"] == pytest.approx(0.2, rel=0.1)                          # vessel.yaml zeta[1]
    assert r["kappa"] == pytest.approx(params["nonlinear"]["roll_quadratic_factor"], rel=0.15)


def test_latency_from_stamps():
    stamp = np.arange(0, 10, 0.2)
    r = ident.latency(stamp, stamp + 0.1 + np.random.default_rng(1).normal(0, 0.005, stamp.size))
    assert r["mean"] == pytest.approx(0.1, abs=0.003)
