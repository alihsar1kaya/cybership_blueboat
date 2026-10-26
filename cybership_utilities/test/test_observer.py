"""
Unit tests for the DP Kalman filter (Fossen 2021, Ch. 13.4.6), no ROS needed: a simulated vessel,
simulated sensors at their own rates, and the filter at 50 Hz.
"""

import math
import os

import numpy as np
import pytest

from cybership_utilities import vessel_model
from cybership_utilities.marine_craft import MarineCraft, Rzyx
from cybership_utilities.marine_craft.sensors import Compass, Gnss
from cybership_utilities.observer import DPKalmanFilter, remove_lever_arm, ssa

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IDX = [0, 1, 5]


def microampere():
    path = os.path.join(REPO, "cybership_config", "config", "microampere", "vessel.yaml")
    return vessel_model.model_parameters(vessel_model.load_vessel_config(path=path)[0])


Q_BIAS, Q_NU = [2.0, 2.0, 0.2], [0.5, 0.5, 0.05]   # as in controller/observer config


def kf_for(params, nonlinear=True, M=None, D=None):
    damping = None
    if nonlinear:
        craft = vessel_model.make_marine_craft(params)
        damping = lambda nu: -craft.damping(np.array([nu[0], nu[1], 0, 0, 0, nu[2]]))[IDX]
    M = vessel_model.M_eff_3dof(params) if M is None else M
    D = vessel_model.D_3dof(params) if D is None else D
    return DPKalmanFilter(M, D, q_bias=Q_BIAS, q_nu=Q_NU, damping=damping)


def run(craft, kf, tau_fn, seconds, tau_ext_ned=np.zeros(3), psi0=0.0, seed=0, h=0.02):
    """Simulate the craft (dt 0.01) with GNSS 5 Hz and compass 50 Hz; KF predicts at 1/h."""
    rng = np.random.default_rng(seed)
    gnss = Gnss(rate=5.0, sigma=0.17, rng=rng)
    compass = Compass(rate=50.0, sigma_deg=0.5, rng=rng)
    craft.eta = np.array([0, 0, 0, 0, 0, psi0])
    kf.x[kf.nw + 2] = psi0
    log = []
    steps = int(round(h / 0.01))
    for k in range(int(round(seconds / h))):
        t = k * h
        tau3 = tau_fn(t)
        R = Rzyx(0, 0, craft.eta[5])
        f_ext = np.r_[R.T @ np.r_[tau_ext_ned[0:2], 0.0], 0, 0, tau_ext_ned[2]]
        for _ in range(steps):
            craft.step(np.array([tau3[0], tau3[1], 0, 0, 0, tau3[2]]), 0.01, tau_ext=f_ext)
        g = gnss.update(t, craft.eta)
        if g is not None:
            kf.correct_gnss(g["ned"][0], g["ned"][1], 0.17)
        c = compass.update(t, craft.eta)
        if c is not None:
            kf.correct_compass(c["psi"], math.radians(0.5))
        log.append((t, *craft.eta[IDX], *craft.nu[IDX], *kf.eta, *kf.nu, *kf.bias))
        kf.predict(tau3, h)
    return np.array(log)


def test_bias_converges_to_an_unknown_constant_force():
    """
    Truth = the filter's own linear model plus a constant NED force the filter is not told about
    (like a current): the bias state must converge to that force, positions/velocities to the truth.
    """
    p = microampere()
    M3 = np.diag(np.diag(vessel_model.M_eff_3dof(p)))
    D3 = np.diag(np.diag(vessel_model.D_3dof(p)))
    M = np.diag([30.2, 30.2, 30.2, 4.0, 3.0, 3.0])
    D = np.diag([0, 0, 200.0, 20.0, 20.0, 0])
    M[np.ix_(IDX, IDX)], D[np.ix_(IDX, IDX)] = M3, D3
    craft = MarineCraft(M, np.zeros((6, 6)), D, np.diag([0, 0, 2581.0, 300.0, 150.0, 0]))
    kf = kf_for(p, nonlinear=False, M=M3, D=D3)
    force = np.array([3.0, 0.0, 0.0])   # [N, N, N m] in NED, along the heading (no turning moments)
    hold = lambda t: np.array([0.0, 0.0, 0.0])
    log = run(craft, kf, hold, 300.0, tau_ext_ned=force)
    last = log[log[:, 0] > 250]
    assert last[:, 13:16].mean(axis=0) == pytest.approx(force, abs=0.5)   # a lively bias (q_bias)
    assert np.sqrt(np.mean((last[:, 7:9] - last[:, 1:3]) ** 2)) < 0.1      # better than GNSS 0.17 m
    assert np.sqrt(np.mean((last[:, 10:13] - last[:, 4:7]) ** 2)) < 0.05


def test_full_microampere_manoeuvring():
    """Full nonlinear model, manoeuvring up to ~0.7 m/s: position better than the raw GNSS (0.17 m),
    heading within 1 deg, velocity within 0.05 m/s (nonlinear damping in the predictor)."""
    p = microampere()
    craft = vessel_model.make_marine_craft(p)
    kf = kf_for(p)
    tau = lambda t: np.array([8.0 * math.sin(0.05 * t), 6.0 * math.sin(0.07 * t), 1.5 * math.sin(0.1 * t)])
    log = run(craft, kf, tau, 300.0)
    last = log[log[:, 0] > 60]
    assert np.sqrt(np.mean(np.sum((last[:, 7:9] - last[:, 1:3]) ** 2, axis=1))) < 0.12
    assert np.degrees(np.abs(ssa(last[:, 9] - last[:, 3])).max()) < 1.0
    assert np.sqrt(np.mean((last[:, 10:12] - last[:, 4:6]) ** 2)) < 0.05


def test_full_microampere_stationkeeping_in_current():
    """
    Low-speed (DP) motion in a 0.15 m/s cross current: the filter does not know the current, the bias
    takes it (Fossen 13.4.6), position stays better than the raw GNSS and the bias points with the
    current (east).
    """
    p = microampere()
    craft = vessel_model.make_marine_craft(p)
    craft.current_ned = np.array([0.0, 0.15, 0.0])
    kf = kf_for(p)
    # thrust that roughly holds the boat against the current, plus small manoeuvres
    tau = lambda t: np.array([1.0 * math.sin(0.05 * t), -25.0 + 2.0 * math.sin(0.07 * t), 0.2 * math.sin(0.1 * t)])
    log = run(craft, kf, tau, 300.0)
    last = log[log[:, 0] > 120]
    assert np.sqrt(np.mean(np.sum((last[:, 7:9] - last[:, 1:3]) ** 2, axis=1))) < 0.17
    assert np.sqrt(np.mean((last[:, 10:12] - last[:, 4:6]) ** 2)) < 0.05
    assert last[:, 14].mean() > 10.0          # bias force towards east [N]


def test_heading_estimate_does_not_jump_at_180_deg():
    """Turning through +-180 deg: the ssa innovation keeps the estimate continuous (Ch. 13.4.3)."""
    p = microampere()
    craft = vessel_model.make_marine_craft(p)
    kf = kf_for(p)
    log = run(craft, kf, lambda t: np.array([0.0, 0.0, 3.0]), 40.0, psi0=math.radians(170))
    err = np.degrees(np.abs([ssa(a) for a in log[:, 9] - log[:, 3]]))
    assert err[log[:, 0] > 2].max() < 2.0
    assert np.abs(np.diff(np.unwrap(log[:, 9]))).max() < math.radians(5)


def test_prediction_only_between_gnss_fixes():
    """Without measurements the covariance grows; a GNSS fix shrinks the position covariance."""
    p = microampere()
    kf = kf_for(p)
    P0 = kf.P[0, 0]
    for _ in range(50):
        kf.predict(np.zeros(3), 0.02)
    assert kf.P[0, 0] > P0
    before = kf.P[0, 0]
    kf.correct_gnss(0.0, 0.0, 0.17)
    assert kf.P[0, 0] < min(before, 0.17**2)


def test_remove_lever_arm():
    p_a = np.array([10.0, 5.0, -0.3])
    p = remove_lever_arm(p_a, [0.0, 0.0, -0.3], 0.0, 0.0, 1.0)
    assert p == pytest.approx([10.0, 5.0, 0.0])


def test_wave_filter_removes_the_wave_frequency_motion():
    """
    Boat at rest, measurements = LF position + WF oscillation at the wave frequency (Fossen 13.2,
    13.4.6): with the wave filter the LF estimate the controller uses stays quiet, without it the
    estimate follows the waves.
    """
    p = microampere()
    w0, amp = 2.7, 0.2
    rng = np.random.default_rng(4)

    def run_kf(wave_filter):
        kf = DPKalmanFilter(vessel_model.M_eff_3dof(p), vessel_model.D_3dof(p), q_bias=Q_BIAS, q_nu=Q_NU,
                            wave_filter=wave_filter)
        est = []
        for k in range(int(120 / 0.02)):
            t = k * 0.02
            wf = amp * math.sin(w0 * t)
            if k % 10 == 0:
                kf.correct_gnss(wf + 0.05 * rng.standard_normal(), 0.5 * wf, 0.05)
            kf.correct_compass(math.radians(2) * math.sin(w0 * t), math.radians(0.2))
            est.append((*kf.eta, *kf.eta_wf))
            kf.predict(np.zeros(3), 0.02)
        return np.array(est)[-2500:]

    with_wf = run_kf({"omega0": w0, "zeta": 0.1, "q": [1.0, 1.0, 0.01]})
    without = run_kf(None)
    assert with_wf[:, 0].std() < 0.3 * without[:, 0].std()       # LF north: waves filtered out
    assert with_wf[:, 0].std() < 0.03
    assert with_wf[:, 3].std() == pytest.approx(amp / math.sqrt(2), rel=0.3)   # WF state takes them
