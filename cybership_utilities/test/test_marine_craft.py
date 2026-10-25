"""
Unit tests for the Fossen marine craft model (no ROS needed):

    cd ~/ros_ws/src/cybership_software_suite && python3 -m pytest -v cybership_utilities/test
"""

import math
import os

import numpy as np
import pytest
import yaml

from cybership_utilities import vessel_model
from cybership_utilities.marine_craft import Rzyx, J, m2c, cross_flow_drag, MarineCraft

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def config(name):
    with open(os.path.join(REPO, "cybership_config", "config", name, "vessel.yaml")) as f:
        return yaml.safe_load(f)


def craft(name):
    return vessel_model.make_marine_craft(vessel_model.model_parameters(config(name)))


def run(model, tau, seconds, dt=0.01):
    for _ in range(int(round(seconds / dt))):
        model.step(tau=np.asarray(tau, dtype=float), dt=dt)
    return model


# --- Kinematics -------------------------------------------------------------------------------

def test_rotation_matrix_is_orthonormal():
    rng = np.random.default_rng(1)
    for phi, theta, psi in rng.uniform(-1.2, 1.2, size=(20, 3)):
        R = Rzyx(phi, theta, psi)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_surge_moves_east_when_heading_east():
    eta = np.array([0, 0, 0, 0, 0, math.pi / 2])
    nu = np.array([1.0, 0, 0, 0, 0, 0])
    assert np.allclose(J(eta) @ nu, [0, 1, 0, 0, 0, 0], atol=1e-12)


# --- Dynamics ---------------------------------------------------------------------------------

def test_coriolis_is_skew_symmetric():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 6))
    M = A @ A.T + 6 * np.eye(6)
    C = m2c(M, rng.normal(size=6))
    assert np.allclose(C + C.T, 0.0, atol=1e-12)


def test_kinetic_energy_is_conserved_without_damping():
    M_RB = np.diag([30.2, 30.2, 30.2, 4.18, 2.72, 2.72])
    M_A = np.diag([2.5, 129.8, 35.7, 2.7, 0.9, 6.4])
    model = MarineCraft(M_RB, M_A, np.zeros((6, 6)), np.zeros((6, 6)),
                        nu0=np.array([0.5, 0.2, 0, 0, 0, 0.3]))
    energy = lambda m: 0.5 * m.nu @ m.M @ m.nu
    e0 = energy(model)
    run(model, np.zeros(6), 10.0)
    assert energy(model) == pytest.approx(e0, rel=1e-4)


def test_cross_flow_drag_opposes_motion():
    sway = cross_flow_drag(1.0, 0.16, 0.27, np.array([0, 0.5, 0, 0, 0, 0]))
    yaw = cross_flow_drag(1.0, 0.16, 0.27, np.array([0, 0, 0, 0, 0, 0.5]))
    assert sway[1] < 0 and abs(sway[5]) < 1e-9   # pure sway: no yaw moment (symmetric strips)
    assert yaw[5] < 0 and abs(yaw[1]) < 1e-9     # pure yaw: no sway force


def test_rigid_body_coriolis_matches_fossen_3dof():
    """C_RB(nu) nu in surge/sway must be the centripetal terms -m v r and m u r (Fossen 2021, Ch. 3)."""
    m = 24.0
    M_RB = np.diag([m, m, m, 1.0, 1.0, 2.0])
    u, v, r = 0.5, 0.2, 0.3
    f = m2c(M_RB, np.array([u, v, 0, 0, 0, r])) @ np.array([u, v, 0, 0, 0, r])
    assert f[0] == pytest.approx(-m * v * r) and f[1] == pytest.approx(m * u * r)


def test_hull_model_matches_shoeboxpy_in_straight_line():
    """
    Without a 'model' section the new model must reproduce the old shoeboxpy simulator.
    Only for straight-line motion: shoeboxpy's C_RB has no centripetal force (C_RB(nu) nu = 0
    in surge/sway), so the two differ as soon as the vessel turns. The new model follows Fossen.
    """
    shoeboxpy = pytest.importorskip("shoeboxpy.model6dof")
    hull = config("voyager")["hull"]
    old = shoeboxpy.Shoebox(L=hull["length"], B=hull["beam"], T=hull["draft"],
                            GM_phi=hull["GM_phi"], GM_theta=hull["GM_theta"])
    new = craft("voyager")
    tau = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(500):
        old.step(tau=tau, dt=0.01)
        new.step(tau=tau, dt=0.01)
    assert np.allclose(new.eta, old.eta, atol=1e-9)
    assert np.allclose(new.nu, old.nu, atol=1e-9)


# --- Microampere: physically reasonable -------------------------------------------------------

def test_microampere_top_speed_from_surge_resistance():
    """Two stern thrusters at 67 N give max_speed (X_|u|u = max_thrust / max_speed^2, Fossen's Otter)."""
    d = config("microampere")["model"]["damping"]
    model = run(craft("microampere"), [d["max_thrust"], 0, 0, 0, 0, 0], 60.0)
    # the linear term has faded to e^(-3 * 3.15) ~ 1e-4 of X_u at top speed
    assert model.nu[0] == pytest.approx(d["max_speed"], rel=0.005)
    assert 2.5 < model.nu[0] < 3.5


def test_surge_damping_blends_linear_into_quadratic():
    """
    X = -(X_u e^(-3|u|) u + X_|u|u |u| u) (MSS otter.m): the linear term X_u = M11 / T_surge at low
    speed, only the quadratic term at high speed, so the damping is not counted twice.
    """
    model = craft("microampere")
    d = config("microampere")["model"]["damping"]
    Xu = model.M[0, 0] / d["surge_time_constant"]
    Xuu = d["max_thrust"] / d["max_speed"] ** 2
    for u in (0.01, 0.2, 1.0, 3.0, -0.5):
        X = model.damping(np.array([u, 0, 0, 0, 0, 0]))[0]
        assert X == pytest.approx(-(Xu * math.exp(-3 * abs(u)) * u + Xuu * abs(u) * u), rel=1e-9)
    # low speed: linear dominates, high speed: linear part below 0.1 % of the total
    slow = model.damping(np.array([0.02, 0, 0, 0, 0, 0]))[0]
    assert slow == pytest.approx(-Xu * 0.02, rel=0.1)
    fast = model.damping(np.array([3.0, 0, 0, 0, 0, 0]))[0]
    assert fast == pytest.approx(-Xuu * 9.0, rel=1e-3)


def test_surge_time_constant_at_low_speed():
    """Small thrust step: the response near u = 0 has time constant M11 / X_u = T_surge."""
    model = craft("microampere")
    T = config("microampere")["model"]["damping"]["surge_time_constant"]
    Xu = model.M[0, 0] / T
    F = 0.05 * Xu          # u_ss ~ 0.05 m/s, where e^(-3u) ~ 0.86 and the quadratic term is small
    dt, t, u63 = 0.01, 0.0, None
    u_ss = run(craft("microampere"), [F, 0, 0, 0, 0, 0], 60.0).nu[0]
    while u63 is None:
        model.step(tau=np.array([F, 0, 0, 0, 0, 0]), dt=dt)
        t += dt
        if model.nu[0] >= 0.632 * u_ss:
            u63 = t
    assert u63 == pytest.approx(T, rel=0.25)


def test_microampere_sideways_and_turning_are_slow():
    side = run(craft("microampere"), [0, 67, 0, 0, 0, 0], 30.0)
    assert 0.1 < side.nu[1] < 0.5            # [m/s] full azimuth sideways
    spin = run(craft("microampere"), [0, 0, 0, 0, 0, 63.7], 30.0)
    assert math.radians(20) < spin.nu[5] < math.radians(80)  # [rad/s] max yaw moment


def test_microampere_roll_period():
    """Free roll decay: period 2 pi / omega_d with omega_n = sqrt(G44 / (Ix + A44))."""
    params = vessel_model.model_parameters(config("microampere"))
    model = vessel_model.make_marine_craft(params, eta0=np.radians([0, 0, 0, 5, 0, 0]))
    m44 = params["MRB"][3] + params["MA"][3]
    wn = math.sqrt(params["G"][3] / m44)
    zeta = params["D"][3] / (2 * m44 * wn)
    expected = 2 * math.pi / (wn * math.sqrt(1 - zeta**2))
    phi, t = [], []
    for k in range(400):
        model.step(dt=0.005)
        phi.append(model.eta[3]); t.append((k + 1) * 0.005)
    crossings = [t[i] for i in range(1, len(phi)) if phi[i - 1] > 0 >= phi[i]]
    assert len(crossings) >= 2
    assert crossings[1] - crossings[0] == pytest.approx(expected, rel=0.05)


def test_microampere_is_stable_at_top_speed():
    """A small roll/pitch disturbance at full speed must not grow (needs the Munk cancellation)."""
    model = vessel_model.make_marine_craft(vessel_model.model_parameters(config("microampere")),
                                           eta0=np.radians([0, 0, 0, 3, 3, 0]))
    run(model, [201, 0, 0, 0, 0, 0], 30.0)
    assert np.all(np.isfinite(model.nu))
    assert abs(model.eta[3]) < math.radians(1) and abs(model.eta[4]) < math.radians(1)
    d = config("microampere")["model"]["damping"]
    Xuu = d["max_thrust"] / d["max_speed"] ** 2
    assert model.nu[0] == pytest.approx(math.sqrt(201 / Xuu), abs=0.05)  # all three thrusters forward


# --- Ocean current (Fossen 2021, Ch. 10.3) ----------------------------------------------------

def test_vessel_drifts_with_the_current_without_turning():
    """
    No thrust in a constant current: the vessel must end up moving with the current (nu_r -> 0)
    and its heading must not change (an irrotational current gives no yaw moment, Property 10.1).
    """
    model = vessel_model.make_marine_craft(vessel_model.model_parameters(config("microampere")),
                                           eta0=np.radians([0, 0, 0, 0, 0, 30]) * [0, 0, 0, 0, 0, 1])
    model.current_ned = np.array([0.2, 0.0, 0.0])    # 0.2 m/s towards north
    run(model, np.zeros(6), 120.0)
    velocity_ned = J(model.eta)[0:3, 0:3] @ model.nu[0:3]
    assert velocity_ned[0:2] == pytest.approx([0.2, 0.0], abs=0.005)
    assert math.degrees(model.eta[5]) == pytest.approx(30.0, abs=0.1)
