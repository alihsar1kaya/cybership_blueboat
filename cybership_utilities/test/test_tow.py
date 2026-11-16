import math
import os

import numpy as np
import pytest

from cybership_utilities import vessel_model
from cybership_utilities.marine_craft.tow import RHO, LoadCell, TowedBody, TowedPointMass, tow_wrench, towed_from_config

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _tow_straight(U, seconds=60.0, dt=0.01, **kw):
    tb = TowedBody(**kw)
    tb.place_behind([0.0, 0.0], 0.0)
    p, tension = np.zeros(2), []
    for _ in range(int(seconds / dt)):
        p1 = p + np.array([U * dt, 0.0])
        tb.step(dt, p, p1)
        p = p1
        tension.append(tb.tension)
    return tb, p, np.array(tension)


@pytest.mark.parametrize("U", [0.3, 0.6])
def test_steady_tension_equals_the_drag(U):
    tb, p, T = _tow_straight(U)
    drag = 0.5 * RHO * tb.cd_x * tb.B * tb.T * U * U + tb.d_lin[0] * U
    assert T[-500:].mean() == pytest.approx(drag, rel=1e-3)
    assert tb.eta[1] == pytest.approx(0.0, abs=1e-6)          # straight behind
    assert p[0] - tb.eta[0] == pytest.approx(tb.line_length + tb.L / 2, abs=0.01)


def test_slack_line_pulls_nothing():
    tb = TowedBody()
    tb.place_behind([0.0, 0.0], 0.0)
    force = tb.step(0.01, [0.0, 0.0], [-0.005, 0.0])          # tow point moves towards the body
    assert np.allclose(force, 0.0) and tb.tension == 0.0


def test_sudden_start_gives_a_snatch_load():
    tb = TowedBody()
    tb.place_behind([0.0, 0.0], 0.0)
    p, T = np.zeros(2), []
    for k in range(1000):
        U = min(0.6, 0.6 * k * 0.01)
        p1 = p + np.array([U * 0.01, 0.0])
        tb.step(0.01, p, p1)
        p = p1
        T.append(tb.tension)
    steady = 0.5 * RHO * tb.cd_x * tb.B * tb.T * 0.36 + tb.d_lin[0] * 0.6
    assert max(T) > 5 * steady                               # the line jerks the load into motion
    assert T[-1] == pytest.approx(steady, rel=0.05)


def test_body_follows_inside_a_turn():
    tb = TowedBody()
    tb.place_behind([0.0, 0.0], 0.0)
    R, U, th, p = 8.0, 0.5, 0.0, np.zeros(2)
    for _ in range(20000):
        th += U / R * 0.01
        p1 = np.array([R * math.sin(th), R - R * math.cos(th)])
        tb.step(0.01, p, p1)
        p = p1
    radius = math.hypot(tb.eta[0], tb.eta[1] - R)
    assert R - 1.0 < radius < R                               # cuts inside the vessel's circle
    assert tb.nu[2] == pytest.approx(U / R, rel=0.02)         # turns with it


def test_wrench_at_the_stern():
    # pulled straight aft: no yaw moment; pulled to starboard at the stern: the bow swings to port
    w = tow_wrench([-10.0, 0.0], 0.0, (-0.6, 0.0, -0.1))
    assert w[0] == pytest.approx(-10.0) and w[5] == pytest.approx(0.0)
    w = tow_wrench([0.0, 10.0], 0.0, (-0.6, 0.0, -0.1))
    assert w[1] == pytest.approx(10.0) and w[5] == pytest.approx(-6.0)


def _vessel_circles():
    return [(-0.3, 0.0, 0.47), (0.0, 0.0, 0.47), (0.3, 0.0, 0.47)]


def test_body_bumps_into_the_stern_instead_of_passing_through():
    # vessel still at the origin heading north; the body drifts towards it on a following current
    tb = TowedBody()
    tb.place_behind([-0.6, 0.0], 0.0)
    for _ in range(int(120 / 0.01)):
        tb.step_with_vessel(0.01, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.6, 0.0), current=(0.15, 0.0),
                            vessel_circles=_vessel_circles())
    bow = tb.eta[0] + tb.L / 2
    assert bow < -0.3 + 0.47 + 0.05          # stopped at the stern circle, not ahead of the vessel
    assert tb.contact_force > 0.0


def test_body_follows_the_wave_particle_motion():
    # a regular wave travelling north: the free body (slack line) oscillates with the water
    w, A = 2 * math.pi / 2.3, 0.05
    k = w * w / 9.81
    tb = TowedBody(line_length=50.0)
    tb.place_behind([0.0, 0.0], 0.0)
    t, us = 0.0, []
    for _ in range(int(60 / 0.01)):
        def waves(dx, dy, t=t):
            th = w * t - k * dx
            return np.array([w * A * math.cos(th), 0.0]), np.array([-w * w * A * math.sin(th), 0.0])
        tb.step_with_vessel(0.01, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.6, 0.0), waves=waves)
        t += 0.01
        us.append(tb.nu[0])
    us = np.array(us[-2000:])
    # surge velocity amplitude close to the particle velocity w A (the position also drifts slowly, as
    # the quadratic drag rectifies the orbit)
    assert 0.8 * w * A < (us.max() - us.min()) / 2 < 1.25 * w * A


# --- point mass on a towbar -----------------------------------------------------------------------------


def _move_tow_point(tb, velocity, seconds, dt=0.01, p=None):
    """Move the tow point at `velocity` (reached over the first second) and log distance and bar force."""
    p = np.asarray(tb.p + tb.line_length * tb._dir if p is None else p, float)
    dist, force = [], []
    for k in range(int(seconds / dt)):
        p1 = p + min(1.0, k * dt) * np.asarray(velocity) * dt
        tb.step(dt, p, p1)
        p = p1
        dist.append(np.hypot(*(p - tb.p)))
        force.append(tb.tension)
    return p, np.array(dist), np.array(force)


@pytest.mark.parametrize("U", [0.3, 0.6])
def test_point_mass_steady_bar_force_equals_the_drag(U):
    tb = TowedPointMass()
    tb.place_behind([0.0, 0.0], 0.0)
    _, dist, f = _move_tow_point(tb, (U, 0.0), 40.0, p=[0.0, 0.0])
    drag = 0.5 * RHO * tb.drag_area * U * U + tb.d_lin * U
    assert f[-500:].mean() == pytest.approx(drag, rel=1e-3)
    assert tb.p[1] == pytest.approx(0.0, abs=1e-9)                       # straight behind
    assert abs(dist - tb.line_length).max() < 0.003                       # rigid: a few mm at most


def test_towbar_pushes_the_load_when_reversing():
    tb = TowedPointMass()
    tb.place_behind([0.0, 0.0], 0.0)
    _, dist, f = _move_tow_point(tb, (-0.3, 0.0), 20.0, p=[0.0, 0.0])
    assert f[-500:].mean() < -1.0                                         # pushes
    assert abs(dist - tb.line_length).max() < 0.003
    assert tb.v[0] == pytest.approx(-0.3, abs=1e-3)                       # goes backwards with the boat


def test_rope_goes_slack_when_reversing():
    tb = TowedPointMass(rigid=False, line_stiffness=2000.0, line_damping=100.0)
    tb.place_behind([0.0, 0.0], 0.0)
    _, dist, f = _move_tow_point(tb, (-0.3, 0.0), 5.0, p=[0.0, 0.0])
    assert np.all(f == 0.0) and dist[-1] < tb.line_length - 1.0


def test_towbar_coupled_to_the_vessel_is_stable():
    # the real Microampere model at the simulator's step: forward, astern, a turn; the bar keeps its length
    cfg = vessel_model.load_vessel_config(path=os.path.join(REPO, "cybership_config", "config", "microampere",
                                                            "vessel.yaml"))[0]
    craft = vessel_model.make_marine_craft(vessel_model.model_parameters(cfg))
    tow_point = np.array([-0.6, 0.0, -0.1])
    tb = towed_from_config({"line": {"length": 2.0}})
    assert isinstance(tb, TowedPointMass) and tb.rigid
    tb.place_behind(tow_point[:2], 0.0)
    dt, stretch = 0.01, []
    for i in range(3000):
        t = i * dt
        tau = np.zeros(6)
        tau[[0, 1, 5]] = (40, 0, 0) if t < 10 else (-40, 0, 0) if t < 20 else (0, 15, 6)
        eta, nu = craft.eta, craft.nu
        c, s = math.cos(eta[5]), math.sin(eta[5])
        vel = (c * nu[0] - s * nu[1], s * nu[0] + c * nu[1], nu[5])
        F, _, _ = tb.step_with_vessel(dt, (eta[0], eta[1], eta[5]), vel, tow_point[:2])
        craft.step(tau, dt, tow_wrench(F, eta[5], tow_point))
        c, s = math.cos(craft.eta[5]), math.sin(craft.eta[5])
        p_tow = craft.eta[:2] + np.array([c * tow_point[0] - s * tow_point[1], s * tow_point[0] + c * tow_point[1]])
        stretch.append(np.hypot(*(p_tow - tb.p)) - tb.line_length)
    assert np.all(np.isfinite(craft.nu))
    assert np.abs(stretch).max() < 0.003
    assert abs(craft.nu[0]) < 1.5 and abs(tb.tension) < 40.0


def test_load_cell_wrench_and_filter():
    # sideways pull at the stern: sway force and a yaw moment x F_y; the filter follows a step in ~T
    lc = LoadCell((-0.6, 0.0, -0.1), filter_time=0.1)
    lc.update(0.0, 0.0, 0.0)
    for k in range(1, 6):
        lc.update(0.02 * k, -10.0, 10.0)
    X, Y, N = lc.wrench()
    assert 0.6 * 10 < Y < 10 and X == pytest.approx(-Y)                # ~1 - e^-1 after 0.1 s
    assert N == pytest.approx(-0.6 * Y)                                  # bow swings to port
    for k in range(6, 100):
        lc.update(0.02 * k, -10.0, 10.0)
    assert np.allclose(lc.wrench(), [-10.0, 10.0, -6.0], atol=1e-3)
