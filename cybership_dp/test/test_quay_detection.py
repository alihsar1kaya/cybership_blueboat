"""
Quay detection from a synthetic Ouster cloud (no ROS):

    cd ~/ros_ws/src/cybership_software_suite && python3 -m pytest -v cybership_dp/test/test_quay_detection.py
"""
import math

import numpy as np
import pytest

from cybership_dp.quay_detection import FLU_TO_FRD, QuayDetector, rot_x, rot_y

LIDAR = [0.011, -0.039, 0.523]                 # FLU from the CO (vessel.yaml)


def synthetic_cloud(d, alpha, roll=0.0, pitch=0.0, noise=0.01, seed=1):
    """A quay face n . p = d (BODY FRD, normal angle alpha) from 0 to 0.3 m above the water, 6 m long, sampled as
    points, plus water returns, the own hull and clutter, then put into the lidar's frame for a heeled boat."""
    rng = np.random.default_rng(seed)
    n = np.array([math.cos(alpha), math.sin(alpha)])
    t = np.array([-n[1], n[0]])
    s = rng.uniform(-3.0, 3.0, 1500)
    h = rng.uniform(0.0, 0.3, 1500)                               # height above the water
    face = np.column_stack([d * n[0] + s * t[0], d * n[1] + s * t[1], -h])
    face[:, :2] += rng.normal(0, noise, (len(s), 2)) * n
    water = np.column_stack([rng.uniform(-8, 8, 300), rng.uniform(-8, 8, 300), np.zeros(300)])
    hull = np.column_stack([rng.uniform(-0.6, 0.6, 200), rng.uniform(-0.45, 0.45, 200), rng.uniform(-0.2, 0, 200)])
    clutter = np.column_stack([rng.uniform(-10, 10, 60), rng.uniform(-10, 10, 60), rng.uniform(-1.5, -0.1, 60)])
    level = np.vstack([face, water, hull, clutter])               # level BODY FRD
    # the boat heels: level = Ry(pitch) Rx(roll) body  ->  body = (Ry Rx)^T level
    body_frd = level @ (rot_y(pitch) @ rot_x(roll))
    body_flu = body_frd @ FLU_TO_FRD.T
    return body_flu - np.array(LIDAR)                             # lidar frame = BODY FLU shifted (no mounting turn)


@pytest.mark.parametrize("d, alpha_deg, roll_deg, pitch_deg", [
    (1.0, -90.0, 0.0, 0.0),        # quay to port, close
    (0.6, -85.0, 3.0, -2.0),       # at the berth, heeled
    (4.0, -60.0, -2.0, 1.0),
    (2.5, 10.0, 0.0, 3.0),         # ahead
])
def test_fit_finds_the_face(d, alpha_deg, roll_deg, pitch_deg):
    det = QuayDetector(LIDAR)
    cloud = synthetic_cloud(d, math.radians(alpha_deg), math.radians(roll_deg), math.radians(pitch_deg))
    r = det.fit(cloud, math.radians(roll_deg), math.radians(pitch_deg))
    assert r["ok"]
    assert r["d"] == pytest.approx(d, abs=0.01)
    assert math.degrees(r["alpha"]) == pytest.approx(alpha_deg, abs=0.3)
    assert r["span"] > 4.0 and r["sigma_d"] < 0.02


def test_water_and_hull_alone_give_no_quay():
    det = QuayDetector(LIDAR)
    cloud = synthetic_cloud(1.0, 0.0)
    rng = np.random.default_rng(3)
    only_water = np.column_stack([rng.uniform(-8, 8, 500), rng.uniform(-8, 8, 500), np.full(500, -0.523)])
    assert not det.fit(only_water)["ok"]


def test_prior_gate_picks_the_expected_face():
    """Two faces (the quay's end too): with a prior the expected one is fitted, not the longer one."""
    det = QuayDetector(LIDAR)
    a = synthetic_cloud(1.0, -math.pi / 2, seed=4)                # 6 m face to port
    b = synthetic_cloud(3.0, 0.0, seed=5)[:600]                   # a shorter face ahead
    cloud = np.vstack([a, b])
    r = det.fit(cloud, prior=(np.array([1.0, 0.0]), 3.0), gate=0.4)
    assert r["ok"] and r["d"] == pytest.approx(3.0, abs=0.02) and abs(r["alpha"]) < math.radians(0.5)
