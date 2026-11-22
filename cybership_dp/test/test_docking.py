"""
Docking geometry and the docking OCP (no ROS):

    cd ~/ros_ws/src/cybership_software_suite && python3 -m pytest -v cybership_dp/test/test_docking.py
"""
import math
import os

import numpy as np
import pytest
import yaml

from cybership_dp.docking import Quay, hull_corners

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
QUAY = [(8.0, -3.0), (9.5, -3.0), (9.5, 3.0), (8.0, 3.0)]
HULL = (1.195, 0.93)


def test_berth_lies_alongside_the_face_with_the_quay_to_port():
    quay = Quay(QUAY)
    b = quay.berth(3, "port", 0.10, HULL)                 # face 3 is the south face, x = 8
    assert b == pytest.approx([8.0 - 0.465 - 0.10, 0.0, math.pi / 2])
    assert quay.face_gap(b, *HULL, face=3) == pytest.approx(0.10)
    # port side (-y BODY) points north, at the quay
    c, s = math.cos(b[2]), math.sin(b[2])
    port = np.array([s, -c])                              # -y_body in NED
    assert port @ np.array([1.0, 0.0]) == pytest.approx(1.0)
    b2 = quay.berth(3, "starboard", 0.10, HULL)
    assert b2[2] == pytest.approx(-math.pi / 2)


def test_clearance_and_halfplane():
    quay = Quay(QUAY)
    a, b = quay.halfplane(3)                              # free water: a . p <= b, i.e. x <= 8
    assert a == pytest.approx([1.0, 0.0]) and b == pytest.approx(8.0)
    assert quay.clearance([7.0, 0.0, 0.0], *HULL) == pytest.approx(8.0 - 7.0 - 0.5975)
    assert quay.clearance([7.6, 0.0, 0.0], *HULL) < 0.0  # bow into the quay
    corners = hull_corners([0.0, 0.0, math.pi / 2], *HULL)
    assert np.max(corners[:, 1]) == pytest.approx(HULL[0] / 2)


@pytest.fixture(scope="module")
def ocp():
    pytest.importorskip("casadi")
    from cybership_utilities import vessel_model
    from cybership_dp.docking_mpc import DockingOCP
    cfg, _ = vessel_model.load_vessel_config(path=os.path.join(REPO, "cybership_config", "config", "microampere",
                                                               "vessel.yaml"))
    return DockingOCP.from_vessel_config(cfg, vessel_model.model_parameters(cfg), free_time=True, gap_min=0.08,
                                         corridor=(0.15, 0.5, 1.5), max_cpu_time=20.0)


@pytest.mark.parametrize("start", [[3.0, -2.0, 0.0], [2.0, 3.0, math.pi], [4.5, -4.0, -0.5]])
def test_docking_plan_reaches_the_berth_without_touching_the_quay(ocp, start):
    quay = Quay(QUAY)
    berth = quay.berth(3, "port", 0.10, HULL)
    r = ocp.solve(np.array(start + [0.0, 0.0, 0.0]), berth, quay.halfplane(3), vc=(0.05, 0.0))
    assert r["ok"]
    assert np.hypot(*(r["X"][:2, -1] - berth[:2])) < 0.05
    assert min(quay.face_gap(r["X"][:3, k], *HULL, face=3) for k in range(r["X"].shape[1])) >= 0.08 - 1e-3
    assert r["slack"] < 1e-3
    # the last part comes in from the front: within 0.3 m of the face the boat is within the funnel of the berth
    for k in range(r["X"].shape[1]):
        e = r["X"][:3, k]
        gap_co = 8.0 - e[0] - HULL[1] / 2
        if gap_co < 0.3:
            assert abs(e[1] - berth[1]) <= 0.15 + 0.5 * gap_co + 1.5 * gap_co ** 2 + 1e-3


def test_warm_replan_starts_from_the_last_plan(ocp):
    quay = Quay(QUAY)
    berth = quay.berth(3, "port", 0.10, HULL)
    r = ocp.solve(np.array([3.0, -2.0, 0.0, 0.0, 0.0, 0.0]), berth, quay.halfplane(3))
    r2 = ocp.solve(r["X"][:, 8], berth, quay.halfplane(3), elapsed=r["t"][8])
    assert r2["ok"] and r2["guess"] == "warm"
    assert np.hypot(*(r2["X"][:2, -1] - berth[:2])) < 0.05


def test_mapped_problem_is_the_inlined_one():
    """The compiled build maps three small functions over the stages; uncompiled, it must give the inlined SX
    problem's solution (same iterations, same plan), with the heading funnel on as in the controller."""
    pytest.importorskip("casadi")
    from cybership_utilities import vessel_model
    from cybership_dp.docking_mpc import DockingOCP
    cfg, _ = vessel_model.load_vessel_config(path=os.path.join(REPO, "cybership_config", "config", "microampere",
                                                               "vessel.yaml"))
    kw = dict(free_time=True, gap_min=0.08, corridor=(0.15, 0.5, 1.5), heading_funnel=(math.radians(5), math.radians(30)),
              max_cpu_time=20.0, compiled=False)
    params = vessel_model.model_parameters(cfg)
    quay = Quay(QUAY)
    berth = quay.berth(3, "port", 0.10, HULL)
    x0 = np.array([3.0, -2.0, 0.3, 0.0, 0.0, 0.0])
    r = [DockingOCP.from_vessel_config(cfg, params, mapped=m, **kw).solve(x0, berth, quay.halfplane(3), vc=(0.05, 0.0))
         for m in (False, True)]
    assert r[0]["ok"] and r[1]["ok"]
    assert r[1]["iterations"] == r[0]["iterations"]
    assert r[1]["T"] == pytest.approx(r[0]["T"], abs=1e-4)
    assert np.max(np.abs(r[1]["X"] - r[0]["X"])) < 1e-4
