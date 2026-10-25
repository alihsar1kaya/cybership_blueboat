"""
Unit tests for the constrained allocation (no ROS needed):

    cd ~/ros_ws/src/cybership_software_suite && python3 -m pytest -v cybership_dp/test
"""

import math
import os

import numpy as np
import pytest
import yaml

from cybership_dp.constrained_allocation import ConstrainedAllocator

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def microampere():
    with open(os.path.join(REPO, "cybership_config", "config", "microampere", "vessel.yaml")) as f:
        return ConstrainedAllocator(yaml.safe_load(f)["thrusters"])


FWD = {"bow_azimuth": 0.0}
SIDE = {"bow_azimuth": math.pi / 2}


def test_configuration_matrix_matches_fossen_T_e():
    """Columns: port (x,-0.29), starboard (x,+0.29), azimuth x and y at (0.37, 0)."""
    a = microampere()
    assert np.allclose(a.T_e, [[1, 1, 1, 0], [0, 0, 0, 1], [0.29, -0.29, 0, 0.37]])


def test_feasible_demand_is_met_exactly():
    a = microampere()
    f, _, s = a.allocate([20, 0, 0], FWD)
    assert np.allclose(s, 0, atol=0.05)
    assert f == pytest.approx([20 / 3] * 3, abs=0.05)   # equal split, least effort


def test_pure_sway_uses_azimuth_and_stern_differential():
    a = microampere()
    f, alpha, s = a.allocate([0, 30, 0], SIDE)
    assert math.degrees(alpha["bow_azimuth"]) == pytest.approx(90)
    assert f[2] == pytest.approx(30, abs=0.1)                       # azimuth sideways
    assert f[0] - f[1] == pytest.approx(-30 * 0.37 / 0.29, abs=0.2)   # cancels the azimuth's yaw moment
    assert np.allclose(s, 0, atol=0.05)


def test_limits_are_respected_and_sway_is_given_up_before_heading():
    a = microampere()
    f, _, s = a.allocate([0, 100, 0], SIDE)           # 100 N sideways is impossible (azimuth 67 N)
    assert np.all(f <= 67 + 1e-6) and np.all(f >= -53.6 - 1e-6)
    assert abs(s[1]) > 30                             # the missing sway force shows up as slack
    assert abs(s[2]) < 1.0                            # heading moment is still met (higher priority)


def test_reverse_is_weaker_than_forward():
    a = microampere()
    f, _, s = a.allocate([-200, 0, 0], {"bow_azimuth": math.pi})   # full astern, azimuth turned back
    assert f[0] == pytest.approx(-53.6, abs=0.1) and f[1] == pytest.approx(-53.6, abs=0.1)


def test_azimuth_pointing_the_wrong_way_gives_no_thrust():
    a = microampere()
    f, alpha, _ = a.allocate([0, 30, 0], {"bow_azimuth": -math.pi / 2})   # wants +Y, points -Y
    assert f[2] == pytest.approx(0, abs=1e-6)                             # no thrust the wrong way
    assert math.degrees(alpha["bow_azimuth"]) == pytest.approx(90)        # but told to turn around


def test_azimuth_set_point_holds_when_demand_is_tiny():
    a = microampere()
    a.allocate([0, 30, 0], SIDE)
    _, alpha, _ = a.allocate([0, -0.1, 0], SIDE)       # tiny opposite demand: do not swing 180 deg
    assert math.degrees(alpha["bow_azimuth"]) == pytest.approx(90)
