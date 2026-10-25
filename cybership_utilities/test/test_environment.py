"""
Unit tests for the environment models (no ROS needed):

    cd ~/ros_ws/src/cybership_software_suite && python3 -m pytest -v cybership_utilities/test
"""

import numpy as np
import pytest

from cybership_utilities.marine_craft.environment import OceanCurrent


def test_constant_current_points_the_right_way():
    assert OceanCurrent(0.2, 0.0).velocity_ned() == pytest.approx([0.2, 0.0, 0.0])    # north
    assert OceanCurrent(0.2, 90.0).velocity_ned() == pytest.approx([0.0, 0.2, 0.0])   # east


def test_constant_current_does_not_change():
    c = OceanCurrent(0.3, 45.0)
    for _ in range(1000):
        c.step(0.01)
    assert c.speed == 0.3


def test_gauss_markov_speed_stays_bounded_and_averages_the_set_speed():
    c = OceanCurrent(0.2, 0.0, mu=0.05, sigma=0.02, speed_min=0.1, speed_max=0.3, seed=1)
    speeds = []
    for _ in range(200_000):        # 2000 s at 100 Hz
        c.step(0.01)
        speeds.append(c.speed)
    speeds = np.array(speeds)
    assert speeds.min() >= 0.1 and speeds.max() <= 0.3
    assert speeds.mean() == pytest.approx(0.2, abs=0.02)
    assert speeds.std() > 0.005     # it does vary


def test_set_changes_speed_and_direction():
    c = OceanCurrent(0.1, 0.0)
    c.set(speed=0.25, direction_deg=180.0)
    assert c.velocity_ned() == pytest.approx([-0.25, 0.0, 0.0], abs=1e-12)
