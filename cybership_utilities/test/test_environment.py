"""
Unit tests for the environment models (no ROS needed):

    cd ~/ros_ws/src/cybership_software_suite && python3 -m pytest -v cybership_utilities/test
"""

import numpy as np
import pytest

import math

from cybership_utilities.marine_craft.environment import BLENDERMANN, OceanCurrent, Wind, blendermann


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


# Microampere areas above the waterline (cybership_config/tools/wind_areas.py on the hull CAD)
AREAS = {"A_Fw": 0.109, "A_Lw": 0.174, "H_Lw": 0.109, "s_L": 0.057, "L_oa": 1.204}
NORTH = np.zeros(6)            # boat at rest, heading north


def test_head_wind_pushes_the_boat_back():
    tau = Wind(AREAS, speed=10.0, direction_deg=180.0).loads(NORTH, np.zeros(6))   # blows towards south
    q = 0.5 * 1.225 * 10.0**2
    CD_t, CD_bow = BLENDERMANN["speed_boat"][:2]
    assert tau[0] == pytest.approx(-q * CD_bow * AREAS["A_Fw"])     # C_X(0) = -CD_lAF(0)
    assert tau[1] == pytest.approx(0.0, abs=1e-9) and tau[5] == pytest.approx(0.0, abs=1e-9)


def test_wind_from_astern_pushes_forward():
    tau = Wind(AREAS, speed=10.0, direction_deg=0.0).loads(NORTH, np.zeros(6))
    assert tau[0] > 0 and abs(tau[1]) < 1e-9


def test_beam_wind_pushes_downwind_and_heels_the_boat_away():
    tau = Wind(AREAS, speed=10.0, direction_deg=90.0).loads(NORTH, np.zeros(6))   # from west = port side
    q = 0.5 * 1.225 * 10.0**2
    assert tau[1] == pytest.approx(q * BLENDERMANN["speed_boat"][0] * AREAS["A_Lw"])   # C_Y(90) = CD_t
    assert tau[3] > 0                          # heel to starboard (downwind)
    assert tau[5] > 0                          # centroid forward of midship: bow turned downwind
    mirror = Wind(AREAS, speed=10.0, direction_deg=270.0).loads(NORTH, np.zeros(6))
    assert mirror[[1, 3, 5]] == pytest.approx(-tau[[1, 3, 5]])
    assert mirror[0] == pytest.approx(tau[0])


def test_wind_is_seen_in_the_body_frame():
    east = np.array([0, 0, 0, 0, 0, math.pi / 2])                    # heading east
    tau = Wind(AREAS, speed=10.0, direction_deg=270.0).loads(east, np.zeros(6))   # head wind from east
    assert tau[0] < 0 and abs(tau[1]) < 1e-9


def test_moving_in_still_air_is_a_head_wind():
    moving = Wind(AREAS, speed=0.0).loads(NORTH, np.array([2.0, 0, 0, 0, 0, 0]))
    head = Wind(AREAS, speed=2.0, direction_deg=180.0).loads(NORTH, np.zeros(6))
    assert moving == pytest.approx(head)


def test_blendermann_coefficients_are_continuous_over_the_beam():
    c = BLENDERMANN["speed_boat"]
    a = blendermann(math.pi / 2 - 1e-6, 0.109, 0.174, 0.057, 1.204, *c)
    b = blendermann(math.pi / 2 + 1e-6, 0.109, 0.174, 0.057, 1.204, *c)
    assert a[1:] == pytest.approx(b[1:], rel=1e-4) and abs(a[0]) < 1e-4 and abs(b[0]) < 1e-4


def test_no_wind_no_load():
    assert Wind(AREAS).loads(NORTH, np.zeros(6)) == pytest.approx(np.zeros(6))
