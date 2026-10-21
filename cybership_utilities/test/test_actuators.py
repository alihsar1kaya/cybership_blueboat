"""
Unit tests for the thruster dynamics (no ROS needed):

    cd ~/ros_ws/src/cybership_software_suite && python3 -m pytest -v cybership_utilities/test
"""

import math

import pytest

from cybership_utilities.marine_craft.actuators import FixedThruster, AzimuthThruster, make_thruster

DT = 0.01


def run(act, seconds):
    for _ in range(int(round(seconds / DT))):
        act.step(DT)
    return act


def test_fixed_thruster_saturates_forward_and_reverse():
    t = FixedThruster(max_force=67, reverse_ratio=0.8)
    t.set_command(100); run(t, 0.1)
    assert t.output()[0] == pytest.approx(67)
    t.set_command(-100); run(t, 0.1)
    assert t.output()[0] == pytest.approx(-53.6)


def test_deadband_gives_no_thrust():
    t = FixedThruster(max_force=67, deadband=0.5)
    t.set_command(0.3); run(t, 0.1)
    assert t.output()[0] == 0.0


def test_thrust_builds_up_with_time_constant():
    t = FixedThruster(max_force=67, time_constant=0.1)
    t.set_command(50); run(t, 0.1)
    assert t.output()[0] == pytest.approx(50 * (1 - math.exp(-1)), rel=1e-6)  # 63 % after T


def test_azimuth_turns_90_degrees_in_one_second():
    a = AzimuthThruster(max_force=67, max_rate=math.radians(90))
    a.set_command(0, 20)                      # straight sideways
    run(a, 0.5)
    assert math.degrees(a.angle) == pytest.approx(45, abs=0.1)
    run(a, 0.5)
    assert math.degrees(a.angle) == pytest.approx(90, abs=0.1)


def test_azimuth_pushes_along_its_actual_angle_while_turning():
    a = AzimuthThruster(max_force=67, max_rate=math.radians(90))
    a.set_command(0, 20)
    run(a, 0.5)                               # halfway: pointing 45 deg
    fx, fy, _ = a.output()
    assert fx == pytest.approx(fy) and math.hypot(fx, fy) == pytest.approx(20)


def test_azimuth_turns_the_shortest_way():
    a = AzimuthThruster(max_force=67, max_rate=math.radians(90), angle=math.radians(170))
    a.set_command(math.cos(math.radians(-170)) * 10, math.sin(math.radians(-170)) * 10)
    run(a, 0.25)                              # 20 deg through 180 takes 0.22 s
    assert abs(math.degrees(a.angle)) == pytest.approx(170, abs=0.1)
    assert math.degrees(a.angle) < 0


def test_azimuth_saturates_the_magnitude_not_each_axis():
    a = AzimuthThruster(max_force=67)
    a.set_command(60, 60)
    run(a, 0.01)
    fx, fy, _ = a.output()
    assert math.hypot(fx, fy) == pytest.approx(67)


def test_defaults_are_instant_and_unlimited():
    """Vessels without actuator data (e.g. voyager) must behave as before."""
    a = make_thruster({"type": "azimuth"})
    a.set_command(0, 1.0); a.step(DT)
    assert math.degrees(a.angle) == pytest.approx(90)
    f = make_thruster({"type": "fixed"})
    f.set_command(-5.0); f.step(DT)
    assert f.output()[0] == -5.0
