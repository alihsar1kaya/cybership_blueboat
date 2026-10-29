"""
Unit tests for the thruster dynamics (no ROS needed):

    cd ~/ros_ws/src/cybership_software_suite && python3 -m pytest -v cybership_utilities/test
"""

import math

import pytest

from cybership_utilities.marine_craft.actuators import FixedThruster, AzimuthThruster, PwmThrusterCurve, make_thruster

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


def test_azimuth_turns_without_thrust_on_a_tiny_command():
    """A command below the deadband still sets the angle (servo), but gives no thrust."""
    a = AzimuthThruster(max_force=67, deadband=0.5, max_rate=math.radians(90))
    a.set_command(0, 1e-3)
    run(a, 1.0)
    assert math.degrees(a.angle) == pytest.approx(90, abs=0.1)
    assert a.output()[1] == 0.0


# --- ESC / PWM model (T200) -------------------------------------------------------------------

def test_pwm_curve_deadzone_and_limits():
    c = PwmThrusterCurve(67.0, reverse_ratio=0.8)
    for pwm in (1475, 1490, 1500, 1510, 1525):
        assert c.force(pwm) == 0.0            # ESC deadzone
    assert c.force(1900) == pytest.approx(67.0)
    assert c.force(1100) == pytest.approx(-0.8 * 67.0)
    assert c.force(1950) == pytest.approx(67.0)   # saturates
    forces = [c.force(p) for p in range(1100, 1901)]
    assert all(b >= a for a, b in zip(forces, forces[1:]))   # monotonic


def test_pwm_driver_skips_the_deadzone():
    """Small forces are not lost: the driver jumps over the deadzone, error below one-us resolution."""
    c = PwmThrusterCurve(67.0, reverse_ratio=0.8)
    for f in (0.05, 0.3, 0.5, 2.0, 10.0, 67.0, -0.3, -20.0, -53.6):
        pwm = c.to_pwm(f)
        assert isinstance(pwm, int)
        assert c.force(pwm) == pytest.approx(f, abs=0.2, rel=0.01)
    assert c.to_pwm(0.0) == 1500
    assert c.to_pwm(0.3) > 1525 and c.to_pwm(-0.3) < 1475


def test_thruster_with_pwm_gives_small_forces():
    """With the ESC model a 0.3 N command gives ~0.3 N (the old 0.5 N force deadband gave 0)."""
    t = make_thruster({"type": "fixed", "max_force": 67, "reverse_ratio": 0.8, "time_constant": 0.1,
                       "pwm": {"neutral": 1500, "deadzone": 25, "half_range": 400}})
    t.set_command(0.3)
    for _ in range(100):
        t.step(0.01)
    assert t.output()[0] == pytest.approx(0.3, abs=0.05)
    assert t.pwm > 1525
    a = make_thruster({"type": "azimuth", "max_force": 67, "pwm": {"neutral": 1500, "deadzone": 25}})
    fx, fy = a.set_command(0.0, 0.3)
    assert math.hypot(fx, fy) == pytest.approx(0.3, abs=0.05) and a.pwm > 1525


def test_azimuth_reverses_instead_of_a_half_turn():
    """
    A demand opposite to where the azimuth points is met in reverse at the same angle (T200 turns both
    ways), not by turning 180 deg at 90 deg/s while pushing the wrong way.
    """
    a = AzimuthThruster(max_force=67, reverse_ratio=0.8, max_rate=math.radians(90))
    a.set_command(0.0, 20.0)                  # +Y
    for _ in range(300):
        a.step(0.01)
    assert math.degrees(a.angle) == pytest.approx(90, abs=1)
    a.set_command(0.0, -20.0)                 # now -Y: reverse at +90 deg
    for _ in range(50):
        a.step(0.01)
    fx, fy, ang = a.output()
    assert math.degrees(ang) == pytest.approx(90, abs=1)      # did not turn
    assert fy == pytest.approx(-20.0, rel=0.05)
    # reverse is limited to 0.8 x 67 N
    a.set_command(0.0, -100.0)
    assert a.command[0] == pytest.approx(-0.8 * 67)


def test_azimuth_mode_has_hysteresis():
    """Near 90 deg off, small wiggles in the demand must not flip between forward and reverse."""
    a = AzimuthThruster(max_force=67, reverse_ratio=0.8, max_rate=math.radians(90))
    modes = []
    for k in range(40):
        ang = math.radians(90 + (5 if k % 2 else -5))     # demand direction wiggles around 90 deg off
        a.set_command(10 * math.cos(ang), 10 * math.sin(ang))
        modes.append(a.command[0] > 0)
    assert len(set(modes)) == 1
