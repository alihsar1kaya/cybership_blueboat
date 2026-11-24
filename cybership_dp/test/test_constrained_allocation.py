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
    f, alpha, s = a.allocate([0, 15, 0], SIDE)
    assert math.degrees(alpha["bow_azimuth"]) == pytest.approx(90)
    assert f[2] == pytest.approx(15, abs=0.1)                       # azimuth sideways
    assert f[0] - f[1] == pytest.approx(-15 * 0.37 / 0.29, abs=0.2)   # cancels the azimuth's yaw moment
    assert np.allclose(s, 0, atol=0.05)


def test_limits_are_respected_and_sway_is_given_up_before_heading():
    a = microampere()
    f, _, s = a.allocate([0, 100, 0], SIDE)           # 100 N sideways is impossible (azimuth 20 N)
    assert np.all(f <= 51.5 + 1e-6) and np.all(f >= -51.5 * 0.78 - 1e-6)
    assert abs(s[1]) > 60                             # the missing sway force shows up as slack
    assert abs(s[2]) < 1.0                            # heading moment is still met (higher priority)


def test_reverse_is_weaker_than_forward():
    a = microampere()
    f, _, s = a.allocate([-200, 0, 0], {"bow_azimuth": math.pi})   # full astern, azimuth turned back
    assert f[0] == pytest.approx(-51.5 * 0.78, abs=0.1) and f[1] == pytest.approx(-51.5 * 0.78, abs=0.1)


def test_azimuth_pointing_the_other_way_reverses_instead_of_turning_around():
    """Wants +Y while pointing -Y: reverse thrust at -90 deg (T200 pushes both ways), no half turn."""
    a = microampere()
    f, alpha, _ = a.allocate([0, 30, 0], {"bow_azimuth": -math.pi / 2})
    assert math.degrees(alpha["bow_azimuth"]) == pytest.approx(-90)       # stays where it points
    assert f[2] < -1.0                                                    # and pushes in reverse
    assert f[2] >= -0.78 * 20 - 1e-6                                      # within the reverse limit


def test_azimuth_set_point_holds_when_demand_is_tiny():
    a = microampere()
    a.allocate([0, 30, 0], SIDE)
    _, alpha, _ = a.allocate([0, -0.1, 0], SIDE)       # tiny opposite demand: do not swing 180 deg
    assert math.degrees(alpha["bow_azimuth"]) == pytest.approx(90)


def test_inflow_compensation_restores_the_thrust_lost_at_speed():
    """
    0.6 m/s ahead with the stern thrusters against each other (holding a yaw moment): the one pushing
    ahead loses thrust and the one in reverse gains, which is a net drag. The compensated commands,
    run through the thruster model, give the forces the allocation wanted.
    """
    from cybership_utilities.marine_craft.actuators import FixedThruster
    with open(os.path.join(REPO, "cybership_config", "config", "microampere", "vessel.yaml")) as f:
        thrusters = yaml.safe_load(f)["thrusters"]
    a = ConstrainedAllocator(thrusters, zero_thrust_speed=5.0)
    f, _, _ = a.allocate([5.0, 0.0, 8.0], FWD)
    assert f[0] > 0 > f[1]
    nu_r = [0.6, 0.0, 0.1]
    cmd = a.compensate_inflow(f, FWD, nu_r)
    assert cmd[0] > f[0] and abs(cmd[1]) < abs(f[1])
    for t, wanted, c in list(zip(thrusters, f, cmd))[:2]:
        thr = FixedThruster(max_force=t["max_force"], reverse_ratio=t["reverse_ratio"], zero_thrust_speed=5.0)
        thr.set_command(c)
        thr.step(0.01)
        thr.inflow = nu_r[0] - nu_r[2] * t["position"][1]
        assert thr.output()[0] == pytest.approx(wanted, abs=1e-6)
    # without an inflow model the forces pass through unchanged
    assert np.allclose(microampere().compensate_inflow(f, FWD, nu_r), f)


def test_azimuth_set_point_ignores_demands_below_the_threshold():
    with open(os.path.join(REPO, "cybership_config", "config", "microampere", "vessel.yaml")) as f:
        a = ConstrainedAllocator(yaml.safe_load(f)["thrusters"], angle_threshold=3.0)
    a.allocate([0, 15, 0], SIDE)                       # within the azimuth's 20 N, so nothing is short
    _, alpha, _ = a.allocate([4.0, 0.0, 0.0], SIDE)     # 1.3 N per thruster ahead: not worth a turn
    assert math.degrees(alpha["bow_azimuth"]) == pytest.approx(90)
    _, alpha, _ = a.allocate([30.0, 0.0, 0.0], SIDE)
    assert abs(math.degrees(alpha["bow_azimuth"])) == pytest.approx(0, abs=1e-6)


def test_azimuth_turns_for_a_small_demand_only_it_can_give():
    # after a transit the azimuth points aft; a 1.6 N sway correction is below the threshold, but the
    # stern thrusters cannot give it, so the azimuth must still turn (it held 11 cm off the goal)
    with open(os.path.join(REPO, "cybership_config", "config", "microampere", "vessel.yaml")) as f:
        a = ConstrainedAllocator(yaml.safe_load(f)["thrusters"], angle_threshold=3.0)
    ahead = {"bow_azimuth": 0.0}
    a.allocate([30.0, 0.0, 0.0], ahead)
    _, alpha, slack = a.allocate([0.0, -1.6, 0.0], ahead)
    assert abs(slack[1]) > 0.5
    _, alpha, _ = a.allocate([0.0, -1.6, 0.0], ahead)
    assert abs(math.degrees(alpha["bow_azimuth"])) == pytest.approx(90, abs=1.0)


def test_frozen_azimuth_does_not_fight_the_stern_thrusters():
    # holding position with the azimuth left at 179 deg after braking: a 0.5 N sway correction is below the
    # threshold, so it does not turn; it must not make the sway by pushing 25 N along x against the stern
    with open(os.path.join(REPO, "cybership_config", "config", "microampere", "vessel.yaml")) as f:
        a = ConstrainedAllocator(yaml.safe_load(f)["thrusters"], angle_threshold=3.0)
    aft = {"bow_azimuth": math.radians(179.0)}
    a.allocate([-30.0, 0.0, 0.0], aft)
    f, alpha, slack = a.allocate([0.0, 0.5, 0.0], aft)
    assert abs(math.degrees(alpha["bow_azimuth"])) == pytest.approx(180, abs=1.0)  # did not turn
    assert np.abs(f).max() <= 0.5 + 3.0 + 1e-6                                  # demand + threshold
    assert abs(slack[0]) < 0.05 and abs(slack[2]) < 0.05                        # surge and yaw still met


def test_forward_only_azimuth_turns_around_instead_of_reversing():
    """The real boat's azimuth only pushes forward (thrust >= 0): pointing the wrong way it must turn, not reverse."""
    with open(os.path.join(REPO, "cybership_config", "config", "microampere", "vessel.yaml")) as f:
        thrusters = yaml.safe_load(f)["thrusters"]
    for t in thrusters:
        if t["type"] == "azimuth":
            t["reverse_ratio"] = 0.0
    a = ConstrainedAllocator(thrusters)
    f, alpha, _ = a.allocate([0, 20, 20 * 0.37], {"bow_azimuth": -math.pi / 2})   # to starboard, azimuth at port
    assert alpha["bow_azimuth"] == pytest.approx(math.pi / 2, abs=0.05)
    assert f[2] >= 0.0
