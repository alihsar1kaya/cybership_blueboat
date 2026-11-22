import math

import numpy as np
import pytest

from cybership_utilities import blueboat_io as bb


def test_gps_time_of_week_counts_milliseconds():
    itow, _ = bb.gps_time(1_800_000_000.0)
    assert 0 <= itow < 604_800_000
    assert bb.gps_time(1_800_000_001.25)[0] - itow == 1250


def test_ecef_of_the_equator_and_the_pole():
    assert np.allclose(bb.llh_to_ecef(0.0, 0.0, 0.0), [6378137.0, 0.0, 0.0])
    b = 6378137.0 * math.sqrt(1.0 - 6.69437999014e-3)
    assert np.allclose(bb.llh_to_ecef(90.0, 0.0, 0.0), [0.0, 0.0, b], atol=1e-6)


@pytest.mark.parametrize("x", [1.23456, -1.23456, 2854123.98765, -0.00004])
def test_ublox_high_precision_split_rebuilds_the_coordinate(x):
    cm, hp = bb.ublox_hp(x)
    assert -99 <= hp <= 99
    assert cm * 1e-2 + hp * 1e-4 == pytest.approx(x, abs=0.6e-4)


def test_accelerometer_at_rest_and_in_a_turn():
    eta = np.zeros(6)
    f = bb.specific_force_frd(eta, np.zeros(6), np.zeros(6), [0.0, 0.0, 0.0])
    assert np.allclose(f, [0.0, 0.0, -9.81])
    assert np.allclose(bb.frd_to_flu(f), [0.0, 0.0, 9.81])            # STIM300 (FLU) reads +g up
    # 1 m/s ahead, turning to starboard at 0.5 rad/s: centripetal 0.5 m/s^2 to starboard
    f = bb.specific_force_frd(eta, [1.0, 0, 0, 0, 0, 0.5], np.zeros(6), [0.0, 0.0, 0.0])
    assert f[1] == pytest.approx(0.5)
    # rolled 90 deg to starboard at rest: gravity's reaction along -y (to port)
    f = bb.specific_force_frd([0, 0, 0, math.pi / 2, 0, 0], np.zeros(6), np.zeros(6), [0.0, 0.0, 0.0])
    assert np.allclose(f, [0.0, -9.81, 0.0], atol=1e-9)


def test_baseline_gives_the_vehicle_heading():
    port, stb = [-0.358, -0.307, -0.122], [-0.358, 0.397, -0.122]
    for psi_deg in (0.0, 30.0, 179.0, 270.0):
        b = bb.baseline_ned([0, 0, 0, 0, 0, math.radians(psi_deg)], port, stb)
        assert np.linalg.norm(b) == pytest.approx(0.704)
        assert bb.vehicle_heading_deg(b) == pytest.approx(psi_deg % 360.0)


def test_pwm_map_of_the_aft_thrusters():
    assert bb.pwm_to_force(1700) == pytest.approx(26.4)
    assert bb.pwm_to_force(1300) == pytest.approx(-13.36)
    assert bb.pwm_to_force(1900) == pytest.approx(52.8)                 # the team's aft limits
    assert bb.pwm_to_force(1100) == pytest.approx(-26.72)
    assert bb.pwm_to_force(1700, contra_rotating=True) == pytest.approx(-13.36)


def test_azimuth_unwrap_crosses_straight_back_the_short_way():
    az = bb.AzimuthUnwrap(turn_limit=1.5)
    assert az.command(math.radians(170), 0.0) == pytest.approx(math.radians(170))
    # -170 deg (wrapped) is 20 deg further round, not 340 deg back
    assert az.command(math.radians(-170), 0.1) == pytest.approx(math.radians(190))
    assert not az.holding(0.1)


def test_azimuth_unwrap_goes_the_other_way_past_the_turn_limit():
    az = bb.AzimuthUnwrap(turn_limit=1.5, rate=math.pi / 2)
    t, a = 0.0, 0.0
    for _ in range(54):                                                 # wind up 540 deg in 10 deg steps
        a += math.radians(10)
        t += 0.1
        az.command(a, t)
    assert az.angle == pytest.approx(math.radians(540))
    cmd = az.command(a + math.radians(10), t + 0.1)                    # one more step would pass 1.5 turns
    assert cmd == pytest.approx(math.radians(550 - 360))                 # back round the other way
    assert az.holding(t + 0.1) and not az.holding(t + 0.1 + 2 * math.pi / (math.pi / 2) + 0.01)


def test_enu_goal_becomes_ned():
    # 3 m east, 4 m north, facing north in ENU (yaw 90 deg) = (4, 3) NED, heading 0
    assert np.allclose(bb.goal_in_ned("enu", 3.0, 4.0, math.pi / 2), [4.0, 3.0, 0.0])
    # facing east in ENU (yaw 0) = heading 90 deg NED
    assert bb.goal_in_ned("enu", 0.0, 0.0, 0.0)[2] == pytest.approx(math.pi / 2)
    assert np.allclose(bb.goal_in_ned("world", 1.0, 2.0, 0.5), [1.0, 2.0, 0.5])
    assert bb.goal_in_ned("map", 1.0, 2.0, 0.0) is None


def test_body_goal_is_relative_to_the_boat():
    pose = (10.0, 5.0, math.pi / 2)                                     # heading east
    # 2 m ahead and a quarter turn to starboard: 2 m further east, heading south
    g = bb.goal_in_ned("base_link", 2.0, 0.0, math.pi / 2, pose)
    assert np.allclose(g[:2], [10.0, 7.0]) and abs(abs(g[2]) - math.pi) < 1e-9
    assert np.allclose(bb.goal_in_ned("base_link", 0.0, 1.0, 0.0, pose)[:2], [9.0, 5.0])   # 1 m to starboard
    assert bb.goal_in_ned("base_link", 1.0, 0.0, 0.0) is None           # no pose yet


def test_dp_gate_needs_autonomous_and_fresh_state():
    g = bb.DPGate(state_timeout=0.5)
    assert g.step(0.0, True) is None and g.reason == "no state"
    g.on_state(0.0)
    assert g.step(0.1, False) is None and g.reason == "not autonomous"
    assert g.step(0.2, True) == "activated" and g.active
    assert g.step(0.3, True) is None
    assert g.step(0.8, True) == "deactivated" and g.reason == "state stale"
    g.on_state(0.9)
    assert g.step(1.0, True) == "activated"
    assert g.step(1.1, False) == "deactivated" and not g.active


def test_byte_field_from_rclpy():
    assert bb.byte_value(b"\x01") == 1 and bb.byte_value(2) == 2


def test_lekf_pose_moves_the_imu_to_the_co():
    """IMU 0.05 m ahead of and 0.1 m above the CO, boat heading east: the CO is 0.05 m west of the IMU."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_euler("ZYX", [math.pi / 2, 0.0, 0.0]).as_quat()
    n, e, psi, roll, pitch = bb.lekf_pose([10.0, 20.0, -0.1], q, [0.05, 0.0, -0.10])
    assert (n, e) == pytest.approx((10.0, 19.95))
    assert psi == pytest.approx(math.pi / 2)
    assert roll == pytest.approx(0.0) and pitch == pytest.approx(0.0)


def test_stamp_clock_maps_sentiboard_time_and_follows_a_restart():
    """Stamps 26 years behind, 20-60 ms transport delay: the mapped time keeps the delays to within the shortest."""
    rng = np.random.default_rng(0)
    clock = bb.StampClock()
    offset = 1.795e9 - bb.SENTIBOARD_EPOCH_UNIX
    errors = []
    for i in range(500):                                       # 10 s at 50 Hz
        stamp = bb.SENTIBOARD_EPOCH_UNIX + 100.0 + 0.02 * i
        delay = 0.02 + 0.04 * rng.random()
        t = clock(stamp, stamp + offset + delay)
        errors.append((stamp + offset + delay - t) - delay)    # mapped delay minus the true one
    assert max(abs(e) for e in errors[50:]) < 0.021           # the shortest delay (~20 ms) is taken as zero
    t = clock(bb.SENTIBOARD_EPOCH_UNIX + 1.0, 1.795e9 + 20.0)  # the SentiBoard node restarted
    assert clock.resets == 1 and t == pytest.approx(1.795e9 + 20.0)
