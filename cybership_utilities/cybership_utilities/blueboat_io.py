"""
Microampere's own interface (TURLab blueboat_interfaces), without ROS: the conversions the simulator needs to
look like the real boat's drivers, and the allocation needs to drive the real boat.

On the boat (blueboat_ws_ua_lp3 / _jetson, read 2026-11-17):

    /control/thrust          ThrusterInputs   aft thrusters: forces [N] (pwm = false) or PWM [us] (pwm = true)
    /actuator/azimuth_cmd    AzimuthTAngle    bow azimuth: thrust [N] signed, angle [rad] ABSOLUTE and UNWRAPPED
    /sensors/imu             Imu              STIM300, frame "IMU" = FLU (x forward, y port, z up), rad/s, m/s^2
                                              specific force (z = +9.81 at rest)
    /sensors/nav_pvt         GNSSNavPvt       port antenna (moving base): lat/lon [deg], height [m], vel NED [m/s]
    /sensors/ecef            GNSSHPPosECEF    port antenna, u-blox scaling: cm + 0.1 mm, p_acc in 0.1 mm
    /sensors/nav_cov         GNSSNavCov       port antenna position covariance NED [m^2]
    /sensors/rel_pos         GNSSRelPos       baseline port -> starboard antenna NED [m]; rel_pos_heading is the
                                              VEHICLE heading [deg, 0..360) (the node subtracts 90 deg from the
                                              baseline's heading)
    /navigation/odometry/ned Odometry         LEKF: frame "ned", child "base_link" (FRD at the IMU), 50 Hz, no
                                              covariance; pose of the IMU, twist in BODY
    /navigation/ned_origin   GpsOrigin        latched: the LEKF's NED origin (the first fix, unless pinned)

All /sensors/* and /navigation/* stamps are SentiBoard time: a counter on the epoch 2000-01-01 that restarts with
its node, 26 years from the computers' clocks (StampClock maps them).

The team's pose controller (blueboat_core_control, pose_controller, read 2026-11-19), which our DP mirrors with
boat_interface:=true so the Foxglove DP tab and the team's allocator work with it:

    /control/opmode          OperationMode    in, latched: the DP drives only in AUTONOMOUS
    /goal_pose               PoseStamped      in: target in frame "ned", "enu" or "base_link" (relative to the boat)
    /control/pose/setpoint   Waypoint         in: target xn, yn [m], psi_n [rad] NED
    /control/tau             WrenchStamped    out: demanded force, BODY (surge, sway: force.x, .y; yaw: torque.z)
    /control/pose/*          debug            target, reference, state (Waypoint), active, status, gains, period, ...
"""
import math

import numpy as np

from cybership_utilities.marine_craft.kinematics import Rzyx

GPS_EPOCH_UNIX = 315964800          # 1980-01-06 00:00:00 UTC
SENTIBOARD_EPOCH_UNIX = 946684800   # 2000-01-01 00:00:00 UTC, the SentiBoard's synthetic epoch
GPS_LEAP_SECONDS = 18               # GPS - UTC since 2017
WEEK = 604800.0

# WGS-84
_A = 6378137.0
_E2 = 6.69437999014e-3


def gps_time(unix_time):
    """(itow [ms], nano [ns]) of the GPS time of week at UNIX time `unix_time` [s]."""
    gps = unix_time - GPS_EPOCH_UNIX + GPS_LEAP_SECONDS
    tow = gps % WEEK
    itow = int(tow * 1000.0)
    return itow, int(round((tow * 1000.0 - itow) * 1e6))


def llh_to_ecef(lat_deg, lon_deg, h):
    """WGS-84 geodetic -> ECEF [m]."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
    return np.array([(n + h) * math.cos(lat) * math.cos(lon),
                     (n + h) * math.cos(lat) * math.sin(lon),
                     (n * (1.0 - _E2) + h) * math.sin(lat)])


def ublox_hp(x):
    """A coordinate [m] as u-blox NAV-HPPOSECEF has it: (cm as int, the rest in 0.1 mm as int, -99..99)."""
    cm = int(math.floor(x * 100.0))
    hp = int(round((x * 100.0 - cm) * 100.0))
    if hp > 99:                       # rounding up to a whole cm
        cm, hp = cm + 1, hp - 100
    return cm, hp


def frd_to_flu(v):
    return np.array([v[0], -v[1], -v[2]])


def specific_force_frd(eta, nu, nu_dot, r, g=9.81):
    """
    What an accelerometer at r (BODY FRD, from the CO) measures, in BODY FRD [m/s^2]:

        f = v_dot + w x v + w_dot x r + w x (w x r) - R^T [0, 0, g]

    eta = (n, e, d, phi, theta, psi), nu = (u, v, w, p, q, r) and its time derivative in BODY.
    At rest it is (0, 0, -g): the reaction to gravity points up.
    """
    v, w = np.asarray(nu[:3], float), np.asarray(nu[3:6], float)
    v_dot, w_dot = np.asarray(nu_dot[:3], float), np.asarray(nu_dot[3:6], float)
    r = np.asarray(r, float)
    a = v_dot + np.cross(w, v) + np.cross(w_dot, r) + np.cross(w, np.cross(w, r))
    return a - Rzyx(*eta[3:6]).T @ np.array([0.0, 0.0, g])


def baseline_ned(eta, port, starboard):
    """Baseline from the port to the starboard antenna (BODY FRD positions) in NED [m]."""
    return Rzyx(*eta[3:6]) @ (np.asarray(starboard, float) - np.asarray(port, float))


def vehicle_heading_deg(baseline):
    """Vehicle heading [deg, 0..360) from a port -> starboard baseline (NED), as the boat's GNSS node gives it."""
    return (math.degrees(math.atan2(baseline[1], baseline[0])) - 90.0) % 360.0


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def enu_to_ned_pose(x, y, yaw):
    """A pose (x east, y north, yaw from east towards north) in ENU as (north, east, heading from north) NED."""
    return np.array([y, x, wrap_pi(math.pi / 2.0 - yaw)])


def goal_in_ned(frame, x, y, yaw, pose=None, ned_frames=("ned", "world", "odom"), enu_frame="enu",
                body_frames=("base_link",)):
    """
    A goal pose as the team's pose controller reads /goal_pose: (x, y, yaw) in NED, or None for an unknown frame.
    NED frames are taken as they are (our 'world' and 'odom' are NED too), ENU is converted, and a body frame
    means relative to the boat's current pose (x forward, y starboard, yaw added to the heading); that needs
    pose = (x, y, psi) NED.
    """
    if frame in ned_frames:
        return np.array([x, y, wrap_pi(yaw)])
    if frame == enu_frame:
        return enu_to_ned_pose(x, y, yaw)
    if frame in body_frames and pose is not None:
        px, py, psi = pose
        c, s = math.cos(psi), math.sin(psi)
        return np.array([px + c * x - s * y, py + s * x + c * y, wrap_pi(psi + yaw)])
    return None


def byte_value(value):
    """A ROS 'byte' field (bytes of length 1 in rclpy) as an int."""
    return value[0] if isinstance(value, (bytes, bytearray)) else int(value)


class DPGate:
    """
    When the DP may drive, by the team's pose_controller rules: active while the boat is in AUTONOMOUS and the
    state is fresh (at most state_timeout s old). On becoming active the boat should hold where it is; on becoming
    inactive the target should be forgotten, so an old target never pulls the boat away when it is next made
    active. Targets are accepted only while active. Times in s on one monotonic clock.
    """

    def __init__(self, state_timeout=0.5):
        self.state_timeout = float(state_timeout)
        self.active = False
        self.reason = "no state"
        self._state_time = None

    def on_state(self, t):
        self._state_time = t

    def step(self, t, autonomous):
        """Update at time t: 'activated', 'deactivated', or None when nothing changed."""
        if not autonomous:
            reason = "not autonomous"
        elif self._state_time is None:
            reason = "no state"
        elif t - self._state_time > self.state_timeout:
            reason = "state stale"
        else:
            reason = None
        was = self.active
        self.active, self.reason = reason is None, reason or ""
        if self.active != was:
            return "activated" if self.active else "deactivated"
        return None


def pwm_to_force(pwm_us, zero_us=1500.0, min_us=1100.0, max_us=1900.0, positive_force_per_us=0.132,
                 negative_force_per_us=0.0668, contra_rotating=False):
    """
    Force [N] from an aft thruster PWM command as the boat's driver map has it (blueboat_simulator
    config/sim.yaml): linear on each side of zero_us. A contra-rotating propeller is mirrored about zero_us
    (to verify with the team).
    """
    pwm = float(np.clip(pwm_us, min_us, max_us))
    if contra_rotating:
        pwm = 2.0 * zero_us - pwm
    d = pwm - zero_us
    return d * (positive_force_per_us if d > 0 else negative_force_per_us)


class AzimuthUnwrap:
    """
    The azimuth angle for a driver that turns to the absolute, unwrapped angle it is given (the boat's stepper
    counts turns; a cable limits how far it may wind up). Each command moves the short way from the last one,
    unless that winds it past turn_limit turns from 0: then it goes the other way round, and the thrust is held
    at 0 while it swings (|swing| / rate seconds), so it does not push in passing directions.
    """

    def __init__(self, turn_limit=1.5, rate=math.pi / 2):
        if turn_limit < 0.5:
            raise ValueError("turn_limit must be at least half a turn")
        self.limit = 2.0 * math.pi * float(turn_limit)
        self.rate = float(rate) if rate > 0 else math.inf
        self.angle = 0.0
        self.hold_until = -math.inf

    def command(self, angle, t):
        """Unwrapped command [rad] for a wanted angle (any wrapping) at time t [s]."""
        step = (angle - self.angle + math.pi) % (2.0 * math.pi) - math.pi
        target = self.angle + step
        if abs(target) > self.limit:
            target -= math.copysign(2.0 * math.pi, step)
            self.hold_until = t + abs(target - self.angle) / self.rate
        self.angle = target
        return target

    def holding(self, t):
        """True while the azimuth swings the long way round (command zero thrust)."""
        return t < self.hold_until


def lekf_pose(position, quat_xyzw, imu_position):
    """
    The CO's pose from the LEKF's odometry: position = the IMU in NED [m], quat_xyzw = BODY FRD -> NED, and the IMU's
    position in BODY FRD from the CO (vessel.yaml boat_interface.imu_position). Returns (north, east, psi, roll, pitch).
    """
    from scipy.spatial.transform import Rotation
    psi, theta, phi = Rotation.from_quat(quat_xyzw).as_euler("ZYX")
    p = np.asarray(position, dtype=float) - Rzyx(phi, theta, psi) @ np.asarray(imu_position, dtype=float)
    return float(p[0]), float(p[1]), float(psi), float(phi), float(theta)


class StampClock:
    """
    Stamps from another clock in local time: the boat's SentiBoard counter, or Gazebo's. The offset is the smallest
    (arrival - stamp) of the last `window` seconds, the clocks' offset plus the shortest transport delay, so
    stamp + offset keeps the time between measurements and puts the freshest one at its arrival. An offset more
    than `jump` seconds from the current one (the SentiBoard node restarted) starts it again.

        clock = StampClock()
        t_meas = clock(stamp_s, now_s)
    """

    def __init__(self, window=10.0, jump=1.0):
        from collections import deque
        self.window, self.jump = float(window), float(jump)
        self.samples = deque()
        self.offset = None
        self.resets = 0

    def __call__(self, stamp, arrival):
        off = arrival - stamp
        if self.offset is not None and abs(off - self.offset) > self.jump:
            self.samples.clear()
            self.resets += 1
        while self.samples and arrival - self.samples[0][0] > self.window:
            self.samples.popleft()
        self.samples.append((arrival, off))
        self.offset = min(o for _, o in self.samples)
        return stamp + self.offset
