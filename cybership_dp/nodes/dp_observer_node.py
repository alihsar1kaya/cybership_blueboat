#!/usr/bin/env python3
"""
Observer block (Fossen 2021, Ch. 13.4.6): DP Kalman filter from GNSS + compass to the estimated
state the controller uses.

    sensor/gnss                 NavSatFix  -> local NED (flat Earth, origin from vessel.yaml), lever arm removed
    sensor/imu                  Imu        -> heading (compass) and roll/pitch (for the lever arm)
    control/force/command/mux   Wrench     -> tau, the input of the model
    control/allocation/slack    Wrench     -> the part of tau the thrusters could not give (added to tau)
    /control/allocator/tau_allocated  WrenchStamped -> with blueboat.interface (boat_interface:=true) instead of the
                                two above: the force the sent thrusts give, from our allocation or the team's
    navigation/odom             Odometry   <- eta_hat (pose) and nu_hat (twist over ground, BODY)
    navigation/bias             Vector3Stamped <- bias force b_hat in NED [N, N, N m]
    navigation/current          Vector3Stamped <- current velocity estimate in NED [m/s] (current.estimate)
    perception/quay             PoseWithCovarianceStamped -> with harbour_file: the lidar's quay face (quay_detector_node,
                                BODY FRD), matched to a mapped face; corrects the position ACROSS that face and the
                                heading (DPKalmanFilter.correct_line). Near a quay GNSS suffers from multipath, and the
                                berth is defined against the mapped quay anyway: this puts the boat where the lidar
                                sees it relative to the quay; along the face GNSS still rules. While faces come
                                (quay.gnss_along_only), GNSS only corrects along the face, so it does not pull back.

Measurements (parameter 'input'):
    sim    sensor/gnss (NavSatFix, the antenna) and sensor/imu (Imu with orientation: compass, roll, pitch), the
           simulator's own sensors
    lekf   /navigation/odometry/ned (Odometry), the real boat's navigation filter (blueboat_lekf, 50 Hz): the IMU's
           pose, moved to the CO with boat_interface.imu_position, as a position (lekf.sigma_position) and a heading
           (lekf.sigma_heading_deg) measurement. The boat has no compass; the LEKF's heading comes from the two
           GNSS antennas. Its stamps are SentiBoard time (epoch 2000), mapped to ROS time by the smallest arrival
           delay (blueboat_io.StampClock). /navigation/ned_origin is checked against the map's origin: the quays are
           in that NED frame only if the LEKF's origin is pinned (ned_reference_ecef) to it.

The filter predicts at 'rate' Hz and corrects when a GNSS fix or a heading arrives (asynchronous,
Ch. 13.4.4). Without GNSS it keeps predicting from the model and the compass (dead reckoning).
M, D and the nonlinear damping come from vessel.yaml, the same model as the controller.

Delayed GNSS (delay_compensation): a fix arrives ~0.1 s after it was taken (header.stamp). The filter
goes back to the stored state at that time, applies the fix and predicts forward again with the stored
inputs and headings, so the old measurement is compared with the state it belongs to.
The GNSS noise is taken from the message covariance (it grows when an RTK receiver loses its fix).

    ros2 run cybership_dp dp_observer_node.py --ros-args -r __ns:=/microampere -p vessel_model:=microampere
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Vector3Stamped, Wrench, WrenchStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, NavSatFix

from cybership_utilities import blueboat_io, vessel_model
from cybership_utilities.marine_craft.tow import LoadCell
from cybership_utilities.marine_craft.sensors import gnss_config, llh2flat
from cybership_utilities.observer import AsyncObserver, DPKalmanFilter, remove_lever_arm

IDX = [0, 1, 5]


class DPObserverNode(Node):
    def __init__(self):
        super().__init__("dp_observer")
        self.declare_parameter("vessel_model", "microampere")
        self.declare_parameter("rate", 50.0)
        # tuned for smooth estimates (2026-10-26 goal tests): lower q gave 25 % less position and 50 %
        # less velocity error than [2, 0.5] and still follows a 0.15 m/s current
        self.declare_parameter("q_bias", [0.2, 0.2, 0.02])     # [N^2/s, N^2/s, (N m)^2/s]
        self.declare_parameter("q_nu", [0.05, 0.05, 0.005])
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("gnss_profile", "standard")   # must match the receiver (sim: sensors.gnss_profile)
        # Wave filter (Fossen 13.4.6): peak frequency of the WF model; 0 = no wave filter
        self.declare_parameter("wave_filter.omega0", 0.0)       # [rad/s], e.g. 2 pi / Tp
        self.declare_parameter("wave_filter.zeta", 0.1)
        self.declare_parameter("wave_filter.q", [0.1, 0.1, 0.001])
        # Ocean current as a state (velocity in NED): the damping then acts on the velocity through the
        # water. q small: the current changes over minutes, and a fast estimate picks up model errors
        # during manoeuvres (1e-4 and up let it drift 0.1-0.3 m/s in the sweep tests).
        self.declare_parameter("current.estimate", True)
        self.declare_parameter("current.q", 1.0e-5)             # [(m/s)^2/s]
        self.declare_parameter("current.p0", 0.04)              # [(m/s)^2] initial variance (0.2 m/s)
        self.declare_parameter("delay_compensation", True)
        self.declare_parameter("input", "sim")                  # "sim" or "lekf" (see the module docstring)
        self.declare_parameter("lekf.topic", "/navigation/odometry/ned")
        self.declare_parameter("lekf.sigma_position", 0.02)     # [m] the LEKF with RTK fixed (ESTIMATE)
        self.declare_parameter("lekf.sigma_heading_deg", 0.5)   # [deg] dual-antenna heading (ESTIMATE)
        self.declare_parameter("gnss_timeout", 1.0)            # [s] warn and dead-reckon after this
        # towed load (launch tow:=true tow_model:=true): the load cell's force (sensor/tow_force) is a known
        # input; else the filter puts the load's pull and inertia into the bias, and the bias feedforward
        # pushes the boat off the goal after a turn
        self.declare_parameter("tow.enabled", False)
        # lidar quay faces (perception/quay) against the map: gates for matching, and a factor on their sigmas
        self.declare_parameter("harbour_file", "")
        self.declare_parameter("harbour_shift", [0.0, 0.0])
        self.declare_parameter("quay.gate", [0.5, 10.0])       # [m, deg]
        self.declare_parameter("quay.sigma_scale", 1.5)
        # the lidar on or off while running (ros2 param set /<ns>/dp_observer quay.enabled false), to see what it does
        self.declare_parameter("quay.enabled", True)
        # while the lidar sees a face (the last 1 s), GNSS only along it: across it the lidar alone (false: both,
        # and the estimate ends between them, half a map error away)
        self.declare_parameter("quay.gnss_along_only", True)
        # the estimated hull is kept out of the mapped quays (DPKalmanFilter.constrain_halfplane): the fenders stop
        # the boat, and without it the estimate went on 5-9 cm into the quay after a docking
        self.declare_parameter("quay.constrain", True)

        config, path = vessel_model.load_vessel_config(self.get_parameter("vessel_model").value)
        self.config = config
        params = vessel_model.model_parameters(config)
        craft = vessel_model.make_marine_craft(params)
        M = vessel_model.M_eff_3dof(params)
        damping = lambda nu: -craft.damping(np.array([nu[0], nu[1], 0, 0, 0, nu[2]]))[IDX]
        w0 = self.get_parameter("wave_filter.omega0").value
        wf = None if w0 <= 0 else {"omega0": w0, "zeta": self.get_parameter("wave_filter.zeta").value,
                                   "q": self.get_parameter("wave_filter.q").value}
        current = None
        if self.get_parameter("current.estimate").value:
            current = {"q": self.get_parameter("current.q").value, "p0": self.get_parameter("current.p0").value}
        self.kf = DPKalmanFilter(M, vessel_model.D_3dof(params),
                                 q_bias=self.get_parameter("q_bias").value,
                                 q_nu=self.get_parameter("q_nu").value, damping=damping, wave_filter=wf,
                                 current=current)
        s = config.get("sensors", {})
        gnss = gnss_config(config, self.get_parameter("gnss_profile").value) if "gnss" in s else {}
        compass = s.get("compass", {})
        self.lever_arm = np.asarray(gnss.get("lever_arm", [0, 0, 0]), dtype=float)
        self.lat0 = math.radians(gnss.get("origin_lat_deg", 63.4305))
        self.lon0 = math.radians(gnss.get("origin_lon_deg", 10.3951))
        # measurement noise: white noise plus the slow GNSS wander the filter does not model
        self.sigma_gnss = math.hypot(gnss.get("sigma", 0.5), gnss.get("bias_sigma", 0.0))
        self.sigma_psi = math.radians(compass.get("sigma_deg", 1.0))

        self.tau = np.zeros(3)
        self.slack, self.t_slack = np.zeros(3), -math.inf
        self.roll_pitch = (0.0, 0.0)
        self.h = 1.0 / self.get_parameter("rate").value
        # the filter with the delayed-fix handling (shared with the offline tests)
        self.observer = AsyncObserver(self.kf, self.h, self.sigma_psi,
                                      delay_compensation=self.get_parameter("delay_compensation").value)
        self.t_gnss = None
        self.gnss_lost = False

        self.input = self.get_parameter("input").value
        if self.input == "lekf":
            self._setup_lekf(config)
        else:
            self.create_subscription(NavSatFix, "sensor/gnss", self.on_gnss, 10)
            self.create_subscription(Imu, "sensor/imu", self.on_imu, 10)
        self.declare_parameter("blueboat.interface", False)
        if self.get_parameter("blueboat.interface").value:
            self.create_subscription(WrenchStamped, "/control/allocator/tau_allocated",
                                     lambda m: self.on_tau(m.wrench), 10)
        else:
            self.create_subscription(Wrench, "control/force/command/mux", self.on_tau, 10)
            self.create_subscription(Wrench, "control/allocation/slack", self.on_slack, 10)
        self.pub_odom = self.create_publisher(Odometry, "navigation/odom", 10)
        self.pub_bias = self.create_publisher(Vector3Stamped, "navigation/bias", 10)
        self.pub_current = self.create_publisher(Vector3Stamped, "navigation/current", 10)
        from cybership_utilities.loop_timing import LoopTimer
        self.timing_step = LoopTimer(self, "filter step", period=self.h)
        self.timing_meas = LoopTimer(self, "measurements")
        self.create_timer(self.h, self.step)
        self.load_cell = None
        if self.get_parameter("tow.enabled").value and config.get("tow"):
            self.load_cell = LoadCell.from_config(config["tow"])
            self.create_subscription(WrenchStamped, "sensor/tow_force", self.on_tow_force, 10)
            self.get_logger().info(f"Towing: load cell force (sensor/tow_force, filter {self.load_cell.T} s) as a known input")
        self._setup_quay()
        if self.input == "lekf":
            measurements = (f"LEKF {self.get_parameter('lekf.topic').value} (sigma {self.sigma_lekf:.2f} m, "
                            f"{math.degrees(self.sigma_lekf_psi):.1f} deg, IMU at {self.imu_position.tolist()} m)")
        else:
            measurements = (f"GNSS sigma {self.sigma_gnss:.2f} m, compass {math.degrees(self.sigma_psi):.1f} deg, "
                            f"lever arm {self.lever_arm.tolist()}")
        self.get_logger().info(f"DP Kalman filter (Fossen 13.4.6, wave filter {wf}, current {current}) from {path}: "
                               f"{measurements}")

    def on_tau(self, msg):
        self.tau = np.array([msg.force.x, msg.force.y, msg.torque.z])

    def _setup_quay(self):
        """Mapped quay faces (unit normal out into the water, offset) for the lidar's measurements."""
        self.faces, self.quay_stats, self.quays = [], [0, 0], []
        path = self.get_parameter("harbour_file").value
        if not path:
            return
        from cybership_dp.docking import load_harbour
        quays, _ = load_harbour(path, shift=self.get_parameter("harbour_shift").value)
        L, B = float(self.config["hull"]["length"]), float(self.config["hull"]["beam"])
        self.hull_body = np.array([[L / 2, B / 2], [L / 2, -B / 2], [-L / 2, -B / 2], [-L / 2, B / 2]])
        self.n_constrained = 0
        for q in quays.values():
            self.quays.append((np.asarray(q.normals, float), np.asarray(q.offsets, float)))
            for i in range(len(q.corners)):
                a, _, n, _ = q.face(i)
                self.faces.append((np.asarray(n, float), float(n @ a)))
        self.create_subscription(PoseWithCovarianceStamped, "perception/quay", self.on_quay, 10)
        self.create_timer(10.0, self._quay_report)
        from std_msgs.msg import Bool
        self._Bool, self.quay_used_recent = Bool, 0
        self.pub_quay_used = self.create_publisher(Bool, "navigation/quay_used", 10)
        self.create_timer(1.0, self._quay_used_tick)

    def _quay_used_tick(self):
        """navigation/quay_used: whether lidar quay faces went into the filter in the last second."""
        self.pub_quay_used.publish(self._Bool(data=self.quay_used_recent > 0))
        self.quay_used_recent = 0

    def on_quay(self, msg):
        """
        A lidar quay face: d (CO to the face) and alpha (the normal from the boat to the quay, BODY). Matched to the
        mapped face it can be (within the gates), then two corrections: the position across the face,
        n . p = d + offset, and the heading, psi = angle(quay normal) - alpha.
        """
        if not self.observer.ready or not self.get_parameter("quay.enabled").value:
            return
        d_m = math.hypot(msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        alpha = 2.0 * math.atan2(q.z, q.w)
        x, y, psi = self.kf.eta
        gate_d, gate_a = self.get_parameter("quay.gate").value
        best = None
        for n_out, off in self.faces:
            d_pred = float(n_out @ np.array([x, y]) - off)
            a_pred = math.atan2(-n_out[1], -n_out[0]) - psi
            ea = (alpha - a_pred + math.pi) % (2 * math.pi) - math.pi
            if d_pred > 0 and abs(d_m - d_pred) < gate_d and abs(ea) < math.radians(gate_a):
                if best is None or abs(d_m - d_pred) < best[0]:
                    best = (abs(d_m - d_pred), n_out, off)
        self.quay_stats[1] += 1
        if best is None:
            return
        _, n_out, off = best
        k = self.get_parameter("quay.sigma_scale").value
        sigma_d = k * max(math.sqrt(max(msg.pose.covariance[0], 0.0)), 0.01)
        sigma_a = k * max(math.sqrt(max(msg.pose.covariance[35], 0.0)), math.radians(0.2))
        psi_m = math.atan2(-n_out[1], -n_out[0]) - alpha
        self.observer.gnss_along_only = self.get_parameter("quay.gnss_along_only").value
        self.observer.line(self.now(), n_out, d_m + off, sigma_d, (psi_m + math.pi) % (2 * math.pi) - math.pi, sigma_a)
        self.quay_stats[0] += 1
        self.quay_used_recent += 1

    def _constrain_to_water(self):
        """
        The estimated hull out of the mapped quays: for each quay, if the hull is behind all its faces, the face it is
        least deep behind (where a fender would push it out) as an inequality, n . p >= offset + the hull's reach.
        """
        if not self.quays or not self.get_parameter("quay.constrain").value:
            return
        x, y, psi = self.kf.eta
        c, s = math.cos(psi), math.sin(psi)
        corners = self.hull_body @ np.array([[c, s], [-s, c]])       # the hull's corners from the CO, NED
        p = np.array([x, y])
        for normals, offsets in self.quays:
            reach = np.max(-(corners @ normals.T), axis=0)              # how far the hull reaches towards each face
            clear = normals @ p - reach - offsets
            i = int(np.argmax(clear))
            if -0.5 < clear[i] < 0.0 and self.kf.constrain_halfplane(normals[i], offsets[i] + reach[i]):
                self.n_constrained += 1

    def _quay_report(self):
        if self.n_constrained:
            self.get_logger().info(f"Estimate kept out of the quay {self.n_constrained} times (fender contact)")
            self.n_constrained = 0
        used, seen = self.quay_stats
        if seen:
            self.get_logger().info(f"Lidar quay: {used}/{seen} faces used (matched to the map)")
        self.quay_stats = [0, 0]

    def on_tow_force(self, msg):
        self.load_cell.update(self.now(), msg.wrench.force.x, msg.wrench.force.y)

    def on_slack(self, msg):
        # saturated thrusters: the boat gets tau + slack, and the filter should be told the same,
        # or the missing force ends up in the bias and the current
        self.slack, self.t_slack = np.array([msg.force.x, msg.force.y, msg.torque.z]), self.now()

    def _setup_lekf(self, config):
        """The boat's LEKF odometry as the position and heading measurement (input lekf)."""
        self.imu_position = np.asarray((config.get("boat_interface") or {}).get("imu_position", [0.0, 0.0, 0.0]), float)
        self.sigma_lekf = float(self.get_parameter("lekf.sigma_position").value)
        self.sigma_lekf_psi = math.radians(float(self.get_parameter("lekf.sigma_heading_deg").value))
        self.observer.sigma_psi = self.sigma_lekf_psi
        self.lekf_clock = blueboat_io.StampClock()
        self.create_subscription(Odometry, self.get_parameter("lekf.topic").value, self.on_lekf, qos_profile_sensor_data)
        try:
            from blueboat_interfaces.msg import GpsOrigin
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(GpsOrigin, "/navigation/ned_origin", self.on_ned_origin, latched)
        except ImportError:
            self.get_logger().warn("No blueboat_interfaces: the LEKF's NED origin is not checked against the map")

    def on_lekf(self, msg):
        with self.timing_meas:
            self._on_lekf(msg)

    def _on_lekf(self, msg):
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        north, east, psi, roll, pitch = blueboat_io.lekf_pose([p.x, p.y, p.z], [o.x, o.y, o.z, o.w], self.imu_position)
        self.roll_pitch = (roll, pitch)
        now = self.now()
        resets = self.lekf_clock.resets
        t_meas = self.lekf_clock(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, now)
        if self.lekf_clock.resets != resets:
            self.get_logger().warn("LEKF stamps jumped (SentiBoard restarted?): clock mapping started again")
        if self.gnss_lost:
            self.get_logger().info(f"LEKF back after {now - self.t_gnss:.1f} s")
            self.gnss_lost = False
        self.t_gnss = now
        self.observer.heading(now, psi)
        self.observer.delay_compensation = self.get_parameter("delay_compensation").value
        self.observer.gnss(now, t_meas, north, east, self.sigma_lekf)

    def on_ned_origin(self, msg):
        """The quays (harbour file) are in the NED frame of the map's origin; the LEKF's must be the same."""
        n, e, _ = llh2flat(math.radians(msg.longitude), math.radians(msg.latitude), msg.altitude, self.lon0, self.lat0)
        off = math.hypot(n, e)
        log = self.get_logger().error if off > 0.5 else self.get_logger().info
        log(f"LEKF NED origin {msg.latitude:.7f}, {msg.longitude:.7f} is {off:.2f} m from the map's origin "
            f"({math.degrees(self.lat0):.7f}, {math.degrees(self.lon0):.7f})"
            + (": pin the LEKF's ned_reference_ecef there, or the quays are off by that much" if off > 0.5 else ""))

    def on_imu(self, msg):
        q = msg.orientation
        phi, theta, psi = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
        self.roll_pitch = (phi, theta)
        self.observer.heading(self.now(), psi)

    def on_gnss(self, msg):
        with self.timing_meas:
            self._on_gnss(msg)

    def _on_gnss(self, msg):
        north, east, down = llh2flat(math.radians(msg.longitude), math.radians(msg.latitude), msg.altitude,
                                     self.lon0, self.lat0)
        p = remove_lever_arm([north, east, down], self.lever_arm, *self.roll_pitch, self.kf.eta[2])
        sigma = self.sigma_gnss
        if msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            sigma = math.sqrt(max(msg.position_covariance[0], msg.position_covariance[4]))
        now = self.now()
        if self.gnss_lost:
            self.get_logger().info(f"GNSS back after {now - self.t_gnss:.1f} s")
            self.gnss_lost = False
        self.t_gnss = now
        t_meas = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.observer.delay_compensation = self.get_parameter("delay_compensation").value
        self.observer.gnss(now, t_meas, p[0], p[1], sigma)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def step(self):
        with self.timing_step:
            self._step()

    def _step(self):
        if not self.observer.ready:
            return
        now = self.now()
        if not self.gnss_lost and self.t_gnss is not None and now - self.t_gnss > self.get_parameter("gnss_timeout").value:
            self.gnss_lost = True
            self.get_logger().warn("No GNSS: dead reckoning on the model and the compass" if self.input != "lekf"
                                   else "No LEKF odometry: dead reckoning on the model")
        self._constrain_to_water()
        self.publish()
        slack = self.slack if now - self.t_slack < 0.5 else 0.0
        tow = self.load_cell.wrench() if self.load_cell is not None else 0.0
        self.observer.step(now, self.tau + slack + tow)

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.get_parameter("world_frame").value
        eta, nu = self.kf.eta, self.kf.nu
        odom.pose.pose.position.x, odom.pose.pose.position.y = float(eta[0]), float(eta[1])
        q = Rotation.from_euler("xyz", [*self.roll_pitch, eta[2]]).as_quat()
        o = odom.pose.pose.orientation
        o.x, o.y, o.z, o.w = (float(v) for v in q)
        P = self.kf.P
        cov = np.zeros((6, 6))
        cov[np.ix_([0, 1, 5], [0, 1, 5])] = P[self.kf.i_eta, self.kf.i_eta]
        odom.pose.covariance = cov.flatten().tolist()
        odom.twist.twist.linear.x, odom.twist.twist.linear.y = float(nu[0]), float(nu[1])
        odom.twist.twist.angular.z = float(nu[2])
        self.pub_odom.publish(odom)
        b = Vector3Stamped()
        b.header.stamp, b.header.frame_id = stamp, odom.header.frame_id
        b.vector.x, b.vector.y, b.vector.z = (float(v) for v in self.kf.bias)
        self.pub_bias.publish(b)
        if self.kf.nc:
            c = Vector3Stamped()
            c.header.stamp, c.header.frame_id = stamp, odom.header.frame_id
            c.vector.x, c.vector.y = (float(v) for v in self.kf.current)
            self.pub_current.publish(c)


def main(args=None):
    rclpy.init(args=args)
    node = DPObserverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
