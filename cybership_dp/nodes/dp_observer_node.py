#!/usr/bin/env python3
"""
Observer block (Fossen 2021, Ch. 13.4.6): DP Kalman filter from GNSS + compass to the estimated
state the controller uses.

    sensor/gnss                 NavSatFix  -> local NED (flat Earth, origin from vessel.yaml), lever arm removed
    sensor/imu                  Imu        -> heading (compass) and roll/pitch (for the lever arm)
    control/force/command/mux   Wrench     -> tau, the input of the model
    navigation/odom             Odometry   <- eta_hat (pose) and nu_hat (twist, BODY)
    navigation/bias             Vector3Stamped <- bias force b_hat in NED [N, N, N m]

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
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped, Wrench
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, NavSatFix

from cybership_utilities import vessel_model
from cybership_utilities.marine_craft.sensors import gnss_config, llh2flat
from cybership_utilities.observer import DPKalmanFilter, remove_lever_arm

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
        self.declare_parameter("delay_compensation", True)
        self.declare_parameter("gnss_timeout", 1.0)            # [s] warn and dead-reckon after this

        config, path = vessel_model.load_vessel_config(self.get_parameter("vessel_model").value)
        params = vessel_model.model_parameters(config)
        craft = vessel_model.make_marine_craft(params)
        damping = lambda nu: -craft.damping(np.array([nu[0], nu[1], 0, 0, 0, nu[2]]))[IDX]
        w0 = self.get_parameter("wave_filter.omega0").value
        wf = None if w0 <= 0 else {"omega0": w0, "zeta": self.get_parameter("wave_filter.zeta").value,
                                   "q": self.get_parameter("wave_filter.q").value}
        self.kf = DPKalmanFilter(vessel_model.M_eff_3dof(params), vessel_model.D_3dof(params),
                                 q_bias=self.get_parameter("q_bias").value,
                                 q_nu=self.get_parameter("q_nu").value, damping=damping, wave_filter=wf)
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
        self.roll_pitch = (0.0, 0.0)
        self.initialized = {"gnss": False, "psi": False}
        self.h = 1.0 / self.get_parameter("rate").value
        # history for delayed fixes: states before each prediction and the headings in between, ~1 s
        self.history = deque(maxlen=int(1.0 / self.h))
        self.headings = deque(maxlen=int(2.0 / self.h))
        self.t_gnss = None
        self.gnss_lost = False

        self.create_subscription(NavSatFix, "sensor/gnss", self.on_gnss, 10)
        self.create_subscription(Imu, "sensor/imu", self.on_imu, 10)
        self.create_subscription(Wrench, "control/force/command/mux", self.on_tau, 10)
        self.pub_odom = self.create_publisher(Odometry, "navigation/odom", 10)
        self.pub_bias = self.create_publisher(Vector3Stamped, "navigation/bias", 10)
        self.create_timer(self.h, self.step)
        self.get_logger().info(f"DP Kalman filter (Fossen 13.4.6, wave filter {wf}) from {path}: "
                               f"GNSS sigma {self.sigma_gnss:.2f} m, "
                               f"compass {math.degrees(self.sigma_psi):.1f} deg, lever arm {self.lever_arm.tolist()}")

    def on_tau(self, msg):
        self.tau = np.array([msg.force.x, msg.force.y, msg.torque.z])

    def on_imu(self, msg):
        q = msg.orientation
        phi, theta, psi = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
        self.roll_pitch = (phi, theta)
        if not self.initialized["psi"]:
            self.kf.x[self.kf.nw + 2] = psi
            self.initialized["psi"] = True
        self.kf.correct_compass(psi, self.sigma_psi)
        self.headings.append((self.now(), psi))

    def on_gnss(self, msg):
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
        if not self.initialized["gnss"]:
            self.kf.x[self.kf.nw:self.kf.nw + 2] = p[0:2]
            self.initialized["gnss"] = True
        t_meas = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if self.get_parameter("delay_compensation").value and now - t_meas > 0.5 * self.h and self.history:
            self.correct_delayed(t_meas, p, sigma)
        else:
            self.kf.correct_gnss(p[0], p[1], sigma)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def correct_delayed(self, t_meas, p, sigma):
        """Apply a fix taken at t_meas: back to the stored state then, correct, predict forward again."""
        hist = list(self.history)
        k = next((i for i in range(len(hist) - 1, -1, -1) if hist[i][0] <= t_meas), None)
        if k is None:                       # older than the history: use it as if it were new
            self.kf.correct_gnss(p[0], p[1], sigma)
            return
        t0, x, P, _ = hist[k]
        self.kf.x, self.kf.P = x.copy(), P.copy()
        self.kf.correct_gnss(p[0], p[1], sigma)
        times = [h[0] for h in hist[k:]] + [self.now()]
        for j, (t_j, _, _, tau_j) in enumerate(hist[k:]):
            self.kf.predict(tau_j, self.h)
            for t_psi, psi in self.headings:
                if t_j < t_psi <= times[j + 1]:
                    self.kf.correct_compass(psi, self.sigma_psi)

    def step(self):
        if not all(self.initialized.values()):
            return
        now = self.now()
        if not self.gnss_lost and self.t_gnss is not None and now - self.t_gnss > self.get_parameter("gnss_timeout").value:
            self.gnss_lost = True
            self.get_logger().warn("No GNSS: dead reckoning on the model and the compass")
        self.publish()
        self.history.append((now, self.kf.x.copy(), self.kf.P.copy(), self.tau.copy()))
        self.kf.predict(self.tau, self.h)

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
