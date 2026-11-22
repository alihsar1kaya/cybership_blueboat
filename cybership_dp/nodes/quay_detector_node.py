#!/usr/bin/env python3
"""
Quay face from the lidar (cybership_dp/quay_detection.py): a RANSAC line through the points that can be a quay,
10 times a second, as a measurement relative to the boat.

    /ouster/points        PointCloud2  in, the Ouster (the real boat's topic; Gazebo's lidar in the simulator)
    navigation/odom       Odometry     in, roll and pitch to level the cloud; with the harbour file also the pose,
                                       to gate the fit to the face where the map says it should be
    perception/quay       PoseWithCovarianceStamped out, BODY FRD (frame <ns>/base_link_ned): position = the foot of
                                       the perpendicular from the CO on the face (d n), orientation = the normal's
                                       angle alpha (from the boat to the quay); covariance [0] = sigma_d^2,
                                       [35] = sigma_alpha^2
    perception/quay_marker Marker      the fitted face, for RViz

    ros2 run cybership_dp quay_detector_node.py --ros-args -r __ns:=/microampere -p vessel_model:=microampere

Stamps: the output carries the cloud's stamp. On the boat the Ouster's stamps are in SentiBoard time (PPS + NMEA),
the same clock as the LEKF and the sensors; never compare them with now().
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker

from cybership_dp.docking import load_harbour
from cybership_dp.quay_detection import QuayDetector
from cybership_utilities import vessel_model


class QuayDetectorNode(Node):
    def __init__(self):
        super().__init__("quay_detector")
        self.declare_parameter("vessel_model", "microampere")
        self.declare_parameter("cloud_topic", "/ouster/points")
        self.declare_parameter("harbour_file", "")
        self.declare_parameter("harbour_shift", [0.0, 0.0])     # [m] NED map error, for tests
        self.declare_parameter("gate", 0.5)                 # [m] around the map's face, when the pose is known
        self.declare_parameter("body_frame", "base_link_ned")
        cfg, _ = vessel_model.load_vessel_config(self.get_parameter("vessel_model").value)
        lid = cfg.get("lidar") or {}
        hull = (float(cfg["hull"]["length"]), float(cfg["hull"]["beam"]))
        self.det = QuayDetector(lid.get("position_flu", [0.0, 0.0, 0.5]), lid.get("orientation_flu_deg", [0, 0, 0]),
                                hull=hull, r_min=float((lid.get("range") or [0.5])[0]))
        self.faces = []                                       # (unit normal into the water, offset, ends) per face
        path = self.get_parameter("harbour_file").value
        if path:
            quays, _ = load_harbour(path, hull, self.get_parameter("harbour_shift").value)
            for q in quays.values():
                for i in range(len(q.corners)):
                    a, b, n, _ = q.face(i)
                    self.faces.append((n, float(n @ a), a, b))
        self.odom = None
        ns = self.get_namespace().strip("/")
        self.frame = f"{ns}/{self.get_parameter('body_frame').value}" if ns else self.get_parameter("body_frame").value
        self.create_subscription(Odometry, "navigation/odom", lambda m: setattr(self, "odom", m), 10)
        self.create_subscription(PointCloud2, self.get_parameter("cloud_topic").value, self.on_cloud,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, "perception/quay", 10)
        self.pub_marker = self.create_publisher(Marker, "perception/quay_marker", 10)
        self.n_ok = self.n_all = 0
        self.create_timer(10.0, self.report)
        from cybership_utilities.loop_timing import LoopTimer
        self.timing = LoopTimer(self, "quay fit", period=0.1)
        self.get_logger().info(f"Quay detector on {self.get_parameter('cloud_topic').value}: lidar at "
                               f"{lid.get('position_flu')} FLU, {len(self.faces)} mapped faces")

    def attitude(self):
        """(roll, pitch, (north, east, psi) or None) from the estimate."""
        if self.odom is None:
            return 0.0, 0.0, None
        q = self.odom.pose.pose.orientation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        psi = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        p = self.odom.pose.pose.position
        return roll, pitch, (p.x, p.y, psi)

    def prior(self, pose):
        """The nearest mapped face the boat is in front of, as a line in BODY FRD (n from the boat to the quay)."""
        if pose is None or not self.faces:
            return None
        x, y, psi = pose
        c, s = math.cos(psi), math.sin(psi)
        best = None
        for n_out, off, a, b in self.faces:
            dist = float(n_out @ np.array([x, y]) - off)      # the CO's distance in front of the face
            t = (b - a) / np.linalg.norm(b - a)
            along = float(t @ (np.array([x, y]) - a))
            if dist <= 0.0 or dist > 15.0 or along < -3.0 or along > np.linalg.norm(b - a) + 3.0:
                continue
            if best is None or dist < best[0]:
                n_b = np.array([c * -n_out[0] + s * -n_out[1], -s * -n_out[0] + c * -n_out[1]])
                best = (dist, n_b)
        return None if best is None else (best[1], best[0])

    def on_cloud(self, msg):
        with self.timing:
            self._on_cloud(msg)

    def _on_cloud(self, msg):
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(float)
        roll, pitch, pose = self.attitude()
        self.n_all += 1
        r = self.det.fit(pts, roll, pitch, prior=self.prior(pose), gate=self.get_parameter("gate").value)
        if not r["ok"]:
            return
        self.n_ok += 1
        out = PoseWithCovarianceStamped()
        out.header.stamp, out.header.frame_id = msg.header.stamp, self.frame
        foot = r["d"] * r["normal"]
        out.pose.pose.position.x, out.pose.pose.position.y = float(foot[0]), float(foot[1])
        out.pose.pose.orientation.z, out.pose.pose.orientation.w = math.sin(r["alpha"] / 2), math.cos(r["alpha"] / 2)
        cov = [0.0] * 36
        cov[0], cov[7], cov[35] = r["sigma_d"] ** 2, r["sigma_d"] ** 2, r["sigma_alpha"] ** 2
        out.pose.covariance = cov
        self.pub.publish(out)
        t = np.array([-r["normal"][1], r["normal"][0]])
        along = r["points"] @ t
        m = Marker()
        m.header = out.header
        m.ns, m.id, m.type, m.action = "quay_fit", 0, Marker.LINE_STRIP, Marker.ADD
        m.pose.orientation.w = 1.0
        for s in (along.min(), along.max()):
            p = foot + (s - float(foot @ t)) * t
            m.points.append(Point(x=float(p[0]), y=float(p[1]), z=-0.2))
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.9, 1.0
        self.pub_marker.publish(m)
        self.last = r

    def report(self):
        if self.n_all:
            msg = f"Quay detector: {self.n_ok}/{self.n_all} scans with a face"
            if self.n_ok and hasattr(self, "last"):
                r = self.last
                msg += (f", last: {r['d']:.2f} m at {math.degrees(r['alpha']):.1f} deg, {r['inliers']} points over "
                        f"{r['span']:.1f} m, rms {100 * r['rms']:.1f} cm")
            self.get_logger().info(msg)
        self.n_ok = self.n_all = 0


def main(args=None):
    rclpy.init(args=args)
    node = QuayDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
