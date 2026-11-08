#!/usr/bin/env python3
"""Logger posisjon (odom) og DP-referanse til CSV mens en test kjører.

Bruk: python3 record_dp.py <vessel_name> <utfil.csv>
"""
import csv
import math
import sys

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Recorder(Node):
    def __init__(self, vessel, path):
        super().__init__("dp_recorder", namespace=vessel)
        self.ref = (math.nan, math.nan, math.nan)
        self.t0 = None
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "x", "y", "psi", "x_d", "y_d", "psi_d"])
        self.create_subscription(Odometry, "measurement/odom", self.on_odom, 10)
        self.create_subscription(PoseStamped, "control/pose/debug/reference_pose", self.on_ref, 10)

    def on_ref(self, msg):
        p = msg.pose
        self.ref = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def on_odom(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        p = msg.pose.pose
        self.w.writerow([f"{t - self.t0:.3f}", p.position.x, p.position.y,
                         yaw_from_quat(p.orientation), *self.ref])


def main():
    rclpy.init()
    node = Recorder(sys.argv[1], sys.argv[2])
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.f.close()


if __name__ == "__main__":
    main()
