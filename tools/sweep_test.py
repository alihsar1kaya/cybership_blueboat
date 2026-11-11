#!/usr/bin/env python3
"""
Drive a running simulator through a fixed list of goals and log what the boat does: true state, desired
force, every thruster's actual force and azimuth angle, and the allocation slack. One goal at a time; the
next starts when the boat has settled (5 cm, 2 deg for 6 s) or after a timeout. Offsets are in the world
frame (north, east) from where the boat is when the goal is given; the heading is absolute [deg].

    python3 tools/sweep_test.py --label calm            # writes ~/ros_ws/results/sweep_<label>.npz
"""
import argparse
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Wrench
from nav_msgs.msg import Odometry

from cybership_interfaces.msg import ThrusterState

# (north [m], east [m], final heading [deg]) offsets, chosen to cover ahead/behind/sideways/diagonal,
# short and long moves, and small and large heading changes
GOALS = [
    (4, 0, 0), (-3, 0, 0), (0, 3, 0), (0, -1, 0), (5, 5, 45), (5, -5, 45), (-6, -2, 180),
    (8, 0, 90), (0, 0, 270), (1, 1, 270), (-8, 4, 0), (3, -6, 270), (0, 0, 0), (10, 0, 0),
]
THRUSTERS = ("port", "starboard", "bow_azimuth")
yaw = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ns", default="/microampere")
    ap.add_argument("--timeout", type=float, default=80.0)
    a = ap.parse_args()
    rclpy.init()
    n = rclpy.create_node("sweep_probe")
    log = {"odom": [], "tau": [], "slack": [], **{k: [] for k in THRUSTERS}}
    now = time.time
    n.create_subscription(Odometry, f"{a.ns}/sim/odom_true", lambda m: log["odom"].append(
        (now(), m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation),
         m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z)), 50)
    n.create_subscription(Wrench, f"{a.ns}/control/force/command/position", lambda m: log["tau"].append(
        (now(), m.force.x, m.force.y, m.torque.z)), 50)
    n.create_subscription(Wrench, f"{a.ns}/control/allocation/slack", lambda m: log["slack"].append(
        (now(), m.force.x, m.force.y, m.torque.z)), 50)
    for k in THRUSTERS:
        n.create_subscription(ThrusterState, f"{a.ns}/thruster/{k}/state", lambda m, k=k: log[k].append(
            (now(), m.force.x, m.force.y, m.angle)), 50)
    pub = n.create_publisher(PoseStamped, "/goal_pose", 10)

    def spin(sec):
        t = now()
        while now() - t < sec:
            rclpy.spin_once(n, timeout_sec=0.02)

    spin(3.0)
    goals = []
    for dn, de, hdg in GOALS:
        x, y, psi = log["odom"][-1][1:4]
        g = (x + dn, y + de, math.radians(hdg))
        m = PoseStamped()
        m.header.frame_id = "world"
        m.pose.position.x, m.pose.position.y = g[0], g[1]
        m.pose.orientation.z, m.pose.orientation.w = math.sin(g[2] / 2), math.cos(g[2] / 2)
        while pub.get_subscription_count() == 0:      # the goal client must be there to hear it
            spin(0.2)
        t0 = now()
        m.header.stamp = n.get_clock().now().to_msg()
        pub.publish(m)
        settled_since = None
        while now() - t0 < a.timeout:
            spin(0.2)
            x, y, psi = log["odom"][-1][1:4]
            ok = math.hypot(g[0] - x, g[1] - y) < 0.05 and abs((g[2] - psi + math.pi) % (2 * math.pi) - math.pi) < math.radians(2)
            settled_since = (settled_since or now()) if ok else None
            if settled_since and now() - settled_since > 6.0:
                break
        goals.append((t0, now(), *g))
        print(f"goal {len(goals):2d} offset ({dn:+.0f}, {de:+.0f}) heading {hdg:3d}: "
              f"{'settled' if settled_since else 'TIMEOUT'} after {now() - t0:.0f} s", flush=True)
    out = os.path.expanduser(f"~/ros_ws/results/sweep_{a.label}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, goals=np.array(goals), offsets=np.array(GOALS, dtype=float),
             **{k: np.array(v) for k, v in log.items()})
    print("saved", out)


if __name__ == "__main__":
    main()
