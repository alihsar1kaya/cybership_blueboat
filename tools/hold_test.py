#!/usr/bin/env python3
"""
Station-keeping test on a running simulator: optionally drive to a goal first, then hold position for a
while and log what DP has to do (true pose, every thruster, allocation slack, towline tension).

    python3 tools/hold_test.py --label calm_tow --hold 120 [--goal "4 0 0"]   # writes results/hold_<label>.npz
"""
import argparse
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Wrench
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

from cybership_interfaces.msg import ThrusterState

THRUSTERS = ("port", "starboard", "bow_azimuth")
yaw = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ns", default="/microampere")
    ap.add_argument("--hold", type=float, default=120.0, help="[s] of station keeping to log")
    ap.add_argument("--goal", default="", help='"north east heading_deg" offsets to drive to first')
    a = ap.parse_args()
    rclpy.init()
    n = rclpy.create_node("hold_test")
    log = {"odom": [], "slack": [], "tow": [], "tow_odom": [], **{k: [] for k in THRUSTERS}}
    now = time.time
    n.create_subscription(Odometry, f"{a.ns}/sim/odom_true", lambda m: log["odom"].append(
        (now(), m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation))), 50)
    n.create_subscription(Wrench, f"{a.ns}/control/allocation/slack", lambda m: log["slack"].append(
        (now(), m.force.x, m.force.y, m.torque.z)), 50)
    n.create_subscription(Float64, f"{a.ns}/sim/tow/tension", lambda m: log["tow"].append((now(), m.data)), 50)
    n.create_subscription(Odometry, f"{a.ns}/sim/tow/odom", lambda m: log["tow_odom"].append(
        (now(), m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation))), 50)
    for k in THRUSTERS:
        n.create_subscription(ThrusterState, f"{a.ns}/thruster/{k}/state", lambda m, k=k: log[k].append(
            (now(), m.force.x, m.force.y, m.angle)), 50)

    def spin(sec):
        t = now()
        while now() - t < sec:
            rclpy.spin_once(n, timeout_sec=0.02)

    spin(3.0)
    while not log["odom"]:                       # the simulator may still be starting
        spin(0.5)
    x, y, psi = log["odom"][-1][1:4]
    ref = (x, y, psi)
    if a.goal:
        dn, de, h = (float(v) for v in a.goal.split())
        ref = (x + dn, y + de, math.radians(h))
        pub = n.create_publisher(PoseStamped, "/goal_pose", 10)
        while pub.get_subscription_count() == 0:
            spin(0.2)
        m = PoseStamped()
        m.header.frame_id = "world"
        m.header.stamp = n.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y = ref[0], ref[1]
        m.pose.orientation.z, m.pose.orientation.w = math.sin(ref[2] / 2), math.cos(ref[2] / 2)
        pub.publish(m)
        t0, since = now(), None
        while now() - t0 < 60.0:
            spin(0.2)
            x, y, p = log["odom"][-1][1:4]
            ok = math.hypot(ref[0] - x, ref[1] - y) < 0.05 and abs(wrap(ref[2] - p)) < math.radians(2)
            since = (since or now()) if ok else None
            if since and now() - since > 2.0:
                break
        print(f"goal reached after {now() - t0:.1f} s" if since else "goal not reached in 60 s", flush=True)
    t_hold = now()
    spin(a.hold)
    out = os.path.expanduser(f"~/ros_ws/results/hold_{a.label}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, ref=np.array(ref), t_hold=t_hold, **{k: np.array(v) for k, v in log.items()})
    print("saved", out)


if __name__ == "__main__":
    main()
