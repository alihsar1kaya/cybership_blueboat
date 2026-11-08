#!/usr/bin/env python3
"""
DP step test: three goals in a row, measured from odometry.

  1. surge: (0,0,0)   -> (3,0,0)     3 m straight ahead
  2. sway:  (3,0,0)   -> (3,3,0)     3 m sideways (heading stays 0)
  3. yaw:   (3,3,0)   -> (3,3,90deg) turn on the spot

For each step: overshoot past the goal, largest sideways deviation from the straight line,
settling time (inside 0.10 m and 5 deg for good), and the final error.

Needs a running sim + DP stack, e.g.
    ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true use_gui:=false

    python3 dp_step_test.py --vessel microampere --label before
Writes ~/ros_ws/results/dp_step_<vessel>_<label>.csv
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

POS_TOL = 0.10            # [m]
YAW_TOL = math.radians(5)  # [rad]

STEPS = [
    ("surge", (3.0, 0.0, 0.0)),
    ("sway", (3.0, 3.0, 0.0)),
    ("yaw", (3.0, 3.0, math.pi / 2)),
]


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vessel", default="microampere")
    p.add_argument("--label", default="run")
    p.add_argument("--step-time", type=float, default=90.0, help="seconds per step")
    p.add_argument("--odom", default="sim/odom_true",
                   help="state to evaluate: sim/odom_true (truth, default) or measurement/odom")
    a = p.parse_args()

    rclpy.init()
    node = rclpy.create_node("dp_step_test")
    log = []  # (t, x, y, yaw, step)
    state = {"step": None}

    def on_odom(m):
        pos = m.pose.pose.position
        log.append((time.time(), pos.x, pos.y, yaw_of(m.pose.pose.orientation), state["step"]))

    node.create_subscription(Odometry, f"/{a.vessel}/{a.odom}", on_odom, 50)
    goal_pub = node.create_publisher(PoseStamped, "/goal_pose", 1)

    # wait for odometry and for the goal subscriber
    t0 = time.time()
    while (not log or goal_pub.get_subscription_count() == 0) and time.time() - t0 < 30:
        rclpy.spin_once(node, timeout_sec=0.1)

    start = (0.0, 0.0, 0.0)
    results = []
    for name, goal in STEPS:
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = goal[0], goal[1]
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(goal[2] / 2), math.cos(goal[2] / 2)
        state["step"] = name
        goal_pub.publish(msg)
        t_goal = time.time()
        while time.time() - t_goal < a.step_time:
            rclpy.spin_once(node, timeout_sec=0.05)

        r = np.array([(t - t_goal, x, y, psi) for t, x, y, psi, s in log if s == name])
        results.append(metrics(name, start, goal, r))
        start = goal

    node.destroy_node()
    rclpy.shutdown()

    os.makedirs(os.path.expanduser("~/ros_ws/results"), exist_ok=True)
    path = os.path.expanduser(f"~/ros_ws/results/dp_step_{a.vessel}_{a.label}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw", "step"])
        w.writerows(log)

    print(f"\nDP step test, {a.vessel}, '{a.label}'  (good = inside {POS_TOL} m and "
          f"{math.degrees(YAW_TOL):.0f} deg)")
    print(f"{'step':>6} {'overshoot':>10} {'sideways':>9} {'yaw dev':>8} {'settle':>8} "
          f"{'final pos':>10} {'final yaw':>10}")
    for r in results:
        settle = f"{r['settle']:.0f} s" if r["settle"] is not None else "never"
        print(f"{r['name']:>6} {r['overshoot']:9.2f}m {r['cross']:8.2f}m {r['yaw_dev']:7.1f}° "
              f"{settle:>8} {r['final_pos']:9.3f}m {r['final_yaw']:9.1f}°")
    print(f"log: {path}")


def metrics(name, start, goal, r):
    t, x, y, psi = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    d = np.array(goal[:2]) - np.array(start[:2])
    length = np.linalg.norm(d)
    rel = np.column_stack([x - start[0], y - start[1]])
    if length > 1e-6:
        u = d / length
        along = rel @ u
        cross = np.abs(rel @ np.array([-u[1], u[0]]))
        overshoot = max(0.0, along.max() - length)
        cross_max = cross.max()
    else:
        overshoot = 0.0
        cross_max = np.linalg.norm(rel, axis=1).max()
    yaw_err = np.array([wrap(goal[2] - v) for v in psi])
    if abs(wrap(goal[2] - start[2])) > 1e-6:
        # yaw step: overshoot = how far it turned past the goal
        sign = math.copysign(1, wrap(goal[2] - start[2]))
        yaw_dev = max(0.0, math.degrees(-(yaw_err * sign).min()))
    else:
        yaw_dev = math.degrees(np.abs(yaw_err).max())
    pos_err = np.hypot(goal[0] - x, goal[1] - y)
    inside = (pos_err < POS_TOL) & (np.abs(yaw_err) < YAW_TOL)
    settle = None
    if inside[-1]:
        outside = np.where(~inside)[0]
        settle = t[outside[-1] + 1] if len(outside) else t[0]
    return {"name": name, "overshoot": overshoot, "cross": cross_max, "yaw_dev": yaw_dev,
            "settle": settle, "final_pos": pos_err[-1], "final_yaw": math.degrees(abs(yaw_err[-1]))}


if __name__ == "__main__":
    main()
