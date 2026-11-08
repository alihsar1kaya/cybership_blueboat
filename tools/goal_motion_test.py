#!/usr/bin/env python3
"""
Record one goal manoeuvre (and the hold after it) from a running sim: true LF track, observer estimate,
GNSS fixes, heading and thruster forces. Saves results/goal_<label>.npz and prints how much "extra"
motion there was (path length vs. straight line, heading travel, thruster activity while holding).

    python3 tools/goal_motion_test.py --label A --goal 5 3 45 --move 90 --hold 60
"""
import argparse
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, WrenchStamped
from nav_msgs.msg import Odometry, Path

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="run")
ap.add_argument("--goal", type=float, nargs=3, default=[5.0, 3.0, 45.0])
ap.add_argument("--move", type=float, default=90.0)
ap.add_argument("--hold", type=float, default=60.0)
a = ap.parse_args()

rclpy.init()
n = rclpy.create_node("goal_probe")
ns = "/microampere"
d = {"lf": [], "est": [], "gnss": [], "true": [], "f": {}}
yaw = lambda q: 2 * math.atan2(q.z, q.w)
n.create_subscription(Odometry, f"{ns}/sim/odom_true", lambda m: d["true"].append(
    (time.time(), m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation))), 10)
n.create_subscription(Odometry, f"{ns}/navigation/odom", lambda m: d["est"].append(
    (time.time(), m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation))), 10)
n.create_subscription(PointStamped, f"{ns}/sensor/gnss/ned", lambda m: d["gnss"].append(
    (time.time(), m.point.x, m.point.y)), 10)
n.create_subscription(Path, f"{ns}/sim/path_lf", lambda m: d.__setitem__("lf", [
    (p.pose.position.x, p.pose.position.y) for p in m.poses]), 1)
for name in ("port", "starboard", "bow_azimuth"):
    n.create_subscription(WrenchStamped, f"{ns}/thruster/{name}/force", lambda m, k=name: d["f"].setdefault(k, []).append(
        (time.time(), m.wrench.force.x, m.wrench.force.y)), 10)
pub = n.create_publisher(PoseStamped, "/goal_pose", 1)
t0 = time.time()
while (not d["true"] or pub.get_subscription_count() == 0) and time.time() - t0 < 30:
    rclpy.spin_once(n, timeout_sec=0.1)
msg = PoseStamped()
msg.header.frame_id = "world"
msg.pose.position.x, msg.pose.position.y = a.goal[0], a.goal[1]
psi = math.radians(a.goal[2])
msg.pose.orientation.z, msg.pose.orientation.w = math.sin(psi / 2), math.cos(psi / 2)
pub.publish(msg)
t_goal = time.time()
while time.time() - t_goal < a.move + a.hold:
    rclpy.spin_once(n, timeout_sec=0.05)

tr = np.array(d["true"]); tr[:, 0] -= t_goal
est = np.array(d["est"]) if d["est"] else np.zeros((0, 4))
if len(est):
    est[:, 0] -= t_goal
gn = np.array(d["gnss"]) if d["gnss"] else np.zeros((0, 3))
if len(gn):
    gn[:, 0] -= t_goal
lf = np.array(d["lf"])
f = {k: np.array(v) for k, v in d["f"].items()}
for v in f.values():
    v[:, 0] -= t_goal
os.makedirs(os.path.expanduser("~/ros_ws/results"), exist_ok=True)
np.savez(os.path.expanduser(f"~/ros_ws/results/goal_{a.label}.npz"), true=tr, est=est, gnss=gn, lf=lf,
         goal=np.array(a.goal), move=a.move, **{f"f_{k}": v for k, v in f.items()})

mv = tr[(tr[:, 0] >= 0) & (tr[:, 0] <= a.move)]
hold = tr[tr[:, 0] > a.move]
straight = math.hypot(a.goal[0] - mv[0, 1], a.goal[1] - mv[0, 2])
lf_len = np.sum(np.hypot(*np.diff(lf, axis=0).T)) if len(lf) > 1 else float("nan")
heading_travel = np.degrees(np.sum(np.abs(np.diff(np.unwrap(mv[:, 3])))))
act = 0.0
for v in f.values():
    h = v[v[:, 0] > a.move]
    if len(h) > 2:
        act += np.mean(np.abs(np.diff(np.hypot(h[:, 1], h[:, 2])) / np.diff(h[:, 0])))
err_hold = np.hypot(hold[:, 1] - a.goal[0], hold[:, 2] - a.goal[1])
print(f"{a.label}: straight {straight:.1f} m, LF path {lf_len:.1f} m | heading travel {heading_travel:.0f} deg "
      f"(needed {abs(a.goal[2]):.0f}) | hold: error mean {err_hold.mean():.2f} m max {err_hold.max():.2f} m, "
      f"thruster activity {act:.1f} N/s")
