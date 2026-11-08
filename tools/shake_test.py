#!/usr/bin/env python3
"""Stationkeeping 60 s: roll/pitch/yaw/position wobble (truth) and thruster activity, azimuth reversals."""
import math, sys, time
import numpy as np, rclpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from scipy.spatial.transform import Rotation as R
rclpy.init(); n = rclpy.create_node("shake"); d = {"o": [], "f": {}}
def odom(m):
    q = m.pose.pose.orientation
    d["o"].append((time.time(), m.pose.pose.position.x, m.pose.pose.position.y, *R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")))
n.create_subscription(Odometry, "/microampere/sim/odom_true", odom, 50)
for k in ("port", "starboard", "bow_azimuth"):
    n.create_subscription(WrenchStamped, f"/microampere/thruster/{k}/force",
                          lambda m, k=k: d["f"].setdefault(k, []).append((time.time(), m.wrench.force.x, m.wrench.force.y)), 10)
t0 = time.time()
while time.time() - t0 < 60: rclpy.spin_once(n, timeout_sec=0.02)
o = np.array(d["o"]); o = o[o[:, 0] - t0 > 10]
s = lambda a: np.degrees(np.std(a))
print(f"{sys.argv[1]}: roll std {s(o[:,3]):.2f} deg, pitch {s(o[:,4]):.2f} deg, yaw {s(np.unwrap(o[:,5])):.2f} deg, "
      f"position std {100*np.std(np.hypot(o[:,1]-o[:,1].mean(), o[:,2]-o[:,2].mean())):.1f} cm")
for k, v in d["f"].items():
    a = np.array(v); a = a[a[:, 0] - t0 > 10]; mag = np.hypot(a[:, 1], a[:, 2])
    line = f"   {k:12s} thrust mean {mag.mean():5.2f} N std {mag.std():5.2f} N, rate of change {np.mean(np.abs(np.diff(mag)/np.diff(a[:,0]))):5.1f} N/s"
    if k == "bow_azimuth":
        ang = np.degrees(np.arctan2(a[:, 2], a[:, 1])); flips = np.sum(np.abs(np.diff(np.unwrap(np.radians(ang)))) > np.radians(120))
        line += f", direction flips {flips}"
    print(line)
