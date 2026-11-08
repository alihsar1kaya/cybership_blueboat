#!/usr/bin/env python3
"""Sender en sekvens av DP-mål og venter til hvert er nådd (mot odometri).

Mellompunkter: løs toleranse, ingen holdetid (båten «flyter» gjennom).
Siste punkt:   stram toleranse, må holdes HOLD_S sekunder.

Bruk: python3 waypoints.py <vessel> "x,y,yaw_deg" ["x,y,yaw_deg" ...]
"""
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry

TOL_VIA = (0.30, math.radians(20), 0.0)    # (pos m, yaw rad, hold s)
TOL_FINAL = (0.10, math.radians(3), 3.0)
TIMEOUT_S = 90.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    vessel = sys.argv[1]
    pts = [tuple(float(v) for v in p.split(",")) for p in sys.argv[2:]]
    rclpy.init()
    node = rclpy.create_node("waypoint_sequencer", namespace=vessel)
    client = ActionClient(node, NavigateToPose, "navigate_to_pose")
    state = {}

    def on_odom(msg):
        q = msg.pose.pose.orientation
        state["x"] = msg.pose.pose.position.x
        state["y"] = msg.pose.pose.position.y
        state["psi"] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))

    node.create_subscription(Odometry, "measurement/odom", on_odom, 10)
    if not client.wait_for_server(timeout_sec=15.0):
        node.get_logger().error("Action server ikke tilgjengelig")
        return 1

    for i, (x, y, yaw_deg) in enumerate(pts):
        final = i == len(pts) - 1
        pos_tol, yaw_tol, hold = TOL_FINAL if final else TOL_VIA
        yaw = math.radians(yaw_deg)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "world"
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        rclpy.spin_until_future_complete(node, client.send_goal_async(goal))
        node.get_logger().info(f"{'Kaiplass' if final else 'Via'}: x={x:.1f} y={y:.1f} psi={yaw_deg:.0f}°")

        t0 = time.monotonic()
        inside_since = None
        while time.monotonic() - t0 < TIMEOUT_S:
            rclpy.spin_once(node, timeout_sec=0.05)
            if "x" not in state:
                continue
            ok = (math.hypot(x - state["x"], y - state["y"]) < pos_tol
                  and abs(wrap(yaw - state["psi"])) < yaw_tol)
            now = time.monotonic()
            inside_since = (inside_since or now) if ok else None
            if inside_since and now - inside_since >= hold:
                node.get_logger().info(f"  nådd etter {now - t0:.1f} s")
                break
        else:
            node.get_logger().warn(f"  timeout ({TIMEOUT_S:.0f} s)")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
