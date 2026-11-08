#!/usr/bin/env python3
"""4-hjørnetest som venter til båten faktisk er i hjørnet før neste mål sendes.

Samme punkter som cybership_tests/four_corner_client.py (3x3 m kvadrat).
Posisjonsregulatoren returnerer action-resultatet umiddelbart, så her
sjekker vi selv mot odometri: innenfor toleranse i HOLD_S sekunder, eller timeout.

Bruk: python3 four_corner_wait.py [vessel_name]
"""
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry

POS_TOL = 0.10      # m
YAW_TOL = 0.05      # rad
HOLD_S = 3.0        # s innenfor toleranse
TIMEOUT_S = 90.0    # s per hjørne


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    vessel = sys.argv[1] if len(sys.argv) > 1 else "voyager"
    rclpy.init()
    node = rclpy.create_node("four_corner_wait", namespace=vessel)
    client = ActionClient(node, NavigateToPose, "navigate_to_pose")
    state = {}

    def on_odom(msg):
        q = msg.pose.pose.orientation
        state["x"] = msg.pose.pose.position.x
        state["y"] = msg.pose.pose.position.y
        state["psi"] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))

    node.create_subscription(Odometry, "measurement/odom", on_odom, 10)

    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, math.pi / 4],
                    [0, 1, math.pi / 4], [0, 0, 0]], dtype=float)
    pts[:, :2] = pts[:, :2] * 3 - 1.5

    if not client.wait_for_server(timeout_sec=15.0):
        node.get_logger().error("Action server ikke tilgjengelig")
        return 1

    for x, y, yaw in pts:
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "world"
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        rclpy.spin_until_future_complete(node, client.send_goal_async(goal))
        node.get_logger().info(f"Mål: x={x:.1f} y={y:.1f} psi={math.degrees(yaw):.0f}°")

        t0 = time.monotonic()
        inside_since = None
        while time.monotonic() - t0 < TIMEOUT_S:
            rclpy.spin_once(node, timeout_sec=0.05)
            if "x" not in state:
                continue
            ok = (math.hypot(x - state["x"], y - state["y"]) < POS_TOL
                  and abs(wrap(yaw - state["psi"])) < YAW_TOL)
            now = time.monotonic()
            inside_since = (inside_since or now) if ok else None
            if inside_since and now - inside_since >= HOLD_S:
                node.get_logger().info(f"  nådd etter {now - t0:.1f} s")
                break
        else:
            node.get_logger().warn(f"  timeout ({TIMEOUT_S:.0f} s) – går videre")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
