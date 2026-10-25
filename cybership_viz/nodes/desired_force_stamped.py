#!/usr/bin/env python3
"""
Republishes the desired force tau_d (geometry_msgs/Wrench, not drawable in RViz) as a WrenchStamped
in the body frame, so RViz can draw it next to the applied force.

    ros2 run cybership_viz desired_force_stamped.py --ros-args -r __ns:=/microampere

Subscribes:  control/force/command/mux (tau_d), control/force/applied (for the body frame id)
Publishes:   control/force/command/stamped
"""

import rclpy
from geometry_msgs.msg import Wrench, WrenchStamped
from rclpy.node import Node


class DesiredForceStamped(Node):
    def __init__(self):
        super().__init__("desired_force_stamped")
        self.body_frame = None
        self.create_subscription(WrenchStamped, "control/force/applied",
                                 lambda m: setattr(self, "body_frame", m.header.frame_id), 10)
        self.create_subscription(Wrench, "control/force/command/mux", self.on_tau_d, 10)
        self.pub = self.create_publisher(WrenchStamped, "control/force/command/stamped", 1)

    def on_tau_d(self, msg):
        if not self.body_frame:
            return
        out = WrenchStamped()
        out.header.frame_id = self.body_frame
        out.header.stamp = self.get_clock().now().to_msg()
        out.wrench = msg
        self.pub.publish(out)


def main():
    rclpy.init()
    node = DesiredForceStamped()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
