#!/usr/bin/env python3
"""
What the DP believes against what is true, near a quay (simulator only: it reads the true state).

    sim/odom_true          the true pose          -> the true hull (white) against the TRUE quay (white outline)
    navigation/odom        the observer's estimate -> the believed hull (orange), drawn where the DP thinks the boat
                                                      lies RELATIVE TO THE QUAY: the estimate minus the map error
                                                      (harbour_shift). Orange on white = the DP knows the gap.
    sensor/gnss/ned        GNSS fixes              -> yellow dots (the last 50): multipath shows as scatter
    navigation/quay_used   lidar faces in the filter? -> in the text

    viz/docking            MarkerArray, frame 'world'
    viz/gap_true           Float64, the true hull's distance to the true quay [m] (< 0: into the fender)
    viz/gap_believed       Float64, the believed hull's distance to the DP's map of the quay [m]

With a map error (launch map_shift) or GNSS multipath (multipath:=true), without the lidar the two pictures differ:
the DP thinks the boat lies 10 cm off while it is pressing on the fender. With the lidar (gazebo:=true, the observer's
quay.enabled true) the believed gap follows the true one. Toggle it live:
    ros2 param set /microampere/dp_observer quay.enabled false
"""
import math
from collections import deque

import rclpy
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float64
from visualization_msgs.msg import Marker, MarkerArray

from cybership_dp.docking import hull_corners, load_harbour
from cybership_utilities import vessel_model


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class DockingViz(Node):
    def __init__(self):
        super().__init__("docking_viz")
        self.declare_parameter("vessel_model", "microampere")
        self.declare_parameter("harbour_file", "")
        self.declare_parameter("harbour_shift", [0.0, 0.0])
        cfg, _ = vessel_model.load_vessel_config(self.get_parameter("vessel_model").value)
        self.hull = (float(cfg["hull"]["length"]), float(cfg["hull"]["beam"]))
        path = self.get_parameter("harbour_file").value
        self.true_quays = list(load_harbour(path, self.hull)[0].values()) if path else []
        self.shift = [float(v) for v in self.get_parameter("harbour_shift").value]
        self.map_quays = list(load_harbour(path, self.hull, self.shift)[0].values()) if path else []
        self.true = self.est = None
        self.lidar = None
        self.fixes = deque(maxlen=50)
        self.create_subscription(Odometry, "sim/odom_true", lambda m: setattr(self, "true", m), 10)
        self.create_subscription(Odometry, "navigation/odom", lambda m: setattr(self, "est", m), 10)
        self.create_subscription(PointStamped, "sensor/gnss/ned", lambda m: self.fixes.append((m.point.x, m.point.y)), 10)
        self.create_subscription(Bool, "navigation/quay_used", lambda m: setattr(self, "lidar", m.data), 10)
        self.pub = self.create_publisher(MarkerArray, "viz/docking", 1)
        self.pub_true = self.create_publisher(Float64, "viz/gap_true", 10)
        self.pub_believed = self.create_publisher(Float64, "viz/gap_believed", 10)
        self.create_timer(0.2, self.tick)

    @staticmethod
    def pose(odom):
        p = odom.pose.pose.position
        return [p.x, p.y, yaw(odom.pose.pose.orientation)]

    def outline(self, mid, corners, rgb, width, z=-0.3):
        m = Marker()
        m.header.frame_id, m.header.stamp = "world", self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "docking_viz", mid, Marker.LINE_STRIP, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = width
        m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, 1.0)
        for c in list(corners) + [corners[0]]:
            m.points.append(Point(x=float(c[0]), y=float(c[1]), z=z))
        return m

    def tick(self):
        if self.true is None or not self.true_quays:
            return
        arr = MarkerArray()
        t = self.pose(self.true)
        gap_true = min(q.clearance(t, *self.hull) for q in self.true_quays)
        arr.markers.append(self.outline(0, hull_corners(t, *self.hull), (1.0, 1.0, 1.0), 0.03))
        for i, q in enumerate(self.true_quays):
            arr.markers.append(self.outline(10 + i, q.corners, (1.0, 1.0, 1.0), 0.02, z=-0.35))
        text = f"gap sant {100 * gap_true:.1f} cm"
        self.pub_true.publish(Float64(data=float(gap_true)))
        if self.est is not None:
            e = self.pose(self.est)
            gap_b = min(q.clearance(e, *self.hull) for q in self.map_quays)
            rel = [e[0] - self.shift[0], e[1] - self.shift[1], e[2]]      # relative to the true quay
            arr.markers.append(self.outline(1, hull_corners(rel, *self.hull), (1.0, 0.55, 0.0), 0.03, z=-0.32))
            text += f" | DP tror {100 * gap_b:.1f} cm"
            self.pub_believed.publish(Float64(data=float(gap_b)))
        text += " | lidar " + ("på" if self.lidar else "av")
        m = Marker()
        m.header.frame_id, m.header.stamp = "world", self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "docking_viz", 2, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = t[0], t[1], -1.2
        m.pose.orientation.w = 1.0
        m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
        m.text = text
        arr.markers.append(m)
        if self.fixes:
            g = Marker()
            g.header = m.header
            g.ns, g.id, g.type, g.action = "docking_viz", 3, Marker.POINTS, Marker.ADD
            g.pose.orientation.w = 1.0
            g.scale.x = g.scale.y = 0.04
            g.color.r, g.color.g, g.color.b, g.color.a = 1.0, 0.95, 0.1, 0.9
            g.points = [Point(x=float(x), y=float(y), z=-0.3) for x, y in self.fixes]
            arr.markers.append(g)
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = DockingViz()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
