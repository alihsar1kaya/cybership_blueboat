#!/usr/bin/env python3
"""
Obstacles for the guidance: circles clicked in RViz or read from a file, for the OCP planner and the display.

    guidance/add_obstacle   PoseWithCovarianceStamped -> a new circle at its position, radius 'radius'. The
                                             microampere.rviz config points the "2D Pose Estimate" tool here: it
                                             hits the water plane anywhere, like "2D Goal Pose"
    /clicked_point          PointStamped  -> the same ("Publish Point" only fires on drawn geometry, not open water)
    guidance/obstacles      MarkerArray   <- all circles as CYLINDER markers: position = centre (NED, frame
                                             'world'), scale.x = diameter. The position controller plans around
                                             them; a lidar node can later publish the same topic.
    ~/clear, ~/undo         Trigger       -> remove all clicked circles / the last one (file obstacles stay)

The file (parameter 'file', YAML) holds fixed obstacles such as a dock:

    obstacles:
      - [8.0, 0.0, 0.8]     # x north, y east, radius [m]

The harbour file (parameter 'harbour', cybership_config/config/<vessel>/harbour.yaml) adds the quays and berths for
docking: each quay drawn as a polygon (namespace 'quay'), each berth as an arrow ('berth'), and each four-cornered
quay covered by circles along its long axis (CYLINDER, 'quay_circles'), so the OCP planner keeps clear of it.

    ros2 run cybership_dp obstacle_manager_node.py --ros-args -r __ns:=/microampere -p radius:=0.5
"""
import math

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray


class ObstacleManager(Node):
    def __init__(self):
        super().__init__("obstacle_manager")
        self.declare_parameter("radius", 0.5)          # [m] of a clicked obstacle
        self.declare_parameter("file", "")              # YAML with fixed obstacles
        self.declare_parameter("frame", "world")
        self.declare_parameter("harbour", "")           # harbour YAML with quays and berths
        self.declare_parameter("harbour_shift", [0.0, 0.0])   # [m] NED map error, for tests
        self.fixed, self.clicked = [], []
        self.quays, self.berths = {}, {}
        harbour = self.get_parameter("harbour").value
        if harbour:
            from cybership_dp.docking import load_harbour
            self.quays, self.berths = load_harbour(harbour, shift=self.get_parameter("harbour_shift").value)
            for q in self.quays.values():
                self.fixed += self.quay_circles(q)
        path = self.get_parameter("file").value
        if path:
            with open(path) as f:
                self.fixed = [tuple(float(v) for v in o) for o in (yaml.safe_load(f) or {}).get("obstacles", [])]
        # transient local: a controller or RViz that starts later still gets the current list
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(MarkerArray, "guidance/obstacles", qos)
        self.create_subscription(PointStamped, "/clicked_point", lambda m: self.add(m.point.x, m.point.y), 10)
        self.create_subscription(PoseWithCovarianceStamped, "guidance/add_obstacle",
                                 lambda m: self.add(m.pose.pose.position.x, m.pose.pose.position.y), 10)
        self.create_service(Trigger, "~/clear", self.on_clear)
        self.create_service(Trigger, "~/undo", self.on_undo)
        self.publish()
        self.get_logger().info(f"Obstacles: {len(self.fixed)} from '{path}' and '{harbour}' ({len(self.quays)} quays, "
                               f"berths {list(self.berths)}); add circles of {self.get_parameter('radius').value} m "
                               f"with the RViz tool 2D Pose Estimate")

    @staticmethod
    def quay_circles(quay):
        """Circles along the long axis of a four-cornered quay that cover it (corners included)."""
        c = quay.corners
        if len(c) != 4:
            return []
        e1, e2 = c[1] - c[0], c[2] - c[1]
        long_, short = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
        length, width = np.linalg.norm(long_), np.linalg.norm(short)
        n = int(math.ceil(length / width))
        step = length / n
        r = math.hypot(width / 2, step / 2)
        centre, axis = quay.centre, long_ / length
        return [(*(centre + axis * (-length / 2 + step * (i + 0.5))), r) for i in range(n)]

    def add(self, x, y):
        r = float(self.get_parameter("radius").value)
        self.clicked.append((x, y, r))
        self.get_logger().info(f"Obstacle at ({x:.2f}, {y:.2f}), radius {r} m")
        self.publish()

    def on_clear(self, _req, res):
        self.clicked = []
        self.publish()
        res.success, res.message = True, "clicked obstacles removed"
        return res

    def on_undo(self, _req, res):
        res.success = bool(self.clicked)
        if self.clicked:
            self.clicked.pop()
        self.publish()
        res.message = f"{len(self.clicked)} clicked obstacles left"
        return res

    def publish(self):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id, clear.action = self.get_parameter("frame").value, Marker.DELETEALL
        arr.markers.append(clear)
        stamp = self.get_clock().now().to_msg()
        n_quay_circles = sum(len(self.quay_circles(q)) for q in self.quays.values())
        for i, (x, y, r) in enumerate(self.fixed + self.clicked):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.get_parameter("frame").value, stamp
            m.ns, m.id, m.type, m.action = "obstacles", i, Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2.0 * float(r)
            m.scale.z = 0.4
            m.color.r, m.color.g, m.color.b, m.color.a = 0.85, 0.15, 0.15, 0.7
            if i < n_quay_circles:                   # the quays' circles are for the planner: barely visible
                m.ns, m.color.a = "quay_circles", 0.12
            arr.markers.append(m)
        frame = self.get_parameter("frame").value
        for k, (name, q) in enumerate(self.quays.items()):
            m = Marker()
            m.header.frame_id, m.header.stamp = frame, stamp
            m.ns, m.id, m.type, m.action = "quay", k, Marker.TRIANGLE_LIST, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.45, 0.35, 0.9
            corners = [Point(x=float(px), y=float(py), z=0.0) for px, py in q.corners]
            for j in range(1, len(corners) - 1):    # a fan over the convex polygon
                m.points += [corners[0], corners[j], corners[j + 1]]
            arr.markers.append(m)
        for k, (name, b) in enumerate(self.berths.items()):
            x, y, psi = b["pose"]
            m = Marker()
            m.header.frame_id, m.header.stamp = frame, stamp
            m.ns, m.id, m.type, m.action = "berth", k, Marker.ARROW, Marker.ADD
            m.pose.position.x, m.pose.position.y = float(x), float(y)
            m.pose.orientation.z, m.pose.orientation.w = math.sin(psi / 2), math.cos(psi / 2)
            m.scale.x, m.scale.y, m.scale.z = 0.8, 0.12, 0.12
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.8, 0.3, 0.9
            arr.markers.append(m)
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
