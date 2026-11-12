#!/usr/bin/env python3
"""
Show the simulated boat in a Gazebo world (for example a VRX harbour): Gazebo draws the water, the
surroundings and, later, the lidar and cameras, while our Fossen model in vessel.py stays the physics.

The node spawns a static model with the vessel's CAD mesh and moves it to the true state on
sim/odom_true at a fixed rate through the world's set_pose service. Gazebo itself applies no forces to it.

Frames: our simulator uses NED (x north, y east, z down) with a body frame x forward, y starboard,
z down; Gazebo uses ENU (x east, y north, z up) with x forward, y left, z up. The NED origin is put at
`origin` [x, y] (ENU metres) in the Gazebo world, on the water surface (z = 0 in VRX).

    ros2 run cybership_simulator gz_pose_bridge.py --ros-args -r __ns:=/microampere \
        -p world:=sydney_regatta -p mesh:=/path/to/microampere.stl
"""
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node as GzNode

# NED world -> ENU world, and body FRD (x forward, y starboard, z down) -> FLU (x forward, y left, z up)
NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
FRD_TO_FLU = np.diag([1.0, -1.0, -1.0])

MODEL_SDF = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="base_link">
      <visual name="hull">
        <pose>0 0 {mesh_z} 0 0 0</pose>
        <geometry><mesh><uri>file://{mesh}</uri></mesh></geometry>
        <material>
          <ambient>0.85 0.85 0.85 1</ambient>
          <diffuse>0.9 0.9 0.9 1</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def ned_to_gz(position, quat_xyzw, origin):
    """Position [m] and orientation (quaternion x, y, z, w) from NED/FRD to the Gazebo ENU/FLU world."""
    p = NED_TO_ENU @ np.asarray(position, dtype=float)
    p[0] += origin[0]
    p[1] += origin[1]
    R = NED_TO_ENU @ Rotation.from_quat(quat_xyzw).as_matrix() @ FRD_TO_FLU
    return p, Rotation.from_matrix(R).as_quat()


class GzPoseBridge(Node):
    def __init__(self):
        super().__init__("gz_pose_bridge")
        self.declare_parameter("world", "sydney_regatta")
        self.declare_parameter("model_name", "microampere")
        self.declare_parameter("mesh", "")
        self.declare_parameter("mesh_z", 0.102)          # [m] mesh deck above the waterline (URDF hull_height - draft)
        self.declare_parameter("origin", [-532.0, 162.0])  # [m] ENU, where NED (0, 0) is in the world
        self.declare_parameter("rate", 30.0)               # [Hz]
        g = lambda n: self.get_parameter(n).value
        self.world, self.name, self.origin = g("world"), g("model_name"), list(g("origin"))
        self.gz = GzNode()
        self.odom = None
        self.spawned = False
        self.mesh, self.mesh_z = g("mesh"), g("mesh_z")
        self.create_subscription(Odometry, "sim/odom_true", self._on_odom, 10)
        self.create_timer(1.0 / g("rate"), self._tick)
        self.get_logger().info(f"Gazebo world '{self.world}': {self.name} at ENU origin {self.origin}")

    def _on_odom(self, msg):
        self.odom = msg

    def _spawn(self):
        req = EntityFactory()
        req.sdf = MODEL_SDF.format(name=self.name, mesh=self.mesh, mesh_z=self.mesh_z)
        req.allow_renaming = False
        ok, rep = self.gz.request(f"/world/{self.world}/create", req, EntityFactory, Boolean, 2000)
        if ok and rep.data:
            self.get_logger().info(f"Spawned '{self.name}' in Gazebo")
        return ok and rep.data

    def _tick(self):
        if not self.spawned:
            self.spawned = self._spawn()          # retried until the Gazebo world is up
            return
        if self.odom is None:
            return
        p, o = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        pos, q = ned_to_gz([p.x, p.y, p.z], [o.x, o.y, o.z, o.w], self.origin)
        req = Pose()
        req.name = self.name
        req.position.x, req.position.y, req.position.z = (float(v) for v in pos)
        req.orientation.x, req.orientation.y, req.orientation.z, req.orientation.w = (float(v) for v in q)
        ok, _ = self.gz.request(f"/world/{self.world}/set_pose", req, Pose, Boolean, 200)
        if not ok:
            self.get_logger().warn("Gazebo set_pose failed", throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = GzPoseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
