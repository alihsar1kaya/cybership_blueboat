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
import array

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
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
      </visual>{sensors}
    </link>
  </model>
</sdf>
"""

# Ouster OS1 as a gpu_lidar: the scan on gz topic <topic>, the cloud on <topic>/points (bridged to /ouster/points)
LIDAR_SDF = """
      <sensor name="os1" type="gpu_lidar">
        <pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>
        <topic>{topic}</topic>
        <gz_frame_id>os_sensor</gz_frame_id>
        <update_rate>{rate}</update_rate>
        <always_on>true</always_on>
        <lidar>
          <scan>
            <horizontal><samples>{h}</samples><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal>
            <vertical><samples>{v}</samples><min_angle>{vmin}</min_angle><max_angle>{vmax}</max_angle></vertical>
          </scan>
          <range><min>{rmin}</min><max>{rmax}</max><resolution>0.003</resolution></range>
          <noise><type>gaussian</type><mean>0.0</mean><stddev>{noise}</stddev></noise>
          <visibility_mask>0x1</visibility_mask>
        </lidar>
      </sensor>"""


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
        # lidar on the boat: vessel.yaml 'lidar' as [x, y, z, roll, pitch, yaw] FLU from the CO [m, deg], empty = none
        self.declare_parameter("lidar_pose", [0.0])
        self.declare_parameter("lidar_config", "")          # vessel.yaml path, for the lidar's other settings
        self.declare_parameter("lidar_topic", "ouster")
        self.declare_parameter("cloud_topic", "/ouster/points")  # the ROS topic for the lidar's cloud
        g = lambda n: self.get_parameter(n).value
        self.world, self.name, self.origin = g("world"), g("model_name"), list(g("origin"))
        self.gz = GzNode()
        self.odom = None
        self.spawned = False
        self.mesh, self.mesh_z = g("mesh"), g("mesh_z")
        self.create_subscription(Odometry, "sim/odom_true", self._on_odom, 10)
        self.create_timer(1.0 / g("rate"), self._tick)
        if len(g("lidar_pose")) == 6:
            self.pub_cloud = self.create_publisher(PointCloud2, g("cloud_topic"), qos_profile_sensor_data)
            self.gz.subscribe(PointCloudPacked, "/" + g("lidar_topic").strip("/") + "/points", self._on_cloud)
        self.get_logger().info(f"Gazebo world '{self.world}': {self.name} at ENU origin {self.origin}")

    def _on_odom(self, msg):
        self.odom = msg

    def _on_cloud(self, m):
        """
        The lidar's cloud from Gazebo (a Gazebo transport thread) to ROS, stamped with ROS time when it arrives.
        Gazebo's clock starts at 0 when the world starts and is not the simulator's wall clock: with its stamps
        RViz and TF found no pose for the cloud and dropped it (ros_gz_bridge keeps them).
        """
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = next((d.value[0] for d in m.header.data if d.key == "frame_id" and d.value), "os_sensor")
        msg.height, msg.width = m.height, m.width
        # Gazebo numbers the types from 0 (INT8), sensor_msgs/PointField from 1
        msg.fields = [PointField(name=f.name, offset=f.offset, datatype=f.datatype + 1, count=max(f.count, 1))
                      for f in m.field]
        msg.is_bigendian, msg.point_step, msg.row_step = m.is_bigendian, m.point_step, m.row_step
        msg.is_dense = m.is_dense
        msg.data = array.array("B", m.data)
        self.pub_cloud.publish(msg)

    def _lidar_sdf(self):
        pose = list(self.get_parameter("lidar_pose").value)
        if len(pose) != 6:
            return ""
        import math
        import yaml
        cfg = {}
        path = self.get_parameter("lidar_config").value
        if path:
            with open(path) as f:
                cfg = (yaml.safe_load(f) or {}).get("lidar") or {}
        half = math.radians(float(cfg.get("vertical_fov_deg", 45.0))) / 2
        rng = cfg.get("range", [0.5, 50.0])
        return LIDAR_SDF.format(
            x=pose[0], y=pose[1], z=pose[2], roll=math.radians(pose[3]), pitch=math.radians(pose[4]),
            yaw=math.radians(pose[5]), topic=self.get_parameter("lidar_topic").value, rate=cfg.get("rate", 10.0),
            h=int(cfg.get("horizontal_samples", 512)), v=int(cfg.get("channels", 32)), vmin=-half, vmax=half,
            rmin=rng[0], rmax=rng[1], noise=cfg.get("noise", 0.01))

    def _spawn(self):
        req = EntityFactory()
        req.sdf = MODEL_SDF.format(name=self.name, mesh=self.mesh, mesh_z=self.mesh_z, sensors=self._lidar_sdf())
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
