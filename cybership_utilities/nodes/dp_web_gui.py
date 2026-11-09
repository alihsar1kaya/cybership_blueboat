#!/usr/bin/env python3
"""
Simple web GUI for DP (one page, Norwegian): map with the boat and goal, click to set a goal, hold,
STOP, sea state (waves and current, only when the simulator runs) and thruster load.

    ros2 run cybership_utilities dp_web_gui.py --ros-args -r __ns:=/microampere -p vessel_model:=microampere
    open http://localhost:8080   (or http://<boat-ip>:8080 from another machine)

The page and the WebSocket (/ws) are served on the same port. The node talks to the DP stack through
its normal interfaces: /goal_pose (as RViz "2D Goal Pose"), the force mux (STOP selects the 'manual'
input and publishes zero force there) and the simulator's environment.* parameters.
"""
import asyncio
import json
import math
import os
import threading
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Wrench, WrenchStamped
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from topic_tools_interfaces.srv import MuxSelect
from websockets.asyncio.server import serve

from cybership_utilities import vessel_model

LABELS = {"port": "Babord", "starboard": "Styrbord", "bow_azimuth": "Baug"}
SEA = {"Hs": "environment.waves.Hs", "Tp": "environment.waves.Tp", "wave_dir": "environment.waves.direction_deg",
       "current": "environment.current.speed", "current_dir": "environment.current.direction_deg",
       "wind": "environment.wind.speed", "wind_dir": "environment.wind.direction_deg"}
DRIVE_INPUT, STOP_INPUT = "control/force/command/position", "control/force/command/manual"


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class DpWebGui(Node):
    def __init__(self):
        super().__init__("dp_web_gui")
        self.declare_parameter("vessel_model", "microampere")
        self.declare_parameter("port", 8080)
        self.declare_parameter("world_frame", "world")
        self.lock = threading.Lock()
        self.pose = {}            # source -> (t, x, y, psi)
        self.goal = None
        self.stopped = False
        self.sea = None           # None: no simulator
        cfg, _ = vessel_model.load_vessel_config(self.get_parameter("vessel_model").value)
        self.thrusters = {t["name"]: {"label": LABELS.get(t["name"], t["name"]), "max": float(t["max_force"]),
                                      "f": 0.0} for t in cfg.get("thrusters", [])}

        self.create_subscription(Odometry, "navigation/odom", lambda m: self._on_odom("est", m), 10)
        self.create_subscription(Odometry, "measurement/odom", lambda m: self._on_odom("meas", m), 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)   # also goals from RViz
        for name in self.thrusters:
            self.create_subscription(WrenchStamped, f"thruster/{name}/force",
                                     lambda m, n=name: self._on_force(n, m), 10)
        self.pub_goal = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.pub_stop = self.create_publisher(Wrench, STOP_INPUT, 10)
        self.mux = self.create_client(MuxSelect, "force_mux/select")
        ns = self.get_namespace().rstrip("/")
        self.sim = AsyncParameterClient(self, f"{ns}/sim")
        self.create_timer(0.1, self._on_timer)
        self.create_timer(2.0, self._read_sea)

    # --- ROS side -------------------------------------------------------------------------------
    def _on_odom(self, src, m):
        p = m.pose.pose
        with self.lock:
            self.pose[src] = (time.time(), p.position.x, p.position.y, yaw_of(p.orientation))

    def _on_goal(self, m):
        with self.lock:
            self.goal = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))

    def _on_force(self, name, m):
        f = m.wrench.force
        with self.lock:
            self.thrusters[name]["f"] = math.copysign(math.hypot(f.x, f.y), f.x if abs(f.x) > 1e-6 else 1.0)

    def _on_timer(self):
        if self.stopped:
            self.pub_stop.publish(Wrench())

    def _read_sea(self):
        if self.sea is not None or not self.sim.services_are_ready():
            return
        fut = self.sim.get_parameters(list(SEA.values()))

        def done(f):
            try:
                vals = [v.double_value for v in f.result().values]
                with self.lock:
                    self.sea = dict(zip(SEA.keys(), vals))
            except Exception as e:   # noqa: BLE001
                self.get_logger().warn(f"Could not read sea state from the simulator: {e}")
        fut.add_done_callback(done)

    def current_pose(self):
        """The observer estimate if it runs, else measurement/odom."""
        est = self.pose.get("est")
        if est and time.time() - est[0] < 1.0:
            return est[1:]
        meas = self.pose.get("meas")
        return meas[1:] if meas else None

    def send_goal(self, x, y, psi):
        m = PoseStamped()
        m.header.frame_id = self.get_parameter("world_frame").value
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y = float(x), float(y)
        m.pose.orientation.z, m.pose.orientation.w = math.sin(psi / 2), math.cos(psi / 2)
        self.pub_goal.publish(m)
        self.get_logger().info(f"Goal from web GUI: x {x:.1f} y {y:.1f} heading {math.degrees(psi):.0f} deg")

    def select(self, topic):
        if not self.mux.service_is_ready():
            self.get_logger().error("force_mux/select is not available; STOP does not work without the mux")
            return
        self.mux.call_async(MuxSelect.Request(topic=topic))

    # --- commands from the page -------------------------------------------------------------------
    def command(self, c):
        kind = c.get("cmd")
        with self.lock:
            pose = self.current_pose()
        if kind == "goal":
            psi = math.radians(float(c["heading_deg"])) if c.get("heading_deg") is not None else (pose[2] if pose else 0.0)
            self.send_goal(float(c["x"]), float(c["y"]), psi)
        elif kind == "hold" and pose:
            self.send_goal(*pose)
        elif kind == "stop":
            self.stopped = True
            self.select(STOP_INPUT)
            self.get_logger().warn("STOP from web GUI: zero thrust")
        elif kind == "resume":
            if pose:
                self.send_goal(*pose)    # hold where it is now, no jump back to the old goal
            self.select(DRIVE_INPUT)
            self.stopped = False
            self.get_logger().info("Resumed from web GUI: holding the current position")
        elif kind == "sea" and self.sea is not None:
            params = [Parameter(SEA[k], value=float(v)) for k, v in c.get("values", {}).items() if k in SEA]
            if params:
                self.sim.set_parameters(params)
                with self.lock:
                    self.sea.update({k: float(v) for k, v in c["values"].items() if k in SEA})

    def state(self):
        with self.lock:
            pose = self.current_pose()
            s = {"pose": pose, "goal": self.goal, "stopped": self.stopped, "sea": self.sea,
                 "source": "observer" if self.pose.get("est") and time.time() - self.pose["est"][0] < 1.0
                 else "måling",
                 "thrusters": [dict(name=n, **v) for n, v in self.thrusters.items()]}
        if pose and self.goal:
            s["dist"] = math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1])
            s["dpsi_deg"] = math.degrees((self.goal[2] - pose[2] + math.pi) % (2 * math.pi) - math.pi)
        return s


async def run_server(node, page):
    clients = set()

    async def handler(ws):
        clients.add(ws)
        try:
            async for raw in ws:
                try:
                    node.command(json.loads(raw))
                except Exception as e:   # noqa: BLE001
                    node.get_logger().warn(f"Bad command from the page: {e}")
        finally:
            clients.discard(ws)

    def process_request(connection, request):
        if request.path.startswith("/ws"):
            return None                  # WebSocket upgrade
        with open(page, encoding="utf-8") as f:
            r = connection.respond(200, f.read())
        r.headers["Content-Type"] = "text/html; charset=utf-8"
        return r

    port = int(node.get_parameter("port").value)
    async with serve(handler, "0.0.0.0", port, process_request=process_request):
        node.get_logger().info(f"DP web GUI on http://localhost:{port}")
        while rclpy.ok():
            if clients:
                msg = json.dumps(node.state())
                for ws in list(clients):
                    try:
                        await ws.send(msg)
                    except Exception:   # noqa: BLE001
                        clients.discard(ws)
            await asyncio.sleep(0.1)


def main():
    rclpy.init()
    node = DpWebGui()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    page = os.path.join(get_package_share_directory("cybership_utilities"), "html", "dp", "index.html")
    try:
        asyncio.run(run_server(node, page))
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
