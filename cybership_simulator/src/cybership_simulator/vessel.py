#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configurable vessel simulator.

Instead of one hand-written class per vessel (voyager.py, enterprise.py, ...), this
simulator builds the hull and thrusters from a YAML file:

    cybership_config/config/<vessel_model>/vessel.yaml

The marine craft (kinematics + dynamics) is Fossen's model in cybership_utilities.marine_craft.

Environment (Fossen Ch. 10.3): ocean current, set with parameters, also while running, e.g.
    ros2 param set /microampere/sim environment.current.speed 0.2
    ros2 param set /microampere/sim environment.current.direction_deg 90.0   # flows towards east

Run:
    ros2 run cybership_simulator vessel.py --ros-args -r __ns:=/voyager \
        -p vessel_model:=voyager -p tf_prefix:=voyager
"""

import numpy as np
import rclpy
import geometry_msgs.msg
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from visualization_msgs.msg import Marker
import skadipy

from cybership_simulator.base import BaseSimulator
from cybership_utilities import vessel_model
from cybership_utilities.marine_craft.actuators import make_thruster
from cybership_utilities.marine_craft.environment import OceanCurrent


class ConfigurableSimulator(BaseSimulator):
    """Simulator whose hull and thrusters come from vessel.yaml."""

    def __init__(self):
        super().__init__(node_name="vessel_simulator")
        if not self._thruster_roll_pitch:
            # Thrusters give only X, Y, N (as in Fossen's Otter): drop the roll/pitch rows of tau = B u
            self.allocator._b_matrix[3:5, :] = 0.0
        self._setup_environment()

    # ------------------------------------------------------------------------
    # Environment (Fossen Ch. 10.3): ocean current
    # ------------------------------------------------------------------------

    def _setup_environment(self):
        p = {
            "environment.current.speed": 0.0,          # [m/s] mean speed
            "environment.current.direction_deg": 0.0,  # [deg] flows towards, 0 = north, 90 = east
            "environment.current.mu": 0.0,             # Gauss-Markov: mean reversion [1/s], 0 = off
            "environment.current.sigma": 0.0,          # Gauss-Markov: noise [m/s/sqrt(s)], 0 = off
            "environment.current.speed_min": 0.0,
            "environment.current.speed_max": 1.0,
            "environment.current.seed": -1,            # -1: random
        }
        for name, default in p.items():
            # dynamic typing: 'ros2 param set ... 90' (an integer) must work as well as 90.0
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
        g = lambda n: self.get_parameter(f"environment.current.{n}").value
        seed = g("seed")
        self._current = OceanCurrent(
            speed=float(g("speed")), direction_deg=float(g("direction_deg")), mu=float(g("mu")),
            sigma=float(g("sigma")), speed_min=float(g("speed_min")), speed_max=float(g("speed_max")),
            seed=None if int(seed) < 0 else int(seed))
        self.add_on_set_parameters_callback(self._on_environment_change)
        self._pub_current = self.create_publisher(
            geometry_msgs.msg.Vector3Stamped, "sim/environment/current", 1)
        self._pub_current_marker = self.create_publisher(Marker, "sim/environment/current_marker", 1)
        self.get_logger().info(
            f"Ocean current: {g('speed')} m/s towards {g('direction_deg')} deg "
            f"(mu {g('mu')}, sigma {g('sigma')})")

    def _on_environment_change(self, params):
        for param in params:
            if param.name == "environment.current.speed":
                self._current.set(speed=float(param.value))
            elif param.name == "environment.current.direction_deg":
                self._current.set(direction_deg=float(param.value))
            elif param.name == "environment.current.mu":
                self._current.mu = float(param.value)
            elif param.name == "environment.current.sigma":
                self._current.sigma = float(param.value)
            else:
                continue
            self.get_logger().info(f"{param.name} = {param.value}")
        return SetParametersResult(successful=True)

    def _publish_environment(self):
        v = self._current.velocity_ned()
        stamp = self.get_clock().now().to_msg()
        msg = geometry_msgs.msg.Vector3Stamped()
        msg.header.frame_id = self.world_frame_id
        msg.header.stamp = stamp
        msg.vector.x, msg.vector.y, msg.vector.z = (float(x) for x in v)
        self._pub_current.publish(msg)

        # Arrow at the vessel, 5 m per m/s, in the world (NED) frame
        eta = self.vessel.eta
        m = Marker()
        m.header.frame_id = self.world_frame_id
        m.header.stamp = stamp
        m.ns, m.id, m.type = "current", 0, Marker.ARROW
        start = geometry_msgs.msg.Point(x=float(eta[0]), y=float(eta[1]), z=0.0)
        end = geometry_msgs.msg.Point(x=float(eta[0] + 5.0 * v[0]), y=float(eta[1] + 5.0 * v[1]), z=0.0)
        m.points = [start, end]
        m.scale.x, m.scale.y, m.scale.z = 0.05, 0.12, 0.15
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 1.0, 0.9
        m.action = Marker.ADD if np.linalg.norm(v) > 1e-6 else Marker.DELETE
        self._pub_current_marker.publish(m)

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    def _load_config(self):
        """Read vessel.yaml once. Called from _create_vessel (the first hook BaseSimulator runs)."""
        if hasattr(self, "config"):
            return
        self.declare_parameter("vessel_model", "voyager")
        self.declare_parameter("vessel_config", "")  # optional explicit path, overrides vessel_model
        self.config, path = vessel_model.load_vessel_config(
            self.get_parameter("vessel_model").value, self.get_parameter("vessel_config").value
        )
        self.get_logger().info(
            f"Loaded vessel '{self.config['name']}' with "
            f"{len(self.config['thrusters'])} thrusters from {path}"
        )

    # ------------------------------------------------------------------------
    # BaseSimulator hooks
    # ------------------------------------------------------------------------

    def _create_vessel(self):
        self._load_config()
        # Mass, added mass, damping and restoring from vessel.yaml ('model' section, else from 'hull')
        params = vessel_model.model_parameters(self.config)
        nl = params["nonlinear"]
        self.get_logger().info(
            f"Model from '{params['source']}': mass {params['mass']:.2f} kg, "
            f"MA {np.round(params['MA'], 2).tolist()}, D {np.round(params['D'], 2).tolist()}, "
            f"G {np.round(params['G'], 1).tolist()}, X_uu {nl['surge_quadratic']}, "
            f"N_rr {nl['yaw_quadratic']:.1f}, cross-flow {'on' if nl['cross_flow'] else 'off'}, "
            f"cancel Munk {nl['cancel_munk']}, thruster roll/pitch moments {nl['thruster_roll_pitch_moments']}"
        )
        self._thruster_roll_pitch = nl["thruster_roll_pitch_moments"]
        return vessel_model.make_marine_craft(params, eta0=self.eta0)

    def _init_allocator(self):
        """Build skadipy actuators from the thruster list. The simulator only uses the B matrix."""
        actuators = []
        self._thruster_index = {}  # name -> first column in u
        col = 0
        for t in self.config["thrusters"]:
            position = skadipy.toolbox.Point(t["position"])
            if t["type"] == "fixed":
                actuators.append(skadipy.actuator.Fixed(
                    position=position,
                    orientation=skadipy.toolbox.Quaternion(
                        axis=(0.0, 0.0, 1.0), radians=np.deg2rad(t.get("direction_deg", 0.0))
                    ),
                ))
                self._thruster_index[t["name"]] = col
                col += 1
            elif t["type"] == "azimuth":
                actuators.append(skadipy.actuator.Azimuth(position=position))
                self._thruster_index[t["name"]] = col
                col += 2
            else:
                raise ValueError(f"Unknown thruster type '{t['type']}' for thruster '{t['name']}'")
        self._n_inputs = col

        dofs = [
            skadipy.allocator._base.ForceTorqueComponent.X,
            skadipy.allocator._base.ForceTorqueComponent.Y,
            skadipy.allocator._base.ForceTorqueComponent.Z,
            skadipy.allocator._base.ForceTorqueComponent.K,
            skadipy.allocator._base.ForceTorqueComponent.M,
            skadipy.allocator._base.ForceTorqueComponent.N,
        ]
        return skadipy.allocator.PseudoInverse(actuators=actuators, force_torque_components=dofs)

    STATE_EVERY = 5  # publish thruster/<name>/state every 5th sim step (20 Hz)

    def setup_thrusters(self):
        self.u = np.zeros((self._n_inputs, 1))
        self._issued_pubs = {}
        self._force_pubs = {}
        self._actuators = {}  # name -> actuator model (thruster dynamics, see marine_craft/actuators.py)
        self._step_count = 0
        for t in self.config["thrusters"]:
            name = t["name"]
            self._actuators[name] = make_thruster(t)
            self._issued_pubs[name] = self.create_publisher(
                geometry_msgs.msg.WrenchStamped, f"thruster/{name}/issued", 1
            )
            # Actual thrust as WrenchStamped, so RViz can draw it (thruster/<name>/state is not drawable)
            self._force_pubs[name] = self.create_publisher(
                geometry_msgs.msg.WrenchStamped, f"thruster/{name}/force", 1
            )
            self.create_subscription(
                geometry_msgs.msg.Wrench, f"thruster/{name}/command",
                lambda msg, t=t: self._on_thruster_command(t, msg), 10,
            )

    # ------------------------------------------------------------------------
    # Actuators: command -> (saturation, deadband) -> thruster dynamics -> actual thrust -> tau = B u
    # ------------------------------------------------------------------------

    def _on_thruster_command(self, t, msg: geometry_msgs.msg.Wrench):
        """Store the command; the actuator model turns it into actual thrust in iterate()."""
        name = t["name"]
        fx, fy = self._actuators[name].set_command(msg.force.x, msg.force.y)

        issued = geometry_msgs.msg.WrenchStamped()
        issued.header.frame_id = self._frame(t["frame"])
        issued.header.stamp = self.get_clock().now().to_msg()
        issued.wrench.force.x = fx
        issued.wrench.force.y = fy
        self._issued_pubs[name].publish(issued)

    def iterate(self):
        """Step environment and thruster dynamics, then the vessel (BaseSimulator.iterate: tau = B u)."""
        self._step_count += 1
        publish_state = self._step_count % self.STATE_EVERY == 0
        self._current.step(self.dt)
        self.vessel.current_ned = self._current.velocity_ned()
        if publish_state:
            self._publish_environment()
        for t in self.config["thrusters"]:
            act = self._actuators[t["name"]]
            act.step(self.dt)
            fx, fy, angle = act.output()
            idx = self._thruster_index[t["name"]]
            self.u[idx, 0] = fx
            if t["type"] == "azimuth":
                self.u[idx + 1, 0] = fy
            if publish_state:
                self.publish_thruster_state(
                    t["name"], t["frame"], fx, fy, azimuth=(t["type"] == "azimuth"),
                    angle=angle, angle_rate=getattr(act, "angle_rate", 0.0))
                force = geometry_msgs.msg.WrenchStamped()
                force.header.frame_id = self._frame(t["frame"])
                force.header.stamp = self.get_clock().now().to_msg()
                force.wrench.force.x, force.wrench.force.y = float(fx), float(fy)
                self._force_pubs[t["name"]].publish(force)
        super().iterate()


def main(args=None):
    rclpy.init(args=args)
    simulator = ConfigurableSimulator()
    try:
        rclpy.spin(simulator)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass  # normal stop (Ctrl+C / launch shutdown)
    finally:
        simulator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
