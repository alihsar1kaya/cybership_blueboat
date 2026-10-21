#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configurable vessel simulator.

Instead of one hand-written class per vessel (voyager.py, enterprise.py, ...), this
simulator builds the hull and thrusters from a YAML file:

    cybership_config/config/<vessel_model>/vessel.yaml

The marine craft (kinematics + dynamics) is Fossen's model in cybership_utilities.marine_craft.

Run:
    ros2 run cybership_simulator vessel.py --ros-args -r __ns:=/voyager \
        -p vessel_model:=voyager -p tf_prefix:=voyager
"""

import numpy as np
import rclpy
import geometry_msgs.msg
import skadipy

from cybership_simulator.base import BaseSimulator
from cybership_utilities import vessel_model
from cybership_utilities.marine_craft.actuators import make_thruster


class ConfigurableSimulator(BaseSimulator):
    """Simulator whose hull and thrusters come from vessel.yaml."""

    def __init__(self):
        super().__init__(node_name="vessel_simulator")
        if not self._thruster_roll_pitch:
            # Thrusters give only X, Y, N (as in Fossen's Otter): drop the roll/pitch rows of tau = B u
            self.allocator._b_matrix[3:5, :] = 0.0

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
        self._actuators = {}  # name -> actuator model (thruster dynamics, see marine_craft/actuators.py)
        self._step_count = 0
        for t in self.config["thrusters"]:
            name = t["name"]
            self._actuators[name] = make_thruster(t)
            self._issued_pubs[name] = self.create_publisher(
                geometry_msgs.msg.WrenchStamped, f"thruster/{name}/issued", 1
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
        """Step the thruster dynamics, then the vessel (BaseSimulator.iterate computes tau = B u)."""
        self._step_count += 1
        publish_state = self._step_count % self.STATE_EVERY == 0
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
