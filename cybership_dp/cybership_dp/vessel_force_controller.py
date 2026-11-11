"""
Configurable thrust allocation.

Builds the skadipy actuators and allocator from the same vessel.yaml the simulator uses:

    cybership_config/config/<vessel_model>/vessel.yaml

so the thruster layout is written in exactly one place. Replaces the per-vessel classes
(voyager/, enterprise/, drillship/) for any vessel that has a vessel.yaml.

allocation.method in vessel.yaml:
  pseudo_inverse   skadipy, unconstrained (default)
  constrained_qp   Fossen 11.2.3: min f'Wf + s'Qs, s.t. T(alpha) f = tau + s, thruster limits,
                   with the azimuths at their ACTUAL angle (from thruster/<name>/state).
                   Publishes the slack s (what the thrusters cannot give) on control/allocation/slack.
                   With allocation.inflow_compensation the commands are corrected for the thrust the
                   propellers lose at speed, from measurement/odom (velocity) and navigation/current
                   (the observer's current estimate, so the speed is through the water).
"""

from __future__ import annotations

import math

import numpy as np
import skadipy
import skadipy.allocator
from geometry_msgs.msg import Vector3Stamped, Wrench
from nav_msgs.msg import Odometry

from cybership_interfaces.msg import ThrusterState

from cybership_utilities import vessel_model

from cybership_dp.force_controller_base import BaseForceControllerROS
from cybership_dp.force_controller_base import ThrusterPublisherSpec
from cybership_dp.constrained_allocation import ConstrainedAllocator

_DOFS = {
    "X": skadipy.allocator.ForceTorqueComponent.X,
    "Y": skadipy.allocator.ForceTorqueComponent.Y,
    "Z": skadipy.allocator.ForceTorqueComponent.Z,
    "K": skadipy.allocator.ForceTorqueComponent.K,
    "M": skadipy.allocator.ForceTorqueComponent.M,
    "N": skadipy.allocator.ForceTorqueComponent.N,
}


class VesselForceControllerROS(BaseForceControllerROS):
    def __init__(self, *args, **kwargs):
        super().__init__(
            node_name="thrust_allocation_node",
            allow_undeclared_parameters=True,
            *args,
            **kwargs,
        )

    # BaseForceControllerROS runs the lifecycle 'configure' step inside __init__,
    # so the config is loaded lazily from the first hook that needs it.
    def _load_config(self) -> None:
        if hasattr(self, "config"):
            return
        if not self.has_parameter("vessel_model"):
            self.declare_parameter("vessel_model", "voyager")
        if not self.has_parameter("vessel_config"):
            self.declare_parameter("vessel_config", "")
        self.config, path = vessel_model.load_vessel_config(
            self.get_parameter("vessel_model").value, self.get_parameter("vessel_config").value
        )
        self.get_logger().info(f"Allocation for vessel '{self.config['name']}' from {path}")

    def _initialize_thrusters(self) -> None:
        self._load_config()
        self.actuators = []
        for t in self.config["thrusters"]:
            alloc = t.get("allocation", {}) or {}
            extra = {
                "name": t["name"],
                "rate_limit": alloc.get("rate_limit", np.inf),
                "saturation_limit": alloc.get("saturation_limit", np.inf),
            }
            position = skadipy.toolbox.Point(t["position"])
            weight = float(alloc.get("weight", 1.0))

            if t["type"] == "fixed":
                self.actuators.append(skadipy.actuator.Fixed(
                    position=position,
                    orientation=skadipy.toolbox.Quaternion(
                        axis=(0.0, 0.0, 1.0), radians=np.deg2rad(t.get("direction_deg", 0.0))
                    ),
                    extra_attributes=extra,
                    weight=weight,
                ))
            elif t["type"] == "azimuth":
                bias = alloc.get("bias_force", [0.0, 0.0])
                extra["desired_force"] = np.array([bias[0], bias[1], 0.0, 0.0, 0.0, 0.0])
                extra["reference_angle"] = np.deg2rad(alloc.get("reference_angle_deg", 0.0))
                self.actuators.append(skadipy.actuator.Azimuth(
                    position=position,
                    extra_attributes=extra,
                    weight=weight,
                ))
            else:
                raise ValueError(f"Unknown thruster type '{t['type']}' for thruster '{t['name']}'")

    def _initialize_allocator(self) -> None:
        self._load_config()
        alloc = self.config.get("allocation", {}) or {}
        method = alloc.get("method", "pseudo_inverse")
        dofs = [_DOFS[d] for d in alloc.get("dofs", ["X", "Y", "N"])]

        self._constrained = method == "constrained_qp"
        if self._constrained:
            if alloc.get("dofs", ["X", "Y", "N"]) != ["X", "Y", "N"]:
                raise ValueError("constrained_qp supports dofs [X, Y, N] only")
            u_0 = (self.config.get("thruster_model", {}) or {}).get("zero_thrust_speed")
            self.allocator = ConstrainedAllocator(
                self.config["thrusters"], slack_weights=alloc.get("slack_weights", (1e4, 1e4, 1e5)),
                angle_threshold=alloc.get("angle_threshold", 1.0),
                zero_thrust_speed=u_0 if alloc.get("inflow_compensation", False) else None)
            self._odom, self._current = None, np.zeros(2)
            if self.allocator.zero_thrust_speed is not None:
                self.create_subscription(Odometry, "measurement/odom", self._on_odom, 10)
                self.create_subscription(Vector3Stamped, "navigation/current", self._on_current, 10)
            self._angles = {}  # actual azimuth angles from thruster/<name>/state
            for t in self.config["thrusters"]:
                if t["type"] == "azimuth":
                    self._angles[t["name"]] = 0.0
                    self.create_subscription(
                        ThrusterState, f"thruster/{t['name']}/state",
                        lambda m, n=t["name"]: self._angles.__setitem__(n, m.angle), 10)
            self._pub_slack = self.create_publisher(Wrench, "control/allocation/slack", 10)
            self.get_logger().info(f"Constrained allocation (QP with slack), Q = {self.allocator.Q.tolist()}")
            return
        if method != "pseudo_inverse":
            raise ValueError(f"Allocation method '{method}' is not supported")
        self.allocator = skadipy.allocator.PseudoInverse(
            actuators=self.actuators,
            force_torque_components=dofs,
        )
        self.allocator.compute_configuration_matrix()

    def _thruster_publisher_specs(self) -> list[ThrusterPublisherSpec]:
        self._load_config()
        specs = []
        for i, t in enumerate(self.config["thrusters"]):
            mapping = {"force.x": 0} if t["type"] == "fixed" else {"force.x": 0, "force.y": 1}
            specs.append(ThrusterPublisherSpec(
                key=t["name"],
                topic=f"thruster/{t['name']}/command",
                actuator_index=i,
                mapping=mapping,
            ))
        return specs

    # ------------------------------------------------------------------------
    # constrained_qp
    # ------------------------------------------------------------------------

    def _on_odom(self, msg: Odometry) -> None:
        q, v = msg.pose.pose.orientation, msg.twist.twist
        psi = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._odom = (self.get_clock().now().nanoseconds * 1e-9, psi,
                      np.array([v.linear.x, v.linear.y, v.angular.z]))

    def _on_current(self, msg: Vector3Stamped) -> None:
        self._current = np.array([msg.vector.x, msg.vector.y])

    def _velocity_through_water(self):
        """nu_r = [u_r, v_r, r] from the last odometry and current estimate, None when the odometry is old."""
        if self._odom is None or self.get_clock().now().nanoseconds * 1e-9 - self._odom[0] > 1.0:
            return None
        _, psi, nu = self._odom
        c, s = math.cos(psi), math.sin(psi)
        vn, ve = self._current
        return nu - np.array([c * vn + s * ve, -s * vn + c * ve, 0.0])

    def timer_callback(self) -> None:
        if not getattr(self, "_constrained", False):
            return super().timer_callback()
        tau = np.asarray(self.tau_cmd, dtype=float).ravel()[[0, 1, 5]]
        f, alpha_cmd, slack = self.allocator.allocate(tau, self._angles)
        nu_r = self._velocity_through_water() if self.allocator.zero_thrust_speed is not None else None
        if nu_r is not None:
            f = self.allocator.compensate_inflow(f, self._angles, nu_r)
        for t, f_i in zip(self.config["thrusters"], f):
            pub = self._pubs.get(t["name"])
            if pub is None:
                continue
            msg = Wrench()
            if t["type"] == "fixed":
                msg.force.x = float(f_i)
            else:
                # Direction = angle set-point, signed length = thrust (< 0: reverse along the angle).
                # A tiny vector turns the azimuth without thrust.
                a = alpha_cmd[t["name"]]
                mag = float(f_i) if abs(f_i) > 1e-3 else 1e-3
                msg.force.x, msg.force.y = mag * math.cos(a), mag * math.sin(a)
            pub.publish(msg)
        s = Wrench()
        s.force.x, s.force.y, s.torque.z = (float(v) for v in slack)
        self._pub_slack.publish(s)
