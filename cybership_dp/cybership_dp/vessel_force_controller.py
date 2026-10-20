"""
Configurable thrust allocation.

Builds the skadipy actuators and allocator from the same vessel.yaml the simulator uses:

    cybership_config/config/<vessel_model>/vessel.yaml

so the thruster layout is written in exactly one place. Replaces the per-vessel classes
(voyager/, enterprise/, drillship/) for any vessel that has a vessel.yaml.
"""

from __future__ import annotations

import numpy as np
import skadipy
import skadipy.allocator

from cybership_utilities import vessel_model

from cybership_dp.force_controller_base import BaseForceControllerROS
from cybership_dp.force_controller_base import ThrusterPublisherSpec

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

        if method != "pseudo_inverse":
            raise ValueError(f"Allocation method '{method}' is not supported yet")
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
