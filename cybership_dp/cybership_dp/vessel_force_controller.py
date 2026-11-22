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

blueboat.output (parameter, constrained_qp only): the commands go out as the real Microampere takes them
(TURLab blueboat_interfaces) instead of thruster/<name>/command: aft forces on /control/thrust (ThrusterInputs,
pwm = false) and the azimuth on /actuator/azimuth_cmd (AzimuthTAngle, absolute unwrapped angle that never winds
past vessel.yaml boat_interface.turn_limit; thrust 0 while it swings the other way round). The azimuth's actual
angle then comes from its driver's AzimuthState. The simulator takes the same topics with boat_interface:=true.
It then works as the team's allocator (blueboat_core_control): the demand comes from /control/tau (WrenchStamped),
thrust goes out only in AUTONOMOUS (/control/opmode; one zero output on leaving it, so it never fights the
joystick), zero output when no tau has come for blueboat.tau_timeout s, and /control/allocator/tau_allocated
(what the sent thrusts give) and /control/allocator/unwinding (Bool) for the Foxglove DP tab.
"""

from __future__ import annotations

import math
import time

import numpy as np
import skadipy
import skadipy.allocator
from geometry_msgs.msg import Vector3Stamped, Wrench, WrenchStamped
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry

from cybership_interfaces.msg import ThrusterState

from cybership_utilities.blueboat_io import AzimuthUnwrap, byte_value

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
            blueboat = self._blueboat_output()
            for t in self.config["thrusters"]:
                if t["type"] == "azimuth":
                    self._angles[t["name"]] = 0.0
                    if not blueboat:
                        self.create_subscription(
                            ThrusterState, f"thruster/{t['name']}/state",
                            lambda m, n=t["name"]: self._angles.__setitem__(n, m.angle), 10)
            if blueboat:
                self._setup_blueboat_output()
            self._pub_slack = self.create_publisher(Wrench, "control/allocation/slack", 10)
            from cybership_utilities.loop_timing import LoopTimer
            self._alloc_timing = LoopTimer(self, "allocation", period=1.0 / float(self.freq))
            self.get_logger().info(f"Constrained allocation (QP with slack), Q = {self.allocator.Q.tolist()}")
            return
        if method != "pseudo_inverse":
            raise ValueError(f"Allocation method '{method}' is not supported")
        self.allocator = skadipy.allocator.PseudoInverse(
            actuators=self.actuators,
            force_torque_components=dofs,
        )
        self.allocator.compute_configuration_matrix()

    def _blueboat_output(self) -> bool:
        if not self.has_parameter("blueboat.output"):
            self.declare_parameter("blueboat.output", False)
        return bool(self.get_parameter("blueboat.output").value)

    def _setup_blueboat_output(self) -> None:
        """Publishers, the azimuth's driver state and the unwrapping for blueboat.output."""
        from blueboat_interfaces.msg import AzimuthState, AzimuthTAngle, OperationMode, ThrusterInputs
        self._ThrusterInputs, self._AzimuthTAngle = ThrusterInputs, AzimuthTAngle
        if not self.has_parameter("blueboat.tau_timeout"):
            self.declare_parameter("blueboat.tau_timeout", 0.5)      # [s] no tau this long: zero output
        self._bb_tau_timeout = float(self.get_parameter("blueboat.tau_timeout").value)
        self._bb_tau, self._bb_tau_time = np.zeros(3), None
        self._bb_autonomous, self._bb_open = False, False
        self._bb_mode_auto = byte_value(OperationMode.AUTONOMOUS)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(WrenchStamped, "/control/tau", self._on_bb_tau, 10)
        self.create_subscription(OperationMode, "/control/opmode", self._on_bb_opmode, latched)
        self._pub_tau_allocated = self.create_publisher(WrenchStamped, "/control/allocator/tau_allocated", 10)
        self._pub_unwinding = self.create_publisher(Bool, "/control/allocator/unwinding", 10)
        bi = self.config.get("boat_interface") or {}
        self._bb_names = {"port": "port", "starboard": "starboard", "azimuth": "bow_azimuth", **(bi.get("thrusters") or {})}
        az = next(t for t in self.config["thrusters"] if t["name"] == self._bb_names["azimuth"])
        rate = math.radians(float(az.get("max_rate_deg_s", 90.0)))
        self._unwrap = AzimuthUnwrap(turn_limit=float(bi.get("turn_limit", 1.5)), rate=rate)
        self._pub_thrust = self.create_publisher(ThrusterInputs, "/control/thrust", 10)
        self._pub_azimuth = self.create_publisher(AzimuthTAngle, "/actuator/azimuth_cmd", 10)
        self.create_subscription(AzimuthState, bi.get("azimuth_state_topic", "/azimuth/state"), self._on_azimuth_state, 10)
        self.get_logger().info(f"blueboat output: /control/tau -> /control/thrust and /actuator/azimuth_cmd in AUTONOMOUS "
                               f"only (azimuth turn limit {bi.get('turn_limit', 1.5)} turns)")

    def force_callback(self, msg: Wrench) -> None:
        if getattr(self, "_constrained", False) and self._blueboat_output():
            return                                        # the demand comes from /control/tau
        super().force_callback(msg)

    def _on_bb_tau(self, msg: WrenchStamped) -> None:
        w = msg.wrench
        self._bb_tau, self._bb_tau_time = np.array([w.force.x, w.force.y, w.torque.z]), time.monotonic()

    def _on_bb_opmode(self, msg) -> None:
        self._bb_autonomous = byte_value(msg.mode) == self._bb_mode_auto

    def _on_azimuth_state(self, msg) -> None:
        a = msg.angle
        self._angles[self._bb_names["azimuth"]] = (a + math.pi) % (2.0 * math.pi) - math.pi

    def _blueboat_step(self) -> None:
        """One allocation step on the boat's topics, through the team allocator's gate."""
        if not self._bb_autonomous:
            if self._bb_open:                             # leaving AUTONOMOUS: one zero output, then quiet
                self._bb_open = False
                self._publish_blueboat_zero()
            return
        self._bb_open = True
        if self._bb_tau_time is None or time.monotonic() - self._bb_tau_time > self._bb_tau_timeout:
            self._publish_blueboat_zero()                 # pose controller gone or inactive
            return
        f, alpha_cmd, slack = self.allocator.allocate(self._bb_tau, self._angles)
        T = self.allocator.T(self._angles)
        nu_r = self._velocity_through_water() if self.allocator.zero_thrust_speed is not None else None
        f_cmd = self.allocator.compensate_inflow(f, self._angles, nu_r) if nu_r is not None else f
        unwinding = self._publish_blueboat(f_cmd, alpha_cmd)
        if unwinding:
            f = np.array(f, dtype=float)
            f[[i for i, t in enumerate(self.config["thrusters"]) if t["name"] == self._bb_names["azimuth"]]] = 0.0
        self._publish_bb_debug(T @ f, unwinding)
        s = Wrench()
        s.force.x, s.force.y, s.torque.z = (float(v) for v in slack)
        self._pub_slack.publish(s)

    def _publish_blueboat_zero(self) -> None:
        """Zero thrust, the azimuth held at its angle."""
        now = self.get_clock().now()
        m = self._ThrusterInputs()
        m.header.stamp, m.pwm = now.to_msg(), False
        self._pub_thrust.publish(m)
        a = self._AzimuthTAngle()
        a.header.stamp, a.angle, a.thrust = now.to_msg(), float(self._unwrap.angle), 0.0
        self._pub_azimuth.publish(a)
        self._publish_bb_debug(np.zeros(3), False)

    def _publish_bb_debug(self, tau_allocated, unwinding) -> None:
        m = WrenchStamped()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), "base_link"
        m.wrench.force.x, m.wrench.force.y, m.wrench.torque.z = (float(v) for v in tau_allocated)
        self._pub_tau_allocated.publish(m)
        self._pub_unwinding.publish(Bool(data=bool(unwinding)))

    def _publish_blueboat(self, f, alpha_cmd) -> bool:
        """Thrust and azimuth commands; True while the azimuth swings the long way round (no azimuth thrust)."""
        now = self.get_clock().now()
        forces = {t["name"]: float(f_i) for t, f_i in zip(self.config["thrusters"], f)}
        m = self._ThrusterInputs()
        m.header.stamp = now.to_msg()
        m.port_prop_in, m.stb_prop_in = forces[self._bb_names["port"]], forces[self._bb_names["starboard"]]
        m.tunnel_prop_in, m.pwm = 0.0, False
        self._pub_thrust.publish(m)
        name = self._bb_names["azimuth"]
        t = now.nanoseconds * 1e-9
        a = self._AzimuthTAngle()
        a.header.stamp = now.to_msg()
        a.angle = float(self._unwrap.command(alpha_cmd[name], t))
        unwinding = self._unwrap.holding(t)
        a.thrust = 0.0 if unwinding else forces[name]
        self._pub_azimuth.publish(a)
        return unwinding

    def _thruster_publisher_specs(self) -> list[ThrusterPublisherSpec]:
        self._load_config()
        if self._blueboat_output():
            return []                                     # the commands go out on the boat's topics
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
        with self._alloc_timing:
            self._constrained_step()

    def _constrained_step(self) -> None:
        if self._blueboat_output():
            return self._blueboat_step()
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
