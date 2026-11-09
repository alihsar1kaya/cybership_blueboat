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

Waves (Fossen Ch. 10.2, MSS waveForceRAO): JONSWAP irregular sea with spreading and first-order loads
from the Capytaine force RAOs in the hydro file, e.g.
    ros2 param set /microampere/sim environment.waves.Hs 0.3
    ros2 param set /microampere/sim environment.waves.direction_deg 180.0   # travelling towards south

Sensors (Fossen Ch. 13.1, vessel.yaml 'sensors'): GNSS (sensor/gnss NavSatFix, sensor/gnss/ned), compass +
gyro (sensor/imu, Imu). The true state is always on sim/odom_true. measurement/odom (what the controller
uses) is the true state, or with parameter sensors.noisy_odom:=true the state plus white noise as in
Fossen's shipSim examples, until an observer (Fossen Ch. 13.4) takes over.

Run:
    ros2 run cybership_simulator vessel.py --ros-args -r __ns:=/voyager \
        -p vessel_model:=voyager -p tf_prefix:=voyager
"""

import copy
import heapq
import math
import os

import numpy as np
import rclpy
import geometry_msgs.msg
from nav_msgs.msg import Odometry, Path
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from visualization_msgs.msg import Marker
import skadipy

from cybership_simulator.base import BaseSimulator
from cybership_utilities import vessel_model
from cybership_utilities.marine_craft.actuators import make_thruster
from cybership_utilities.marine_craft.environment import OceanCurrent, Wind
from cybership_utilities.marine_craft.sensors import Gnss, gnss_config, make_sensors
from cybership_utilities.marine_craft.waves import make_waves
from cybership_utilities import hydro
from std_msgs.msg import Float64


class ConfigurableSimulator(BaseSimulator):
    """Simulator whose hull and thrusters come from vessel.yaml."""

    def __init__(self):
        super().__init__(node_name="vessel_simulator")
        if not self._thruster_roll_pitch:
            # Thrusters give only X, Y, N (as in Fossen's Otter): drop the roll/pitch rows of tau = B u
            self.allocator._b_matrix[3:5, :] = 0.0
        self._setup_environment()
        self._setup_sensors()

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
            "environment.wind.speed": 0.0,             # [m/s] mean wind speed (10 m/s = Beaufort 5)
            "environment.wind.direction_deg": 0.0,     # [deg] blows towards, 0 = north, 90 = east
            "environment.wind.mu": 0.0,                # gusts, Gauss-Markov: mean reversion [1/s], 0 = off
            "environment.wind.sigma": 0.0,             # gusts: noise [m/s/sqrt(s)], 0 = off
            "environment.wind.seed": -1,
            "environment.waves.Hs": 0.0,               # [m] significant wave height, 0 = no waves
            "environment.waves.Tp": 2.3,               # [s] peak period (w0 = 2 pi / Tp)
            "environment.waves.direction_deg": 0.0,    # [deg] travel towards, 0 = north
            "environment.waves.gamma": 3.3,            # JONSWAP peakedness
            "environment.waves.spreading": 1,          # 1: short-crested (cos^2), 0: long-crested
            "environment.waves.max_freq": 5.0,         # [rad/s] spectrum cut, MSS default
            "environment.waves.seed": -1,
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
        # Wind (Fossen 10.1): areas and Blendermann coefficients from vessel.yaml (model.wind)
        w = (self.config.get("model") or {}).get("wind")
        gw = lambda n: self.get_parameter(f"environment.wind.{n}").value
        self._wind = None
        if w:
            seed = int(gw("seed"))
            self._wind = Wind(w["areas"], w.get("coefficients", "speed_boat"), w.get("rho_a", 1.225),
                              speed=float(gw("speed")), direction_deg=float(gw("direction_deg")),
                              mu=float(gw("mu")), sigma=float(gw("sigma")), seed=None if seed < 0 else seed)
            self.get_logger().info(f"Wind: {gw('speed')} m/s towards {gw('direction_deg')} deg "
                                   f"(Blendermann '{w.get('coefficients', 'speed_boat')}', "
                                   f"A_Fw {w['areas']['A_Fw']} m2, A_Lw {w['areas']['A_Lw']} m2)")
        elif float(gw("speed")) > 0:
            self.get_logger().warn("Wind needs model.wind (areas) in vessel.yaml; no wind")
        self._pub_wind = self.create_publisher(geometry_msgs.msg.Vector3Stamped, "sim/environment/wind", 1)
        self.add_on_set_parameters_callback(self._on_environment_change)
        self._pub_current = self.create_publisher(
            geometry_msgs.msg.Vector3Stamped, "sim/environment/current", 1)
        self._pub_current_marker = self.create_publisher(Marker, "sim/environment/current_marker", 1)
        self.get_logger().info(
            f"Ocean current: {g('speed')} m/s towards {g('direction_deg')} deg "
            f"(mu {g('mu')}, sigma {g('sigma')})")
        # Waves: force RAOs from the hydro file of vessel.yaml (model.hydro.file)
        self._hydro = None
        h = (self.config.get("model") or {}).get("hydro")
        if h:
            path = h["file"] if os.path.isabs(h["file"]) else os.path.join(
                os.path.dirname(self.config["_path"]), h["file"])
            data = hydro.load_hydro(path)
            if "force_rao" in data:
                # motion RAOs from the same model: X = [-w^2 (M_RB + A) + i w (B + Bv) + G]^-1 F
                params = vessel_model.model_parameters(self.config)
                Bv = params["D"] - params["B_eq"]
                data["motion_rao"] = hydro.motion_raos(data["freqs"], data["A"], data["B"], data["force_rao"],
                                                       data["rao_freqs"], params["MRB"], Bv, params["G"])
                self._hydro = data
        self._waves, self._waves_dirty = None, True
        self._pub_wave = self.create_publisher(Float64, "sim/environment/wave_elevation", 10)

    def _on_environment_change(self, params):
        for param in params:
            if param.name == "environment.current.speed":
                self._current.set(speed=float(param.value))
            elif param.name.startswith("environment.wind.") and self._wind is not None:
                n = param.name.split(".")[-1]
                if n == "speed":
                    self._wind.set(speed=float(param.value))
                elif n == "direction_deg":
                    self._wind.set(direction_deg=float(param.value))
                elif n == "mu":
                    self._wind.gusts.mu = float(param.value)
                elif n == "sigma":
                    self._wind.gusts.sigma = float(param.value)
            elif param.name == "environment.current.direction_deg":
                self._current.set(direction_deg=float(param.value))
            elif param.name == "environment.current.mu":
                self._current.mu = float(param.value)
            elif param.name == "environment.current.sigma":
                self._current.sigma = float(param.value)
            elif param.name.startswith("environment.waves."):
                self._waves_dirty = True        # rebuilt in iterate(), after the new value is set
            else:
                continue
            self.get_logger().info(f"{param.name} = {param.value}")
        return SetParametersResult(successful=True)

    def _update_waves(self):
        self._waves_dirty = False
        g = lambda n: self.get_parameter(f"environment.waves.{n}").value
        if float(g("Hs")) <= 0.0:
            self._waves = None
            self.vessel.eta_wf, self.vessel.nu_wf = np.zeros(6), np.zeros(6)
            return
        if self._hydro is None:
            self.get_logger().warn("Waves need force RAOs (model.hydro file with force_rao); no waves")
            self._waves = None
            return
        env = {"Hs": float(g("Hs")), "Tp": float(g("Tp")), "direction_deg": float(g("direction_deg")),
               "gamma": float(g("gamma")), "spreading": bool(int(g("spreading"))),
               "max_freq": float(g("max_freq"))}
        seed = int(g("seed"))
        self._waves = make_waves(env, self._hydro, seed=None if seed < 0 else seed)
        self._wave_t0 = self._step_count * self.dt
        self.get_logger().info(f"Waves: Hs {env['Hs']} m, Tp {env['Tp']} s, towards {env['direction_deg']} deg, "
                               f"{'short' if env['spreading'] else 'long'}-crested (JONSWAP gamma {env['gamma']})")

    def _publish_environment(self):
        v = self._current.velocity_ned()
        stamp = self.get_clock().now().to_msg()
        msg = geometry_msgs.msg.Vector3Stamped()
        msg.header.frame_id = self.world_frame_id
        msg.header.stamp = stamp
        msg.vector.x, msg.vector.y, msg.vector.z = (float(x) for x in v)
        self._pub_current.publish(msg)
        if self._wind is not None:
            wmsg = geometry_msgs.msg.Vector3Stamped()
            wmsg.header = msg.header
            wmsg.vector.x, wmsg.vector.y, wmsg.vector.z = (float(x) for x in self._wind.velocity_ned())
            self._pub_wind.publish(wmsg)

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
    # Actuators: command -> (saturation, deadband or ESC/PWM driver + curve) -> thruster dynamics -> tau = B u
    # ------------------------------------------------------------------------

    def _on_thruster_command(self, t, msg: geometry_msgs.msg.Wrench):
        """A command arrives; it acts after the command latency (computer -> ESC -> motor)."""
        self._later(getattr(self, "_latency", {}).get("thrusters", 0.0),
                    lambda: self._apply_thruster_command(t, msg))

    def _apply_thruster_command(self, t, msg: geometry_msgs.msg.Wrench):
        """Store the command; the actuator model turns it into actual thrust in iterate()."""
        name = t["name"]
        fx, fy = self._actuators[name].set_command(msg.force.x, msg.force.y)

        issued = geometry_msgs.msg.WrenchStamped()
        issued.header.frame_id = self._frame(t["frame"])
        issued.header.stamp = self.get_clock().now().to_msg()
        issued.wrench.force.x = fx
        issued.wrench.force.y = fy
        self._issued_pubs[name].publish(issued)

    # ------------------------------------------------------------------------
    # Sensors (Fossen Ch. 13.1): GNSS, compass, gyro at their own rates
    # ------------------------------------------------------------------------

    def _setup_sensors(self):
        self.declare_parameter("sensors.noisy_odom", False)
        self.declare_parameter("sensors.seed", -1)
        self.declare_parameter("sensors.gnss_profile", "standard")   # 'standard' or 'rtk' (vessel.yaml)
        seed = self.get_parameter("sensors.seed").value
        self._sensors = make_sensors(self.config, seed=None if seed < 0 else seed,
                                     gnss_profile=self.get_parameter("sensors.gnss_profile").value)
        self._noise_rng = np.random.default_rng(None if seed < 0 else seed + 1)
        # Latency (vessel.yaml 'latency', seconds): sensor messages are published that long after the
        # measurement (stamped with the measurement time), thruster commands act that long after arrival.
        lat = self.config.get("latency") or {}
        self._latency = {k: float(lat.get(k, 0.0)) for k in ("gnss", "imu", "thrusters")}
        self._queue, self._queue_n = [], 0
        # GNSS outages and RTK fix loss (Fossen 13.4.4: the observer dead-reckons without GNSS)
        for name, default in (("sensors.gnss.available", True), ("sensors.gnss.rtk_fix", True),
                              ("sensors.gnss.outage_rate", 0.0), ("sensors.gnss.outage_duration", 10.0)):
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
        self._gnss_outage_until = -1.0
        self._gnss_float = None
        if self.get_parameter("sensors.gnss_profile").value == "rtk" and "gnss" in self._sensors:
            # what the RTK receiver gives without a fix: the standard receiver's errors
            self._gnss_float = Gnss(rng=self._noise_rng, **gnss_config(self.config, "standard"))
        self._pub_gnss = self.create_publisher(NavSatFix, "sensor/gnss", 10)
        self._pub_gnss_ned = self.create_publisher(geometry_msgs.msg.PointStamped, "sensor/gnss/ned", 10)
        self._pub_imu = self.create_publisher(Imu, "sensor/imu", 10)
        self._pub_odom_true = self.create_publisher(Odometry, "sim/odom_true", 1)
        # LF track (without the wave motion) for RViz, 2 Hz, last 5 minutes
        self._pub_path = self.create_publisher(Path, "sim/path_lf", 1)
        self._path = Path()
        self._last_psi = None
        self.get_logger().info(f"Sensors: {sorted(self._sensors)}, GNSS "
                               f"{self.get_parameter('sensors.gnss_profile').value}, noisy odom "
                               f"{self.get_parameter('sensors.noisy_odom').value}, latency {self._latency} s")

    def _later(self, delay, fn):
        """Run fn after 'delay' seconds of simulation time (immediately if delay <= 0)."""
        if delay <= 0:
            fn()
            return
        self._queue_n += 1
        heapq.heappush(self._queue, (self._step_count * self.dt + delay, self._queue_n, fn))

    def _run_due(self):
        t = self._step_count * self.dt
        while self._queue and self._queue[0][0] <= t + 1e-9:
            heapq.heappop(self._queue)[2]()

    def _gnss_available(self, t):
        g = lambda n: self.get_parameter(f"sensors.gnss.{n}").value
        if not g("available"):
            return False
        if t < self._gnss_outage_until:
            return False
        rate = float(g("outage_rate"))      # outages per minute
        if rate > 0 and self._noise_rng.random() < rate / 60.0 * self.dt:
            self._gnss_outage_until = t + float(g("outage_duration"))
            self.get_logger().warn(f"GNSS outage for {float(g('outage_duration')):.0f} s")
            return False
        return True

    def _publish_path(self):
        pose = geometry_msgs.msg.PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.world_frame_id
        pose.pose.position.x, pose.pose.position.y = float(self.vessel.eta[0]), float(self.vessel.eta[1])
        self._path.header = pose.header
        self._path.poses.append(pose)
        self._path.poses = self._path.poses[-600:]
        self._pub_path.publish(self._path)

    def _publish_sensors(self):
        t = self._step_count * self.dt
        if self._step_count % 50 == 0:
            self._publish_path()
        eta, nu = self.vessel.get_states()
        stamp = self.get_clock().now().to_msg()
        if "gnss" in self._sensors:
            rtk_lost = self._gnss_float is not None and not self.get_parameter("sensors.gnss.rtk_fix").value
            gnss = self._gnss_float if rtk_lost else self._sensors["gnss"]
            m = gnss.update(t, eta, nu)
            if m is not None and self._gnss_available(t):
                fix = NavSatFix()
                fix.header.stamp, fix.header.frame_id = stamp, self._frame("gnss_link")
                rtk = self._gnss_float is not None and not rtk_lost
                fix.status.status = NavSatStatus.STATUS_GBAS_FIX if rtk else NavSatStatus.STATUS_FIX
                fix.status.service = NavSatStatus.SERVICE_GPS
                fix.latitude, fix.longitude, fix.altitude = m["lat"], m["lon"], m["alt"]
                fix.position_covariance = [m["sigma"] ** 2, 0.0, 0.0, 0.0, m["sigma"] ** 2, 0.0, 0.0, 0.0, 1.0]
                fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
                ned = geometry_msgs.msg.PointStamped()
                ned.header.stamp, ned.header.frame_id = stamp, self.world_frame_id
                ned.point.x, ned.point.y, ned.point.z = (float(v) for v in m["ned"])
                self._later(self._latency["gnss"], lambda f=fix, n=ned: (self._pub_gnss.publish(f),
                                                                        self._pub_gnss_ned.publish(n)))
        if "compass" in self._sensors:
            m = self._sensors["compass"].update(t, eta, nu)
            if m is not None:
                self._last_psi = m["psi"]
        if "gyro" in self._sensors:
            m = self._sensors["gyro"].update(t, eta, nu)
            if m is not None:
                imu = Imu()
                imu.header.stamp, imu.header.frame_id = stamp, self._frame(self.body_frame_id)
                psi = eta[5] if self._last_psi is None else self._last_psi
                q = Rotation.from_euler("xyz", [eta[3], eta[4], psi]).as_quat()
                imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = (float(v) for v in q)
                imu.angular_velocity.x, imu.angular_velocity.y = float(m["p"]), float(m["q"])
                imu.angular_velocity.z = float(m["r"])
                imu.linear_acceleration_covariance[0] = -1.0  # no accelerometer data
                self._later(self._latency["imu"], lambda m=imu: self._pub_imu.publish(m))

    def publish_odom(self):
        """True state on sim/odom_true; measurement/odom is the true state or, if noisy, state + noise."""
        if not hasattr(self, "_pub_odom_true"):     # before _setup_sensors
            return super().publish_odom()
        noisy = self.get_parameter("sensors.noisy_odom").value
        real_pub = self.publisher_odom
        self.publisher_odom = self._pub_odom_true
        super().publish_odom()                      # fills self.odom (used for TF) with the true state
        self.publisher_odom = real_pub
        if not noisy:
            real_pub.publish(self.odom)
            return
        # White noise as in Fossen's shipSim_v2-v4: +-0.5 m, +-0.1 deg, +-0.05 m/s at 3 sigma, or the
        # sensor sigmas from vessel.yaml
        rng = self._noise_rng
        s_pos = getattr(self._sensors.get("gnss"), "error", None)
        s_pos = 0.5 / 3 if s_pos is None else float(s_pos.sigma[0])
        s_psi = getattr(self._sensors.get("compass"), "error", None)
        s_psi = math.radians(0.1) / 3 if s_psi is None else float(s_psi.sigma[0])
        s_r = getattr(self._sensors.get("gyro"), "error", None)
        s_r = math.radians(0.1) / 3 if s_r is None else float(s_r.sigma[0])
        msg = copy.deepcopy(self.odom)
        msg.pose.pose.position.x += s_pos * rng.standard_normal()
        msg.pose.pose.position.y += s_pos * rng.standard_normal()
        eta, _ = self.vessel.get_states()
        q = Rotation.from_euler("xyz", [eta[3], eta[4], eta[5] + s_psi * rng.standard_normal()]).as_quat()
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = (float(v) for v in q)
        msg.twist.twist.linear.x += 0.05 / 3 * rng.standard_normal()
        msg.twist.twist.linear.y += 0.05 / 3 * rng.standard_normal()
        msg.twist.twist.angular.z += s_r * rng.standard_normal()
        real_pub.publish(msg)

    def iterate(self):
        """Step environment and thruster dynamics, then the vessel (BaseSimulator.iterate: tau = B u)."""
        self._step_count += 1
        publish_state = self._step_count % self.STATE_EVERY == 0
        self._run_due()
        self._current.step(self.dt)
        self.vessel.current_ned = self._current.velocity_ned()
        if self._wind is not None:
            # wind loads on the LF state (Fossen 10.1), relative to the boat's velocity over ground
            self._wind.step(self.dt)
            self.vessel.tau_ext = self._wind.loads(self.vessel.eta, self.vessel.nu)
        if self._waves_dirty:
            self._update_waves()
        if self._waves is not None:
            # total motion = LF + WF (Fossen 10.2.4, 13.2): the WF motion from the motion RAOs is added to
            # what the sensors and RViz see; the LF model gets no first-order wave forces
            t = self._step_count * self.dt - self._wave_t0
            psi, u = self.vessel.eta[5], self.vessel.nu[0]
            self.vessel.eta_wf, self.vessel.nu_wf = self._waves.motion(t, psi, U=u)
            if publish_state:
                self._pub_wave.publish(Float64(data=self._waves.loads(t, psi, U=u)[1]))
        if publish_state:
            self._publish_environment()
        self._publish_sensors()
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
