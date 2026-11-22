"""
The real Microampere's own topics on the simulator (launch boat_interface:=true). The boat's software then runs
against the simulation as it would on the boat: TURLab's blueboat_core_control, Foxglove's microAmpere.json
layout, and our allocation with blueboat output.

    in   /control/thrust            ThrusterInputs   aft thrusters, force [N] or PWM [us]; used in AUTONOMOUS only
         /actuator/azimuth_cmd      AzimuthTAngle    bow azimuth: thrust [N], absolute unwrapped angle [rad]
         /start_operation           Bool             Foxglove's START / STOP OPERATION: AUTONOMOUS / STOPPED
         /control/joystick          Joy              Foxglove's Emergency Stop (buttons[2] = 1): STOPPED
    out  /control/opmode            OperationMode    latched; parameter boat_interface.opmode (1 = AUTONOMOUS)
         /navigation/odometry/ned   Odometry         like the boat's LEKF: pose and BODY (FRD) twist of the IMU
         /navigation/ned_origin     GpsOrigin        latched
         /sensors/imu               Imu              STIM300: frame "IMU" (FLU), specific force and rate + noise
         /sensors/nav_pvt, /sensors/ecef, /sensors/nav_cov    port antenna (moving base)
         /sensors/rel_pos           GNSSRelPos       baseline to the starboard antenna, vehicle heading
         /azimuth/state             AzimuthState     unwrapped angle
         /battery                   BatteryState
         /tf_static                 world -> ned (the same NED frame), ned -> enu, <sim body> -> base_link (the IMU,
                                    FRD), so Foxglove's 3D panel can show and send poses in the team's frames

The formats follow the boat's drivers (blueboat_sentinode, blueboat_lekf; see cybership_utilities.blueboat_io).
Stamps: ROS time, or with the parameter boat_interface.clock "sentiboard" (launch boat_clock:=sentiboard) the
SentiBoard's: seconds since the simulator started on the epoch 2000-01-01, as on the boat. The odometry follows
the GNSS fix's multipath error near a quay (sensors.gnss.multipath), as the boat's LEKF follows its GNSS.
Antenna and IMU positions, rates, noise and the PWM map are in vessel.yaml 'boat_interface'.
The simulator's own topics (sim/odom_true, sensor/gnss, thruster/<name>/command, ...) stay as they are.
"""
import datetime
import math
import time

import geometry_msgs.msg
import numpy as np
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import BatteryState, Imu, Joy
from std_msgs.msg import Bool
from tf2_ros import StaticTransformBroadcaster

from cybership_utilities import blueboat_io as bb
from cybership_utilities.marine_craft.kinematics import Rzyx
from cybership_utilities.marine_craft.sensors import Gnss, gnss_config

STOPPED, AUTONOMOUS = 0, 1


class BoatInterface:
    def __init__(self, sim, config, gnss_profile):
        try:
            from blueboat_interfaces import msg
        except ImportError as e:
            raise RuntimeError("boat_interface needs the blueboat_interfaces package in the workspace "
                               "(https://turlab.itk.ntnu.no/turlab/blueboat_interfaces)") from e
        self.msg, self.sim = msg, sim
        cfg = config.get("boat_interface") or {}
        self.imu_pos = np.asarray(cfg.get("imu_position", [0.0, 0.0, 0.0]), float)
        ant = cfg.get("antennas") or {}
        self.port = self.imu_pos + np.asarray(ant.get("port", [-0.358, -0.307, -0.122]), float)
        self.stb = self.imu_pos + np.asarray(ant.get("starboard", [-0.358, 0.397, -0.122]), float)
        rates = cfg.get("rates") or {}
        defaults = {"imu": 100.0, "gnss": 5.0, "odometry": 50.0, "azimuth": 20.0, "battery": 1.0, "opmode": 1.0}
        self.period = {k: 1.0 / float(rates.get(k, v)) for k, v in defaults.items()}
        self.due = {k: 0.0 for k in defaults}
        noise = cfg.get("noise") or {}
        rate_imu = 1.0 / self.period["imu"]
        self.sigma_gyro = float(noise.get("gyro_density", 4.36e-5)) * math.sqrt(rate_imu)
        self.sigma_acc = float(noise.get("accel_density", 1.1667e-3)) * math.sqrt(rate_imu)
        self.sigma_odom = float(noise.get("odometry_position", 0.01))
        self.sigma_odom_psi = math.radians(float(noise.get("odometry_heading_deg", 0.2)))
        self.sigma_baseline = float(noise.get("baseline", 0.01))
        self.pwm = cfg.get("pwm_map") or {}
        names = cfg.get("thrusters") or {}
        self.names = {k: names.get(k, d) for k, d in (("port", "port"), ("starboard", "starboard"),
                                                       ("azimuth", "bow_azimuth"))}
        self.thrusters = {t["name"]: t for t in config["thrusters"]}
        g = dict(gnss_config(config, gnss_profile))
        g.update(rate=1.0 / self.period["gnss"], lever_arm=self.port.tolist())
        self.gnss = Gnss(rng=sim._noise_rng, **g)
        self.rtk = gnss_profile == "rtk"
        self.origin = (g.get("origin_lat_deg", 63.4305), g.get("origin_lon_deg", 10.3951))
        self.rng = sim._noise_rng
        self.nu_prev = None

        sim.declare_parameter("boat_interface.opmode", AUTONOMOUS)    # 0 STOPPED, 1 AUTONOMOUS, 2 JOYSTICK
        sim.declare_parameter("boat_interface.clock", "ros")          # "ros" or "sentiboard"
        self.sentiboard = sim.get_parameter("boat_interface.clock").value == "sentiboard"
        self.t0_ns = sim.get_clock().now().nanoseconds
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = {
            "opmode": sim.create_publisher(msg.OperationMode, "/control/opmode", latched),
            "origin": sim.create_publisher(msg.GpsOrigin, "/navigation/ned_origin", latched),
            "odom": sim.create_publisher(Odometry, "/navigation/odometry/ned", 10),
            "imu": sim.create_publisher(Imu, "/sensors/imu", 50),
            "pvt": sim.create_publisher(msg.GNSSNavPvt, "/sensors/nav_pvt", 10),
            "ecef": sim.create_publisher(msg.GNSSHPPosECEF, "/sensors/ecef", 10),
            "cov": sim.create_publisher(msg.GNSSNavCov, "/sensors/nav_cov", 10),
            "relpos": sim.create_publisher(msg.GNSSRelPos, "/sensors/rel_pos", 10),
            "azimuth": sim.create_publisher(msg.AzimuthState, cfg.get("azimuth_state_topic", "/azimuth/state"), 10),
            "battery": sim.create_publisher(BatteryState, "/battery", 1),
        }
        sim.create_subscription(msg.ThrusterInputs, "/control/thrust", self.on_thrust, 10)
        sim.create_subscription(msg.AzimuthTAngle, "/actuator/azimuth_cmd", self.on_azimuth, 10)
        sim.create_subscription(Bool, "/start_operation", self.on_start_operation, 10)
        sim.create_subscription(Joy, "/control/joystick", self.on_joystick, 10)
        self.static_tf = StaticTransformBroadcaster(sim)
        self.static_tf.sendTransform(self.frames(sim.world_frame_id, sim._frame(sim.body_frame_id)))
        origin = msg.GpsOrigin()
        origin.header.frame_id = "ned"
        origin.latitude, origin.longitude, origin.altitude = float(self.origin[0]), float(self.origin[1]), 0.0
        self.pub["origin"].publish(origin)
        sim.get_logger().info(
            f"Boat interface: /control/thrust + /actuator/azimuth_cmd in; /navigation/odometry/ned, /sensors/*, "
            f"/control/opmode out (IMU at {self.imu_pos.tolist()}, antennas {self.port.round(3).tolist()} and "
            f"{self.stb.round(3).tolist()} m BODY)")

    # --- commands --------------------------------------------------------------------------------------------

    def opmode(self):
        return int(self.sim.get_parameter("boat_interface.opmode").value)

    def set_opmode(self, mode, why):
        if mode != self.opmode():
            self.sim.set_parameters([Parameter("boat_interface.opmode", value=int(mode))])
            self.sim.get_logger().info(f"Operation mode {('STOPPED', 'AUTONOMOUS', 'JOYSTICK')[mode]} ({why})")
        self.publish_opmode()

    def on_start_operation(self, m):
        self.set_opmode(AUTONOMOUS if m.data else STOPPED, f"/start_operation {m.data}")

    def on_joystick(self, m):
        if len(m.buttons) > 2 and m.buttons[2]:
            self.set_opmode(STOPPED, "emergency stop on /control/joystick")

    def frames(self, world, body):
        """Static transforms to the team's frames: ned (= our world), enu, base_link (FRD at the IMU)."""
        out = []
        for parent, child, xyz, q in ((world, "ned", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                                      ("ned", "enu", (0.0, 0.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)),
                                      (body, "base_link", tuple(self.imu_pos), (0.0, 0.0, 0.0, 1.0))):
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = self.stamp(), parent, child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in xyz)
            r = t.transform.rotation
            r.x, r.y, r.z, r.w = q
            out.append(t)
        return out

    def on_thrust(self, m):
        """Aft thrusters: forces, or PWM through the driver's map. The boat's control_node passes them on in
        autonomous mode only; otherwise the thrusters get zero."""
        if m.pwm:
            p = self.pwm
            kw = {k: float(p[k]) for k in ("zero_us", "min_us", "max_us", "positive_force_per_us",
                                           "negative_force_per_us") if k in p}
            f_port = bb.pwm_to_force(m.port_prop_in, contra_rotating=bool(p.get("port_contra_rotating", False)), **kw)
            f_stb = bb.pwm_to_force(m.stb_prop_in, contra_rotating=bool(p.get("starboard_contra_rotating", True)), **kw)
        else:
            f_port, f_stb = float(m.port_prop_in), float(m.stb_prop_in)
        if self.opmode() != AUTONOMOUS:
            f_port = f_stb = 0.0
        for key, f in (("port", f_port), ("starboard", f_stb)):
            w = geometry_msgs.msg.Wrench()
            w.force.x = f
            self.sim._on_thruster_command(self.thrusters[self.names[key]], w)

    def on_azimuth(self, m):
        name = self.names["azimuth"]
        thrust = float(m.thrust) if self.opmode() == AUTONOMOUS else 0.0
        self.sim._later(self.sim._latency.get("thrusters", 0.0),
                        lambda: self.sim._actuators[name].set_angle_command(thrust, float(m.angle)))

    # --- measurements ----------------------------------------------------------------------------------------

    def step(self, t):
        """Called every simulator step with its time t [s]; publishes what is due."""
        eta, nu = self.sim.vessel.get_states()
        dt = self.sim.dt
        nu_dot = np.zeros(6) if self.nu_prev is None else (nu - self.nu_prev) / dt
        self.nu_prev = nu.copy()
        lat = self.sim._latency
        for key, fn in (("imu", lambda: self.imu(eta, nu, nu_dot, lat.get("imu", 0.0))),
                        ("odometry", lambda: self.odometry(eta, nu)),
                        ("gnss", lambda: self.gnss_messages(t, eta, nu, lat.get("gnss", 0.0))),
                        ("azimuth", self.azimuth_state),
                        ("battery", self.battery),
                        ("opmode", self.publish_opmode)):
            if t + 1e-9 >= self.due[key]:
                self.due[key] = t + self.period[key]
                fn()

    def stamp(self):
        now = self.sim.get_clock().now()
        if self.sentiboard:
            return Time(nanoseconds=now.nanoseconds - self.t0_ns + bb.SENTIBOARD_EPOCH_UNIX * 10**9).to_msg()
        return now.to_msg()

    def imu(self, eta, nu, nu_dot, delay):
        f = bb.frd_to_flu(bb.specific_force_frd(eta, nu, nu_dot, self.imu_pos)) + self.sigma_acc * self.rng.standard_normal(3)
        w = bb.frd_to_flu(nu[3:6]) + self.sigma_gyro * self.rng.standard_normal(3)
        m = Imu()
        m.header.stamp, m.header.frame_id = self.stamp(), "IMU"
        m.orientation_covariance[0] = -1.0                       # the STIM300 gives no orientation
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (float(v) for v in w)
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (float(v) for v in f)
        self.sim._later(delay, lambda: self.pub["imu"].publish(m))

    def odometry(self, eta, nu):
        """The IMU's pose (NED) and twist (BODY FRD), as blueboat_lekf publishes it, with RTK-like noise."""
        R = Rzyx(*eta[3:6])
        p = eta[:3] + R @ self.imu_pos
        p[:2] += self.sigma_odom * self.rng.standard_normal(2) + getattr(self.sim, "_mp_error", 0.0)
        psi = eta[5] + self.sigma_odom_psi * self.rng.standard_normal()
        v = nu[:3] + np.cross(nu[3:6], self.imu_pos)
        m = Odometry()
        m.header.stamp, m.header.frame_id, m.child_frame_id = self.stamp(), "ned", "base_link"
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = (float(x) for x in p)
        q = Rotation.from_euler("ZYX", [psi, eta[4], eta[3]]).as_quat()
        o = m.pose.pose.orientation
        o.x, o.y, o.z, o.w = (float(x) for x in q)
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = (float(x) for x in v)
        m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z = (float(x) for x in nu[3:6])
        self.pub["odom"].publish(m)

    def gnss_messages(self, t, eta, nu, delay):
        """Port antenna (NAV-PVT, HPPOSECEF, COV) and the baseline to the starboard antenna (RELPOSNED)."""
        if not self.sim._gnss_available(t):
            return
        g = self.gnss.update(t, eta)
        if g is None:
            return
        msg = self.msg
        rtk_fix = self.rtk and bool(self.sim.get_parameter("sensors.gnss.rtk_fix").value)
        sigma = g["sigma"] if rtk_fix or not self.rtk else max(g["sigma"], 0.3)
        carr = 2 if rtk_fix else 0
        R = Rzyx(*eta[3:6])
        v_ant = R @ (nu[:3] + np.cross(nu[3:6], self.port))
        itow, nano = bb.gps_time(time.time())
        utc = datetime.datetime.now(datetime.timezone.utc)
        stamp = self.stamp()

        pvt = msg.GNSSNavPvt()
        pvt.header.stamp, pvt.header.frame_id = stamp, "gnss_prt"
        pvt.itow, pvt.nano = itow, nano
        pvt.year, pvt.month, pvt.day, pvt.hour, pvt.min, pvt.sec = (utc.year, utc.month, utc.day, utc.hour,
                                                                     utc.minute, utc.second)
        pvt.valid = pvt.VALID_DATE | pvt.VALID_TIME | pvt.VALID_FULLY_RESOLVED
        pvt.confirmed_avail = pvt.confirmed_date = pvt.confirmed_time = True
        pvt.t_acc, pvt.fix_type, pvt.gnss_fix_ok, pvt.diff_soln = 20, 3, True, rtk_fix
        pvt.carr_soln, pvt.num_sv = carr, 22
        pvt.lat, pvt.lon, pvt.height = float(g["lat"]), float(g["lon"]), float(g["alt"])
        pvt.h_msl = float(g["alt"])
        pvt.h_acc, pvt.v_acc = float(sigma), float(2.0 * sigma)
        pvt.vel_n, pvt.vel_e, pvt.vel_d = (float(x) for x in v_ant)
        pvt.g_speed = float(math.hypot(v_ant[0], v_ant[1]))
        pvt.course = float(math.degrees(math.atan2(v_ant[1], v_ant[0])) % 360.0)
        pvt.p_dop = 1.2

        x, y, z = bb.llh_to_ecef(g["lat"], g["lon"], g["alt"])
        ecef = msg.GNSSHPPosECEF()
        ecef.header.stamp, ecef.header.frame_id = stamp, "gnss_prt"
        ecef.itow = itow
        (ecef.ecef_x, ecef.ecef_x_hp), (ecef.ecef_y, ecef.ecef_y_hp), (ecef.ecef_z, ecef.ecef_z_hp) = (
            bb.ublox_hp(x), bb.ublox_hp(y), bb.ublox_hp(z))
        ecef.p_acc = int(sigma * 1e4)                             # 0.1 mm

        cov = msg.GNSSNavCov()
        cov.header.stamp, cov.header.frame_id = stamp, "gnss_prt"
        cov.itow, cov.pos_cov_valid, cov.vel_cov_valid = itow, 1, 1
        cov.pos_cov_nn = cov.pos_cov_ee = float(sigma ** 2)
        cov.pos_cov_dd = float((2.0 * sigma) ** 2)
        cov.vel_cov_nn = cov.vel_cov_ee = cov.vel_cov_dd = 0.02 ** 2

        b = bb.baseline_ned(eta, self.port, self.stb)
        if rtk_fix or not self.rtk:
            b = b + self.sigma_baseline * self.rng.standard_normal(3)
        else:
            b = b + 0.2 * self.rng.standard_normal(3)              # float: the heading falls apart
        rel = msg.GNSSRelPos()
        rel.header.stamp, rel.header.frame_id = stamp, "gnss_strb"
        rel.itow = itow
        rel.rel_pos_n, rel.rel_pos_e, rel.rel_pos_d = (float(v) for v in b)
        rel.rel_pos_length = float(np.linalg.norm(b))
        rel.rel_pos_heading = float(bb.vehicle_heading_deg(b))
        rel.acc_n = rel.acc_e = rel.acc_d = rel.acc_length = float(self.sigma_baseline)
        rel.acc_heading = float(math.degrees(self.sigma_baseline / max(rel.rel_pos_length, 1e-3)))
        rel.gnss_fix_ok, rel.diff_soln, rel.rel_pos_valid, rel.carr_soln = True, True, True, carr
        rel.is_moving, rel.rel_pos_heading_valid = True, carr == 2

        def send():
            self.pub["pvt"].publish(pvt)
            self.pub["ecef"].publish(ecef)
            self.pub["cov"].publish(cov)
            self.pub["relpos"].publish(rel)
        self.sim._later(delay, send)

    def azimuth_state(self):
        act = self.sim._actuators[self.names["azimuth"]]
        m = self.msg.AzimuthState()
        m.header.stamp = self.stamp()
        m.angle = float(act.angle_unwrapped)
        m.esc_pulse_us = float(act.pwm) if getattr(act, "pwm", None) is not None else 1500.0
        m.esc_duty = m.esc_pulse_us / 20000.0
        m.encoder_valid, m.command_timeout, m.esc_armed = True, False, True
        self.pub["azimuth"].publish(m)

    def battery(self):
        m = BatteryState()
        m.header.stamp = self.stamp()
        m.voltage = float(self.sim.get_parameter("battery.voltage").value)
        m.present = True
        self.pub["battery"].publish(m)

    def publish_opmode(self):
        m = self.msg.OperationMode()
        m.header.stamp = self.stamp()
        m.mode = bytes([self.opmode()])                          # 'byte' field
        self.pub["opmode"].publish(m)
