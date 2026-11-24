#!/usr/bin/env python3

# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository
# Created By: Jan-Erik Hygen
# Created Date: 2023-01-30,
# Revised: 2025-01-31 Kristian Magnus Roen   Now fitting the MC-Gym for csad.
# Revised: 2025-03-26 Emir Cem Gezer Applied to the Cybership Software Suite
# Tested:
# Copyright (C) 2025: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import json
import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Wrench, WrenchStamped, Pose2D, PoseStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
import math
from scipy.spatial.transform import Rotation as R
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer, GoalResponse, CancelResponse, ActionClient
from rclpy.executors import MultiThreadedExecutor
from cybership_tests.go_to_client import NavigateToPoseClient
from shoeboxpy.model3dof import Shoebox
from cybership_utilities import vessel_model
from cybership_utilities import blueboat_io
from cybership_utilities.marine_craft.tow import towed_from_config
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float32MultiArray, Float64, String
from std_srvs.srv import SetBool, Trigger, Empty
from cybership_controller.position.reference_filter import ThirdOrderReferenceFilter
from cybership_dp.los import ALOS
from cybership_dp.dubins import DubinsPath
from cybership_dp import trajectory_planner
from cybership_dp import docking
from rclpy.duration import Duration
from rclpy.task import Future
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
try:
    from cybership_interfaces.msg import PerformanceMetrics
except ImportError:
    print("cybership_msgs not found. Skipping performance metrics.")


def wrap_to_pi(angle):
    """
    Wrap an angle in radians to the interval [-pi, pi].
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def Rz(psi):
    """3DOF Rotation matrix about z-axis.

    Parameters
    ----------
    psi : float
        Yaw angle (rad)

    Returns
    -------
    Rz : array_like
        3x3 rotation matrix.

    """
    return np.array(
        [[np.cos(psi), -np.sin(psi), 0],
         [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
    )


def saturate(x, z):
    """
    Saturation function: returns x / (|x| + z)
    This ensures a smooth saturation between -1 and 1.
    """
    return x / (np.abs(x) + z)


class PositionController(Node):
    def __init__(self):
        super().__init__("position_controller")

        # Declare parameters with default values
        self.declare_parameters(
            "",
            [
                # Controller gains
                ('control.p_gain.pos', 1.0),
                ('control.i_gain.pos', 0.2),
                ('control.d_gain.pos', 0.2),
                ('control.p_gain.vel', 0.7),
                ('control.i_gain.vel', 0.1),
                ('control.d_gain.vel', 0.5),
                ('control.p_gain.yaw', 1.3),
                ('control.i_gain.yaw', 0.2),
                ('control.d_gain.yaw', 1.0),

                # Model-based DP law (Fossen): gains from the vessel model, all in the body frame.
                # false = original law with the fixed gains above (kept for voyager etc.)
                ('control.model_based', False),
                ('control.omega_n', [0.3, 0.3, 0.6]),          # closed-loop natural frequency [rad/s]
                ('control.zeta', [0.9, 0.9, 0.9]),             # closed-loop relative damping
                ('control.ki_ratio', 0.1),                     # Ki = ki_ratio * omega_n * Kp
                # Fossen's DP law with an observer: tau = ... - R^T b_hat (navigation/bias) instead of the
                # integral action; the observer's bias already holds the slowly varying forces
                ('control.bias_feedforward', False),
                ('control.max_integral_force', [10.0, 10.0, 5.0]),  # anti-windup [N, N, Nm]

                # Guidance "turn then go" (before the control law): far from the goal, first turn
                # the bow towards it on the spot, then drive there bow first; DP with the goal
                # heading only inside approach_radius. false = plain set-point DP (original).
                ('guidance.turn_then_go', False),
                ('guidance.approach_radius', 1.5),     # [m] switch from transit to station (final heading)
                # [m] only goals farther than this use turn-then-go; closer ones are plain DP (crab)
                ('guidance.min_transit_distance', 0.0),
                # [m] after a transit, turn to the final heading only this close to the goal
                ('guidance.settle_radius', 0.3),
                ('guidance.final_turn_rate_deg_s', 15.0),   # [deg/s] turn to the final heading at the goal
                # manoeuvre choice (forward / reverse / dp) by estimated time
                ('guidance.forward_speed', 0.6),       # [m/s] typical transit speed bow first
                ('guidance.reverse_speed', 0.35),      # [m/s] stern first (reverse thrust 80 %, blunt stern)
                ('guidance.turn_rate_deg_s', 20.0),    # [deg/s] typical turning rate incl. accelerating
                ('guidance.max_reverse_distance', 6.0),  # [m] never back up farther than this
                ('guidance.align_angle_deg', 20.0),    # [deg] start driving when the bow is this close
                # limits on how fast the reference moves (0 = no limit). The speeds are through the water
                # when the observer estimates the current: surge forward_speed / reverse_speed, sideways:
                ('guidance.sway_speed', 0.0),          # [m/s] the bow azimuth alone carries the sway force
                ('guidance.min_speed', 0.05),          # [m/s] over ground, when the current uses up the limit
                ('guidance.max_turn_rate_deg_s', 0.0),  # [deg/s] heading reference
                ('guidance.max_dp_distance', 0.0),     # [m] consider moving directly (dp) up to this distance
                # Transit as path following along the straight line start -> goal (Fossen Ch. 12.5,
                # TTK4190 L6): ALOS heading + surge speed control, no sway force (like the Otter USV).
                # false: the transit is DP towards the goal through the reference filter (as before).
                ('guidance.path_following', False),
                ('guidance.dubins_radius', 0.0),
                ('guidance.los_method', 'alos'),
                ('guidance.approach_length', 0.0),
                ('guidance.approach_ratio', 1.0),
                ('guidance.dubins_radii', [0.0]),
                ('guidance.path_turn_rate_deg_s', 17.0),
                ('guidance.turn_penalty', 0.0),
                ('guidance.spot_turn_penalty', 0.0),
                ('guidance.turn_trajectory', False),
                ('guidance.weathervane', False),
                ('guidance.weathervane_rate_deg_s', 3.0),
                ('guidance.weathervane_deadband_deg', 10.0),
                ('guidance.weathervane_min_current', 0.03),
                ('guidance.weathervane_min_force', 3.0),
                ('guidance.turn_accel_deg_s2', 20.0),
                ('guidance.force_manoeuvre', ''),
                ('guidance.planner', 'rules'),
                ('guidance.ocp_min_distance', 0.5),
                ('guidance.ocp_max_distance', 25.0),
                ('guidance.ocp_threaded', True),
                ('guidance.ocp.w_T', 1.0),
                ('guidance.ocp.w_v', 30.0),
                ('guidance.ocp.w_r', 5.0),
                ('guidance.ocp.w_rev', 8.0),
                ('guidance.ocp.w_xte', 0.0),
                ('guidance.ocp.w_hold', 40.0),
                ('guidance.ocp.w_f', 2.0),
                ('guidance.ocp.w_df', 20.0),
                ('guidance.ocp.obstacle_margin', 0.5),   # [m] clearance to obstacles on top of hull and radius
                # Docking (cybership_dp/docking_mpc.py, results/docking_mpc_plan.md): berths from a harbour YAML,
                # started with guidance/dock (berth name) or the ~/dock service (the first berth)
                ('docking.harbour_file', ''),
                ('docking.harbour_shift', [0.0, 0.0]),   # [m] NED: a map error, for tests
                ('docking.standoff', 2.0),               # [m] out from the berth: where docking starts from
                ('docking.max_start_distance', 10.0),    # [m] further from the berth: go to the stand-off first
                ('docking.gap_margin', 0.02),            # [m] the plan keeps berth gap - this from the face
                ('docking.corridor', [0.15, 0.5, 1.5]),  # approach funnel w0 + k1 gap + k2 gap^2 [m]
                # [deg, deg/m] lined up before close: |heading error| <= 3 deg + 15 deg/m gap. (5, 30) let the plan
                # still turn 11 deg and back at 0.13 m/s 0.3 m off the face; with the DP's yaw lag a corner came
                # within 0-2 cm of the quay from one start in current (3, 15: 6.5 cm and more, plans ~2 s longer)
                ('docking.heading_funnel_deg', [3.0, 15.0]),
                ('docking.replan_error', [0.25, 10.0]),  # [m, deg] off the plan by this much: plan again
                ('docking.replan_min_interval', 2.0),    # [s]
                ('docking.lock_gap', 1.0),               # [m] no new plans closer than this to the face
                # alongside: push this hard towards the quay, so the boat rests on its fenders (0 = hold the berth
                # 10 cm off with DP). Surge and yaw stay DP; across the face the force is max(push, DP's own)
                ('docking.push_force', 8.0),             # [N]
                # push on the fenders only once the boat lies at the berth: within [along the face, off the face] m
                # of it (the hull's gap to the face), or this many seconds after the plan ended. Pushing as soon as
                # the plan's time was up left a boat that lagged the plan 0.6 m along the face pressed on the fender,
                # and the fender's friction let it slide to the berth at 2-3 cm/s
                ('docking.alongside_tolerance', [0.15, 0.20]),
                ('docking.alongside_timeout', 15.0),     # [s]
                ('guidance.max_path_turn_deg', 270.0),
                ('guidance.max_path_ratio', 0.0),
                ('guidance.time_model.path', [6.0, 1.667, 3.82]),
                ('guidance.time_model.pivot', [6.0, 1.667, 3.82, 2.86]),
                ('guidance.time_model.forward', [0.0, 1.667, 2.86]),
                ('guidance.time_model.reverse', [0.0, 2.857, 2.86]),
                ('guidance.time_model.dp', [0.0, 1.0, 3.82]),
                ('guidance.arrival_turn_penalty', 0.0),
                ('guidance.approach_dp', False),
                ('guidance.ilos_kappa', 0.2),
                ('guidance.path_end_speed', 0.1),
                ('guidance.reverse_penalty', 1.0),
                ('guidance.los_lookahead', 2.5),       # [m] Delta (K_p = 1/Delta)
                ('guidance.alos_gamma', 0.02),         # [rad/(m s)] crab-angle adaptation, 0 = plain LOS
                ('guidance.alos_max_crab_deg', 30.0),  # [deg] limit on the crab-angle estimate
                ('guidance.transit_accel', 0.1),       # [m/s^2] speed ramp up and down in transit
                ('guidance.arrive_speed', 0.2),        # [m/s] speed when reaching approach_radius

                # Controller limits
                ('control.max_integral_error.pos', 1.0),
                ('control.max_integral_error.yaw', 1.4),
                ('control.saturation.pos', 0.1),
                ('control.saturation.yaw', 0.1),

                # Tolerances
                ('control.tolerance.pos', 0.25),
                ('control.tolerance.yaw', 0.1),
                # Require being within tolerance for this long before success
                ('control.success_hold_time', 1.0),

                # Vessel properties. If vessel_model is set, the model comes from
                # cybership_config/config/<vessel_model>/vessel.yaml (same as the simulator)
                # and vessel.length/beam/draft are ignored.
                ('vessel_model', ''),
                # Towed load (vessel.yaml 'tow', launch tow:=true tow_model:=true): the load's mass and drag in
                # surge in the DP law and the OCP planner
                ('tow.enabled', False),
                # The team's pose controller interface (launch boat_interface:=true, blueboat_io.py): drive only in
                # AUTONOMOUS (/control/opmode), goals from /goal_pose and /control/pose/setpoint, the force on
                # /control/tau, debug on /control/pose/* (the Foxglove DP tab)
                ('blueboat.interface', False),
                ('blueboat.state_timeout', 0.5),       # [s] older state makes the DP inactive
                ('vessel.length', 1.0),
                ('vessel.beam', 0.3),
                ('vessel.draft', 0.02),

                # Reference filter parameters
                ('filter.omega', [0.15, 0.15, 0.15]),
                ('filter.delta', [0.8, 0.8, 0.8]),

                # Performance metrics
                ('metrics.window_size', 200),
                ('metrics.interval', 1.0),

                # Time step
                ('dt', 0.1),
            ]
        )

        # Initialize the 3rd order reference filter.
        # Here, we treat [x, y, yaw] as the 3D pose to be smoothed.
        self.ref_filter = None

        self.start_time = None

        self.error_window = []  # For position error
        self.error_yaw_window = []  # For yaw error
        self.sample_count = 0
        self.last_metrics_time = 0.0

        #Initialize TF buffer and listener for frame transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Action server goal tracking (for non-blocking execution)
        self.active_goal_handle = None
        self._goal_done = None              # Future the execute callback waits on
        self.goal_within_tolerance_since = None

        # Add parameter callback for runtime updates
        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.add_post_set_parameters_callback(self._post_parameter_update)

        # Publisher to send control commands (force and torque)
        self.control_pub = self.create_publisher(
            Wrench, "control/force/command", 10)
        # Debug publishers for tracking performance metrics
        self.debug_pose_pub = self.create_publisher(
            PoseStamped, "control/pose/debug/reference_pose", 10)
        self.debug_vel_pub = self.create_publisher(
            TwistStamped, "control/pose/debug/reference_velocity", 10)
        self.debug_error_pose_pub = self.create_publisher(
            Pose2D, "control/pose/debug/tracking_error_position", 10)
        self.debug_error_vel_pub = self.create_publisher(
            TwistStamped, "control/pose/debug/tracking_error_velocity", 10)

        self.debug_metrics_pub = None
        if PerformanceMetrics is not None:
            self.debug_metrics_pub = self.create_publisher(
                PerformanceMetrics, "control/pose/debug/performance_metrics", 10)

        # self.create_subscription(PoseStamped, "/goal_pose", self.goal_pose_callback, 10)
        self.create_subscription(
            Odometry, "measurement/odom", self.odom_callback, 10)

        self.marker_pub = self.create_publisher(
            Marker, "visualization_marker", 10)

        # Get time step from parameters
        self.dt = self.get_parameter('dt').value

        # Initialize controller parameters from ROS parameters
        self._snapshot_parameters()

        # Latest odometry message storage
        self.latest_odom = None

        # --- Target state ---
        # Set initial desired target position and orientation.
        self.target_x = None  # target x position (meters)
        self.target_y = None  # target y position (meters)
        self.target_yaw = None  # target yaw (radians)

        # Track integral error
        self.integral_error_pos = np.zeros(2)
        self.integral_error_yaw = 0.0
        self.integral_mb = np.zeros(3)  # model-based law: integral of [x, y, yaw] error
        # obstacles for the OCP planner: CYLINDER markers (centre, scale.x = diameter) from the obstacle manager
        self._obstacles = []
        self.create_subscription(MarkerArray, 'guidance/obstacles', self._on_obstacles,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._plan_pub = self.create_publisher(Path, 'guidance/plan', 1)
        self.bias_hat = None            # observer bias b_hat in NED [N, N, N m] (navigation/bias)
        self.create_subscription(Vector3Stamped, 'navigation/bias', self._on_bias, 10)
        self.current_hat = None         # observer current velocity in NED [m/s, m/s] (navigation/current)
        self.create_subscription(Vector3Stamped, 'navigation/current', self._on_current, 10)

        self.error_pos = np.zeros(2)
        self.error_yaw = 0.0

        self._bb = None
        self._tau_parts = None          # (feedforward, feedback) of the last model-based tau, for /control/pose/*
        if self.get_parameter('blueboat.interface').value:
            self._setup_blueboat()

        self.get_logger().info(
            "Goto Point Controller (Reference Filter Version) Initialized with parameters."
        )

        # Timer for periodic control updates
        # build the OCP planner now: building the CasADi problem is plain Python (~1 s) and would stop the
        # 10 Hz loop if it happened at the first goal
        if self.get_parameter('guidance.planner').value == 'ocp' and getattr(self, 'model_based', False) \
                and getattr(self, '_vessel_cfg', None) is not None:
            t = time.time()
            self._ocp_planner()
            self.get_logger().info(f"OCP planner built in {time.time() - t:.1f} s")
        self._dock = None
        self._setup_docking()
        # compute time and period on /diagnostics (step 6: will it keep up on the boat's computer?)
        from cybership_utilities.loop_timing import LoopTimer
        self._timing = dict(loop=LoopTimer(self, "control loop", period=self.dt),
                            dock=LoopTimer(self, "docking OCP solve", report=5.0),
                            ocp=LoopTimer(self, "transit OCP solve", report=5.0))
        self.timer = self.create_timer(self.dt, self._timed_control_loop)

        # --- Action Server using nav2_msgs/NavigateToPose ---
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            "navigate_to_pose",
            execute_callback=self.execute_callback,
            goal_callback=self.action_goal_callback,
            cancel_callback=self.action_cancel_callback,
        )
        # Service to enable/disable the controller
        self.enabled = True
        self.state_service = self.create_service(
            SetBool,
            f"{self.get_name()}/change_state",
            self.change_state_callback
        )

    def _snapshot_parameters(self):
        """Update controller configuration from parameters"""
        # Get controller gains
        self.Kp_pos = self.get_parameter('control.p_gain.pos').value
        self.Ki_pos = self.get_parameter('control.i_gain.pos').value
        self.Kd_pos = self.get_parameter('control.d_gain.pos').value
        self.Kp_vel = self.get_parameter('control.p_gain.vel').value
        self.Ki_vel = self.get_parameter('control.i_gain.vel').value
        self.Kd_vel = self.get_parameter('control.d_gain.vel').value
        self.Kp_yaw = self.get_parameter('control.p_gain.yaw').value
        self.Ki_yaw = self.get_parameter('control.i_gain.yaw').value
        self.Kd_yaw = self.get_parameter('control.d_gain.yaw').value

        # Get controller limits
        self.max_integral_error_pos = self.get_parameter('control.max_integral_error.pos').value
        self.max_integral_error_yaw = self.get_parameter('control.max_integral_error.yaw').value
        self.saturation_pos = self.get_parameter('control.saturation.pos').value
        self.saturation_yaw = self.get_parameter('control.saturation.yaw').value

        # Get tolerances
        self.pos_tol = self.get_parameter('control.tolerance.pos').value
        self.yaw_tol = self.get_parameter('control.tolerance.yaw').value
        self.success_hold_time = self.get_parameter('control.success_hold_time').value

        # Get vessel properties
        vessel_length = self.get_parameter('vessel.length').value
        vessel_beam = self.get_parameter('vessel.beam').value
        vessel_draft = self.get_parameter('vessel.draft').value

        # Create vessel model
        model_name = self.get_parameter('vessel_model').value
        if model_name:
            config, path = vessel_model.load_vessel_config(model_name)
            params = vessel_model.model_parameters(config)
            self._vessel_cfg, self._vessel_params = config, params      # for the OCP planner
            self.M_eff = vessel_model.M_eff_3dof(params)
            self.D_eff = vessel_model.D_3dof(params)
            self._tow_surge = None
            if self.get_parameter('tow.enabled').value and config.get('tow'):
                # a load on a towbar moves with the boat in surge: its mass and drag in the DP law and the OCP
                # planner (a rope only pulls: just the drag). The observer takes the load cell's force as an
                # input, so the sideways pull in a turn does not end up in the bias. Cancelling the measured
                # force here instead was unstable in yaw: it holds the load's inertia (45 kg x 0.6^2 m^2 at
                # the stern, more than the boat's 9.4 kg m^2), and the filter delays it
                tb = towed_from_config(config['tow'])
                m_t, a_t, b_t = tb.surge_model()
                self._tow_surge = (m_t if tb.rigid else 0.0, a_t, b_t)
                self.M_eff = self.M_eff.copy()
                self.M_eff[0, 0] += self._tow_surge[0]
                self.get_logger().info(
                    f"Towed load in the model ({tb.describe()}): +{self._tow_surge[0]:.1f} kg in surge, drag "
                    f"{a_t:.1f} u + {b_t:.1f} u|u| N")
            # Full model, so the feedforward can use the nonlinear damping D(nu_d) nu_d
            self._craft = vessel_model.make_marine_craft(params)
            self.get_logger().info(
                f"Vessel model from {path}: M_eff diag {np.round(np.diag(self.M_eff), 2).tolist()}, "
                f"D diag {np.round(np.diag(self.D_eff), 2).tolist()}")
        else:
            shoebox = Shoebox(
                L=vessel_length,
                B=vessel_beam,
                T=vessel_draft,
            )
            self.M_eff = shoebox.M_eff
            self.D_eff = shoebox.D
            self._craft = None

        # Model-based gains (pole placement per DOF): M*e'' + (D + Kd)*e' + Kp*e = 0
        self.model_based = self.get_parameter('control.model_based').value
        if self.model_based:
            m = np.diag(self.M_eff)
            d = np.diag(self.D_eff)
            w = np.asarray(self.get_parameter('control.omega_n').value, dtype=float)
            z = np.asarray(self.get_parameter('control.zeta').value, dtype=float)
            self.Kp_mb = m * w**2
            self.Kd_mb = np.maximum(2.0 * z * m * w - d, 0.0)
            self.Ki_mb = self.get_parameter('control.ki_ratio').value * w * self.Kp_mb
            self.max_integral_force = np.asarray(
                self.get_parameter('control.max_integral_force').value, dtype=float)
            self.get_logger().info(
                f"Model-based DP: Kp {np.round(self.Kp_mb, 2).tolist()}, "
                f"Kd {np.round(self.Kd_mb, 2).tolist()}, Ki {np.round(self.Ki_mb, 3).tolist()}")

        # Get performance metrics parameters
        self.window_size = self.get_parameter('metrics.window_size').value
        self.metrics_interval = self.get_parameter('metrics.interval').value

        # Reset reference filter if parameters change
        if self.ref_filter is not None and self.latest_odom is not None:
            # Get filter parameters
            filter_omega = self.get_parameter('filter.omega').value
            filter_delta = self.get_parameter('filter.delta').value

            # Save current state
            current_eta = self.ref_filter.eta_d

            # Create new filter with updated parameters
            self.ref_filter = ThirdOrderReferenceFilter(
                dt=self.dt,
                omega=filter_omega,
                delta=filter_delta,
                initial_eta=current_eta,
            )
            self.ref_filter.eta_d = current_eta

        self.get_logger().info(
            f"Updated configuration - vessel: [{vessel_length}, {vessel_beam}, {vessel_draft}], " +
            f"POS gains: P={self.Kp_pos}, I={self.Ki_pos}, D={self.Kd_pos}, " +
            f"YAW gains: P={self.Kp_yaw}, I={self.Ki_yaw}, D={self.Kd_yaw}"
        )

    def _on_parameter_update(self, params):
        """Handle parameter updates"""

        # Return SetParametersResult to properly accept the parameter changes
        return SetParametersResult(successful=True)

    def _post_parameter_update(self, params):
        """Handle post-update actions, such as logging."""
        updated_params = [param.name for param in params]

        for param in params:

            if param.name == 'dt':
                self.dt = param.value
                # Recreate the timer with new dt
                self.timer.cancel()
                self.timer = self.create_timer(self.dt, self._timed_control_loop)
                self._timing['loop'].period = self.dt

        self.get_logger().info(f"Parameters updated: {updated_params}")
        self._snapshot_parameters()  # Ensure configuration is updated after parameters change
        if self._bb is not None:
            self._bb_publish_gains()

    def odom_callback(self, msg: Odometry):
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        _, _, current_yaw = R.from_quat([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]).as_euler("xyz", degrees=False)

        # Always store latest odom
        self.latest_odom = msg
        if self._bb is not None:
            self._bb['gate'].on_state(time.monotonic())

        if self.ref_filter is None:
            # Initialize the filter to CURRENT pose (not the target)
            filter_omega = self.get_parameter('filter.omega').value
            filter_delta = self.get_parameter('filter.delta').value
            self.ref_filter = ThirdOrderReferenceFilter(
                dt=self.dt,
                omega=filter_omega,
                delta=filter_delta,
                initial_eta=[current_x, current_y, current_yaw],
            )
            self.ref_filter.eta_d = np.array(
                [current_x, current_y, current_yaw])

            # Only seed a target from odom if no goal has ever been set
            if self.target_x is None:
                self.target_x = current_x
                self.target_y = current_y
                self.target_yaw = current_yaw

    def process_performance_metrics(self, desired_pose, desired_vel, error_pos, error_vel, error_yaw):
        """
        Process and publish debug information using a moving window approach.
        """
        # Publish reference pose (from reference filter)
        debug_ref_pose = PoseStamped()
        debug_ref_pose.header.stamp = self.get_clock().now().to_msg()
        debug_ref_pose.header.frame_id = "world"
        debug_ref_pose.pose.position.x = desired_pose[0]
        debug_ref_pose.pose.position.y = desired_pose[1]
        debug_ref_pose.pose.position.z = 0.0
        # Convert yaw to quaternion
        q = R.from_euler('z', desired_pose[2]).as_quat()
        debug_ref_pose.pose.orientation.x = q[0]
        debug_ref_pose.pose.orientation.y = q[1]
        debug_ref_pose.pose.orientation.z = q[2]
        debug_ref_pose.pose.orientation.w = q[3]
        self.debug_pose_pub.publish(debug_ref_pose)

        debug_ref_vel = TwistStamped()
        debug_ref_vel.header.stamp = self.get_clock().now().to_msg()
        debug_ref_vel.header.frame_id = "base_link"
        debug_ref_vel.twist.linear.x = desired_vel[0]
        debug_ref_vel.twist.linear.y = desired_vel[1]
        debug_ref_vel.twist.angular.z = desired_vel[2]
        self.debug_vel_pub.publish(debug_ref_vel)

        debug_error_vel = TwistStamped()
        debug_error_vel.header.stamp = self.get_clock().now().to_msg()
        debug_error_vel.header.frame_id = "base_link"
        debug_error_vel.twist.linear.x = error_vel[0]
        debug_error_vel.twist.linear.y = error_vel[1]
        debug_error_vel.twist.angular.z = error_yaw
        self.debug_error_vel_pub.publish(debug_error_vel)

        # Publish error information
        debug_error_pose = Pose2D()
        debug_error_pose.x = error_pos[0]
        debug_error_pose.y = error_pos[1]
        debug_error_pose.theta = error_yaw
        self.debug_error_pose_pub.publish(debug_error_pose)

        # Update metrics using moving window
        error_norm = np.sqrt(error_pos[0]**2 + error_pos[1]**2)

        # Initialize start time if not already set
        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9

        # Add newest error to window
        self.error_window.append(error_norm)
        self.error_yaw_window.append(abs(error_yaw))

        # Keep window at fixed size
        if len(self.error_window) > self.window_size:
            self.error_window.pop(0)
            self.error_yaw_window.pop(0)

        self.sample_count += 1

        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_metrics_time >= self.metrics_interval and len(self.error_window) > 0:
            self.last_metrics_time = current_time

            # Calculate window statistics
            error_array = np.array(self.error_window)

            metrics_msg = PerformanceMetrics()
            metrics_msg.header.stamp = self.get_clock().now().to_msg()
            metrics_msg.header.frame_id = self.get_namespace().strip("/") or "vessel"
            metrics_msg.message = "Position tracking error (meters)"

            # Calculate statistics over the window
            metrics_msg.mean = np.mean(error_array)
            metrics_msg.median = np.median(error_array)
            metrics_msg.rms = np.sqrt(np.mean(np.square(error_array)))
            metrics_msg.min = np.min(error_array)
            metrics_msg.max = np.max(error_array)
            metrics_msg.stddev = np.std(error_array)

            self.debug_metrics_pub.publish(metrics_msg)

            self.get_logger().info(
                f"Position tracking metrics - Mean: {metrics_msg.mean:.3f}m, "
                f"RMS: {metrics_msg.rms:.3f}m, Max: {metrics_msg.max:.3f}m"
            )

    def check_goal_progress(self):
        """
        Check if active goal has reached target. Called from control_loop.
        Non-blocking approach for single-threaded executor compatibility.
        """
        if self.active_goal_handle is None:
            return

        if self.latest_odom is None:
            return

        # Extract current pose from odometry
        current_odom = self.latest_odom.pose.pose
        current_x = current_odom.position.x
        current_y = current_odom.position.y

        # Extract yaw
        _, _, current_yaw = R.from_quat([
            current_odom.orientation.x,
            current_odom.orientation.y,
            current_odom.orientation.z,
            current_odom.orientation.w,
        ]).as_euler("xyz", degrees=False)

        # Check for cancellation request
        if self.active_goal_handle.is_cancel_requested:
            self._finish_goal("canceled")
            return

        # Compute remaining distance and angle error
        error_norm = np.sqrt(
            (self.target_x - current_x) ** 2
            + (self.target_y - current_y) ** 2
        )
        error_yaw = wrap_to_pi(self.target_yaw - current_yaw)

        # Publish feedback
        feedback_msg = NavigateToPose.Feedback()
        current_feedback_pose = PoseStamped()
        current_feedback_pose.header.stamp = self.get_clock().now().to_msg()
        current_feedback_pose.header.frame_id = "odom"
        current_feedback_pose.pose.position = current_odom.position
        current_feedback_pose.pose.orientation = current_odom.orientation

        feedback_msg.current_pose = current_feedback_pose
        feedback_msg.distance_remaining = float(error_norm)

        try:
            self.active_goal_handle.publish_feedback(feedback_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish feedback: {e}")
            self._finish_goal("abort")
            return

        # Check if target is reached (both position and yaw) and held
        now_t = self.get_clock().now().nanoseconds / 1e9
        if error_norm < self.pos_tol and abs(error_yaw) < self.yaw_tol:
            if self.goal_within_tolerance_since is None:
                self.goal_within_tolerance_since = now_t
            if (now_t - self.goal_within_tolerance_since) >= self.success_hold_time:
                self._finish_goal("succeed")
                # Mark large errors to avoid lingering metrics effects
                self.error_pos = np.array([np.inf, np.inf])
                self.error_yaw = np.inf
        else:
            self.goal_within_tolerance_since = None

    def _finish_goal(self, how):
        """End the active goal with succeed, canceled or abort and let its execute callback return."""
        handle, done = self.active_goal_handle, self._goal_done
        self.active_goal_handle, self._goal_done = None, None
        self.goal_within_tolerance_since = None
        if handle is None:
            return
        try:
            getattr(handle, how)()
            self.get_logger().info(f"NavigateToPose goal ended: {how}")
        except Exception as e:
            self.get_logger().warn(f"Could not end goal with {how}: {e}")
        if done is not None and not done.done():
            done.set_result(how)

    def _timed_control_loop(self):
        with self._timing['loop']:
            self.control_loop()

    def control_loop(self):
        """
        Periodic control loop to update reference filter and publish control commands.
        This version uses a 3rd order reference filter to generate a smooth desired trajectory.
        """
        # Skip control loop if controller is disabled
        if not self.enabled:
            return

        if self.latest_odom is None:
            return

        if self.ref_filter is None:
            return

        if self._bb is not None and not self._bb_tick():
            return
        self._tau_parts = None

        # Check goal progress (non-blocking)
        self.check_goal_progress()

        # Extract current position from odometry (global frame)
        pos = self.latest_odom.pose.pose.position
        current_x = pos.x
        current_y = pos.y

        # Extract yaw (heading) from quaternion orientation
        orientation = self.latest_odom.pose.pose.orientation
        rot = R.from_quat([orientation.x, orientation.y,
                          orientation.z, orientation.w])
        _, _, current_yaw = rot.as_euler("xyz", degrees=False)

        # Also get current velocities from odometry for the derivative term.
        current_vx = self.latest_odom.twist.twist.linear.x
        current_vy = self.latest_odom.twist.twist.linear.y
        current_yaw_rate = self.latest_odom.twist.twist.angular.z

        # Update the reference filter with the latest target pose.
        # This "command" is smoothed by the filter.
        ref = self._guidance_reference(current_x, current_y, current_yaw)
        if ref is None:
            # path following (transit): ALOS heading + surge speed, the reference filter follows the
            # boat so DP takes over smoothly when the transit ends
            nu = np.array([current_vx, current_vy, current_yaw_rate])
            eta = np.array([current_x, current_y, current_yaw])
            self._reset_ref_filter(eta, Rz(current_yaw) @ nu)
            psi_d, u_d, kappa = self._path_command
            self._reference = None
            self.process_performance_metrics(eta, np.zeros(3), np.zeros(2), np.zeros(2),
                                             wrap_to_pi(psi_d - current_yaw))
            self._publish_wrench(*self._path_following_tau(current_yaw, nu, psi_d, u_d, kappa))
            return
        traj = getattr(self, '_traj', None)
        if traj is not None:
            # final approach: guidance gives the trajectory itself; keep the filter on it for the handover
            desired_pose, desired_vel, desired_acc = traj
            self._reset_ref_filter(desired_pose, desired_vel)
        else:
            self.ref_filter.set_eta_r(self._limit_reference(ref, current_yaw))
            self.ref_filter.update()

            # Retrieve filtered outputs: desired pose, velocity, and acceleration.
            desired_pose = self.ref_filter.eta_d  # [x, y, yaw]
            desired_vel = self.ref_filter.eta_d_dot  # velocity
            # acceleration (feedforward term)
            desired_acc = self.ref_filter.eta_d_ddot

        self._reference = np.asarray(desired_pose, dtype=float)

        # Compute errors (position and yaw) between the filtered desired state and current state.
        error_pos = np.array(
            [desired_pose[0] - current_x, desired_pose[1] - current_y])
        error_yaw = wrap_to_pi(desired_pose[2] - current_yaw)

        self.error_pos = error_pos
        self.error_yaw = error_yaw

        # Compute velocity errors
        current_vel = np.array([current_vx, current_vy])
        error_vel = np.array(
            [desired_vel[0] - current_vx, desired_vel[1] - current_vy])
        error_yaw_rate = desired_vel[2] - current_yaw_rate

        if self.model_based:
            self.process_performance_metrics(
                desired_pose, desired_vel, error_pos, error_vel, error_yaw)
            control_x, control_y, control_yaw = self._model_based_tau(
                current_yaw, np.array([error_pos[0], error_pos[1], error_yaw]),
                desired_vel, desired_acc,
                np.array([current_vx, current_vy, current_yaw_rate]))
            control_x, control_y = self._dock_push(current_yaw, control_x, control_y)
            self._publish_wrench(control_x, control_y, control_yaw)
            return

        acc = self.M_eff @ self.ref_filter.get_nu_d()

        self.integral_error_pos += error_pos * self.dt
        self.integral_error_yaw += error_yaw * self.dt

        # Apply saturation to the integral error
        self.integral_error_pos = np.clip(
            self.integral_error_pos, -self.max_integral_error_pos, self.max_integral_error_pos)
        self.integral_error_yaw = np.clip(
            self.integral_error_yaw, -self.max_integral_error_yaw, self.max_integral_error_yaw)

        # Compute control commands.
        # For position, we use feedforward desired acceleration plus a PID correction.
        world_x = (
            acc[0]
            + self.Kp_pos * error_pos[0]
            + self.Kd_vel * error_vel[0]
            + self.Ki_pos * self.integral_error_pos[0]
        )
        world_y = (
            acc[1]
            + self.Kp_pos * error_pos[1]
            + self.Kd_vel * error_vel[1]
            + self.Ki_pos * self.integral_error_pos[1]
        )
        world_yaw = (
            acc[2]
            + self.Kp_yaw * error_yaw
            + self.Kd_yaw * error_yaw_rate
            + self.Ki_yaw * self.integral_error_yaw
        )

        # Process and publish performance metrics
        self.process_performance_metrics(
            desired_pose, desired_vel, error_pos, error_vel, error_yaw)

        control_x, control_y, control_yaw = Rz(
            current_yaw).T @ np.array([world_x, world_y, world_yaw])

        # Optionally apply saturation.
        # control_x = saturate(control_x, self.saturation_pos)
        # control_y = saturate(control_y, self.saturation_pos)
        # control_yaw = saturate(control_yaw, self.saturation_yaw)

        # Publish the control command.
        self._publish_wrench(control_x, control_y, control_yaw)

        # Debug logging.
        self.get_logger().debug(
            f"Current: ({current_x:.2f}, {current_y:.2f}, {current_yaw:.2f}), "
            f"Filtered desired: ({desired_pose[0]:.2f}, {desired_pose[1]:.2f}, {desired_pose[2]:.2f}), "
            f"Control: ({control_x:.2f}, {control_y:.2f}, {control_yaw:.2f})"
        )

    def _on_bias(self, msg):
        self.bias_hat = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def _on_current(self, msg):
        self.current_hat = np.array([msg.vector.x, msg.vector.y])

    def _relative_reference(self, psi, nu_d, nu_d_dot):
        """
        Desired velocity and acceleration relative to the water (Fossen 10.3): nu_rd = nu_d - nu_c with
        nu_c = R^T [V_N, V_E, 0] from the observer, and nu_c_dot = -r S nu_c for a current that is constant
        in NED. Without a current estimate the two are returned unchanged.
        """
        if self.current_hat is None or not self.get_parameter('control.bias_feedforward').value:
            return nu_d, nu_d_dot
        nu_c = Rz(psi).T @ np.array([self.current_hat[0], self.current_hat[1], 0.0])
        nu_c_dot = -nu_d[2] * np.array([-nu_c[1], nu_c[0], 0.0])
        return nu_d - nu_c, nu_d_dot - nu_c_dot

    def _model_based_tau(self, psi, error_world, eta_d_dot, eta_d_ddot, nu):
        """
        Model-based DP law (Fossen), everything in the body frame:

            tau = M nu_d_dot + D(nu_d) nu_d + Kp R^T e + Kd (nu_d - nu) + Ki R^T int(e) dt

        When the observer estimates the current (navigation/current) the feedforward uses the desired
        velocity relative to the water, nu_d - R^T v_c: holding still in a current means moving through
        the water, and the drag of that follows the hull when the boat turns.

        With control.bias_feedforward (observer running) the integral term is replaced by -R^T b_hat,
        the observer's estimate of the slowly varying forces (Fossen 13.4.6).
        D(nu_d) nu_d is the full (linear + nonlinear) damping of the vessel model, if available.

        error_world = eta_d - eta (yaw wrapped), eta_d_dot/eta_d_ddot from the reference filter
        (world frame), nu = [u, v, r] from odometry (body frame).
        """
        R = Rz(psi)
        e_body = R.T @ error_world
        nu_d = R.T @ eta_d_dot
        nu_d_dot = R.T @ eta_d_ddot

        if self.get_parameter('control.bias_feedforward').value and self.bias_hat is not None:
            # Fossen (13.4.6 + Ch. 15): cancel the estimated slowly varying forces, no integral action
            tau_i = -R.T @ self.bias_hat
        else:
            tau_i = self._integral_action(R, error_world)
        nu_rd, nu_rd_dot = self._relative_reference(psi, nu_d, nu_d_dot)
        if self._craft is not None:
            damping_ff = -self._craft.damping(np.array([nu_rd[0], nu_rd[1], 0, 0, 0, nu_rd[2]]))[[0, 1, 5]]
        else:
            damping_ff = self.D_eff @ nu_rd
        damping_ff[0] += self._tow_drag(nu_rd[0])
        tau_ff = self.M_eff @ nu_rd_dot + damping_ff
        tau_fb = self.Kp_mb * e_body + self.Kd_mb * (nu_d - nu) + tau_i
        self._tau_parts = (tau_ff, tau_fb)
        tau = tau_ff + tau_fb
        return float(tau[0]), float(tau[1]), float(tau[2])

    def _on_obstacles(self, msg):
        self._obstacles = [(m.pose.position.x, m.pose.position.y, m.scale.x / 2.0) for m in msg.markers
                           if m.action == Marker.ADD and m.type == Marker.CYLINDER]

    def _publish_plan(self, plan):
        """The planned track as a Path in 'world' for RViz (guidance/plan)."""
        path = Path()
        path.header.frame_id, path.header.stamp = 'world', self.get_clock().now().to_msg()
        for x, y, psi in plan['eta']:
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x, p.pose.position.y = float(x), float(y)
            p.pose.orientation.z, p.pose.orientation.w = math.sin(psi / 2), math.cos(psi / 2)
            path.poses.append(p)
        self._plan_pub.publish(path)

    def _tow_drag(self, u_r):
        """Drag of a towed load at surge speed u_r through the water (0 without one)."""
        if getattr(self, '_tow_surge', None) is None:
            return 0.0
        _, a, b = self._tow_surge
        return a * u_r + b * u_r * abs(u_r)

    def _reset_ref_filter(self, eta, eta_dot):
        f = self.ref_filter
        f.eta_d, f.eta_d_dot, f.eta_d_ddot = eta.copy(), eta_dot.copy(), np.zeros(3)
        f._eta_r = eta.copy()
        f._x = np.concatenate([eta, eta_dot, np.zeros(3)])
        self._carrot = eta.copy()

    def _path_following_tau(self, psi, nu, psi_d, u_d, kappa=0.0):
        """
        Path following in transit (Fossen Ch. 12.5): a surge speed controller and a heading autopilot,
        no sway force; ALOS has put the crab angle into psi_d. Same model-based gains as the DP law.
        kappa is the path curvature (1/R on a Dubins arc): the yaw rate r_d = u_ref kappa is fed forward.

            X = m11 u_ref_dot + d(u_ref) + Kd_u (u_ref - u),   N = Kp_r e_psi + Kd_r (r_d - r) + d_r(r_d),   Y = 0
        """
        a = self.get_parameter('guidance.transit_accel').value
        step = np.clip(u_d - self._u_ref, -a * self.dt, a * self.dt)
        self._u_ref += step
        u_ref_dot = step / self.dt
        r_d = self._u_ref * kappa
        # damping on the speed through the water when the observer estimates the current
        u_r = self._relative_reference(psi, np.array([self._u_ref, 0.0, 0.0]), np.zeros(3))[0][0]
        if self._craft is not None:
            dmp = -self._craft.damping(np.array([u_r, 0, 0, 0, 0, r_d]))
            d, d_r = dmp[0], dmp[5]
        else:
            d, d_r = self.D_eff[0, 0] * u_r, self.D_eff[2, 2] * r_d
        d += self._tow_drag(u_r)
        X = self.M_eff[0, 0] * u_ref_dot + d + self.Kd_mb[0] * (self._u_ref - nu[0])
        N = self.Kp_mb[2] * wrap_to_pi(psi_d - psi) + self.Kd_mb[2] * (r_d - nu[2]) + d_r
        if self.get_parameter('control.bias_feedforward').value and self.bias_hat is not None:
            # wind and other slow loads from the observer, as in the DP law (a head wind stopped the boat
            # 0.3 m short of the goal at path_end_speed); sway is left to ALOS's crab angle
            b = -Rz(psi).T @ self.bias_hat
            X, N = X + b[0], N + b[2]
        return float(X), 0.0, float(N)

    def _integral_action(self, R, error_world):
        """Integral in the world frame (does not rotate with the boat), with conditional integration."""
        self.integral_mb += error_world * self.dt
        tau_i = self.Ki_mb * (R.T @ self.integral_mb)
        if np.any(np.abs(tau_i) > self.max_integral_force):
            self.integral_mb -= error_world * self.dt
            tau_i = np.clip(self.Ki_mb * (R.T @ self.integral_mb),
                            -self.max_integral_force, self.max_integral_force)
        return tau_i

    def _speed_limit(self, course, yaw):
        """
        Largest speed over ground [m/s] along `course` that keeps the velocity through the water inside
        the boat's envelope: an ellipse in the body frame with forward_speed / reverse_speed along surge
        and sway_speed along sway. Sway is the narrow one: the linear sway damping is large and only
        the bow azimuth pushes sideways, so 0.1 m/s already takes most of its 20 N.
        """
        g = lambda n: self.get_parameter(f'guidance.{n}').value
        d = np.array([np.cos(course - yaw), np.sin(course - yaw)])
        c = np.zeros(2)
        if self.current_hat is not None:
            c = (Rz(yaw).T @ np.array([self.current_hat[0], self.current_hat[1], 0.0]))[:2]
        best = 0.0
        for vx, ahead in ((g('forward_speed'), True), (g('reverse_speed'), False)):
            w = 1.0 / np.array([vx, g('sway_speed')])**2
            a, b, k = w @ (d * d), w @ (d * c), w @ (c * c) - 1.0
            disc = b * b - a * k
            if disc < 0.0:
                continue
            s = (b + np.sqrt(disc)) / a
            if (s * d[0] - c[0] >= 0.0) == ahead:
                best = max(best, s)
        return max(best, g('min_speed'))

    def _limit_reference(self, ref, yaw):
        """
        Rate limit the guidance reference before the reference filter. The filter alone has no speed
        limit: its peak speed grows with the distance to the goal, so a 4 m sideways move asked for more
        sway force than the azimuth has, and a 180 deg heading step gave 35 deg/s. The point the filter
        follows moves towards the goal no faster than _speed_limit(), and the heading no faster than
        max_turn_rate_deg_s.
        """
        g = lambda n: self.get_parameter(f'guidance.{n}').value
        c = getattr(self, '_carrot', None)
        if c is None:
            c = self._carrot = np.array(self.ref_filter.eta_d, dtype=float)
        d = ref[:2] - c[:2]
        distance = np.hypot(d[0], d[1])
        if g('sway_speed') <= 0.0:
            c[:2] = ref[:2]
        elif distance > 1e-9:
            speed = self._speed_limit(np.arctan2(d[1], d[0]), yaw)
            c[:2] += d * min(1.0, speed * self.dt / distance)
        step = wrap_to_pi(ref[2] - c[2])
        rate = np.radians(g('max_turn_rate_deg_s'))
        c[2] = wrap_to_pi(c[2] + (np.clip(step, -rate * self.dt, rate * self.dt) if rate > 0.0 else step))
        return c.copy()

    def _use_ocp(self, distance, yaw):
        """The optimal-control planner for moves up to ocp_max_distance (longer ones: Dubins / ALOS transit)."""
        g = lambda n: self.get_parameter(f'guidance.{n}').value
        if g('planner') != 'ocp' or not self.model_based or getattr(self, '_vessel_cfg', None) is None:
            return False
        if getattr(self, '_ocp_failed', False):
            return False
        turn = abs(wrap_to_pi(self.target_yaw - yaw))
        return distance <= g('ocp_max_distance') and (distance > g('ocp_min_distance') or turn > np.radians(20.0))

    def _ocp_planner(self):
        g = lambda n: self.get_parameter(f'guidance.ocp.{n}').value
        weights = {k: g(k) for k in ('w_T', 'w_v', 'w_r', 'w_rev', 'w_xte', 'w_hold', 'w_f', 'w_df', 'obstacle_margin')}
        if getattr(self, '_ocp', None) is None or self._ocp_weights != weights:
            G = lambda n: self.get_parameter(f'guidance.{n}').value
            self._ocp = trajectory_planner.OCPPlanner.from_vessel_config(
                self._vessel_cfg, self._vessel_params, u_max=G('forward_speed'), u_min=-G('reverse_speed'),
                r_max_deg=G('max_turn_rate_deg_s'), tow=getattr(self, '_tow_surge', None), **weights)
            self._ocp_weights = weights
        return self._ocp

    def _ocp_reference(self, x, y, yaw, phase):
        """
        Plan once per goal with the optimal-control planner (in a thread on the boat, so the 10 Hz loop keeps
        holding position meanwhile), then feed the planned pose, velocity and acceleration to the DP law as a
        trajectory (Fossen Ch. 12.1, trajectory tracking). At the end, station keeping on the goal.
        """
        if phase is None:
            nu = self.latest_odom.twist.twist
            start = [x, y, yaw, nu.linear.x, nu.linear.y, getattr(nu.angular, 'z', 0.0)]
            goal = [self.target_x, self.target_y, self.target_yaw]
            self._ocp_id = getattr(self, '_ocp_id', 0) + 1
            self._ocp_result, self._ocp_hold = None, np.array([x, y, yaw])
            obstacles = list(self._obstacles)
            job = lambda i=self._ocp_id: setattr(self, '_ocp_result', (
                i, self._ocp_planner().plan(start, goal, obstacles=obstacles)))
            if self.get_parameter('guidance.ocp_threaded').value:
                threading.Thread(target=job, daemon=True).start()
            else:
                job()
            self._guidance_phase = phase = 'ocp_plan'
            self._traj = None
        if phase == 'ocp_plan':
            res = self._ocp_result
            if res is None or res[0] != self._ocp_id:
                return self._ocp_hold                            # still planning: hold where we were
            plan = res[1]
            for ox, oy, r in plan.get('ignored', []):
                self.get_logger().warn(f"Guidance: obstacle at ({ox:.1f}, {oy:.1f}) ignored (the boat overlaps it at the "
                                       f"start or the goal, or too many obstacles)")
            for (ox, oy, r), mg in zip(plan.get('obstacles', []), plan.get('margins', [])):
                if mg < self.get_parameter('guidance.ocp.obstacle_margin').value - 1e-6:
                    self.get_logger().info(f"Guidance: obstacle at ({ox:.1f}, {oy:.1f}) close to the start or the goal, "
                                           f"clearance {mg:.2f} m there")
            for ox, oy, r in plan.get('blocked', []):
                self.get_logger().error(f"Guidance: the boat would overlap the obstacle at ({ox:.1f}, {oy:.1f}) at "
                                        f"the goal, pick a goal further from it")
            if not plan.get('ok') and (plan.get('obstacles') or plan.get('blocked')):
                # the manoeuvre rules do not know about obstacles: stay put instead of driving through one
                self.get_logger().error("Guidance: no collision-free plan found, holding position")
                self.target_x, self.target_y, self.target_yaw = self._ocp_hold
                self._reset_ref_filter(self._ocp_hold.copy(), np.zeros(3))
                self._guidance_phase, self._final_turn = 'station', False
                self._finish_goal("abort")
                return self._ocp_hold
            if not plan.get('ok'):
                self.get_logger().warn("Guidance: OCP planner failed, falling back to the manoeuvre rules")
                self._ocp_failed, self._guidance_phase = True, None
                return self._ocp_hold
            self._ocp_plan, self._ocp_t = plan, 0.0
            self._timing['ocp'].record(plan['solve_time'])
            self._publish_plan(plan)
            n_obs = len(plan.get('obstacles', []))
            self.get_logger().info(
                f"Guidance: OCP plan {plan['T']:.1f} s, solved in {plan['solve_time']:.2f} s ({plan['guess']} guess)"
                + (f", around {n_obs} obstacle{'s' if n_obs > 1 else ''}" if n_obs else ""))
            self._guidance_phase = phase = 'ocp'
        self._ocp_t += self.dt
        eta_d, eta_dot, eta_ddot, done = trajectory_planner.sample(self._ocp_plan, self._ocp_t)
        if done:
            self._traj = None
            goal = np.array([self.target_x, self.target_y, self.target_yaw])
            self._reset_ref_filter(goal, np.zeros(3))
            self._guidance_phase, self._final_turn = 'station', False
            self.get_logger().info("Guidance: station (end of OCP plan)")
            return goal
        self._traj = (eta_d, eta_dot, eta_ddot)
        return eta_d

    # ------------------------------------------------------------------------
    # Docking (results/docking_mpc_plan.md): plan with the docking OCP, track with the DP law
    # ------------------------------------------------------------------------

    def _setup_docking(self):
        """Berths from docking.harbour_file and one docking OCP per berth gap, built now: compiled to C the first
        time (~30 s, more on the boat's computer), then loaded from ~/.cache/cybership_dp."""
        path = self.get_parameter('docking.harbour_file').value
        self._berths = {}
        if not path:
            return
        if not (getattr(self, 'model_based', False) and getattr(self, '_vessel_cfg', None) is not None):
            self.get_logger().warn("Docking needs control.model_based and vessel_model: docking is off")
            return
        from cybership_dp.docking_mpc import DockingOCP
        cfg = self._vessel_cfg
        self._hull = (float(cfg['hull']['length']), float(cfg['hull']['beam']))
        self._quays, self._berths = docking.load_harbour(path, self._hull,
                                                         self.get_parameter('docking.harbour_shift').value)
        self._dock_solver_busy = False
        g = lambda n: self.get_parameter(f'docking.{n}').value
        hf = g('heading_funnel_deg')
        t = time.time()
        # the plan's minimum gap is fixed when the problem is built: one problem per berth gap
        self._dock_ocps = {gap: DockingOCP.from_vessel_config(
            cfg, self._vessel_params, free_time=True, gap_min=gap - g('gap_margin'), corridor=tuple(g('corridor')),
            heading_funnel=(math.radians(hf[0]), math.radians(hf[1])), max_cpu_time=8.0,
            # replans try the multipliers first (30 iterations at most), then the shifted plan alone: 0.08 s for a
            # small offset, 0.14 s from 0.3 m / 10 deg (tools/docking_ocp_speed.py). On one slow core the typical
            # replan time is what keeps the boat from chasing plans (0.1-0.5 s with, 0.4-2.8 s without)
            dual_warm=True)
            for gap in {b['gap'] for b in self._berths.values()}}
        self.create_subscription(String, 'guidance/dock', lambda m: self._start_dock(m.data), 10)
        self.create_service(Trigger, f'{self.get_name()}/dock', self._on_dock_service)
        self._dock_status_pub = self.create_publisher(String, 'guidance/dock_status', 10)
        self.get_logger().info(f"Docking: berths {list(self._berths)} from {path}, planners built in "
                               f"{time.time() - t:.1f} s")

    def _on_dock_service(self, request, response):
        response.success, response.message = self._start_dock('')
        return response

    def _berth_for_goal(self, x, y, yaw):
        """A goal within 1 m of a berth means: dock there (an RViz click next to the quay). Of berths at the same
        place, the one whose heading is closest to the goal's (so the arrow picks the side). None otherwise."""
        if self.latest_odom is not None:
            px, py, psi = self._odom_pose()
            if math.hypot(px - x, py - y) < 0.3 and abs(wrap_to_pi(psi - yaw)) < math.radians(10):
                return None                                # a goal where the boat already is: "hold here"
        near = [(abs(wrap_to_pi(b['pose'][2] - yaw)), name) for name, b in getattr(self, '_berths', {}).items()
                if math.hypot(b['pose'][0] - x, b['pose'][1] - y) < 1.0]
        return min(near)[1] if near else None

    def _start_dock(self, name):
        """Dock at berth `name` ('' = the first in the harbour file)."""
        if not self._berths:
            return False, "no berths (docking.harbour_file)"
        name = name.strip() or next(iter(self._berths))
        if name not in self._berths:
            self.get_logger().error(f"Docking: no berth '{name}' (have {list(self._berths)})")
            return False, f"no berth '{name}'"
        if self._bb is not None and not self._bb['gate'].active:
            self.get_logger().warn(f"Docking refused: not active ({self._bb['gate'].reason})")
            return False, f"not active ({self._bb['gate'].reason})"
        if self.latest_odom is None:
            return False, "no state yet"
        if self._dock is not None and self._dock['name'] == name:
            return True, f"already docking to {name} ({self._dock['stage']})"     # e.g. the button pressed twice
        b = self._berths[name]
        self._finish_goal("abort")
        self._set_target(*b['pose'])                       # also ends a docking in progress
        self._dock = dict(name=name, berth=np.array(b['pose'], float), hp=b['halfplane'], gap=b['gap'],
                          quay=b['quay'], face=b['face'], stage=None, plan=None, t_plan=None, result=None,
                          solving=False, last_solve=-math.inf, failures=0, hold=None, retry_at=0.0)
        self.get_logger().info(f"Docking: to {name} at ({b['pose'][0]:.2f}, {b['pose'][1]:.2f}), heading "
                               f"{math.degrees(b['pose'][2]):.0f} deg, {b['side']} side {b['gap']:.2f} m off the face")
        return True, f"docking to {name}"

    def _dock_stage(self, stage):
        d = self._dock
        if d['stage'] != stage:
            d['stage'] = stage
            self._dock_status_pub.publish(String(data=f"{d['name']}: {stage}"))
        if stage != 'transit':
            self._guidance_phase = f'dock_{stage}'

    def _dock_solve(self, x, y, yaw, warm):
        """Solve the docking OCP in a thread from the current state (velocity through the water, the observer's
        current and bias in the model); the result is picked up by _dock_reference."""
        d, now = self._dock, time.monotonic()
        tw = self.latest_odom.twist.twist
        vc = np.zeros(2) if self.current_hat is None else np.asarray(self.current_hat, float)
        c, s = math.cos(yaw), math.sin(yaw)
        x0 = np.array([x, y, yaw, tw.linear.x - (c * vc[0] + s * vc[1]), tw.linear.y - (-s * vc[0] + c * vc[1]),
                       tw.angular.z])
        bias = (np.asarray(self.bias_hat, float) if self.get_parameter('control.bias_feedforward').value
                and self.bias_hat is not None else np.zeros(3))
        elapsed = (now - d['t_plan']) if (warm and d['plan'] is not None) else None
        ocp = self._dock_ocps[d['gap']]
        d['solving'], d['last_solve'] = True, now
        self._dock_solver_busy = True

        def job():
            try:
                res = ocp.solve(x0, d['berth'], d['hp'], vc=vc, bias=bias, elapsed=elapsed)
            except Exception as e:                        # a failed solve must not kill the control loop
                res = dict(ok=False, status=str(e), solve_time=time.monotonic() - now)
            finally:
                self._dock_solver_busy = False
            res['t0'] = now
            d['result'], d['solving'] = res, False

        threading.Thread(target=job, daemon=True).start()

    def _dock_push(self, psi, fx, fy):
        """
        Alongside, push on the fenders: across the face the force towards the quay is at least docking.push_force;
        along it, and in yaw, the DP law is unchanged. Not a position target inside the quay: the observer would
        take the fender's push for a disturbance and the bias feedforward would squeeze the fender ever harder.
        Clamping instead, the observer sees the force actually sent, so it settles. If wind or current pull the
        boat off harder than the push, the DP's own force towards the quay is larger and it holds the berth.
        """
        d = getattr(self, '_dock', None)
        push = self.get_parameter('docking.push_force').value
        if d is None or d['stage'] != 'alongside' or push <= 0.0:
            return fx, fy
        a = np.asarray(d['hp'][0], float)
        a = a / np.linalg.norm(a)                            # NED direction into the quay
        c, s = math.cos(psi), math.sin(psi)
        into = np.array([c * a[0] + s * a[1], -s * a[0] + c * a[1]])     # the same in BODY
        f = np.array([fx, fy])
        f_into = float(f @ into)
        f = f + (max(push, f_into) - f_into) * into
        return float(f[0]), float(f[1])

    def _dock_reference(self, x, y, yaw):
        """
        Docking phases: transit to a stand-off pose (the OCP planner, the quay as obstacles) if the boat starts far
        off or behind the face; plan with the docking OCP (holding position meanwhile); track the plan, replanning
        only when the boat is off it (and not in the last metre: the DP's); alongside: hold the berth.
        """
        d, now = self._dock, time.monotonic()
        g = lambda n: self.get_parameter(f'docking.{n}').value
        a, b = np.asarray(d['hp'][0], float), float(d['hp'][1])
        berth = d['berth']
        gap_co = b - float(a @ np.array([x, y])) - self._hull[1] / 2      # CO's distance to the face - half beam
        if d['stage'] is None:
            out = -a / np.linalg.norm(a)
            along = float(np.array([-out[1], out[0]]) @ (np.array([x, y]) - berth[:2]))
            face_len = self._quays[d['quay']].face(d['face'])[3]
            near = math.hypot(x - berth[0], y - berth[1]) <= g('max_start_distance')
            if gap_co >= 0.5 and near and abs(along) <= face_len / 2 + 4.0:
                self._dock_stage('plan')
            else:
                d['standoff'] = np.array([*(berth[:2] + out * g('standoff')), berth[2]])
                self.target_x, self.target_y, self.target_yaw = d['standoff']
                self._guidance_phase, self._ocp_failed = None, False
                self._dock_stage('transit')
                self.get_logger().info(f"Docking: to the stand-off pose {g('standoff'):.1f} m out first")
        if d['stage'] == 'transit':
            so = d['standoff']
            if getattr(self, '_ocp_failed', False):
                self.get_logger().error("Docking: no plan to the stand-off pose, holding position")
                self._dock = None
                self.target_x, self.target_y, self.target_yaw = x, y, yaw
                return np.array([x, y, yaw])
            if not (self._guidance_phase == 'station' and math.hypot(x - so[0], y - so[1]) < 0.3
                    and abs(wrap_to_pi(yaw - so[2])) < math.radians(10)):
                return self._ocp_reference(x, y, yaw, self._guidance_phase)
            self.target_x, self.target_y, self.target_yaw = berth
            self._dock_stage('plan')
        # one solve at a time: the CasADi solver is not thread safe, and two at once (a new docking while the last
        # one's solve still ran) crashed the node
        if d['stage'] == 'plan' and not d['solving'] and now >= d['retry_at'] and not self._dock_solver_busy:
            d['hold'] = np.array([x, y, yaw])
            self._dock_solve(x, y, yaw, warm=False)
            self._dock_stage('planning')
        res = d['result']
        if res is not None:
            d['result'] = None
            if 'solve_time' in res:
                self._timing['dock'].record(res['solve_time'])
                if res['ok']:
                    d['solve_time'] = res['solve_time']      # for the replan interval: successful solves only
            if res['ok']:
                d['plan'] = dict(t=res['t'], eta=res['eta'], nu=res['nu'], nu_dot=res['nu_dot'], T=res['T'])
                d['t_plan'], d['failures'] = res['t0'], 0
                self._publish_plan(d['plan'])
                self.get_logger().info(f"Docking: plan {res['T']:.1f} s, solved in {res['solve_time']:.2f} s "
                                       f"({res.get('guess', '')})")
                self._dock_stage('track')
            elif d['plan'] is None:
                d['failures'] += 1
                if d['failures'] >= 3:
                    self.get_logger().error(f"Docking: no plan after 3 tries ({res.get('status', '')}), holding")
                    self._dock = None
                    self.target_x, self.target_y, self.target_yaw = d['hold']
                    return d['hold']
                self.get_logger().warn(f"Docking: plan failed ({res.get('status', '')}), trying again in 2 s")
                d['retry_at'] = now + 2.0
                self._dock_stage('plan')
            else:
                # a failed replan: keep following the last plan (the shifted last plan is still a good one, L5)
                self.get_logger().warn(f"Docking: replan failed ({res.get('status', '')}), keeping the last plan")
        if d['plan'] is None:
            self._traj = None
            return d['hold'] if d['hold'] is not None else np.array([x, y, yaw])
        if d['stage'] == 'track':
            eta_d, eta_dot, eta_ddot, done = trajectory_planner.sample(d['plan'], now - d['t_plan'])
            out = -a / np.linalg.norm(a)
            along = float(np.array([-out[1], out[0]]) @ (np.array([x, y]) - berth[:2]))
            tol = g('alongside_tolerance')
            late = now - d['t_plan'] - d['plan']['T']
            if done and (abs(along) > tol[0] or gap_co > tol[1]) and late < g('alongside_timeout'):
                # the plan is over but the boat is not at the berth yet: the DP takes it there before pushing
                if d.get('reaching') is None:
                    d['reaching'] = now
                    self._reset_ref_filter(np.array([x, y, yaw]), np.zeros(3))
                    self.get_logger().info(f"Docking: plan over, {along:+.2f} m along and {gap_co:.2f} m off the "
                                           f"face; going to the berth before pushing")
                self._traj = None
                return berth.copy()
            if done:
                self._traj = None
                self._reset_ref_filter(berth.copy(), np.zeros(3))
                self._dock_stage('alongside')
                push = self.get_parameter('docking.push_force').value
                self.get_logger().info(f"Docking: alongside {d['name']}" + (f", pushing {push:.0f} N on the fenders"
                                                                            if push > 0 else ""))
                return berth.copy()
            err = math.hypot(eta_d[0] - x, eta_d[1] - y), abs(wrap_to_pi(eta_d[2] - yaw))
            lim = g('replan_error')
            # at least replan_min_interval between plans, and 3 times the last solve (the plan's rule, step 6): on a
            # slow computer a plan that took 1 s otherwise arrived with the boat off it again, and it chased plans
            interval = max(g('replan_min_interval'), 3.0 * d.get('solve_time', 0.0))
            if (not d['solving'] and not self._dock_solver_busy and now - d['last_solve'] >= interval
                    and gap_co >= g('lock_gap')
                    and (err[0] > lim[0] or err[1] > math.radians(lim[1]))):
                self.get_logger().info(f"Docking: {err[0]:.2f} m / {math.degrees(err[1]):.0f} deg off the plan, "
                                       f"planning again")
                self._dock_solve(x, y, yaw, warm=True)
            self._traj = (eta_d, eta_dot, eta_ddot)
            return eta_d
        self._traj = None                                  # alongside: hold the berth
        return berth.copy()

    def _plan_dubins(self, x, y, heading, mode='path', first_turn=0.0):
        """
        Quickest Dubins path (by the time model) from (x, y, heading) to the goal over arrival headings:
        the goal heading, or one with a shorter path plus a turn on the spot at the goal. Paths that turn
        more than max_path_turn_deg in total are left out while others exist: they were loops that took
        about as long as the alternatives but looked like the boat had lost its way.

        Like a skipper, it may turn tighter at lower speed: every radius in dubins_radii is tried, with the
        speed on that path limited to radius * path_turn_rate (0.6 m/s on R = 2 m, 0.3 m/s on R = 1 m at
        17 deg/s). The time model sees the path length scaled by forward_speed / path speed. Short goals also
        get a shorter approach leg (approach_ratio * distance, at least 0.5 m, at most approach_length):
        with R = 2 m and a fixed 1.5 m leg, goals 3-5 m away with another final heading only had loops.
        Returns (time, features, path, arrival heading, path speed) or None.
        """
        g = lambda n: self.get_parameter(f'guidance.{n}').value
        distance = np.hypot(self.target_x - x, self.target_y - y)
        approach = min(g('approach_length'), max(0.5, g('approach_ratio') * distance))
        radii = [r for r in g('dubins_radii') if r > 0.0] or [g('dubins_radius')]
        cand = []
        for R in radii:
            speed = min(g('forward_speed'), R * np.radians(g('path_turn_rate_deg_s')))
            for psi_a in [self.target_yaw] + list(np.linspace(-np.pi, np.pi, 24, endpoint=False)):
                path = DubinsPath.with_approach((x, y, heading), (self.target_x, self.target_y, psi_a), R, approach)
                path.approach, path.radius = approach, R
                f = [path.length * g('forward_speed') / speed, abs(wrap_to_pi(self.target_yaw - psi_a))]
                if mode == 'pivot':
                    f.append(first_turn)
                t = float(np.dot(g(f'time_model.{mode}'), [1.0] + f))
                # turning costs extra: the quickest path was often a small loop that looked needless; turning
                # on the spot (before or after the path) costs more again: it looks like a fidget spinner
                t += g('turn_penalty') * (path.turning + f[1] + first_turn) + g('spot_turn_penalty') * (f[1] + first_turn)
                cand.append((t, f, path, psi_a, speed))
        # also no paths much longer than the distance: with the 1.5 m approach leg and R = 2 m, goals 3-6 m
        # away got loops 2-3 times the distance that looked like the boat had lost its way
        # (only for driving off directly; a pivot faces the goal first, so its path is short anyway and it
        # stays available as the fallback)
        ok = [c for c in cand if c[2].turning <= np.radians(g('max_path_turn_deg'))] or cand
        limit = g('max_path_ratio') * distance + approach
        if mode == 'path' and g('max_path_ratio') > 0.0:
            ok = [c for c in ok if c[2].length <= limit]
            if not ok:
                return None
        best = None
        for c in ok:
            if best is None or c[0] < best[0] - 0.5:     # prefer the exact goal heading on a near tie
                best = c
        return best

    def _dubins_enabled(self):
        return (self.get_parameter('guidance.path_following').value and self.model_based
                and self.get_parameter('guidance.dubins_radius').value > 0.0)

    def _choose_manoeuvre(self, distance, bearing, yaw, x0=0.0, y0=0.0):
        """
        How to get to the goal, chosen once per goal by an estimated time (seconds), like a skipper would:

            path:     bow first along a Dubins path (turning while moving) with a straight final approach
            pivot:    turn on the spot towards the goal first, then the Dubins path (goals behind the boat)
            forward:  turn the bow to the goal, drive straight, turn to the final heading (no Dubins)
            reverse:  turn the stern to the goal, back up, turn to the final heading (short distances)
            dp:       move directly (sideways / diagonally with the azimuth) while turning slowly to the
                      final heading; always for small corrections, and up to max_dp_distance

        Each estimate is linear in a few features of the manoeuvre, T = c . [1, features...], with the
        coefficients guidance.time_model.<manoeuvre> fitted to simulated runs of every manoeuvre to random
        goals (the hand-made estimates picked a slower manoeuvre for about a third of the goals):

            path:    [1, path length, |final turn on the spot|]
            pivot:   [1, path length, |final turn on the spot|, |turn on the spot first|]
            forward / reverse: [1, distance, |turn to the course| + |turn to the final heading|]
            dp:      [1, distance / speed limit, |turn to the final heading|]
        """
        g = lambda n: self.get_parameter(f'guidance.{n}').value
        if distance <= max(g('approach_radius'), g('min_transit_distance')):
            return 'dp', {}
        T = lambda mode, f: float(np.dot(g(f'time_model.{mode}'), [1.0] + list(f)))
        times, self._features, self._dubins_opts = {}, {}, {}
        for mode, course in (('forward', bearing), ('reverse', wrap_to_pi(bearing + np.pi))):
            if mode == 'reverse' and distance > g('max_reverse_distance'):
                continue
            turns = abs(wrap_to_pi(course - yaw)) + abs(wrap_to_pi(self.target_yaw - course))
            self._features[mode] = [distance, turns]
        if self._dubins_enabled():
            # The arrival heading is free: the goal heading, or one that gives a shorter path plus a turn on
            # the spot at the goal, whichever is quicker (a fixed goal heading on the "wrong" side made the
            # boat drive a full loop).
            # 'pivot' turns on the spot first, all the way to the goal or only part of it (30 deg steps), then
            # drives off on an arc: a short twist often saves a long loop, like a skipper would do it
            pivots = [bearing] + [wrap_to_pi(yaw + np.radians(k)) for k in range(-180, 181, 30) if k != 0]
            for mode, headings in (('path', [yaw]), ('pivot', pivots)):
                best = None
                for pivot in headings:
                    if mode == 'pivot' and abs(wrap_to_pi(pivot - yaw)) < np.radians(g('align_angle_deg')):
                        continue
                    c = self._plan_dubins(x0, y0, pivot, mode, abs(wrap_to_pi(pivot - yaw)))
                    if c is not None and (best is None or c[0] < best[0]):
                        best, best_pivot = c, pivot
                if best is None:
                    continue
                self._features[mode] = best[1]
                self._dubins_opts[mode] = (best[2], best[3], best[4])
                if mode == 'pivot':
                    self._pivot_heading = best_pivot
            self._features.pop('forward')   # the paths replace turn on the spot - drive straight - turn
        if g('sway_speed') > 0.0 and distance <= g('max_dp_distance'):
            # the heading changes on the way: use the slower of the start and the final heading
            speed = min(self._speed_limit(bearing, yaw), self._speed_limit(bearing, self.target_yaw))
            self._features['dp'] = [distance / speed, abs(wrap_to_pi(self.target_yaw - yaw))]
        times = {k: T(k, f) * (g('reverse_penalty') if k == 'reverse' else 1.0) for k, f in self._features.items()}
        for k in ('reverse', 'forward', 'dp'):            # the same turning cost as the Dubins paths, all on the spot
            if k in times:
                times[k] += (g('turn_penalty') + g('spot_turn_penalty')) * self._features[k][1]
        for k in ('path', 'pivot'):
            if k in times:
                path = self._dubins_opts[k][0]
                spot = self._features[k][1] + (self._features[k][2] if k == 'pivot' else 0.0)
                times[k] += g('turn_penalty') * (path.turning + spot) + g('spot_turn_penalty') * spot
        best = min(times, key=times.get)
        forced = g('force_manoeuvre')
        if forced:
            best = forced if forced in times else best
        if best in self._dubins_opts:
            self._dubins, self._path_arrival, self._path_speed = self._dubins_opts[best]
            self._path_pivot = best == 'pivot'
        return best, times

    def _guidance_reference(self, x, y, yaw):
        """Reference eta_r = [x, y, psi] for the reference filter (guidance block)."""
        if getattr(self, '_dock', None) is not None:
            return self._dock_reference(x, y, yaw)
        goal = np.array([self.target_x, self.target_y, self.target_yaw])
        if not self.get_parameter('guidance.turn_then_go').value:
            return goal
        radius = self.get_parameter('guidance.approach_radius').value
        align = np.radians(self.get_parameter('guidance.align_angle_deg').value)
        dx, dy = self.target_x - x, self.target_y - y
        distance = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx)
        phase = getattr(self, '_guidance_phase', None)

        if phase in ('ocp_plan', 'ocp') or (phase is None and self._use_ocp(distance, yaw)):
            return self._ocp_reference(x, y, yaw, phase)
        if phase is None:
            if self._obstacles:
                self.get_logger().warn("Guidance: the manoeuvre rules (moves over ocp_max_distance or after a failed "
                                       "plan) do not avoid obstacles")
            self._manoeuvre, times = self._choose_manoeuvre(distance, bearing, yaw, x, y)
            self.get_logger().info(
                f"Guidance: {self._manoeuvre} to the goal {distance:.1f} m away"
                + (" (estimated " + ", ".join(f"{k} {v:.0f} s" for k, v in times.items()) + ")" if times else ""))
            phase = 'station' if self._manoeuvre == 'dp' else 'turn'
            if self._manoeuvre == 'path':
                phase = 'path'
                self.get_logger().info(f"Guidance: Dubins path {self._dubins.word}, {self._dubins.length:.1f} m")
            self._guidance_hold = (x, y)
            self._alos = ALOS(self.get_parameter('guidance.los_lookahead').value,
                              self.get_parameter('guidance.alos_gamma').value,
                              self.get_parameter('guidance.alos_max_crab_deg').value,
                              self.get_parameter('guidance.los_method').value,
                              self.get_parameter('guidance.ilos_kappa').value)
            self._alos.set_path((x, y), (self.target_x, self.target_y))
            self._max_cross_track = 0.0
            if self._manoeuvre == 'dp':
                # small correction: change the heading slowly (rate limited) while moving; turning
                # 90 deg in 4 s while translating pushed the boat 0.65 m the wrong way
                self._guidance_heading, self._final_turn = yaw, True
        if phase in ('path', 'approach'):
            return self._dubins_reference(x, y, yaw)
        # the course the bow (forward) or the stern (reverse) points along
        reverse = getattr(self, '_manoeuvre', 'forward') == 'reverse'
        path_following = (self.get_parameter('guidance.path_following').value and self.model_based
                          and getattr(self, '_alos', None) is not None)
        if path_following:
            # along the path (start -> goal), and in transit the ALOS heading (L6 slide 61)
            course = self._alos.pi_h
            if phase == 'transit':
                course = self._alos.update(x, y, self.dt)
                self._max_cross_track = max(self._max_cross_track, abs(self._alos.errors(x, y)[1]))
        else:
            course = bearing
        if phase == 'turn' and getattr(self, '_manoeuvre', None) == 'pivot':
            course = getattr(self, '_pivot_heading', course)
        course = wrap_to_pi(course + np.pi) if reverse else course
        heading_error = abs(wrap_to_pi(course - yaw))
        settle = self.get_parameter('guidance.settle_radius').value
        if phase in ('turn', 'transit') and distance < radius:
            # arrive: keep the transit heading while stopping; the final heading comes when settled,
            # so the boat does not turn ~180 deg while it still has speed (it drifted 1 m past)
            phase, self._guidance_heading = 'arrive', course if phase == 'transit' else yaw
        elif phase == 'arrive' and distance < settle:
            phase, self._final_turn = 'station', True   # turn to the final heading, rate limited
        elif phase == 'turn' and (heading_error < (np.radians(3.0) if getattr(self, '_manoeuvre', None) == 'pivot'
                                                   else align)
                                  or (heading_error < align and getattr(self, '_yaw_traj', None) is not None
                                      and self._yaw_traj['t'] > self._yaw_traj['T'] + 3.0)):
            # a pivot turns all the way: from 20 deg off, the Dubins path to a goal 3 m away was a full loop
            phase = 'transit'
            self._traj, self._yaw_traj = None, None
            if getattr(self, '_manoeuvre', None) == 'pivot':
                # turned towards the goal on the spot: now the Dubins path from here
                # plan again from where the boat is now (it may have drifted while turning)
                _, _, self._dubins, self._path_arrival, self._path_speed = self._plan_dubins(x, y, yaw)
                self.get_logger().info(f"Guidance: Dubins path {self._dubins.word}, {self._dubins.length:.1f} m")
                return self._dubins_reference(x, y, yaw)
        elif phase == 'transit' and heading_error > 2 * align:
            phase, self._guidance_hold = 'turn', (x, y)   # pushed off course: turn again

        if phase != getattr(self, '_guidance_phase', None):
            self.get_logger().info(f"Guidance: {phase} (distance {distance:.1f} m)")
            if path_following and getattr(self, '_guidance_phase', None) == 'transit':
                self.get_logger().info(
                    f"ALOS: crab angle estimate {np.degrees(self._alos.beta_hat):.1f} deg, "
                    f"max cross-track error {self._max_cross_track:.2f} m")
            if path_following and phase == 'transit':
                self._u_ref = float(self.latest_odom.twist.twist.linear.x)
        self._guidance_phase = phase
        if phase != 'turn':
            self._traj, self._yaw_traj = None, None
        if phase == 'turn':
            if self.get_parameter('guidance.turn_trajectory').value:
                psi_d, r_d, a_d, self._turn_done = self._turn_trajectory(yaw, course)
                self._traj = (np.array([*self._guidance_hold, psi_d]), np.array([0.0, 0.0, r_d]),
                              np.array([0.0, 0.0, a_d]))
                return np.array([*self._guidance_hold, psi_d])
            return np.array([*self._guidance_hold, course])
        if phase == 'transit' and path_following:
            # speed along the path: ramp down to arrive_speed at approach_radius
            g = lambda n: self.get_parameter(f'guidance.{n}').value
            a = g('transit_accel')
            left = max(self._alos.remaining(x, y) - radius, 0.0)
            speed = g('reverse_speed') if reverse else g('forward_speed')
            u_d = min(speed, np.sqrt(g('arrive_speed')**2 + 2 * a * left))
            self._path_command = (course, -u_d if reverse else u_d, 0.0)
            return None
        if phase == 'transit':
            return np.array([self.target_x, self.target_y, course])
        if phase == 'arrive':
            return np.array([self.target_x, self.target_y, self._guidance_heading])
        if getattr(self, '_final_turn', False):
            # ramp the heading reference to the final heading: a fast large turn at the goal needs
            # nearly all the thrust and the allocation gives up sway, so the boat slid ~0.5 m away
            rate = np.radians(self.get_parameter('guidance.final_turn_rate_deg_s').value)
            step = wrap_to_pi(self.target_yaw - self._guidance_heading)
            step_max = rate * self.dt
            self._guidance_heading = wrap_to_pi(self._guidance_heading + np.clip(step, -step_max, step_max))
            if abs(step) < 1e-3:
                self._final_turn = False
            return np.array([self.target_x, self.target_y, self._guidance_heading])
        if self.get_parameter('guidance.weathervane').value:
            return np.array([self.target_x, self.target_y, self._weathervane_heading()])
        return goal

    def _weathervane_heading(self):
        """
        Weather-optimal heading while holding position (Fossen, L6 / Ch. 12.1: align the vessel against the
        mean resulting environmental force). The bow turns slowly into the current estimated by the observer,
        or, without current, against the bias force (wind, waves, a towline); the heading is otherwise free.
        With a towed load this keeps the load downstream, so the line stays taut instead of going slack and
        jerking. Small changes inside the deadband are ignored, so the heading does not hunt.
        """
        g = lambda n: self.get_parameter(f'guidance.{n}').value
        if getattr(self, '_wv_heading', None) is None:
            self._wv_heading, self._wv_turning = self.target_yaw, False
        target = None
        vc = self.current_hat
        b = self.bias_hat
        if vc is not None and np.hypot(vc[0], vc[1]) > g('weathervane_min_current'):
            target = np.arctan2(-vc[1], -vc[0])                       # bow towards where the current comes from
        elif b is not None and np.hypot(b[0], b[1]) > g('weathervane_min_force'):
            target = np.arctan2(-b[1], -b[0])                         # bow against the mean force
        if target is not None:
            err = wrap_to_pi(target - self._wv_heading)
            if abs(err) > np.radians(g('weathervane_deadband_deg')):
                self._wv_turning = True
            if self._wv_turning:
                step = np.radians(g('weathervane_rate_deg_s')) * self.dt
                self._wv_heading = wrap_to_pi(self._wv_heading + np.clip(err, -step, step))
                if abs(err) < np.radians(1.0):
                    self._wv_turning = False
        self.target_yaw = self._wv_heading       # the goal's heading becomes the weather heading
        return self._wv_heading

    def _crab_feedforward(self, pi_h):
        """
        Crab angle that cancels the estimated current across the path: with speed U through the water
        along psi, the ground track stays on pi_h when U sin(psi - pi_h) = -c_perp, so
        psi = pi_h - asin(c_perp / U). ALOS then only adapts what is left (wind, model errors);
        on its own it needed most of a 4-10 m leg to learn the angle and drifted up to 0.9 m off.
        """
        if self.current_hat is None:
            return 0.0
        c_perp = -np.sin(pi_h) * self.current_hat[0] + np.cos(pi_h) * self.current_hat[1]
        U = max(self._u_ref, 0.2)
        return float(np.arcsin(np.clip(c_perp / U, -0.7, 0.7)))

    def _dubins_reference(self, x, y, yaw):
        """
        Follow the Dubins path with ALOS (heading + surge speed, no sway force) and slow down towards its
        end; near the end hand over to DP station keeping. The path ends with the goal heading, so the
        boat drives there like a boat (turning while moving) instead of turning on the spot twice.
        Returns None while following (the command is in _path_command), else the station reference.
        """
        g = lambda n: self.get_parameter(f'guidance.{n}').value
        path = self._dubins
        if self._guidance_phase == 'approach':
            return self._approach_reference()
        if self._guidance_phase != 'path':
            self.get_logger().info("Guidance: path")
            self._u_ref = float(self.latest_odom.twist.twist.linear.x)
            self._guidance_phase, self._path_moving, self._path_stall = 'path', False, 0.0
        s, _, pi_h, ye, kappa = path.project(x, y)
        self._max_cross_track = max(self._max_cross_track, abs(ye))
        left = path.length - s
        dx, dy = self.target_x - x, self.target_y - y
        distance = np.hypot(dx, dy)
        ahead = np.cos(yaw) * dx + np.sin(yaw) * dy           # goal distance along the bow
        u = float(self.latest_odom.twist.twist.linear.x)
        self._path_moving = self._path_moving or u > 0.15
        self._path_stall = self._path_stall + self.dt if (self._path_moving and u < 0.05) else 0.0
        leg = getattr(path, 'approach', g('approach_length'))
        dp_from = leg if g('approach_dp') else g('settle_radius')
        if (left < max(dp_from, g('settle_radius')) or distance < g('settle_radius')
                or (distance < g('approach_radius') and ahead < g('settle_radius'))   # passed or abeam
                or self._path_stall > 3.0):                                           # stopped short
            self.get_logger().info(
                f"Guidance: station (end of path, ALOS crab estimate {np.degrees(self._alos.beta_hat):.1f} deg, "
                f"max cross-track error {self._max_cross_track:.2f} m)")
            if g('approach_dp') and leg > 0 and left > g('settle_radius'):
                # DP follows the approach leg itself: a point sliding along it and braking evenly to rest
                # at the goal, with its speed and deceleration fed forward. The azimuth corrects the
                # sideways error on the way in, and there is no filter lag to catch up at the end.
                u0 = max(float(self.latest_odom.twist.twist.linear.x), 0.05)
                self._approach = dict(s=s, v=u0, decel=u0**2 / (2.0 * left))
                self._guidance_phase = 'approach'
                self.get_logger().info(f"Guidance: approach ({left:.1f} m at {u0:.2f} m/s)")
                return self._approach_reference()
            self._guidance_phase, self._final_turn, self._guidance_heading = 'station', True, yaw
            return np.array([self.target_x, self.target_y, yaw])
        # a lookahead longer than the turning radius cuts the arcs: scale it down on tight paths
        self._alos.delta = min(g('los_lookahead'), max(1.0, 1.25 * getattr(path, 'radius', g('dubins_radius'))))
        psi_d = wrap_to_pi(self._alos.heading(pi_h, ye, self.dt) - self._crab_feedforward(pi_h))
        a = g('transit_accel')
        # slow down to path_end_speed where DP takes over (settle_radius before the end)
        u_d = min(getattr(self, '_path_speed', g('forward_speed')), np.sqrt(g('path_end_speed')**2 + 2 * a * max(left - max(dp_from, g('settle_radius')), 0.0)))
        self._path_command = (psi_d, u_d, kappa)
        return None

    def _turn_trajectory(self, yaw, target):
        """
        Heading trajectory for a turn on the spot: accelerate at turn_accel, hold max_turn_rate, brake to rest
        exactly on the target heading. DP tracks it with the yaw rate and acceleration fed forward. The
        reference filter took 15 s for 166 deg (it lagged 3-4 s at the start and crept over the last 5 deg);
        this takes about 8 s. Returns (psi_d, r_d, r_d_dot, finished).
        """
        g = lambda n: self.get_parameter(f'guidance.{n}').value
        tr = getattr(self, '_yaw_traj', None)
        if tr is None:
            a, v = np.radians(g('turn_accel_deg_s2')), np.radians(g('max_turn_rate_deg_s'))
            d = wrap_to_pi(target - yaw)
            D = abs(d)
            if D < v * v / a:                     # never reaches full rate: accelerate, then brake
                v = np.sqrt(D * a)
            t1 = v / a
            t2 = t1 + max(D - v * v / a, 0.0) / v if v > 0 else t1
            tr = self._yaw_traj = dict(psi0=yaw, sign=np.sign(d), D=D, a=a, v=v, t1=t1, t2=t2, T=t2 + t1, t=0.0)
        tr['t'] += self.dt
        t, a, v, t1, t2, T = tr['t'], tr['a'], tr['v'], tr['t1'], tr['t2'], tr['T']
        if t < t1:
            s_, r_, acc = 0.5 * a * t * t, a * t, a
        elif t < t2:
            s_, r_, acc = 0.5 * a * t1 * t1 + v * (t - t1), v, 0.0
        elif t < T:
            tau = T - t
            s_, r_, acc = tr['D'] - 0.5 * a * tau * tau, a * tau, -a
        else:
            s_, r_, acc = tr['D'], 0.0, 0.0
        k = tr['sign']
        return wrap_to_pi(tr['psi0'] + k * s_), k * r_, k * acc, t >= T

    def _approach_reference(self):
        """Pose, velocity and acceleration on the approach leg (constant deceleration to the goal)."""
        ap, path = self._approach, self._dubins
        ap['v'] = max(ap['v'] - ap['decel'] * self.dt, 0.0)
        ap['s'] = min(ap['s'] + ap['v'] * self.dt, path.length)
        x, y, h = path.at(ap['s'])
        if ap['v'] <= 0.0 or ap['s'] >= path.length:
            self._traj = None
            self._reset_ref_filter(np.array([self.target_x, self.target_y, h]), np.zeros(3))
            self._guidance_phase, self._final_turn, self._guidance_heading = 'station', True, h
            self.get_logger().info("Guidance: station (end of approach)")
            return np.array([self.target_x, self.target_y, h])
        c, sn = np.cos(h), np.sin(h)
        self._traj = (np.array([x, y, h]), np.array([ap['v'] * c, ap['v'] * sn, 0.0]),
                      np.array([-ap['decel'] * c, -ap['decel'] * sn, 0.0]))
        return np.array([x, y, h])

    def _publish_wrench(self, fx, fy, mz):
        if self._bb is not None:
            self._bb_publish(np.array([fx, fy, mz], dtype=float))
            return
        wrench_msg = Wrench()
        wrench_msg.force.x = fx
        wrench_msg.force.y = fy
        wrench_msg.torque.z = mz
        self.control_pub.publish(wrench_msg)

    # ------------------------------------------------------------------------
    # The team's pose controller interface (blueboat.interface)
    # ------------------------------------------------------------------------

    def _setup_blueboat(self):
        """Topics and services as blueboat_core_control's pose_controller has them (see blueboat_io.py)."""
        from blueboat_interfaces.msg import OperationMode, Waypoint
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        debug = '/control/pose'
        self._bb = {
            'gate': blueboat_io.DPGate(self.get_parameter('blueboat.state_timeout').value),
            'autonomous': False, 'last_tick': None, 'Waypoint': Waypoint, 'warned': set(),
            'AUTONOMOUS': blueboat_io.byte_value(OperationMode.AUTONOMOUS),
            'tau': self.create_publisher(WrenchStamped, '/control/tau', 10),
            'gains': self.create_publisher(String, f'{debug}/gains', latched),
        }
        for name, kind in [('active', Bool), ('status', String), ('target', Waypoint), ('reference', Waypoint),
                           ('state', Waypoint), ('target_pose', PoseStamped), ('reference_pose', PoseStamped),
                           ('error', Vector3Stamped), ('integral', Vector3Stamped), ('tau_feedforward', WrenchStamped),
                           ('tau_feedback', WrenchStamped), ('period', Float64)]:
            self._bb[name] = self.create_publisher(kind, f'{debug}/{name}', 10)
        self.create_subscription(OperationMode, '/control/opmode', self._bb_on_opmode, latched)
        self.create_subscription(PoseStamped, '/goal_pose', self._bb_on_goal_pose, 10)
        self.create_subscription(Waypoint, f'{debug}/setpoint', self._bb_on_setpoint, 10)
        self.create_service(Trigger, f'{debug}/reset', self._bb_on_reset)
        self.create_service(Trigger, f'{debug}/reset_integral', self._bb_on_reset_integral)
        self.create_service(Trigger, f'{debug}/hold', self._bb_on_hold)
        self._bb_publish_gains()
        self.get_logger().info("Team interface: drives in AUTONOMOUS only; goals from /goal_pose and "
                               f"{debug}/setpoint; force on /control/tau, debug on {debug}/*")

    def _bb_on_opmode(self, msg):
        self._bb['autonomous'] = blueboat_io.byte_value(msg.mode) == self._bb['AUTONOMOUS']

    def _odom_pose(self):
        """(x, y, psi) NED of the latest odometry."""
        p, q = self.latest_odom.pose.pose.position, self.latest_odom.pose.pose.orientation
        return np.array([p.x, p.y, math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))])

    def _bb_tick(self):
        """The gate, once per control step: True when the DP may drive."""
        bb, now = self._bb, time.monotonic()
        if bb['last_tick'] is not None:
            bb['period'].publish(Float64(data=now - bb['last_tick']))
        bb['last_tick'] = now
        event = bb['gate'].step(now, bb['autonomous'])
        if event == 'activated':
            self._bb_hold()
            self.get_logger().info("Team interface: active, holding the current pose")
        elif event == 'deactivated':
            self._finish_goal("abort")
            self._guidance_phase, self._traj, self._dock = None, None, None
            self._bb_publish(np.zeros(3), debug=False)          # stop at once, before the allocator notices
            self.get_logger().info(f"Team interface: inactive ({bb['gate'].reason})")
        active = bb['gate'].active
        bb['active'].publish(Bool(data=active))
        bb['status'].publish(String(data=f"active: {getattr(self, '_guidance_phase', None) or 'station'}"
                                    if active else f"inactive: {bb['gate'].reason}"))
        return active

    def _bb_hold(self):
        """Hold the current pose: the target is where the boat is, the filter and the integral start over there."""
        eta = self._odom_pose()
        tw = self.latest_odom.twist.twist
        self._reset_ref_filter(eta, Rz(eta[2]) @ np.array([tw.linear.x, tw.linear.y, tw.angular.z]))
        self.integral_mb = np.zeros(3)
        self.integral_error_pos, self.integral_error_yaw = np.zeros(2), 0.0
        self._set_target(*eta)

    def _bb_target(self, target, source):
        """A target (x, y, psi) NED from /goal_pose or the setpoint: accepted only while active."""
        if not np.all(np.isfinite(target)):
            self.get_logger().warn(f"Target {target} from {source} ignored: not finite")
        elif not self._bb['gate'].active:
            self.get_logger().warn(f"Target from {source} ignored: not active ({self._bb['gate'].reason})")
        elif self._berth_for_goal(*target) is not None:
            berth = self._berth_for_goal(*target)
            self.get_logger().info(f"Target from {source} next to berth {berth}: {self._start_dock(berth)[1]}")
        else:
            self._finish_goal("abort")                           # replaces a NavigateToPose goal in progress
            self._set_target(*target)
            self.get_logger().info(f"Target from {source}: x {target[0]:.2f} m, y {target[1]:.2f} m, "
                                   f"heading {math.degrees(target[2]):.1f} deg NED")

    def _bb_on_goal_pose(self, msg):
        q, p = msg.pose.orientation, msg.pose.position
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        frame = msg.header.frame_id.lstrip('/')
        pose = self._odom_pose() if self.latest_odom is not None else None
        target = blueboat_io.goal_in_ned(frame, p.x, p.y, yaw, pose)
        if target is None:
            if frame not in self._bb['warned']:
                self._bb['warned'].add(frame)
                self.get_logger().warn(f"/goal_pose in frame '{frame}' ignored: expected ned, world, enu or base_link")
            return
        self._bb_target(target, f"/goal_pose ({frame})")

    def _bb_on_setpoint(self, msg):
        self._bb_target(np.array([msg.xn, msg.yn, msg.psi_n]), "/control/pose/setpoint")

    def _bb_on_reset(self, request, response):
        if self.latest_odom is not None:
            eta, tw = self._odom_pose(), self.latest_odom.twist.twist
            self._reset_ref_filter(eta, Rz(eta[2]) @ np.array([tw.linear.x, tw.linear.y, tw.angular.z]))
        self.integral_mb = np.zeros(3)
        self.integral_error_pos, self.integral_error_yaw = np.zeros(2), 0.0
        response.success, response.message = True, "integral cleared, filter seeded on the current state"
        return response

    def _bb_on_reset_integral(self, request, response):
        self.integral_mb = np.zeros(3)
        self.integral_error_pos, self.integral_error_yaw = np.zeros(2), 0.0
        response.success = True
        response.message = "integral cleared" + (" (unused: the observer's bias is fed forward)"
                                                 if self.get_parameter('control.bias_feedforward').value else "")
        return response

    def _bb_on_hold(self, request, response):
        response.success = self._bb['gate'].active and self.latest_odom is not None
        if response.success:
            self._finish_goal("abort")
            self._bb_hold()
        response.message = "holding the current pose" if response.success else \
            f"not active ({self._bb['gate'].reason})"
        return response

    def _bb_publish_gains(self):
        gains = {'law': 'model_based' if self.model_based else 'pid',
                 'planner': self.get_parameter('guidance.planner').value,
                 'bias_feedforward': self.get_parameter('control.bias_feedforward').value,
                 'filter_omega': list(self.get_parameter('filter.omega').value),
                 'filter_delta': list(self.get_parameter('filter.delta').value)}
        if self.model_based:
            gains.update(Kp=np.round(self.Kp_mb, 3).tolist(), Kd=np.round(self.Kd_mb, 3).tolist(),
                         Ki=np.round(self.Ki_mb, 4).tolist(),
                         omega_n=list(self.get_parameter('control.omega_n').value),
                         zeta=list(self.get_parameter('control.zeta').value))
        self._bb['gains'].publish(String(data=json.dumps(gains)))

    def _bb_publish(self, tau, debug=True):
        """/control/tau and, with debug, the pose controller's debug topics for this step."""
        bb, stamp = self._bb, self.get_clock().now().to_msg()

        def wrench(v):
            m = WrenchStamped()
            m.header.stamp, m.header.frame_id = stamp, 'base_link'
            m.wrench.force.x, m.wrench.force.y, m.wrench.torque.z = (float(x) for x in v)
            return m

        def waypoint(eta):
            m = bb['Waypoint']()
            m.header.stamp, m.header.frame_id = stamp, 'ned'
            m.xn, m.yn, m.psi_n = float(eta[0]), float(eta[1]), float(wrap_to_pi(eta[2]))
            return m

        def pose(eta):
            m = PoseStamped()
            m.header.stamp, m.header.frame_id = stamp, 'ned'
            m.pose.position.x, m.pose.position.y = float(eta[0]), float(eta[1])
            m.pose.orientation.z, m.pose.orientation.w = math.sin(eta[2] / 2.0), math.cos(eta[2] / 2.0)
            return m

        def vector(v):
            m = Vector3Stamped()
            m.header.stamp, m.header.frame_id = stamp, 'ned'
            m.vector.x, m.vector.y, m.vector.z = (float(x) for x in v)
            return m

        bb['tau'].publish(wrench(tau))
        if not debug or self.latest_odom is None or self.target_x is None:
            return
        eta = self._odom_pose()
        target = np.array([self.target_x, self.target_y, self.target_yaw])
        bb['state'].publish(waypoint(eta))
        bb['target'].publish(waypoint(target))
        bb['target_pose'].publish(pose(target))
        ref = getattr(self, '_reference', None)
        if ref is not None:
            bb['reference'].publish(waypoint(ref))
            bb['reference_pose'].publish(pose(ref))
            err = ref - eta
            err[2] = wrap_to_pi(err[2])
            bb['error'].publish(vector(err))
        bb['integral'].publish(vector(self.integral_mb))
        if self._tau_parts is not None:
            bb['tau_feedforward'].publish(wrench(self._tau_parts[0]))
            bb['tau_feedback'].publish(wrench(self._tau_parts[1]))

    def _set_target(self, x, y, yaw):
        """A new target (NED): guidance decides turn/transit/station again. A new goal ends docking."""
        self._dock = None
        self.target_x, self.target_y, self.target_yaw = float(x), float(y), float(yaw)
        self._guidance_phase = None
        self._traj, self._yaw_traj = None, None
        self._ocp_failed = False
        self._wv_heading = None
        self._final_turn = False

    def action_goal_callback(self, goal_request):
        self.get_logger().info("Received NavigateToPose goal request")
        if self._bb is not None and not self._bb['gate'].active:
            self.get_logger().warn(f"NavigateToPose goal rejected: not active ({self._bb['gate'].reason})")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def action_cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request for NavigateToPose")
        return CancelResponse.ACCEPT

    def change_state_callback(self, request, response):
        """Service callback to enable/disable the controller."""
        if request.data:
            self.enabled = True
            # Reset controller state
            self.start_time = None
            self.error_window.clear()
            self.error_yaw_window.clear()
            self.sample_count = 0
            self.last_metrics_time = 0.0
            self.integral_error_pos = np.zeros(2)
            self.integral_error_yaw = 0.0
            self.integral_mb = np.zeros(3)
            self.error_pos = np.zeros(2)
            self.error_yaw = 0.0
            self.ref_filter = None

            response.success = True
            response.message = "Controller enabled and reset."
        else:
            self.enabled = False
            # Publish zero forces and torques to reset the controller
            self._publish_wrench(0.0, 0.0, 0.0)

            response.success = True
            response.message = "Controller disabled."
        return response

    async def execute_callback(self, goal_handle):
        """
        Action server callback. It sets the target and then waits, without blocking the executor, until
        check_goal_progress() (called from control_loop) ends the goal: reached, canceled, or replaced by
        a newer goal. The goal must have a final state before this returns, else rclpy marks it aborted.
        """
        self.get_logger().info("Executing NavigateToPose goal...")

        # Extract target pose from the goal message
        target_pose: PoseStamped = goal_handle.request.pose

        if not target_pose.header.frame_id:
            self.get_logger().error("Rejected goal without frame_id")
            goal_handle.abort()
            return NavigateToPose.Result()
        try:
            if target_pose.header.frame_id in("odom","/odom"):
                target_in_odom = target_pose
            else:
                target_in_odom = self.tf_buffer.transform(
                    target_pose, "odom", timeout=Duration(seconds=0.2))
        except TransformException as error:
            self.get_logger().error(
                f"Failed to transform target pose from {target_pose.header.frame_id} to odom: {error}")
            goal_handle.abort()
            return NavigateToPose.Result()

        # Convert quaternion to yaw angle
        orientation = target_in_odom.pose.orientation
        rot = R.from_quat([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w]
        )
        _, _, yaw = rot.as_euler("xyz", degrees=False)
        berth = self._berth_for_goal(target_in_odom.pose.position.x, target_in_odom.pose.position.y, yaw)
        if berth is not None:
            ok, msg = self._start_dock(berth)
            self.get_logger().info(f"Goal next to berth {berth}: {msg}")
            (goal_handle.succeed if ok else goal_handle.abort)()
            return NavigateToPose.Result()
        self._set_target(target_in_odom.pose.position.x, target_in_odom.pose.position.y, yaw)

        self.get_logger().info(
            f"Target position: ({self.target_x}, {self.target_y})")

        # Publish target pose marker
        self.publish_target_pose_marker(target_in_odom)

        # A new goal replaces the one in progress
        self._finish_goal("abort")
        self.active_goal_handle = goal_handle
        self._goal_done = Future()
        self.goal_within_tolerance_since = None

        await self._goal_done
        return NavigateToPose.Result()

    def publish_target_pose_marker(self, pose_stamped: PoseStamped):
        """Publish a simple marker (e.g., an arrow) in RViz to visualize the requested pose."""
        marker = Marker()
        marker.header = pose_stamped.header
        marker.ns = "target_pose"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Pose (position/orientation)
        marker.pose = pose_stamped.pose

        # Scale and color
        marker.scale.x = 0.5  # Arrow length
        marker.scale.y = 0.1  # Arrow width
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # Lifetime (0 = forever)
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    server_node = PositionController()
    rclpy.spin(server_node)
    server_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
