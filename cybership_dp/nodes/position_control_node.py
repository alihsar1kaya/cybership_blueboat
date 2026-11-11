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

import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Wrench, Pose2D, PoseStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
import math
from scipy.spatial.transform import Rotation as R
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer, GoalResponse, CancelResponse, ActionClient
from rclpy.executors import MultiThreadedExecutor
from cybership_tests.go_to_client import NavigateToPoseClient
from shoeboxpy.model3dof import Shoebox
from cybership_utilities import vessel_model
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, Trigger, Empty
from cybership_controller.position.reference_filter import ThirdOrderReferenceFilter
from cybership_dp.los import ALOS
from cybership_dp.dubins import DubinsPath
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
                ('guidance.force_manoeuvre', ''),
                ('guidance.max_path_turn_deg', 270.0),
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
        self.bias_hat = None            # observer bias b_hat in NED [N, N, N m] (navigation/bias)
        self.create_subscription(Vector3Stamped, 'navigation/bias', self._on_bias, 10)
        self.current_hat = None         # observer current velocity in NED [m/s, m/s] (navigation/current)
        self.create_subscription(Vector3Stamped, 'navigation/current', self._on_current, 10)

        self.error_pos = np.zeros(2)
        self.error_yaw = 0.0

        self.get_logger().info(
            "Goto Point Controller (Reference Filter Version) Initialized with parameters."
        )

        # Timer for periodic control updates
        self.timer = self.create_timer(self.dt, self.control_loop)

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
            self.M_eff = vessel_model.M_eff_3dof(params)
            self.D_eff = vessel_model.D_3dof(params)
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
                self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(f"Parameters updated: {updated_params}")
        self._snapshot_parameters()  # Ensure configuration is updated after parameters change

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

        wrench_msg = Wrench()

        # Publish these instead
        wrench_msg.force.x = control_x
        wrench_msg.force.y = control_y
        wrench_msg.force.z = 0.0  # No vertical force
        wrench_msg.torque.x = 0.0
        wrench_msg.torque.y = 0.0
        wrench_msg.torque.z = control_yaw

        # Publish the control command.
        self.control_pub.publish(wrench_msg)

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
        tau = (self.M_eff @ nu_rd_dot + damping_ff
               + self.Kp_mb * e_body + self.Kd_mb * (nu_d - nu) + tau_i)
        return float(tau[0]), float(tau[1]), float(tau[2])

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
        the bow azimuth pushes sideways, so 0.2 m/s already takes about 45 N of its 67 N.
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

    def _plan_dubins(self, x, y, heading, mode='path', first_turn=0.0):
        """
        Quickest Dubins path (by the time model) from (x, y, heading) to the goal over arrival headings:
        the goal heading, or one with a shorter path plus a turn on the spot at the goal. Paths that turn
        more than max_path_turn_deg in total are left out while others exist: they were loops that took
        about as long as the alternatives but looked like the boat had lost its way.
        Returns (time, features, path, arrival heading) or None.
        """
        g = lambda n: self.get_parameter(f'guidance.{n}').value
        cand = []
        for psi_a in [self.target_yaw] + list(np.linspace(-np.pi, np.pi, 24, endpoint=False)):
            path = DubinsPath.with_approach((x, y, heading), (self.target_x, self.target_y, psi_a),
                                            g('dubins_radius'), g('approach_length'))
            f = [path.length, abs(wrap_to_pi(self.target_yaw - psi_a))]
            if mode == 'pivot':
                f.append(first_turn)
            t = float(np.dot(g(f'time_model.{mode}'), [1.0] + f))
            cand.append((t, f, path, psi_a))
        ok = [c for c in cand if c[2].turning <= np.radians(g('max_path_turn_deg'))] or cand
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
            for mode, pivot in (('path', yaw), ('pivot', bearing)):
                if mode == 'pivot' and abs(wrap_to_pi(bearing - yaw)) < np.radians(g('align_angle_deg')):
                    continue
                best = self._plan_dubins(x0, y0, pivot, mode, abs(wrap_to_pi(pivot - yaw)))
                if best is None:
                    continue
                self._features[mode] = best[1]
                self._dubins_opts[mode] = (best[2], best[3])
            self._features.pop('forward')   # the paths replace turn on the spot - drive straight - turn
        if g('sway_speed') > 0.0 and distance <= g('max_dp_distance'):
            # the heading changes on the way: use the slower of the start and the final heading
            speed = min(self._speed_limit(bearing, yaw), self._speed_limit(bearing, self.target_yaw))
            self._features['dp'] = [distance / speed, abs(wrap_to_pi(self.target_yaw - yaw))]
        times = {k: T(k, f) * (g('reverse_penalty') if k == 'reverse' else 1.0) for k, f in self._features.items()}
        best = min(times, key=times.get)
        forced = g('force_manoeuvre')
        if forced:
            best = forced if forced in times else best
        if best in self._dubins_opts:
            self._dubins, self._path_arrival = self._dubins_opts[best]
            self._path_pivot = best == 'pivot'
        return best, times

    def _guidance_reference(self, x, y, yaw):
        """Reference eta_r = [x, y, psi] for the reference filter (guidance block)."""
        goal = np.array([self.target_x, self.target_y, self.target_yaw])
        if not self.get_parameter('guidance.turn_then_go').value:
            return goal
        radius = self.get_parameter('guidance.approach_radius').value
        align = np.radians(self.get_parameter('guidance.align_angle_deg').value)
        dx, dy = self.target_x - x, self.target_y - y
        distance = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx)
        phase = getattr(self, '_guidance_phase', None)

        if phase is None:
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
        course = wrap_to_pi(course + np.pi) if reverse else course
        heading_error = abs(wrap_to_pi(course - yaw))
        settle = self.get_parameter('guidance.settle_radius').value
        if phase in ('turn', 'transit') and distance < radius:
            # arrive: keep the transit heading while stopping; the final heading comes when settled,
            # so the boat does not turn ~180 deg while it still has speed (it drifted 1 m past)
            phase, self._guidance_heading = 'arrive', course if phase == 'transit' else yaw
        elif phase == 'arrive' and distance < settle:
            phase, self._final_turn = 'station', True   # turn to the final heading, rate limited
        elif phase == 'turn' and heading_error < (np.radians(3.0) if getattr(self, '_manoeuvre', None) == 'pivot'
                                                  else align):
            # a pivot turns all the way: from 20 deg off, the Dubins path to a goal 3 m away was a full loop
            phase = 'transit'
            if getattr(self, '_manoeuvre', None) == 'pivot':
                # turned towards the goal on the spot: now the Dubins path from here
                # plan again from where the boat is now (it may have drifted while turning)
                _, _, self._dubins, self._path_arrival = self._plan_dubins(x, y, yaw)
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
        if phase == 'turn':
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
        return goal

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
        dp_from = g('approach_length') if g('approach_dp') else g('settle_radius')
        if (left < max(dp_from, g('settle_radius')) or distance < g('settle_radius')
                or (distance < g('approach_radius') and ahead < g('settle_radius'))   # passed or abeam
                or self._path_stall > 3.0):                                           # stopped short
            self.get_logger().info(
                f"Guidance: station (end of path, ALOS crab estimate {np.degrees(self._alos.beta_hat):.1f} deg, "
                f"max cross-track error {self._max_cross_track:.2f} m)")
            if g('approach_dp') and g('approach_length') > 0 and left > g('settle_radius'):
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
        psi_d = wrap_to_pi(self._alos.heading(pi_h, ye, self.dt) - self._crab_feedforward(pi_h))
        a = g('transit_accel')
        # slow down to path_end_speed where DP takes over (settle_radius before the end)
        u_d = min(g('forward_speed'), np.sqrt(g('path_end_speed')**2 + 2 * a * max(left - max(dp_from, g('settle_radius')), 0.0)))
        self._path_command = (psi_d, u_d, kappa)
        return None

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
        wrench_msg = Wrench()
        wrench_msg.force.x = fx
        wrench_msg.force.y = fy
        wrench_msg.torque.z = mz
        self.control_pub.publish(wrench_msg)

    def action_goal_callback(self, goal_request):
        self.get_logger().info("Received NavigateToPose goal request")
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
            zero_wrench = Wrench()
            self.control_pub.publish(zero_wrench)

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

        self.target_x = target_in_odom.pose.position.x
        self.target_y = target_in_odom.pose.position.y
        self._guidance_phase = None  # new goal: guidance decides turn/transit/station again
        self._traj = None
        self._final_turn = False

        self.get_logger().info(
            f"Target position: ({self.target_x}, {self.target_y})")

        # Convert quaternion to yaw angle
        orientation = target_in_odom.pose.orientation
        rot = R.from_quat([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w]
        )
        _, _, self.target_yaw = rot.as_euler("xyz", degrees=False)

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
