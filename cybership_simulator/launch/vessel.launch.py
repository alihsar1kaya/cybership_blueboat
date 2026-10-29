"""
Generic launch file for the configurable simulator.

    ros2 launch cybership_simulator vessel.launch.py vessel:=voyager
    ros2 launch cybership_simulator vessel.launch.py vessel:=microampere use_gui:=false
    ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true gnss:=rtk waves:=true wave_hs:=0.1
    then open http://localhost:8080 (web GUI, web:=false to skip)

Loads cybership_config/config/<vessel>/vessel.yaml. Unlike <vessel>.launch.py, the
vessel name is not restricted to a fixed list. If a vessel has no URDF or RViz config
yet, the generic 'any' model/config is used instead.
"""

import math
import os

import launch
import launch.actions
import launch.conditions
import launch.substitutions
import launch_ros.actions
import launch_ros.parameter_descriptions
from ament_index_python.packages import get_package_share_directory

from cybership_utilities.launch import anon


def _launch_setup(context, *args, **kwargs):
    vessel = launch.substitutions.LaunchConfiguration("vessel").perform(context)
    vessel_name = launch.substitutions.LaunchConfiguration("vessel_name").perform(context) or vessel
    use_gui = launch.substitutions.LaunchConfiguration("use_gui").perform(context).lower() == "true"
    use_dp = launch.substitutions.LaunchConfiguration("dp").perform(context).lower() == "true"
    noisy_odom = launch.substitutions.LaunchConfiguration("noisy_odom").perform(context).lower() == "true"
    use_observer = launch.substitutions.LaunchConfiguration("observer").perform(context).lower() == "true"
    wave_filter_w0 = float(launch.substitutions.LaunchConfiguration("wave_filter_omega0").perform(context))
    lc = lambda name: launch.substitutions.LaunchConfiguration(name).perform(context)
    use_waves = lc("waves").lower() == "true"
    wave_hs, wave_tp, wave_dir = float(lc("wave_hs")), float(lc("wave_tp")), float(lc("wave_dir"))
    if wave_filter_w0 <= 0.0:
        # observer wave filter always on at the wave peak frequency: waves can be switched on while running
        # (web GUI), and the filter costs nothing in calm water (tools/observer_fair_test.py)
        wave_filter_w0 = 2 * math.pi / wave_tp
    gnss = launch.substitutions.LaunchConfiguration("gnss").perform(context)

    # URDF: vessel-specific if it exists, else the generic 'any' model
    description_dir = get_package_share_directory("cybership_description")
    urdf = os.path.join(description_dir, "urdf", "model", vessel, "base.urdf.xacro")
    if not os.path.exists(urdf):
        urdf = os.path.join(description_dir, "urdf", "model", "any", "base.urdf.xacro")

    # RViz config: vessel-specific if it exists, else 'any.rviz'
    viz_dir = get_package_share_directory("cybership_viz")
    rviz_config = os.path.join(viz_dir, "config", f"{vessel}.rviz")
    if not os.path.exists(rviz_config):
        rviz_config = os.path.join(viz_dir, "config", "any.rviz")

    sim_config = os.path.join(
        get_package_share_directory("cybership_simulator"), "config", "simulation.yaml"
    )

    actions = [
        launch.actions.LogInfo(msg=f"Vessel '{vessel}' as /{vessel_name}, URDF: {urdf}"),
        launch_ros.actions.Node(
            package="cybership_simulator",
            executable="vessel.py",
            namespace=vessel_name,
            name="sim",
            output="screen",
            parameters=[
                sim_config,
                {"vessel_model": vessel, "tf_prefix": vessel_name, "sensors.noisy_odom": noisy_odom,
                 "sensors.gnss_profile": gnss,
                 "environment.waves.Hs": wave_hs if use_waves else 0.0,
                 "environment.waves.Tp": wave_tp, "environment.waves.direction_deg": wave_dir},
            ],
        ),
        launch_ros.actions.Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=vessel_name,
            name=f"robot_state_publisher_{anon()}",
            parameters=[{
                # value_type=str: otherwise ROS tries to parse the URDF text as YAML
                "robot_description": launch_ros.parameter_descriptions.ParameterValue(
                    launch.substitutions.Command(
                        ["xacro ", urdf, " gazebo:=ignition namespace:=", vessel_name]
                    ),
                    value_type=str,
                ),
                "frame_prefix": f"{vessel_name}/",
                "use_sim_time": False,  # sim stamps messages with wall-clock time
            }],
        ),
    ]
    if use_dp:
        web_port = int(lc("web_port")) if lc("web").lower() == "true" else 0
        actions += _dp_actions(vessel, vessel_name, use_observer, wave_filter_w0, gnss, web_port)
    if use_gui:
        # Desired force tau_d as WrenchStamped, so RViz can draw it (green arrow)
        actions.append(launch_ros.actions.Node(
            package="cybership_viz",
            executable="desired_force_stamped.py",
            namespace=vessel_name,
        ))
        actions.append(launch_ros.actions.Node(
            package="rviz2",
            executable="rviz2",
            namespace=vessel_name,
            name=f"rviz_{anon()}",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": False}],  # must match the sim's wall-clock stamps
        ))
    return actions


def _dp_actions(vessel, vessel_name, use_observer=False, wave_filter_w0=0.0, gnss="standard", web_port=0):
    """Allocation, force mux, position controller and goal client (the DP stack), optionally the observer."""
    config_dir = os.path.join(get_package_share_directory("cybership_config"), "config")
    controller_params = os.path.join(config_dir, vessel, "controller_position.yaml")
    actions = []
    if not os.path.exists(controller_params):
        actions.append(launch.actions.LogInfo(
            msg=f"No controller_position.yaml for '{vessel}', using voyager's gains"))
        controller_params = os.path.join(config_dir, "voyager", "controller_position.yaml")

    actions += [
        # Workaround: position_control_node transforms goals into frame 'odom',
        # which nothing publishes in simulation (see known bugs).
        launch_ros.actions.Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"world_to_odom_{vessel_name}",
            arguments=["--frame-id", "world", "--child-frame-id", "odom"],
        ),
        # Thrust allocation from vessel.yaml
        launch_ros.actions.Node(
            package="cybership_dp",
            executable="vessel_allocation_node.py",
            namespace=vessel_name,
            output="screen",
            parameters=[{"vessel_model": vessel, "frequency": 20.0}],
            remappings=[("control/force/command", "control/force/command/mux")],
        ),
        # Force mux. The first input is selected at start, so the position controller drives the boat.
        launch_ros.actions.Node(
            package="topic_tools",
            executable="mux",
            namespace=vessel_name,
            name="force_mux",
            arguments=[
                "control/force/command/mux",       # output
                "control/force/command/position",  # inputs (first = selected at start)
                "control/force/command",
                "control/force/command/manual",
                "control/force/command/velocity",
                "control/force/command/los",
            ],
        ),
        # Position (DP) controller and the client that turns /goal_pose (RViz "2D Goal Pose") into goals
        launch_ros.actions.Node(
            package="cybership_dp",
            executable="position_control_node.py",
            namespace=vessel_name,
            name="position_controller",
            output="screen",
            # vessel_model: mass/added mass from vessel.yaml, the same model the simulator uses
            parameters=[controller_params, {"vessel_model": vessel,
                                            "control.bias_feedforward": use_observer}],
            remappings=[("control/force/command", "control/force/command/position")]
            # with the observer the controller sees the estimate (Fossen 13.4.6), not the true state
            + ([("measurement/odom", "navigation/odom")] if use_observer else []),
        ),
        launch_ros.actions.Node(
            package="cybership_dp",
            executable="position_control_client_node.py",
            namespace=vessel_name,
            name="position_control_client",
        ),
    ]
    if web_port:
        actions.append(launch_ros.actions.Node(
            package="cybership_utilities",
            executable="dp_web_gui.py",
            namespace=vessel_name,
            name="dp_web_gui",
            output="screen",
            parameters=[{"vessel_model": vessel, "port": web_port}],
        ))
    if use_observer:
        actions.append(launch_ros.actions.Node(
            package="cybership_dp",
            executable="dp_observer_node.py",
            namespace=vessel_name,
            name="dp_observer",
            output="screen",
            parameters=[{"vessel_model": vessel, "wave_filter.omega0": wave_filter_w0, "gnss_profile": gnss}],
        ))
    return actions


def generate_launch_description():
    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            "vessel", default_value="voyager",
            description="Vessel model; loads cybership_config/config/<vessel>/vessel.yaml"),
        launch.actions.DeclareLaunchArgument(
            "vessel_name", default_value="",
            description="Namespace for the vessel (default: same as vessel)"),
        launch.actions.DeclareLaunchArgument(
            "use_gui", default_value="true", description="Start RViz"),
        launch.actions.DeclareLaunchArgument(
            "dp", default_value="false",
            description="Also start the DP stack (allocation, mux, position controller)"),
        launch.actions.DeclareLaunchArgument(
            "noisy_odom", default_value="false",
            description="Controller sees the state plus sensor noise (until an observer is used)"),
        launch.actions.DeclareLaunchArgument(
            "observer", default_value="false",
            description="DP Kalman filter on the simulated sensors; the controller uses its estimate"),
        launch.actions.DeclareLaunchArgument(
            "waves", default_value="false",
            description="Irregular waves (JONSWAP, Capytaine RAOs); also turns on the observer's wave filter. "
                        "Can be changed while running: ros2 param set /<ns>/sim environment.waves.Hs <m>"),
        launch.actions.DeclareLaunchArgument("wave_hs", default_value="0.1", description="Significant wave height [m]"),
        launch.actions.DeclareLaunchArgument("wave_tp", default_value="2.3", description="Peak period [s]"),
        launch.actions.DeclareLaunchArgument("wave_dir", default_value="0.0",
                                             description="Direction the waves travel towards [deg], 0 = north"),
        launch.actions.DeclareLaunchArgument(
            "gnss", default_value="standard",
            description="GNSS receiver: 'standard' (~0.2 m + slow wander) or 'rtk' (~2 cm), vessel.yaml sensors"),
        launch.actions.DeclareLaunchArgument(
            "wave_filter_omega0", default_value="0.0",
            description="Observer wave filter peak frequency [rad/s] (2 pi / Tp of the waves), 0 = off"),
        launch.actions.DeclareLaunchArgument(
            "web", default_value="true",
            description="With dp:=true: simple web GUI (map, goals, STOP, sea state) on http://localhost:<web_port>"),
        launch.actions.DeclareLaunchArgument("web_port", default_value="8080", description="Port of the web GUI"),
        launch.actions.OpaqueFunction(function=_launch_setup),
    ])
