"""
Generic launch file for the configurable simulator.

    ros2 launch cybership_simulator vessel.launch.py vessel:=voyager
    ros2 launch cybership_simulator vessel.launch.py vessel:=microampere use_gui:=false
    ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true gnss:=rtk waves:=true wave_hs:=0.05
    then open http://localhost:8080 (web GUI, web:=false to skip)
    ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true gazebo:=true
    (also shows the boat in a VRX Gazebo world; our model stays the physics, Gazebo only draws it)

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
    wind_speed, wind_dir = float(lc("wind_speed")), float(lc("wind_dir"))
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

    map_shift = [float(v) for v in lc("map_shift").split()]
    harbour = lc("harbour")
    if harbour.lower() in ("false", "none", "off", "''", '""'):
        harbour = ""
    if harbour == "auto":
        harbour = os.path.join(get_package_share_directory("cybership_config"), "config", vessel, "harbour.yaml")
        harbour = harbour if os.path.exists(harbour) else ""

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
                 "environment.waves.Tp": wave_tp, "environment.waves.direction_deg": wave_dir,
                 "environment.wind.speed": wind_speed, "environment.wind.direction_deg": wind_dir,
                 "tow.enabled": lc("tow").lower() == "true",
                 "boat_interface.enabled": lc("boat_interface").lower() == "true",
                 "boat_interface.clock": lc("boat_clock"),
                 "initial_conditions.position.x": float(lc("x0")),
                 "initial_conditions.position.y": float(lc("y0")),
                 "initial_conditions.position.yaw": math.radians(float(lc("yaw0_deg"))),
                 "harbour_file": harbour,
                 "sensors.gnss.multipath": lc("multipath").lower() == "true"},
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
        tow_model = lc("tow").lower() == "true" and lc("tow_model").lower() == "true"
        actions += _dp_actions(vessel, vessel_name, use_observer, wave_filter_w0, gnss, web_port, tow_model,
                               lc("obstacles"), lc("boat_interface").lower() == "true", harbour, map_shift,
                               lc("observer_input"), lc("dp_prefix") or None)
        if harbour:
            # what the DP believes against the truth near the quays (viz/docking, viz/gap_true, viz/gap_believed)
            actions.append(launch_ros.actions.Node(
                package="cybership_dp", executable="docking_viz_node.py", namespace=vessel_name, name="docking_viz",
                parameters=[{"vessel_model": vessel, "harbour_file": harbour, "harbour_shift": map_shift}],
                remappings=[] if use_observer else [("navigation/odom", "measurement/odom")]))
    if lc("foxglove").lower() == "true":
        # Foxglove Studio connects to ws://localhost:8765 (the team's microAmpere.json layout, blueboat_foxglove)
        actions.append(launch_ros.actions.Node(
            package="foxglove_bridge", executable="foxglove_bridge", name="foxglove_bridge",
            parameters=[{"port": 8765}]))
    if lc("gazebo").lower() == "true":
        actions += _gazebo_actions(vessel, vessel_name, lc("gz_world"), [float(v) for v in lc("gz_origin").split()],
                                   gui=lc("gz_gui").lower() == "true", lidar=lc("lidar").lower() == "true")
        if lc("lidar").lower() == "true":
            # the lidar's frame for RViz (the cloud is in 'os_sensor', as on the boat), from vessel.yaml
            import yaml
            with open(os.path.join(get_package_share_directory("cybership_config"), "config", vessel,
                                   "vessel.yaml")) as f:
                lid = (yaml.safe_load(f) or {}).get("lidar") or {}
            if lid:
                x, y, z = lid["position_flu"]
                r, p, yw = (math.radians(v) for v in lid.get("orientation_flu_deg", [0, 0, 0]))
                actions.append(launch_ros.actions.Node(
                    package="tf2_ros", executable="static_transform_publisher", name=f"lidar_frame_{vessel_name}",
                    arguments=["--x", str(x), "--y", str(y), "--z", str(z), "--roll", str(r), "--pitch", str(p),
                               "--yaw", str(yw), "--frame-id", f"{vessel_name}/base_link", "--child-frame-id",
                               "os_sensor"]))
            # the quay face from the lidar (perception/quay); attitude and pose from the observer if there is one
            actions.append(launch_ros.actions.Node(
                package="cybership_dp",
                executable="quay_detector_node.py",
                namespace=vessel_name,
                name="quay_detector",
                output="screen",
                parameters=[{"vessel_model": vessel, "harbour_file": harbour, "harbour_shift": map_shift}],
                remappings=[] if use_observer else [("navigation/odom", "measurement/odom")],
            ))
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


def _gazebo_actions(vessel, vessel_name, world, origin, gui=True, lidar=True):
    """
    Gazebo world (our small_harbour, or a VRX world by name), the node that moves our boat's mesh in it, and the
    boat's lidar (vessel.yaml 'lidar'), published on /ouster/points by gz_pose_bridge with ROS time as on the real
    boat. gui false: the server only, rendering headless (for tests).
    """
    import yaml
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    mesh = os.path.join(get_package_share_directory("cybership_description"), "meshes", vessel, f"{vessel}.stl")
    own = os.path.join(get_package_share_directory("cybership_simulator"), "worlds", f"{world}.sdf")
    vessel_yaml = os.path.join(get_package_share_directory("cybership_config"), "config", vessel, "vessel.yaml")
    with open(vessel_yaml) as f:
        lidar_cfg = (yaml.safe_load(f) or {}).get("lidar") if lidar else None
    lidar_pose = ([*lidar_cfg["position_flu"], *lidar_cfg.get("orientation_flu_deg", [0.0, 0.0, 0.0])]
                  if lidar_cfg else [0.0])
    if os.path.exists(own):
        gz = launch.actions.IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
            launch_arguments={"gz_args": f"-r {own}" if gui else f"-r -s --headless-rendering {own}"}.items())
    else:
        vrx = os.path.join(get_package_share_directory("vrx_gz"), "launch", "vrx_environment.launch.py")
        gz = launch.actions.IncludeLaunchDescription(
            PythonLaunchDescriptionSource(vrx),
            launch_arguments={"world": world, "sim_mode": "sim", "bridge_competition_topics": "False"}.items())
    return [
        gz,
        launch_ros.actions.Node(
            package="cybership_simulator",
            executable="gz_pose_bridge.py",
            namespace=vessel_name,
            parameters=[{"world": world, "mesh": mesh, "origin": origin, "lidar_pose": lidar_pose,
                         "lidar_config": vessel_yaml}],
        ),
    ]


def _dp_actions(vessel, vessel_name, use_observer=False, wave_filter_w0=0.0, gnss="standard", web_port=0,
                tow_model=False, obstacles="", boat_interface=False, harbour="", map_shift=(0.0, 0.0),
                observer_input="auto", prefix=None):
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
            prefix=prefix,
            # boat_interface: commands on the real boat's topics (/control/thrust, /actuator/azimuth_cmd)
            parameters=[{"vessel_model": vessel, "frequency": 20.0, "blueboat.output": boat_interface}],
            remappings=[("control/force/command", "control/force/command/mux")]
            # velocity for the thrust-loss compensation: the estimate when there is an observer
            + ([("measurement/odom", "navigation/odom")] if use_observer else []),
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
            prefix=prefix,
            # vessel_model: mass/added mass from vessel.yaml, the same model the simulator uses
            # boat_interface: the team's pose controller interface (/control/opmode, /goal_pose, /control/tau,
            # /control/pose/*), so Foxglove's DP tab and the team's allocator work with it
            parameters=[controller_params, {"vessel_model": vessel,
                                            "control.bias_feedforward": use_observer,
                                            "tow.enabled": tow_model,
                                            "blueboat.interface": boat_interface,
                                            "docking.harbour_file": harbour, "docking.harbour_shift": map_shift}],
            remappings=[("control/force/command", "control/force/command/position")]
            # with the observer the controller sees the estimate (Fossen 13.4.6), not the true state
            + ([("measurement/odom", "navigation/odom")] if use_observer else []),
        ),
    ]
    if not boat_interface:              # with the team interface the controller reads /goal_pose itself
        actions.append(launch_ros.actions.Node(
            package="cybership_dp",
            executable="position_control_client_node.py",
            namespace=vessel_name,
            name="position_control_client",
        ))
    actions += [
        # obstacles for the OCP planner: click them in RViz (Publish Point) or list them in a file
        launch_ros.actions.Node(
            package="cybership_dp",
            executable="obstacle_manager_node.py",
            namespace=vessel_name,
            name="obstacle_manager",
            output="screen",
            parameters=[{"file": obstacles, "harbour": harbour, "harbour_shift": map_shift}],
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
            prefix=prefix,
            parameters=[{"vessel_model": vessel, "wave_filter.omega0": wave_filter_w0, "gnss_profile": gnss,
                         "tow.enabled": tow_model, "blueboat.interface": boat_interface,
                         "input": observer_input if observer_input != "auto" else ("lekf" if boat_interface else "sim"),
                         "harbour_file": harbour, "harbour_shift": map_shift}],
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
        launch.actions.DeclareLaunchArgument("wave_hs", default_value="0.05", description="Significant wave height [m] (0.05: calm water, the BlueBoat is not for rough seas)"),
        launch.actions.DeclareLaunchArgument("wave_tp", default_value="2.3", description="Peak period [s]"),
        launch.actions.DeclareLaunchArgument("wave_dir", default_value="0.0",
                                             description="Direction the waves travel towards [deg], 0 = north"),
        launch.actions.DeclareLaunchArgument(
            "wind_speed", default_value="0.0",
            description="Mean wind speed [m/s] (Blendermann loads, vessel.yaml model.wind). "
                        "Can be changed while running: ros2 param set /<ns>/sim environment.wind.speed <m/s>"),
        launch.actions.DeclareLaunchArgument("wind_dir", default_value="0.0",
                                             description="Direction the wind blows towards [deg], 0 = north"),
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
        launch.actions.DeclareLaunchArgument(
            "tow", default_value="false",
            description="Tow a load behind the boat (vessel.yaml 'tow'; bar force on sim/tow/tension)"),
        launch.actions.DeclareLaunchArgument(
            "boat_interface", default_value="false",
            description="The real Microampere's topics: /control/thrust and /actuator/azimuth_cmd in, "
                        "/navigation/odometry/ned and /sensors/* out (needs blueboat_interfaces); our DP then speaks "
                        "the team's interface: drives only in AUTONOMOUS (/start_operation), goals on /goal_pose and "
                        "/control/pose/setpoint, force on /control/tau, debug on /control/pose/* and /control/allocator/*; "
                        "the observer reads the LEKF odometry (/navigation/odometry/ned) instead of sensor/gnss and imu"),
        launch.actions.DeclareLaunchArgument(
            "dp_prefix", default_value="",
            description="Command put in front of the DP nodes (observer, position controller, allocation), to run them "
                        "on fewer cores as on the boat's computer, e.g. dp_prefix:='taskset -c 0-1'"),
        launch.actions.DeclareLaunchArgument(
            "observer_input", default_value="auto",
            description="The observer's measurements: 'sim' (sensor/gnss, sensor/imu), 'lekf' (/navigation/odometry/ned) "
                        "or 'auto' (lekf with boat_interface:=true)"),
        launch.actions.DeclareLaunchArgument(
            "boat_clock", default_value="ros",
            description="With boat_interface:=true: stamps of the boat's topics, 'ros' or 'sentiboard' (seconds since "
                        "the start on the epoch 2000-01-01, as the boat's SentiBoard)"),
        launch.actions.DeclareLaunchArgument(
            "foxglove", default_value="false", description="Start foxglove_bridge on ws://localhost:8765"),
        launch.actions.DeclareLaunchArgument(
            "obstacles", default_value="",
            description="YAML file with fixed circular obstacles for the planner (obstacles: [[x, y, r], ...]); "
                        "more can be clicked in RViz with Publish Point"),
        launch.actions.DeclareLaunchArgument(
            "harbour", default_value="auto",
            description="Quays and berths for docking (YAML); auto = cybership_config/config/<vessel>/harbour.yaml if "
                        "it exists; false = no quay. Dock with: ros2 topic pub --once /<ns>/guidance/dock std_msgs/String \"data: ''\""),
        launch.actions.DeclareLaunchArgument(
            "map_shift", default_value="0.0 0.0",
            description="Test a map error: the DP's harbour map moved by 'north east' [m]; the simulator keeps the true quay"),
        launch.actions.DeclareLaunchArgument(
            "multipath", default_value="false",
            description="Test GNSS multipath: within 3 m of a quay the fix gets 0.1 m noise and a slow 0.15 m bias"),
        launch.actions.DeclareLaunchArgument("x0", default_value="0.0", description="Start north [m]"),
        launch.actions.DeclareLaunchArgument("y0", default_value="0.0", description="Start east [m]"),
        launch.actions.DeclareLaunchArgument("yaw0_deg", default_value="0.0", description="Start heading [deg]"),
        launch.actions.DeclareLaunchArgument(
            "tow_model", default_value="true",
            description="With tow:=true: DP law and OCP planner use the load's mass and drag in surge, the observer the load cell (sensor/tow_force)"),
        launch.actions.DeclareLaunchArgument(
            "gazebo", default_value="false",
            description="Also show the boat in a VRX Gazebo world (needs ros_gz and vrx_gz); physics stays ours"),
        launch.actions.DeclareLaunchArgument(
            "gz_world", default_value="small_harbour",
            description="small_harbour (ours, light) or a VRX world such as sydney_regatta (large, slow)"),
        launch.actions.DeclareLaunchArgument(
            "gz_gui", default_value="true", description="With gazebo:=true: the Gazebo window (false: server only)"),
        launch.actions.DeclareLaunchArgument(
            "lidar", default_value="true",
            description="With gazebo:=true: the boat's Ouster lidar (vessel.yaml 'lidar') on /ouster/points"),
        launch.actions.DeclareLaunchArgument(
            "gz_origin", default_value="0 0",
            description="Where our NED origin is in the Gazebo world [m, ENU x y]; -532 162 for sydney_regatta"),
        launch.actions.OpaqueFunction(function=_launch_setup),
    ])
