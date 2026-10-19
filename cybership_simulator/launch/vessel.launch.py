"""
Generic launch file for the configurable simulator.

    ros2 launch cybership_simulator vessel.launch.py vessel:=voyager
    ros2 launch cybership_simulator vessel.launch.py vessel:=helga use_gui:=false

Loads cybership_config/config/<vessel>/vessel.yaml. Unlike <vessel>.launch.py, the
vessel name is not restricted to a fixed list. If a vessel has no URDF or RViz config
yet, the generic 'any' model/config is used instead.
"""

import os

import launch
import launch.actions
import launch.conditions
import launch.substitutions
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

from cybership_utilities.launch import anon


def _launch_setup(context, *args, **kwargs):
    vessel = launch.substitutions.LaunchConfiguration("vessel").perform(context)
    vessel_name = launch.substitutions.LaunchConfiguration("vessel_name").perform(context) or vessel
    use_gui = launch.substitutions.LaunchConfiguration("use_gui").perform(context).lower() == "true"

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
                {"vessel_model": vessel, "tf_prefix": vessel_name},
            ],
        ),
        launch_ros.actions.Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=vessel_name,
            name=f"robot_state_publisher_{anon()}",
            parameters=[{
                "robot_description": launch.substitutions.Command(
                    ["xacro ", urdf, " gazebo:=ignition namespace:=", vessel_name]
                ),
                "frame_prefix": f"{vessel_name}/",
                "use_sim_time": True,
            }],
        ),
    ]
    if use_gui:
        actions.append(launch_ros.actions.Node(
            package="rviz2",
            executable="rviz2",
            namespace=vessel_name,
            name=f"rviz_{anon()}",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": True}],
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
        launch.actions.OpaqueFunction(function=_launch_setup),
    ])
