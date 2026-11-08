#!/usr/bin/env bash
# 4-hjørnetest med RViz som viser kraftpiler (total kraft + hver thruster).
source /opt/ros/jazzy/setup.bash
source ~/ros_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
rviz2 -d ~/ros_ws/tools/voyager_forces.rviz > /tmp/rviz_forces.log 2>&1 &
RVIZ=$!
trap 'kill $RVIZ 2>/dev/null' EXIT
GUI=false bash ~/ros_ws/tools/run_four_corner_sim.sh
