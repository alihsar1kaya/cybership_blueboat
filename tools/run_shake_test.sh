#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash; source ~/ros_ws/venv/bin/activate; source ~/ros_ws/install/setup.bash
export ROS_DOMAIN_ID=82
setsid ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true use_gui:=false "${@:2}" > /tmp/shake_$1.log 2>&1 &
sleep 12; python3 ~/ros_ws/tools/shake_test.py $1
pkill -INT -f "[v]essel.launch.py vessel:=microampere dp:=true use_gui:=false"; sleep 3
for p in $(pgrep -f "[i]nstall/cybership_|[r]obot_state_publisher --ros-args|[s]tatic_transform_publisher --frame-id world|[t]opic_tools/mux"); do
  grep -q "ROS_DOMAIN_ID=82" /proc/$p/environ 2>/dev/null && kill -9 $p; done
