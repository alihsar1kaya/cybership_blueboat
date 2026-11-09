#!/usr/bin/env bash
# Goal manoeuvre test headless: bash tools/run_goal_motion_test.sh <label> <Hs> [launch args]
source /opt/ros/jazzy/setup.bash; source ~/ros_ws/venv/bin/activate; source ~/ros_ws/install/setup.bash
export ROS_DOMAIN_ID=81
setsid ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true use_gui:=false web:=false "${@:3}" > /tmp/goal_$1.log 2>&1 &
sleep 12
if [ "$2" != "0" ]; then
  ros2 param set /microampere/sim environment.waves.seed 3 >/dev/null
  ros2 param set /microampere/sim environment.waves.Hs $2 >/dev/null
fi
sleep 5
if [ -n "$CURRENT" ]; then ros2 param set /microampere/sim environment.current.speed ${CURRENT% *} >/dev/null; ros2 param set /microampere/sim environment.current.direction_deg ${CURRENT#* } >/dev/null; sleep 20; fi
if [ -n "$WIND" ]; then ros2 param set /microampere/sim environment.wind.speed ${WIND% *} >/dev/null; ros2 param set /microampere/sim environment.wind.direction_deg ${WIND#* } >/dev/null; sleep 20; fi
[ -n "$BATTERY" ] && ros2 param set /microampere/sim battery.voltage $BATTERY >/dev/null
[ -n "$PF" ] && ros2 param set /microampere/position_controller guidance.path_following $PF >/dev/null
python3 "$(dirname "$(readlink -f "$0")")"/goal_motion_test.py --label $1 ${GOAL:+--goal $GOAL}
pkill -INT -f "[v]essel.launch.py vessel:=microampere dp:=true use_gui:=false web:=false"; sleep 3
for p in $(pgrep -f "[i]nstall/cybership_|[r]obot_state_publisher --ros-args|[s]tatic_transform_publisher --frame-id world|[t]opic_tools/mux"); do
  grep -q "ROS_DOMAIN_ID=81" /proc/$p/environ 2>/dev/null && kill -9 $p; done
