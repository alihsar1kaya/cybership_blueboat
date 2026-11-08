#!/usr/bin/env bash
# Starter sim + DP uten GUI, kjører DP-stegtesten og stopper alt igjen.
#   bash ~/ros_ws/tools/run_dp_step_test.sh microampere before [launch args, e.g. noisy_odom:=true]
VESSEL=${1:-microampere}
LABEL=${2:-run}
WS=~/ros_ws

source /opt/ros/jazzy/setup.bash
source $WS/venv/bin/activate
source $WS/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-49}

setsid ros2 launch cybership_simulator vessel.launch.py vessel:=$VESSEL dp:=true use_gui:=false "${@:3}" \
  > /tmp/dp_step_$VESSEL.log 2>&1 &
LAUNCH=$!
cleanup() {
  kill -INT -- -$LAUNCH 2>/dev/null; sleep 3; kill -KILL -- -$LAUNCH 2>/dev/null
}
trap cleanup EXIT
sleep 10

python3 $WS/tools/dp_step_test.py --vessel $VESSEL --label $LABEL
