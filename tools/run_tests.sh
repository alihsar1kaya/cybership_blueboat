#!/usr/bin/env bash
# Starter simulatoren, kjører kontrakttestene og stopper simulatoren igjen.
#   bash ~/ros_ws/tools/run_tests.sh            # Voyager
#   VESSEL=voyager bash ~/ros_ws/tools/run_tests.sh
VESSEL=${VESSEL:-voyager}
WS=~/ros_ws
REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)

source /opt/ros/jazzy/setup.bash
source $WS/venv/bin/activate
source $WS/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-48}

echo "Starter simulator for $VESSEL ..."
setsid ros2 run cybership_simulator vessel.py --ros-args -r __ns:=/$VESSEL -r __node:=sim -p tf_prefix:=$VESSEL -p vessel_model:=$VESSEL \
  > /tmp/sim_$VESSEL.log 2>&1 &
SIM=$!
trap 'kill -INT -- -$SIM 2>/dev/null; sleep 1; kill -KILL -- -$SIM 2>/dev/null' EXIT
sleep 6

cd $REPO
VESSEL=$VESSEL python -m pytest -v cybership_tests/test/test_thruster_contract.py
