#!/usr/bin/env bash
# Kjører 4-hjørne DP-test på Voyager i simulator og lagrer resultatet i ~/ros_ws/results.
#   bash ~/ros_ws/tools/run_four_corner_sim.sh          # uten RViz
#   GUI=true bash ~/ros_ws/tools/run_four_corner_sim.sh # med RViz
set -e
GUI=${GUI:-false}
VESSEL=voyager
WS=~/ros_ws
OUT=$WS/results/four_corner_${VESSEL}_sim.csv
LOGDIR=$WS/results/logs
mkdir -p "$LOGDIR"

source /opt/ros/jazzy/setup.bash
source $WS/venv/bin/activate
source $WS/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}

PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -INT -- -"$p" 2>/dev/null || true; done
  sleep 2
  for p in "${PIDS[@]}"; do kill -KILL -- -"$p" 2>/dev/null || true; done
}
trap cleanup EXIT

start() {  # start i egen prosessgruppe så alt kan stoppes samlet
  local name=$1; shift
  setsid "$@" > "$LOGDIR/$name.log" 2>&1 &
  PIDS+=($!)
}

start sim  ros2 launch cybership_simulator voyager.launch.py vessel_name:=$VESSEL use_gui:=$GUI
# Midlertidig fiks: position_control_node transformerer mål til "odom" (commit ae90975),
# men ingen publiserer den rammen i simulering.
start tf   ros2 run tf2_ros static_transform_publisher --frame-id world --child-frame-id odom
sleep 5
# Startes hver for seg: dp.launch.py gir posisjonsregulatoren controller_velocity.yaml
start alloc ros2 launch cybership_dp thrust_allocation.launch.py vessel_name:=$VESSEL vessel_model:=$VESSEL
start pos   ros2 launch cybership_dp position_controller.launch.py vessel_name:=$VESSEL vessel_model:=$VESSEL
sleep 6
ros2 service call /$VESSEL/force_mux/select topic_tools_interfaces/srv/MuxSelect \
  '{topic: "control/force/command/position"}' > /dev/null
start rec  python3 "$(dirname "$(readlink -f "$0")")"/record_dp.py $VESSEL "$OUT"
sleep 1

python3 "$(dirname "$(readlink -f "$0")")"/four_corner_wait.py $VESSEL
echo "Data lagret i $OUT"
