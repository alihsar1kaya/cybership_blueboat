#!/usr/bin/env bash
# Enkel docking-test (geometri inspirert av Fig. 7 i Gezer et al.) med dagens DP-regulator.
#   bash ~/ros_ws/tools/run_docking_sim.sh direkte     # ett mål rett i kaiplassen
#   bash ~/ros_ws/tools/run_docking_sim.sh veipunkter  # håndplukket rute rundt hindringene
#   GUI=true bash ... for RViz
set -e
SCENARIO=${1:-direkte}
GUI=${GUI:-false}
VESSEL=voyager
WS=~/ros_ws
OUT=$WS/results/docking_${SCENARIO}_${VESSEL}_sim.csv
LOGDIR=$WS/results/logs
mkdir -p "$LOGDIR"

# Start og kaiplass (x, y, yaw_deg)
START_X=3.0; START_Y=-2.3; START_YAW=90
BERTH="-3.5,1.5,180"
case $SCENARIO in
  direkte)    POINTS=("$BERTH") ;;
  veipunkter) POINTS=("3.0,1.2,180" "-0.3,1.2,180" "-0.6,-1.4,180" "-3.4,-1.4,180" "-3.5,0.3,180" "$BERTH") ;;
  *) echo "Ukjent scenario: $SCENARIO (bruk direkte|veipunkter)"; exit 1 ;;
esac

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
start() { local name=$1; shift; setsid "$@" > "$LOGDIR/$name.log" 2>&1 & PIDS+=($!); }

# Simulatoren startes direkte (ikke via launch) for å kunne gi startposisjon som parameter.
# simulator/reset-tjenesten kan ikke brukes: den krasjer (vessel.eta får form (6,1), base.py:176).
START_YAW_RAD=$(python3 -c "import math;print(math.radians($START_YAW))")
start sim ros2 run cybership_simulator voyager.py --ros-args -r __ns:=/$VESSEL -r __node:=sim \
  -p tf_prefix:=$VESSEL \
  -p initial_conditions.position.x:=$START_X \
  -p initial_conditions.position.y:=$START_Y \
  -p initial_conditions.position.yaw:=$START_YAW_RAD
if [ "$GUI" = "true" ]; then
  start desc ros2 launch cybership_description description.launch.py vessel_model:=$VESSEL vessel_name:=$VESSEL
  start viz  ros2 launch cybership_viz viz.launch.py vessel_model:=$VESSEL vessel_name:=$VESSEL
fi
sleep 5

# Midlertidige fikser, se run_four_corner_sim.sh
start tf    ros2 run tf2_ros static_transform_publisher --frame-id world --child-frame-id odom
start alloc ros2 launch cybership_dp thrust_allocation.launch.py vessel_name:=$VESSEL vessel_model:=$VESSEL
start pos   ros2 launch cybership_dp position_controller.launch.py vessel_name:=$VESSEL vessel_model:=$VESSEL
sleep 6
ros2 service call /$VESSEL/force_mux/select topic_tools_interfaces/srv/MuxSelect \
  '{topic: "control/force/command/position"}' > /dev/null
start rec python3 "$(dirname "$(readlink -f "$0")")"/record_dp.py $VESSEL "$OUT"
sleep 1

python3 "$(dirname "$(readlink -f "$0")")"/waypoints.py $VESSEL "${POINTS[@]}"
echo "Data lagret i $OUT"
