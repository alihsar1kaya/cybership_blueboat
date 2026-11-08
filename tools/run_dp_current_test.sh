#!/usr/bin/env bash
# usage: dp_current_test.sh <speed> <direction_deg> <label>
# Kjører DP i strøm (speed m/s mot retning grader) i 120 s og skriver avvik. Fra steg 3 (havstrøm).
source /opt/ros/jazzy/setup.bash; source ~/ros_ws/venv/bin/activate; source ~/ros_ws/install/setup.bash
export ROS_DOMAIN_ID=78
setsid ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true use_gui:=false "${@:4}" > /tmp/dpcur_$3.log 2>&1 &
sleep 12
ros2 param set /microampere/sim environment.current.direction_deg $2 >/dev/null
ros2 param set /microampere/sim environment.current.speed $1 >/dev/null
python3 - "$3" <<'PY'
import sys, time, math, rclpy, numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench
from cybership_interfaces.msg import ThrusterState
rclpy.init(); n = rclpy.create_node("probe"); s = {"odom": [], "slack": None, "az": None}
n.create_subscription(Odometry, "/microampere/sim/odom_true",
    lambda m: s["odom"].append((time.time(), m.pose.pose.position.x, m.pose.pose.position.y,
                                2*math.atan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w))), 10)
n.create_subscription(Wrench, "/microampere/control/allocation/slack", lambda m: s.__setitem__("slack", m), 10)
n.create_subscription(ThrusterState, "/microampere/thruster/bow_azimuth/state", lambda m: s.__setitem__("az", m), 10)
t0 = time.time()
while time.time() - t0 < 120: rclpy.spin_once(n, timeout_sec=0.1)
o = np.array([(t - t0, x, y, p) for t, x, y, p in s["odom"]])
late = o[o[:, 0] > 90]
err = np.hypot(late[:, 1], late[:, 2])
az = s["az"]; sl = s["slack"]
print(f"{sys.argv[1]}: maks avvik {np.hypot(o[:,1],o[:,2]).max():.2f} m, siste 30 s: snitt {err.mean():.2f} m, "
      f"kurs {math.degrees(late[-1,3]):.1f} deg, azimut {math.hypot(az.force.x, az.force.y):.1f} N @ "
      f"{math.degrees(az.angle):.0f} deg, slakk Y {sl.force.y:.1f} N")
PY
pkill -INT -f "[v]essel.launch.py vessel:=microampere dp:=true use_gui:=false"; sleep 3
for p in $(pgrep -f "[i]nstall/cybership_|[r]obot_state_publisher --ros-args|[s]tatic_transform_publisher --frame-id world|[t]opic_tools/mux"); do
  grep -q "ROS_DOMAIN_ID=78" /proc/$p/environ 2>/dev/null && kill -9 $p; done
