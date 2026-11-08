#!/usr/bin/env bash
# DP stationkeeping in waves (Fossen Ch. 10.2): 150 s at the origin, then position error (truth) and
# thruster activity (mean |dF/dt|, wear). Fase 6.
#   bash tools/run_dp_wave_test.sh <Hs> <Tp> <direction_deg> <label> [launch args, e.g. observer:=true wave_filter_omega0:=2.7]
source /opt/ros/jazzy/setup.bash; source ~/ros_ws/venv/bin/activate; source ~/ros_ws/install/setup.bash
export ROS_DOMAIN_ID=79
setsid ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true use_gui:=false "${@:5}" > /tmp/dpwave_$4.log 2>&1 &
sleep 12
ros2 param set /microampere/sim environment.waves.Tp $2 >/dev/null
ros2 param set /microampere/sim environment.waves.direction_deg $3 >/dev/null
ros2 param set /microampere/sim environment.waves.seed 7 >/dev/null
ros2 param set /microampere/sim environment.waves.Hs $1 >/dev/null
python3 - "$4" <<'PY'
import sys, time, math, rclpy, numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64
rclpy.init(); n = rclpy.create_node("probe"); s = {"odom": [], "f": {}, "z": []}
n.create_subscription(Odometry, "/microampere/sim/odom_true",
    lambda m: s["odom"].append((time.time(), m.pose.pose.position.x, m.pose.pose.position.y,
                                2*math.atan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w))), 10)
n.create_subscription(Float64, "/microampere/sim/environment/wave_elevation", lambda m: s["z"].append(m.data), 10)
for name in ("port", "starboard", "bow_azimuth"):
    n.create_subscription(WrenchStamped, f"/microampere/thruster/{name}/force",
        lambda m, k=name: s["f"].setdefault(k, []).append((time.time(), math.hypot(m.wrench.force.x, m.wrench.force.y))), 10)
t0 = time.time()
while time.time() - t0 < 150: rclpy.spin_once(n, timeout_sec=0.05)
o = np.array([(t - t0, x, y, p) for t, x, y, p in s["odom"]]); late = o[o[:, 0] > 30]
err = np.hypot(late[:, 1], late[:, 2])
act = 0.0
for k, v in s["f"].items():
    a = np.array(v); a = a[a[:, 0] - t0 > 30]
    act += np.mean(np.abs(np.diff(a[:, 1]) / np.diff(a[:, 0])))
z = np.array(s["z"])
print(f"{sys.argv[1]}: Hs realised {4*z.std():.2f} m | pos error mean {err.mean():.2f} m, max {err.max():.2f} m | "
      f"heading std {math.degrees(np.std(late[:, 3])):.1f} deg | thruster activity {act:.0f} N/s")
PY
pkill -INT -f "[v]essel.launch.py vessel:=microampere dp:=true use_gui:=false"; sleep 3
for p in $(pgrep -f "[i]nstall/cybership_|[r]obot_state_publisher --ros-args|[s]tatic_transform_publisher --frame-id world|[t]opic_tools/mux"); do
  grep -q "ROS_DOMAIN_ID=79" /proc/$p/environ 2>/dev/null && kill -9 $p; done
