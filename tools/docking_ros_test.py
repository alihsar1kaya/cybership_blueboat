#!/usr/bin/env python3
"""
Docking through ROS: the whole stack (simulator, observer, position controller with docking, allocation) launched
headless once per case, each in its own ROS domain, docked with guidance/dock and measured on the true state
(sim/odom_true) against the harbour file's quay. The same start poses and success test as tools/docking_bench.py,
so the two can be compared.

    python3 tools/docking_ros_test.py                       # 6 starts x calm, current onto the quay, wind onto it
    python3 tools/docking_ros_test.py --envs calm --starts 2 --label try

Writes ~/ros_ws/results/docking_ros_<label>.json and prints one line per run and a summary.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, f"{REPO}/cybership_dp")
from cybership_dp.docking import hull_corners, load_harbour, wrap  # noqa: E402

HARBOUR = f"{REPO}/cybership_config/config/microampere/harbour.yaml"
HULL = (1.195, 0.93)
ENVS = {
    "calm": [],
    "current_on": [("environment.current.speed", 0.10), ("environment.current.direction_deg", 0.0)],
    "current_along": [("environment.current.speed", 0.10), ("environment.current.direction_deg", 90.0)],
    "wind_on": [("environment.wind.speed", 6.0), ("environment.wind.direction_deg", 0.0)],
    "current_off": [("environment.current.speed", 0.10), ("environment.current.direction_deg", 180.0)],
}

PROBE = r'''
import json, math, sys, time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import WrenchStamped
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
args = json.loads(sys.argv[1])
rclpy.init()
n = Node("docking_probe")
state = {"odom": None, "status": [], "contact": 0.0}
n.create_subscription(Odometry, "/microampere/sim/odom_true", lambda m: state.__setitem__("odom", m), 10)
n.create_subscription(String, "/microampere/guidance/dock_status", lambda m: state["status"].append((time.time(), m.data)), 10)
n.create_subscription(WrenchStamped, "/microampere/sim/contact", lambda m: state.__setitem__("contact", math.hypot(m.wrench.force.x, m.wrench.force.y)), 10)
pub = n.create_publisher(String, "/microampere/guidance/dock", 10)
def spin(t):
    end = time.time() + t
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.02)
# wait for the stack: odometry and a subscriber on guidance/dock (the controller)
t0 = time.time()
while (state["odom"] is None or pub.get_subscription_count() == 0) and time.time() - t0 < 60:
    spin(0.2)
if args["env"]:
    cli = n.create_client(SetParameters, "/microampere/sim/set_parameters")
    cli.wait_for_service(timeout_sec=10)
    req = SetParameters.Request()
    for name, v in args["env"]:
        req.parameters.append(Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(v))))
    f = cli.call_async(req)
    rclpy.spin_until_future_complete(n, f, timeout_sec=5)
spin(args["settle"])            # the observer converges, the current/wind settle
t_start = time.time()
pub.publish(String(data=args["berth"]))
log = []
while time.time() - t_start < args["timeout"]:
    spin(0.1)
    o = state["odom"]
    if o is None:
        continue
    q = o.pose.pose.orientation
    psi = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    log.append([round(time.time() - t_start, 2), o.pose.pose.position.x, o.pose.pose.position.y, psi,
                o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z, state["contact"]])
    if any("alongside" in s for _, s in state["status"]) and time.time() - max(t for t, s in state["status"] if "alongside" in s) > 25:
        break
print(json.dumps({"log": log, "status": [(round(t - t_start, 2), s) for t, s in state["status"]]}))
'''


def run_case(case):
    """Launch the stack in domain `case['domain']`, dock, return the measurements."""
    env = dict(os.environ, ROS_DOMAIN_ID=str(case["domain"]), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
               GZ_PARTITION=f"docking_test_{case['domain']}")       # Gazebo instances apart too
    x0, y0, yaw0 = case["start"]
    launch = subprocess.Popen(
        ["ros2", "launch", "cybership_simulator", "vessel.launch.py", "vessel:=microampere", "dp:=true",
         "observer:=true", "gnss:=rtk", "use_gui:=false", "web:=false", f"x0:={x0}", f"y0:={y0}",
         f"yaw0_deg:={math.degrees(yaw0)}"] + case.get("launch_args", []),
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        args = dict(env=ENVS[case["env"]], berth=case["berth"], timeout=case["timeout"], settle=5.0)
        out = subprocess.run([sys.executable, "-c", PROBE, json.dumps(args)], env=env, capture_output=True,
                             text=True, timeout=case["timeout"] + 120)
        data = json.loads(out.stdout.strip().splitlines()[-1]) if out.stdout.strip() else {"log": [], "status": []}
    except Exception as e:
        data = {"log": [], "status": [], "error": str(e)}
    finally:
        os.killpg(launch.pid, signal.SIGINT)
        try:
            launch.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(launch.pid, signal.SIGKILL)
    return dict(case, **measure(case, data))


def measure(case, data):
    quays, berths = load_harbour(HARBOUR, HULL)
    b = berths[case["berth"] or next(iter(berths))]
    quay, face, berth = quays[b["quay"]], b["face"], b["pose"]
    n = quay.normals[face]
    t_along = next((t for t, st in data["status"] if "alongside" in st), None)
    min_clear, towards_near, t_ok, since, t_min = math.inf, 0.0, None, None, None
    pen_along, f_along, f_hist = 0.0, 0.0, []
    for row in data["log"]:
        t, x, y, psi, u, v, r = row[:7]
        fc = row[7] if len(row) > 7 else 0.0
        eta = np.array([x, y, psi])
        clear = quay.clearance(eta, *HULL)
        if t_along is None or t < t_along:                   # the approach: a touch here is a fault
            if clear < min_clear:
                min_clear, t_min = clear, t
        else:                                                 # alongside: resting on the fenders is the point
            pen_along, f_along = max(pen_along, -clear), max(f_along, fc)
            f_hist.append(fc)
        gap = quay.face_gap(eta, *HULL, face)
        c, s = math.cos(psi), math.sin(psi)
        v_ned = np.array([c * u - s * v, s * u + c * v])
        if gap < 0.3:
            for corner in hull_corners(eta, *HULL):
                rel = corner - eta[:2]
                towards_near = max(towards_near, float(-n @ (v_ned + r * np.array([-rel[1], rel[0]]))))
        err = eta - berth
        # alongside: the hull 0-12 cm off the face (at most 2 cm into the fender), heading within 5 deg, within
        # 0.3 m along the face, still
        along = abs(float(np.array([-n[1], n[0]]) @ err[:2]))
        ok = (-0.02 <= gap <= 0.12 and abs(wrap(err[2])) < math.radians(5) and along < 0.3
              and math.hypot(u, v) < 0.05)
        since = (since if since is not None else t) if ok else None
        if since is not None and t - since >= 5.0 and t_ok is None:
            t_ok = since
    last = data["log"][-1] if data["log"] else None
    final = (math.hypot(last[1] - berth[0], last[2] - berth[1]) if last else None)
    if min_clear == math.inf:
        min_clear = None
    return dict(docked=t_ok is not None and (min_clear is None or min_clear > 0), t_dock=t_ok, min_clear=min_clear,
                penetration_alongside=pen_along, fender_force_max=f_along,
                fender_force_mean_last10=float(np.mean(f_hist[-100:])) if f_hist else 0.0,
                t_min_clear=t_min, log=[[round(v, 3) for v in row] for row in data["log"][::5]],
                towards_near=towards_near, final_err=final, status=data["status"], error=data.get("error"),
                samples=len(data["log"]))


def starts(n, seed=7):
    """As docking_bench.starts_for: 4-7 m from the berth on the open-water side, any heading."""
    _, berths = load_harbour(HARBOUR, HULL)
    berth = next(iter(berths.values()))["pose"]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        b, d = math.radians(rng.uniform(120, 240)), rng.uniform(4.0, 7.0)
        out.append([round(berth[0] + d * math.cos(b), 2), round(berth[1] + d * math.sin(b), 2),
                    round(rng.uniform(-math.pi, math.pi), 3)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="calm,current_on,wind_on")
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--berth", default="")
    ap.add_argument("--timeout", type=float, default=120.0)
    # 8 at once on the 16-thread laptop starved the boat_interface runs (12/18 against 17/18 at 4-6 at once)
    ap.add_argument("--parallel", type=int, default=4, help="runs at once (domains 91..)")
    ap.add_argument("--label", default="v1")
    ap.add_argument("--only", default="", help="run only starts whose index is in this comma list (0-based)")
    ap.add_argument("--arg", action="append", default=[],
                    help="extra launch argument, repeatable: --arg gazebo:=true --arg gz_gui:=false "
                         "--arg 'map_shift:=0.1 0.0' --arg multipath:=true")
    a = ap.parse_args()
    keep = {int(i) for i in a.only.split(",")} if a.only else None
    cases = [dict(env=e, start=s, berth=a.berth, timeout=a.timeout, launch_args=a.arg)
             for e in a.envs.split(",") for i, s in enumerate(starts(a.starts)) if keep is None or i in keep]
    for i, c in enumerate(cases):
        c["domain"] = 91 + i % a.parallel
    print(f"{len(cases)} dockings through ROS, {a.parallel} at a time ...", flush=True)
    t0, results = time.time(), []
    # one case per domain at a time: run them in waves of `parallel`
    for k in range(0, len(cases), a.parallel):
        with ThreadPoolExecutor(a.parallel) as ex:
            for r in ex.map(run_case, cases[k:k + a.parallel]):
                results.append(r)
                td = f"{r['t_dock']:.1f} s" if r["t_dock"] is not None else "-"
                mc = f"{r['min_clear']:.3f}" if r["min_clear"] is not None else "-"
                print(f"  {len(results):3d}/{len(cases)} {r['env']:14s} start {r['start']}  "
                      f"{'DOCKED' if r['docked'] else 'not docked':10s} t {td:8s} approach clearance {mc} m"
                      f"{' TOUCH' if r['min_clear'] is not None and r['min_clear'] <= 0 else ''}  alongside: "
                      f"{1000 * r['penetration_alongside']:.1f} mm in, fender {r['fender_force_mean_last10']:.1f} N "
                      f"(max {r['fender_force_max']:.0f})  ({time.time() - t0:.0f} s)", flush=True)
    out = os.path.expanduser(f"~/ros_ws/results/docking_ros_{a.label}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    docked = [r for r in results if r["docked"]]
    print(f"docked {len(docked)}/{len(results)}, touches {sum(1 for r in results if r['min_clear'] is not None and r['min_clear'] <= 0)}, "
          f"median time {np.median([r['t_dock'] for r in docked]) if docked else float('nan'):.1f} s -> {out}")


if __name__ == "__main__":
    main()
