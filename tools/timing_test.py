#!/usr/bin/env python3
"""
Step 6: does the DP keep up on a slower computer? The whole stack on the boat's topics (boat_interface, LEKF input,
SentiBoard stamps) docks as in tools/docking_ros_test.py, one run at a time, while the DP nodes (observer, position
controller, allocation) report their compute times on /diagnostics (cybership_utilities/loop_timing.py).

The boat's LattePanda Delta 3 (Celeron N5105, 4 cores) is roughly half as fast per core as this laptop and also runs
the LEKF, the sensor and IO nodes. To come closer, the DP nodes can be put on fewer cores (--cores, taskset through
the launch argument dp_prefix) with a busy loop on each of them (--burn), which halves what each core gives the DP.
The simulator and Gazebo keep the rest of the machine, so the "world" stays real time.

    python3 tools/timing_test.py --label laptop                       # no limit
    python3 tools/timing_test.py --cores 2 --burn --label n5105_2      # 2 cores at half speed

Prints per loop: compute time (mean, p95, max), the longest time between two calls, the CPU share, and the docking
solves; then each docking's result. Writes ~/ros_ws/results/timing_<label>.json.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import docking_ros_test as drt  # noqa: E402

DIAG = r'''
import json, sys, time, rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticArray
rclpy.init()
n = Node("timing_probe")
rows, end = [], time.time() + float(sys.argv[1])
def on(m):
    for s in m.status:
        if "microampere" in s.name:
            level = s.level[0] if isinstance(s.level, (bytes, bytearray)) else int(s.level)
            rows.append([time.time(), s.name, level, {kv.key: kv.value for kv in s.values}])
n.create_subscription(DiagnosticArray, "/diagnostics", on, 50)
try:
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
except KeyboardInterrupt:          # the runner stops it when the docking is over
    pass
print(json.dumps(rows), flush=True)
'''


def run(case, cores, burn, domain, extra):
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
               GZ_PARTITION=f"timing_test_{domain}")
    x0, y0, yaw0 = case["start"]
    args = ["ros2", "launch", "cybership_simulator", "vessel.launch.py", "vessel:=microampere", "dp:=true",
            "observer:=true", "gnss:=rtk", "use_gui:=false", "web:=false", f"x0:={x0}", f"y0:={y0}",
            f"yaw0_deg:={yaw0 * 180 / 3.141592653589793}", "boat_interface:=true", "boat_clock:=sentiboard"] + extra
    cpus = f"0-{cores - 1}" if cores else ""
    if cpus:
        args.append(f"dp_prefix:=taskset -c {cpus}")
    burners = [subprocess.Popen(["taskset", "-c", str(c), sys.executable, "-c", "while True: pass"],
                                start_new_session=True) for c in (range(cores) if burn and cores else [])]
    log = tempfile.NamedTemporaryFile(prefix="timing_launch_", suffix=".log", delete=False)
    launch = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        diag = subprocess.Popen([sys.executable, "-c", DIAG, str(case["timeout"] + 100)], env=env,
                                stdout=subprocess.PIPE, text=True, start_new_session=True)
        probe = dict(env=drt.ENVS[case["env"]], berth=case["berth"], timeout=case["timeout"], settle=5.0)
        out = subprocess.run([sys.executable, "-c", drt.PROBE, json.dumps(probe)], env=env, capture_output=True,
                             text=True, timeout=case["timeout"] + 120)
        data = json.loads(out.stdout.strip().splitlines()[-1]) if out.stdout.strip() else {"log": [], "status": []}
        os.killpg(diag.pid, signal.SIGINT)
        rows = json.loads(diag.communicate(timeout=20)[0] or "[]")
    finally:
        for p in [launch] + burners:
            try:
                os.killpg(p.pid, signal.SIGINT if p is launch else signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            launch.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(launch.pid, signal.SIGKILL)
    result = drt.measure(case, data)
    result.pop("log", None)
    # each docking plan from the controller's log: (plan duration, solve time, guess); failed replans as None
    text = open(log.name, errors="replace").read()
    log.close()
    os.unlink(log.name)
    solves = [(float(T), float(ts), g) for T, ts, g in
              re.findall(r"Docking: plan ([\d.]+) s, solved in ([\d.]+) s \(([^)]*)\)", text)]
    solves += [(None, None, "failed")] * len(re.findall(r"Docking: (?:re)?plan failed", text))
    return dict(case, **result, diagnostics=rows, solves=solves)


def summary(results):
    """Per loop over all runs: compute mean / p95 / max, longest period, CPU share; levels seen."""
    loops = {}
    for r in results:
        for _, name, level, v in r["diagnostics"]:
            loops.setdefault(name.split(": ", 1)[-1], []).append((level, v))
    print(f"{'loop':22s} {'calls':>6s} {'mean ms':>8s} {'p95 ms':>7s} {'max ms':>8s} {'period max ms':>14s} "
          f"{'nominal':>8s} {'CPU':>6s}  warn/err")
    out = {}
    for name, rows in sorted(loops.items()):
        f = lambda k: [float(v[k]) for _, v in rows if k in v and v[k] != "nan"]
        calls = sum(int(v["calls"]) for _, v in rows)
        mean = sum(float(v["compute_mean_ms"]) * int(v["calls"]) for _, v in rows) / max(calls, 1)
        per = f("period_max_ms")
        s = dict(calls=calls, mean=mean, p95=max(f("compute_p95_ms")), max=max(f("compute_max_ms")),
                 period_max=max(per) if per else float("nan"), nominal=(f("period_nominal_ms") or [float("nan")])[0],
                 load=sum(f("cpu_load")) / len(rows), warn=sum(1 for lv, _ in rows if lv == 1),
                 err=sum(1 for lv, _ in rows if lv == 2))
        out[name] = s
        print(f"{name:22s} {calls:6d} {s['mean']:8.1f} {s['p95']:7.1f} {s['max']:8.1f} {s['period_max']:14.1f} "
              f"{s['nominal']:8.1f} {100 * s['load']:5.1f}%  {s['warn']}/{s['err']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=0, help="DP nodes on cores 0..N-1 (0: no limit)")
    ap.add_argument("--burn", action="store_true", help="a busy loop on each of those cores (half speed)")
    ap.add_argument("--envs", default="current_on")
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--domain", type=int, default=93)
    ap.add_argument("--label", default="laptop")
    ap.add_argument("--arg", action="append", default=[], help="extra launch argument, repeatable")
    a = ap.parse_args()
    cases = [dict(env=e, start=s, berth="", timeout=a.timeout) for e in a.envs.split(",")
             for s in drt.starts(a.starts)]
    print(f"{len(cases)} dockings one at a time, DP on {a.cores or 'all'} cores{' at half speed' if a.burn else ''}",
          flush=True)
    results = []
    for i, c in enumerate(cases):
        t = time.time()
        r = run(c, a.cores, a.burn, a.domain, a.arg)
        results.append(r)
        clear = r["min_clear"]
        sv = [f"{ts:.1f}{'' if g_ == 'warm' else ' ' + g_}" if ts is not None else "failed" for _, ts, g_ in r["solves"]]
        print(f"  {i + 1}/{len(cases)} {c['env']:12s} {'DOCKED' if r['docked'] else 'not docked':10s} "
              f"t {r['t_dock'] if r['t_dock'] is None else round(r['t_dock'], 1)} s, approach clearance "
              f"{'-' if clear is None else f'{clear:.3f}'} m ({time.time() - t:.0f} s); plans [s]: {', '.join(sv)}",
              flush=True)
    loops = summary(results)
    docked = sum(r["docked"] for r in results)
    print(f"docked {docked}/{len(results)}")
    path = os.path.expanduser(f"~/ros_ws/results/timing_{a.label}.json")
    with open(path, "w") as f:
        json.dump(dict(cores=a.cores, burn=a.burn, loops=loops,
                       runs=[{k: v for k, v in r.items() if k != "diagnostics"} for r in results]), f, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
