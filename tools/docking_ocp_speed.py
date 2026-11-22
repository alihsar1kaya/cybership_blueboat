#!/usr/bin/env python3
"""
How fast the docking OCP (cybership_dp/docking_mpc.py) solves, built as the position controller builds it, on a fixed
set of problems: the starts of tools/docking_ros_test.py in calm water and in 0.1 m/s current onto the quay, a cold
solve from rest and then a replan 3 s into that plan from a state 0.15 m and 5 deg off it (--perturb; warm start).

    python3 tools/docking_ocp_speed.py                          # the current settings
    python3 tools/docking_ocp_speed.py --set N=30 --label n30   # a variant (DockingOCP keyword arguments)

Prints per problem the solve time, IPOPT iterations, the guess that worked, how much of the time went to CasADi's
function evaluations, and the plan (duration, closest approach of the hull to the quay); then the totals. Solves run
in this process one after the other (one thread, as in the controller).
"""
import argparse
import ast
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
for pkg in ("cybership_dp", "cybership_utilities"):
    sys.path.insert(0, f"{REPO}/{pkg}")
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from cybership_dp.docking import load_harbour  # noqa: E402
from cybership_dp.docking_mpc import DockingOCP  # noqa: E402
from cybership_utilities import vessel_model  # noqa: E402
import docking_ros_test as drt  # noqa: E402

ENVS = {"calm": np.zeros(2), "current_on": np.array([0.10, 0.0])}


def build(**kw):
    cfg, _ = vessel_model.load_vessel_config(path=f"{REPO}/cybership_config/config/microampere/vessel.yaml")
    params = vessel_model.model_parameters(cfg)
    hull = (float(cfg["hull"]["length"]), float(cfg["hull"]["beam"]))
    _, berths = load_harbour(drt.HARBOUR, hull)
    berth = next(iter(berths.values()))
    opts = dict(free_time=True, gap_min=berth["gap"] - 0.02, corridor=(0.15, 0.5, 1.5),
                heading_funnel=(math.radians(5.0), math.radians(30.0)), max_cpu_time=8.0)
    opts.update(kw)
    t = time.perf_counter()
    ocp = DockingOCP.from_vessel_config(cfg, params, **opts)
    return ocp, berth, hull, time.perf_counter() - t


def clearance(ocp_res, quay, hull):
    return min(quay.clearance(e, *hull) for e in ocp_res["eta"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="DockingOCP keyword, e.g. N=30 (repeatable)")
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--label", default="base")
    ap.add_argument("--perturb", type=float, nargs=2, default=[0.15, 5.0], help="replan from this far off [m, deg]")
    a = ap.parse_args()
    kw = {k: ast.literal_eval(v) for k, v in (s.split("=", 1) for s in a.set)}
    ocp, berth, hull, t_build = build(**kw)
    quays, _ = load_harbour(drt.HARBOUR, hull)
    quay = quays[berth["quay"]]
    print(f"built in {t_build:.1f} s ({kw or 'defaults'})")
    rows = []
    for env, vc in ENVS.items():
        for i, (x, y, psi) in enumerate(drt.starts(a.starts)):
            ocp.last = None
            x0 = np.array([x, y, psi, 0.0, 0.0, 0.0])
            cold = ocp.solve(x0, berth["pose"], berth["halfplane"], vc=vc)
            out = [("cold", cold)]
            if cold["ok"]:
                k = int(np.searchsorted(cold["t"], 3.0))
                xs = cold["X"][:, k].copy()
                xs[0] += a.perturb[0] * math.cos(xs[2] + 1.0)
                xs[1] += a.perturb[0] * math.sin(xs[2] + 1.0)
                xs[2] += math.radians(a.perturb[1])
                out.append(("warm", ocp.solve(xs, berth["pose"], berth["halfplane"], vc=vc, elapsed=float(cold["t"][k]))))
            for kind, r in out:
                c = clearance(r, quay, hull) if r["ok"] else float("nan")
                rows.append(dict(env=env, start=i, kind=kind, ok=r["ok"], time=r["solve_time"], it=r["iterations"],
                                 guess=r.get("guess", ""), eval=r["t_eval"], T=r["T"], clear=c))
                print(f"  {env:10s} start {i} {kind:4s} {'ok ' if r['ok'] else 'FAIL'} {r['solve_time']:6.2f} s "
                      f"{r['iterations']:4d} it  guess {r.get('guess', ''):8s} eval {100 * r['t_eval'] / max(r['solve_time'], 1e-9):3.0f} %"
                      f"  plan {r['T']:5.1f} s  closest {100 * c:5.1f} cm", flush=True)
    for kind in ("cold", "warm"):
        t = np.array([r["time"] for r in rows if r["kind"] == kind])
        ok = sum(r["ok"] for r in rows if r["kind"] == kind)
        it = np.array([r["it"] for r in rows if r["kind"] == kind])
        ev = sum(r["eval"] for r in rows if r["kind"] == kind) / max(t.sum(), 1e-9)
        print(f"{kind}: {ok}/{len(t)} ok, time mean {t.mean():.2f} s, median {np.median(t):.2f}, max {t.max():.2f}; "
              f"iterations mean {it.mean():.0f}; CasADi evaluations {100 * ev:.0f} % of the time")
    path = os.path.expanduser(f"~/ros_ws/results/docking_ocp_speed_{a.label}.json")
    with open(path, "w") as f:
        json.dump(dict(settings={k: str(v) for k, v in kw.items()}, build=t_build, rows=rows), f, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
