#!/usr/bin/env python3
"""
Model report for a vessel.yaml, without ROS: what the simulator model does open loop.

  1. Modal properties from M, D, G (port of Fossen's MSS modalFromMDG.m): time constants
     T1, T2, T6 and natural periods / damping ratios in heave, roll, pitch.
  2. Open-loop tests with the full nonlinear model (MarineCraft, RK4): top speed, surge rise
     time, stopping, max sway speed, max yaw rate.

    python3 tools/model_report.py --label baseline
    python3 tools/model_report.py --yaml path/to/vessel.yaml --label test

Writes results/model_report_<vessel>_<label>.json so runs can be compared.
"""

import argparse
import json
import os

import numpy as np

from cybership_utilities.vessel_model import load_vessel_config, make_marine_craft, model_parameters

WS = os.path.expanduser("~/ros_ws")
DEFAULT_YAML = os.path.join(WS, "src/cybership_software_suite/cybership_config/config/microampere/vessel.yaml")


def modal_from_mdg(M, D, G):
    """Python port of MSS modalFromMDG.m (floating vessel): T1, T2, T6 and (w, zeta) for 3, 4, 5."""
    n = 6
    A = np.block([[np.zeros((n, n)), np.eye(n)],
                  [-np.linalg.solve(M, G), -np.linalg.solve(M, D)]])
    ev, V = np.linalg.eig(A)

    def assign(idx, dofs):
        """Greedy assignment of modes to DOFs by largest participation, as in MSS."""
        mag = np.abs(V[:n][np.ix_(dofs, idx)]).astype(float)
        out = {}
        for _ in range(min(mag.shape)):
            i, j = np.unravel_index(np.argmax(mag), mag.shape)
            if not np.isfinite(mag[i, j]):
                break
            out[dofs[i]] = idx[j]
            mag[i, :] = -np.inf
            mag[:, j] = -np.inf
        return out

    real = [k for k in range(2 * n) if abs(ev[k].imag) < 1e-8 and ev[k].real < 0]
    cplx = [k for k in range(2 * n) if ev[k].imag > 0]
    T = {d: -1.0 / ev[k].real for d, k in assign(real, [0, 1, 5]).items()}
    osc = {}
    for d, k in assign(cplx, [2, 3, 4]).items():
        wn = abs(ev[k])
        osc[d] = (wn, -ev[k].real / wn)
    return T, osc


def simulate(craft, tau, t_end, dt=0.01, nu0=None):
    craft.eta = np.zeros(6)
    craft.nu = np.zeros(6) if nu0 is None else nu0
    t = np.arange(0.0, t_end, dt)
    nus = np.empty((len(t), 6))
    etas = np.empty((len(t), 6))
    for k in range(len(t)):
        nus[k], etas[k] = craft.nu, craft.eta
        craft.step(tau, dt)
    return t, nus, etas


def rise_time(t, y, frac=0.632):
    target = frac * y[-1]
    i = np.argmax(y >= target)
    return float(t[i]) if y[i] >= target else float("nan")


def thruster_limits(config):
    """Max forward thrust of the fixed thrusters, azimuth max force, and yaw moment from differential thrust."""
    fixed = [t for t in config["thrusters"] if t["type"] == "fixed"]
    azi = [t for t in config["thrusters"] if t["type"] == "azimuth"]
    f_fixed = sum(float(t.get("max_force", 0.0)) for t in fixed)
    f_azi = sum(float(t.get("max_force", 0.0)) for t in azi)
    # differential: forward on one side, reverse on the other
    n_diff = sum(float(t.get("max_force", 0.0)) * float(t.get("reverse_ratio", 1.0)) * abs(t["position"][1])
                 for t in fixed)
    return f_fixed, f_azi, n_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    config, _ = load_vessel_config(path=args.yaml)
    params = model_parameters(config)
    craft = make_marine_craft(params)
    M = craft.M
    D = craft.D_L
    G = craft.G
    T, osc = modal_from_mdg(M, D, G)

    f_fixed, f_azi, n_diff = thruster_limits(config)
    rep = {"vessel": config["name"], "label": args.label,
           "modal": {"T1": T.get(0), "T2": T.get(1), "T6": T.get(5),
                     "heave": {"period": 2 * np.pi / osc[2][0], "zeta": osc[2][1]},
                     "roll": {"period": 2 * np.pi / osc[3][0], "zeta": osc[3][1]},
                     "pitch": {"period": 2 * np.pi / osc[4][0], "zeta": osc[4][1]}}}

    tests = {}
    # Surge: stern thrusters only (transit), and all thrusters forward
    for name, F in (("stern", f_fixed), ("all", f_fixed + f_azi)):
        t, nu, _ = simulate(craft, [F, 0, 0, 0, 0, 0], 60.0)
        tests[f"top_speed_{name}"] = {"thrust_N": F, "u_ms": float(nu[-1, 0]),
                                      "rise_time_63_s": rise_time(t, nu[:, 0])}
    # Low-speed surge (DP regime): 10 N
    t, nu, _ = simulate(craft, [10.0, 0, 0, 0, 0, 0], 60.0)
    tests["surge_10N"] = {"u_ms": float(nu[-1, 0]), "rise_time_63_s": rise_time(t, nu[:, 0])}
    # Stopping from the stern-thruster top speed, no thrust
    u0 = tests["top_speed_stern"]["u_ms"]
    t, nu, eta = simulate(craft, np.zeros(6), 60.0, nu0=np.array([u0, 0, 0, 0, 0, 0]))
    i = np.argmax(nu[:, 0] < 0.1)
    tests["stop_from_top"] = {"u0_ms": u0, "time_to_0.1ms_s": float(t[i]), "distance_m": float(eta[i, 0])}
    # Sway: azimuth sideways, pure force (yaw moment removed to isolate sway)
    t, nu, _ = simulate(craft, [0, f_azi, 0, 0, 0, 0], 60.0)
    tests["sway_azimuth"] = {"force_N": f_azi, "v_ms": float(nu[-1, 1]), "rise_time_63_s": rise_time(t, nu[:, 1])}
    # Yaw: differential stern thrust
    t, nu, _ = simulate(craft, [0, 0, 0, 0, 0, n_diff], 30.0)
    tests["yaw_differential"] = {"moment_Nm": n_diff, "r_degs": float(np.degrees(nu[-1, 5])),
                                 "rise_time_63_s": rise_time(t, nu[:, 5])}
    rep["open_loop"] = tests

    def f(x):
        return f"{x:7.2f}" if x is not None else "   -   "

    m = rep["modal"]
    print(f"== {rep['vessel']} [{args.label}]")
    print(f"Time constants (modalFromMDG, linear D):  T1 {f(m['T1'])} s  T2 {f(m['T2'])} s  T6 {f(m['T6'])} s")
    for k in ("heave", "roll", "pitch"):
        print(f"{k:6s} period {f(m[k]['period'])} s   zeta {m[k]['zeta']:.2f}")
    print("Open loop (full nonlinear model):")
    for k, v in tests.items():
        print(f"  {k:18s} " + "  ".join(f"{a} {b:.2f}" for a, b in v.items()))

    out = os.path.join(WS, "results", f"model_report_{rep['vessel']}_{args.label}.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
