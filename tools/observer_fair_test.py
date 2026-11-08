#!/usr/bin/env python3
"""
Fair comparison: our DPKalmanFilter vs. the repo's LuenbergerObserver (cybership_observer).

Both observers get the same recorded sensor data (GNSS 5 Hz sigma 0.17 m, compass 50 Hz sigma 0.5 deg) and
force input, the same (possibly wrong) model matrices, and are each tuned with the same cost and
optimizer on separate tuning seeds. Errors are measured against the true LF state (what DP needs).

    python3 tools/observer_fair_test.py            # tune both, evaluate on 10 seeds, save results
"""
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

REPO = os.path.expanduser("~/ros_ws/src/cybership_software_suite")
sys.path.insert(0, f"{REPO}/cybership_observer/src")
sys.path.insert(0, f"{REPO}/cybership_utilities/test")
from cybership_observer.observer import LuenbergerObserver  # noqa: E402
import test_observer as T  # noqa: E402
from cybership_utilities import hydro, vessel_model  # noqa: E402
from cybership_utilities.marine_craft.sensors import Compass, Gnss  # noqa: E402
from cybership_utilities.marine_craft.waves import make_waves  # noqa: E402
from cybership_utilities.observer import DPKalmanFilter, ssa  # noqa: E402

H = 0.02                   # observer rate 50 Hz (as the Luenberger node default)
SECONDS, T_SKIP = 300.0, 60.0
IDX = [0, 1, 5]
SCENARIOS = ["manoeuvre", "current", "current+waves", "manoeuvre+waves"]
SIG_GNSS, SIG_PSI = 0.17, math.radians(0.5)
OUT = os.path.expanduser("~/ros_ws/results/observer_fair")
P = T.microampere()
M3, D3 = vessel_model.M_eff_3dof(P), vessel_model.D_3dof(P)
_CRAFT = vessel_model.make_marine_craft(P)


def true_damping(nu):
    return -_CRAFT.damping(np.array([nu[0], nu[1], 0, 0, 0, nu[2]]))[IDX]


def hydro_data():
    cfg = vessel_model.load_vessel_config(path=f"{REPO}/cybership_config/config/microampere/vessel.yaml")[0]
    d = hydro.load_hydro(os.path.join(os.path.dirname(cfg["_path"]), cfg["model"]["hydro"]["file"]))
    Bv = P["D"] - P["B_eq"]
    d["motion_rao"] = hydro.motion_raos(d["freqs"], d["A"], d["B"], d["force_rao"], d["rao_freqs"],
                                        P["MRB"], Bv, P["G"])
    return d


def make_data(args):
    """Simulate one scenario/seed (open loop) and record truth (LF), sensors and tau."""
    scen, seed = args
    path = f"{OUT}/data_{scen}_{seed}.npz"
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(seed)
    g, c = Gnss(rate=5.0, sigma=SIG_GNSS, rng=rng), Compass(rate=50.0, sigma_deg=0.5, rng=rng)
    craft = vessel_model.make_marine_craft(P)
    cur = 0.15 if "current" in scen else 0.0
    cdir = rng.uniform(0, 2 * math.pi)
    craft.current_ned = np.array([cur * math.cos(cdir), cur * math.sin(cdir), 0.0])
    waves = None
    if "waves" in scen:
        waves = make_waves({"Hs": 0.1, "Tp": 2.3, "direction_deg": rng.uniform(0, 360)}, hydro_data(),
                           seed=seed)
    ph = rng.uniform(0, 2 * math.pi, 3)
    if scen.startswith("manoeuvre"):
        amp, w = np.array([8.0, 6.0, 1.5]), np.array([0.05, 0.07, 0.10])
    else:
        amp, w = np.array([1.0, 2.0, 0.2]), np.array([0.05, 0.07, 0.10])
    log, gn, cm = [], [], []
    for k in range(int(SECONDS / H)):
        t = k * H
        tau = amp * np.sin(w * t + ph)
        if cur:   # a rough feedforward against the current, rotated to body
            psi = craft.eta[5]
            f = -60.0 * cur * np.array([math.cos(cdir - psi), math.sin(cdir - psi)])
            tau[0:2] += f
        if waves is not None:
            craft.eta_wf, craft.nu_wf = waves.motion(t, craft.eta[5], U=craft.nu[0])
        eta_tot, _ = craft.get_states()
        gm, cmm = g.update(t, eta_tot), c.update(t, eta_tot)
        gn.append(gm["ned"][0:2] if gm is not None else [np.nan, np.nan])
        cm.append(cmm["psi"] if cmm is not None else np.nan)
        log.append(np.r_[t, craft.eta[IDX], craft.nu[IDX], tau])
        for _ in range(2):
            craft.step(np.array([tau[0], tau[1], 0, 0, 0, tau[2]]), H / 2)
    np.savez(path, log=np.array(log), gnss=np.array(gn), compass=np.array(cm))
    return path


def model_error(seed, on):
    """Same random +-20 % error on diag(M) and diag(D) for both observers."""
    if not on:
        return M3.copy(), D3.copy(), np.ones(3)
    rng = np.random.default_rng(1000 + seed)
    eM, eD = rng.uniform(-0.2, 0.2, 3), rng.uniform(-0.2, 0.2, 3)
    S = np.diag(np.sqrt(1 + eM))
    return S @ M3 @ S, np.diag(1 + eD) @ D3, 1 + eD


# --- observers: theta is the log10 tuning vector ---------------------------------------------------
KF0 = np.log10([0.2, 0.02, 0.05, 0.005, 0.1, 0.001])          # q_bias xy/psi, q_nu xy/psi, q_wave xy/psi


def run_kf(data, M, D, dscale, theta, linear=False):
    q = 10 ** np.asarray(theta)
    kf = DPKalmanFilter(M, D, q_bias=[q[0], q[0], q[1]], q_nu=[q[2], q[2], q[3]],
                        damping=None if linear else (lambda nu: dscale * true_damping(nu)),
                        wave_filter={"omega0": 2 * math.pi / 2.3, "zeta": 0.1, "q": [q[4], q[4], q[5]]})
    log, gn, cm = data["log"], data["gnss"], data["compass"]
    out = np.zeros((len(log), 6))
    started = False
    for k in range(len(log)):
        if not np.isnan(gn[k, 0]):
            if not started:     # initialize on the first fix, like the node
                kf.x[kf.i_eta] = [gn[k, 0], gn[k, 1], cm[k]]
                started = True
            kf.correct_gnss(gn[k, 0], gn[k, 1], SIG_GNSS)
        if started and not np.isnan(cm[k]):
            kf.correct_compass(cm[k], SIG_PSI)
        out[k] = np.r_[kf.eta, kf.nu]
        kf.predict(log[k, 7:10], H)
    return out


LO0 = np.log10([1.0, 2.0, 0.2, 0.4, 5.0, 5.0, 142.0])          # l1 xy/psi, l2 xy/psi, l3 xy/psi, Tb


def run_lo(data, M, D, dscale, theta, hold=True):
    """hold=True: last GNSS fix is reused until max_age 0.2 s (as observer_ros.py); False: fresh fixes only."""
    l = 10 ** np.asarray(theta)
    ob = LuenbergerObserver(M=M, D=D, Tb=l[6] * np.eye(3))
    L1, L2, L3 = np.diag([l[0], l[0], l[1]]), M @ np.diag([l[2], l[2], l[3]]), np.diag([l[4], l[4], l[5]])
    log, gn, cm = data["log"], data["gnss"], data["compass"]
    out = np.zeros((len(log), 6))
    y, ty, psi = None, -9.0, 0.0
    for k in range(len(log)):
        t = log[k, 0]
        if not np.isnan(cm[k]):
            psi = cm[k]
        if not np.isnan(gn[k, 0]):
            if y is None:
                ob.eta_hat = np.array([[gn[k, 0]], [gn[k, 1]], [psi]])
            y, ty = np.array([gn[k, 0], gn[k, 1], psi]), t
        if y is None:
            continue
        if hold:
            ym, dr = np.array([y[0], y[1], psi]), (t - ty) > 0.2
            e, n, _ = ob.step(ym, log[k, 7:10], L1, L2, L3, H, dead_reckoning=dr)
        else:
            fresh = ty == t
            ym = np.array([y[0], y[1], psi]) if fresh else np.r_[ob.eta_hat[0:2, 0], psi]
            L1k = L1 if fresh else np.diag([0, 0, l[1]])
            e, n, _ = ob.step(ym, log[k, 7:10], L1k, L2, L3, H)
        out[k] = np.r_[e.ravel(), n.ravel()]
        if not np.all(np.isfinite(out[k])) or np.abs(out[k]).max() > 1e4:
            out[:] = np.nan
            break
    return out


def metrics(data, est):
    log = data["log"]
    m = log[:, 0] > T_SKIP
    tr, es = log[m, 1:7], est[m]
    if not np.all(np.isfinite(es)):
        return np.array([np.inf] * 3)
    pos = np.sqrt(np.mean(np.sum((es[:, 0:2] - tr[:, 0:2]) ** 2, 1)))
    vel = np.sqrt(np.mean(np.sum((es[:, 3:5] - tr[:, 3:5]) ** 2, 1)))
    hd = math.degrees(np.sqrt(np.mean(ssa(es[:, 2] - tr[:, 2]) ** 2)))
    return np.array([pos, vel, hd])


def one(args):
    name, theta, scen, seed, err = args
    data = np.load(f"{OUT}/data_{scen}_{seed}.npz")
    M, D, ds = model_error(seed, err)
    fn = {"kf": run_kf, "kf_linear": lambda *a: run_kf(*a, linear=True),
          "lo_hold": run_lo, "lo_fresh": lambda *a: run_lo(*a, hold=False)}[name]
    return metrics(data, fn(data, M, D, ds, theta))


REF = np.array([0.10, 0.03, 0.5])      # normalize: 10 cm, 3 cm/s, 0.5 deg


def cost_fn(pool, name, seeds):
    def cost(theta):
        r = pool.map(one, [(name, theta, s, sd, True) for s in SCENARIOS for sd in seeds])
        c = float(np.mean([np.sum(m / REF) for m in r]))
        return c if np.isfinite(c) else 1e6
    return cost


def tune(pool, name, x0, seeds, n_random=0, maxfev=80):
    cost = cost_fn(pool, name, seeds)
    rng = np.random.default_rng(7)
    starts = [np.asarray(x0)] + [np.asarray(x0) + rng.uniform(-1.5, 1.5, len(x0)) for _ in range(n_random)]
    scored = sorted(((cost(s), tuple(s)) for s in starts))
    best = minimize(cost, np.array(scored[0][1]), method="Nelder-Mead",
                    options={"maxfev": maxfev, "xatol": 0.02, "fatol": 0.01})
    print(f"  {name}: start {cost(np.asarray(x0)):.2f} -> tuned {best.fun:.2f}  theta {np.round(10**best.x, 4)}",
          flush=True)
    return best.x, best.fun


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    TUNE_SEEDS, EVAL_SEEDS = [100, 101, 102, 103], list(range(10))
    with Pool(16) as pool:
        pool.map(make_data, [(s, sd) for s in SCENARIOS for sd in TUNE_SEEDS + EVAL_SEEDS])
        print("data ready; tuning on seeds", TUNE_SEEDS, "(all scenarios, with model error)", flush=True)
        tuned = {}
        tuned["kf"] = tune(pool, "kf", KF0, TUNE_SEEDS, n_random=4, maxfev=60)
        tuned["kf_linear"] = tune(pool, "kf_linear", tuned["kf"][0], TUNE_SEEDS, n_random=0, maxfev=40)
        tuned["lo_hold"] = tune(pool, "lo_hold", LO0, TUNE_SEEDS, n_random=20, maxfev=120)
        tuned["lo_fresh"] = tune(pool, "lo_fresh", tuned["lo_hold"][0], TUNE_SEEDS, n_random=10, maxfev=120)
        res = {}
        for name, (theta, _) in tuned.items():
            for err in (False, True):
                r = pool.map(one, [(name, theta, s, sd, err) for s in SCENARIOS for sd in EVAL_SEEDS])
                res[(name, err)] = np.array(r).reshape(len(SCENARIOS), len(EVAL_SEEDS), 3)
    np.savez(f"{OUT}/results.npz", **{f"{k[0]}_{'err' if k[1] else 'exact'}": v for k, v in res.items()},
             **{f"theta_{k}": v[0] for k, v in tuned.items()})
    for err in (False, True):
        print(f"\n=== eval seeds 0-9, model {'with +-20 % error in M, D' if err else 'exact'}: "
              "RMS pos [cm] / vel [cm/s] / heading [deg], mean (worst seed) ===")
        print(f"{'scenario':17s}" + "".join(f"| {n:^27s}" for n in tuned))
        for i, s in enumerate(SCENARIOS):
            row = f"{s:17s}"
            for n in tuned:
                a = res[(n, err)][i]
                mu, wo = a.mean(0), a.max(0)
                row += f"| {mu[0]*100:4.1f}({wo[0]*100:4.1f}) {mu[1]*100:3.1f}({wo[1]*100:3.1f}) {mu[2]:.2f} "
            print(row)
