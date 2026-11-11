#!/usr/bin/env python3
"""
Analyse sweep_test.py runs: per goal the manoeuvre the guidance chose, time, path, overshoot, heading,
and what each thruster did. Writes a table (CSV) and figures to ~/ros_ws/results/<--out>/ (default sweep).

    python3 tools/sweep_analysis.py calm current wind waves [--logs <dir with sweep_<label>.log>]
"""
import argparse
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THR = ("port", "starboard", "bow_azimuth")
LIMIT = {"fwd": 67.0, "rev": 0.8 * 67.0}
RES = os.path.expanduser("~/ros_ws/results")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def guidance_events(logfile):
    """[(wall time, text)] of the controller's guidance lines (manoeuvre choice and phases)."""
    out = []
    if not logfile or not os.path.exists(logfile):
        return out
    pat = re.compile(r"\[(\d+\.\d+)\] \[[^\]]*position_controller\]: Guidance: (.*)")
    for line in open(logfile, errors="ignore"):
        m = pat.search(line)
        if m:
            out.append((float(m.group(1)), m.group(2)))
    return out


def analyse(label, logdir):
    d = np.load(f"{RES}/sweep_{label}.npz")
    odom, goals, offs = d["odom"], d["goals"], d["offsets"]
    events = guidance_events(os.path.join(logdir, f"sweep_{label}.log") if logdir else None)
    rows = []
    for i, (t0, t1, gx, gy, gpsi) in enumerate(goals):
        o = odom[(odom[:, 0] >= t0) & (odom[:, 0] <= t1)]
        if len(o) < 5:
            continue
        t = o[:, 0] - t0
        x, y, psi, u, v, r = o[:, 1:].T
        x0, y0, psi0 = x[0], y[0], psi[0]
        dist = math.hypot(gx - x0, gy - y0)
        err = np.hypot(gx - x, gy - y)
        herr = np.abs(wrap(gpsi - psi))
        inside = (err < 0.05) & (herr < math.radians(2))
        # settle time: first time it stays inside until the end of the window
        settle = next((t[k] for k in range(len(t)) if inside[k:].all()), np.nan)
        path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        if dist > 0.3:
            ph = math.atan2(gy - y0, gx - x0)
            along = (x - x0) * math.cos(ph) + (y - y0) * math.sin(ph)
            cross = -(x - x0) * math.sin(ph) + (y - y0) * math.cos(ph)
            overshoot, max_cross = float(max(along.max() - dist, 0.0)), float(np.abs(cross).max())
            bearing_body = math.degrees(wrap(ph - psi0))
        else:
            overshoot, max_cross, bearing_body = float(err.max()), float(err.max()), float("nan")
        heading_travel = float(np.degrees(np.sum(np.abs(np.diff(np.unwrap(psi))))))
        heading_needed = abs(math.degrees(wrap(gpsi - psi0)))
        choice = next((txt.split()[0] for te, txt in events
                       if t0 - 1 <= te <= t1 and re.match(r"(forward|reverse|dp|path|pivot) to the goal", txt)), "?")
        row = dict(scenario=label, goal=i + 1, dn=offs[i][0], de=offs[i][1], hdg=offs[i][2], dist=dist,
                   bearing_body=bearing_body, choice=choice, settle=settle, path=path,
                   path_ratio=path / dist if dist > 0.3 else np.nan, overshoot=overshoot, max_cross=max_cross,
                   heading_travel=heading_travel, heading_needed=heading_needed,
                   max_speed=float(np.hypot(u, v).max()), max_sway=float(np.abs(v).max()),
                   max_r=float(np.degrees(np.abs(r).max())), final_err=float(err[-1]))
        for k in THR:
            f = d[k]
            f = f[(f[:, 0] >= t0) & (f[:, 0] <= t1)]
            if len(f) < 2:
                continue
            tt, mag = f[:, 0], np.hypot(f[:, 1], f[:, 2]) * np.sign(f[:, 1] + 1e-9 * f[:, 2])
            if k == "bow_azimuth":
                mag = np.hypot(f[:, 1], f[:, 2])      # the azimuth's force along its own angle
                row["az_travel"] = float(np.degrees(np.sum(np.abs(np.diff(np.unwrap(f[:, 3]))))))
            dt = np.diff(tt, prepend=tt[0])
            row[f"{k}_impulse"] = float(np.sum(np.abs(mag) * dt))
            row[f"{k}_energy"] = float(np.sum(np.abs(mag) ** 1.5 * dt))     # P ~ T^1.5 (momentum theory)
            row[f"{k}_peak"] = float(np.abs(mag).max())
            sat = (mag >= 0.95 * LIMIT["fwd"]) | (mag <= -0.95 * LIMIT["rev"])
            row[f"{k}_sat"] = float(np.sum(dt[sat]))
        s = d["slack"]
        s = s[(s[:, 0] >= t0) & (s[:, 0] <= t1)]
        row["slack_max"] = float(np.abs(s[:, 1:]).max()) if len(s) else 0.0
        rows.append(row)
    return d, rows


def fig_paths(datas, out):
    fig, axs = plt.subplots(1, len(datas), figsize=(5.2 * len(datas), 5.4))
    for ax, (label, d) in zip(np.atleast_1d(axs), datas):
        o = d["odom"]
        ax.plot(o[:, 2], o[:, 1], lw=1.2, color="#2a6fdb")
        for i, (t0, t1, gx, gy, gp) in enumerate(d["goals"]):
            ax.plot(gy, gx, "o", ms=5, color="#d1453b")
            ax.annotate(str(i + 1), (gy, gx), textcoords="offset points", xytext=(4, 4), fontsize=8)
        for k in range(0, len(o), 60):           # heading every ~3 s
            ax.arrow(o[k, 2], o[k, 1], 0.5 * math.sin(o[k, 3]), 0.5 * math.cos(o[k, 3]),
                     head_width=0.18, color="#555", lw=0.5, length_includes_head=True)
        ax.set_title(label); ax.set_xlabel("east [m]"); ax.set_ylabel("north [m]")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_thrust_share(rows, out):
    first = rows[0]["scenario"]
    calm = [r for r in rows if r["scenario"] == first]
    labels = [f"{r['goal']}: ({r['dn']:+.0f},{r['de']:+.0f}) {r['hdg']:.0f}°" for r in calm]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bottom = np.zeros(len(calm))
    for k, c in zip(THR, ("#2a6fdb", "#7fb2ff", "#e39b2f")):
        vals = np.array([r.get(f"{k}_impulse", 0.0) for r in calm])
        ax.bar(labels, vals, bottom=bottom, label=k, color=c)
        bottom += vals
    ax.set_ylabel("impulse ∫|F| dt [N s]"); ax.set_title(f"Thrust per goal, {first}")
    ax.legend(); plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_timeseries(d, goal, out, title):
    t0, t1 = d["goals"][goal - 1][:2]
    o = d["odom"]; o = o[(o[:, 0] >= t0) & (o[:, 0] <= t1)]
    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(o[:, 0] - t0, o[:, 4], label="u (surge)"); axs[0].plot(o[:, 0] - t0, o[:, 5], label="v (sway)")
    axs[0].set_ylabel("m/s"); axs[0].legend()
    axs[1].plot(o[:, 0] - t0, np.degrees(np.unwrap(o[:, 3])), label="heading")
    axs[1].axhline(math.degrees(d["goals"][goal - 1][4]) % 360, ls="--", color="gray")
    axs[1].set_ylabel("deg"); axs[1].legend()
    for k in THR:
        f = d[k]; f = f[(f[:, 0] >= t0) & (f[:, 0] <= t1)]
        mag = np.hypot(f[:, 1], f[:, 2]) if k == "bow_azimuth" else f[:, 1]
        axs[2].plot(f[:, 0] - t0, mag, label=k)
        if k == "bow_azimuth":
            axs[3].plot(f[:, 0] - t0, np.degrees(f[:, 3]), color="#e39b2f", label="azimuth angle")
    axs[2].set_ylabel("thrust [N]"); axs[2].legend(); axs[3].set_ylabel("deg"); axs[3].legend()
    axs[3].set_xlabel("time after goal [s]"); axs[0].set_title(title)
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--logs", default="")
    ap.add_argument("--out", default="sweep", help="folder under ~/ros_ws/results for the table and figures")
    a = ap.parse_args()
    out = f"{RES}/{a.out}"
    os.makedirs(out, exist_ok=True)
    datas, rows = [], []
    for label in a.labels:
        d, r = analyse(label, a.logs)
        datas.append((label, d)); rows += r
    keys = list(rows[0].keys()) + sorted({k for r in rows for k in r} - set(rows[0].keys()))
    with open(f"{out}/sweep_table.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    fig_paths(datas, f"{out}/paths.png")
    fig_thrust_share(rows, f"{out}/thrust_share.png")
    for label, d in datas:
        for g in (2, 7, 8):
            fig_timeseries(d, g, f"{out}/ts_{label}_goal{g}.png", f"{label}, goal {g}")
    print(f"wrote {out}/sweep_table.csv and figures")
    hdr = ("scen", "g", "offset", "hdg", "choice", "settle", "path/d", "over", "cross", "hdgTr/need",
           "vmax", "|v|max", "rmax", "port", "stbd", "bow", "azTr", "slack")
    print(" ".join(f"{h:>9s}" for h in hdr))
    for r in rows:
        print(" ".join(f"{x:>9}" for x in (
            r["scenario"][:7], r["goal"], f"{r['dn']:+.0f},{r['de']:+.0f}", f"{r['hdg']:.0f}", r["choice"],
            f"{r['settle']:.0f}", f"{r['path_ratio']:.2f}", f"{r['overshoot']:.2f}", f"{r['max_cross']:.2f}",
            f"{r['heading_travel']:.0f}/{r['heading_needed']:.0f}", f"{r['max_speed']:.2f}", f"{r['max_sway']:.2f}",
            f"{r['max_r']:.0f}", f"{r.get('port_impulse', 0):.0f}", f"{r.get('starboard_impulse', 0):.0f}",
            f"{r.get('bow_azimuth_impulse', 0):.0f}", f"{r.get('az_travel', 0):.0f}", f"{r['slack_max']:.1f}")))


if __name__ == "__main__":
    main()
