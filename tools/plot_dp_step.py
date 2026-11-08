#!/usr/bin/env python3
"""
Compare DP step tests (dp_step_test.py), e.g. before/after a controller change.

    python3 plot_dp_step.py ~/ros_ws/results/dp_step_microampere_before.csv \
        ~/ros_ws/results/dp_step_microampere_after.csv --out ~/ros_ws/results/dp_step_microampere.png
"""

import argparse
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOALS = {"surge": (3.0, 0.0, 0.0), "sway": (3.0, 3.0, 0.0), "yaw": (3.0, 3.0, 90.0)}


def load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["step"] not in ("", "None")]
    t = np.array([float(r["t"]) for r in rows])
    return {"t": t - t[0], "x": np.array([float(r["x"]) for r in rows]),
            "y": np.array([float(r["y"]) for r in rows]),
            "yaw": np.degrees([float(r["yaw"]) for r in rows]), "step": [r["step"] for r in rows]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--out", default=os.path.expanduser("~/ros_ws/results/dp_step.png"))
    a = p.parse_args()

    runs = [(os.path.basename(c).replace("dp_step_", "").replace(".csv", ""), load(c)) for c in a.csv]
    fig = plt.figure(figsize=(14, 9))
    ax_xy = fig.add_subplot(1, 2, 1)
    axes = [fig.add_subplot(3, 2, k) for k in (2, 4, 6)]

    for name, r in runs:
        ax_xy.plot(r["y"], r["x"], lw=1.5, label=name)
        for ax, key in zip(axes, ("x", "y", "yaw")):
            ax.plot(r["t"], r[key], lw=1.2, label=name)

    # goals and the straight lines between them
    pts = [(0, 0)] + [(g[0], g[1]) for g in GOALS.values()]
    ax_xy.plot([p[1] for p in pts], [p[0] for p in pts], "k--", lw=0.8, label="straight line")
    for name, g in GOALS.items():
        ax_xy.plot(g[1], g[0], "k*", ms=12)
    ax_xy.annotate("start", (0, 0), textcoords="offset points", xytext=(6, -12))
    ax_xy.annotate("1: 3 m ahead", (0, 3), textcoords="offset points", xytext=(-80, 6))
    ax_xy.annotate("2: 3 m sideways\n3: turn 90°", (3, 3), textcoords="offset points", xytext=(6, 6))
    ax_xy.set_xlabel("y east [m]"); ax_xy.set_ylabel("x north [m]")
    ax_xy.set_aspect("equal"); ax_xy.grid(alpha=0.3); ax_xy.legend()
    ax_xy.set_title("Path seen from above")

    t_step = max(r["t"][-1] for _, r in runs) / 3
    targets = {"x": [3, 3, 3], "y": [0, 3, 3], "yaw": [0, 0, 90]}
    for ax, key, unit in zip(axes, ("x", "y", "yaw"), ("m", "m", "deg")):
        for i, v in enumerate(targets[key]):
            ax.hlines(v, i * t_step, (i + 1) * t_step, colors="k", linestyles="--", lw=0.8)
        for i in (1, 2):
            ax.axvline(i * t_step, color="0.7", lw=0.8)
        ax.set_ylabel(f"{key} [{unit}]"); ax.grid(alpha=0.3)
    axes[0].set_title("Position and heading over time (-- = goal)")
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
