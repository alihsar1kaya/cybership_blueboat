#!/usr/bin/env python3
"""Plotter docking-forsøk og sjekker kollisjon/klaring for hele skroget.

Bruk: python3 plot_docking.py <csv> [<csv> ...]   (flere kjøringer i samme figur)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Geometri inspirert av Fig. 7 i Gezer et al. (x0, x1, y0, y1)
BASIN = (-4.5, 4.5, -3.0, 3.0)
OBSTACLES = [(-2.0, -1.5, -0.2, 3.0), (1.0, 1.5, -3.0, 0.2)]
BERTH = (-4.0, -3.0, 1.0, 2.0)
L, B = 1.0, 0.3   # C/S Voyager (avrundet)
COLORS = {"direkte": "#d1495b", "veipunkter": "#2a5bd7"}


def hull_points(x, y, psi, n=9):
    """Punkter langs skrogets omkrets (rektangel L x B) i verdensramme."""
    s = np.linspace(-0.5, 0.5, n)
    local = np.vstack([np.c_[s * L, np.full(n, B / 2)], np.c_[s * L, np.full(n, -B / 2)],
                       np.c_[np.full(3, L / 2), np.linspace(-B / 2, B / 2, 3)],
                       np.c_[np.full(3, -L / 2), np.linspace(-B / 2, B / 2, 3)]])
    c, sn = np.cos(psi), np.sin(psi)
    return local @ np.array([[c, sn], [-sn, c]]) + [x, y]


def signed_clearance(p, box):
    """Avstand fra punkter til boks (negativ = inni)."""
    x0, x1, y0, y1 = box
    dx = np.maximum(np.maximum(x0 - p[:, 0], p[:, 0] - x1), 0)
    dy = np.maximum(np.maximum(y0 - p[:, 1], p[:, 1] - y1), 0)
    outside = np.hypot(dx, dy)
    inside = -np.minimum(np.minimum(p[:, 0] - x0, x1 - p[:, 0]), np.minimum(p[:, 1] - y0, y1 - p[:, 1]))
    is_in = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
    return np.where(is_in, inside, outside)


def analyse(d):
    clear = []
    for x, y, psi in zip(d["x"], d["y"], d["psi"]):
        p = hull_points(x, y, psi)
        c_obs = min(signed_clearance(p, o).min() for o in OBSTACLES)
        c_wall = min((p[:, 0] - BASIN[0]).min(), (BASIN[1] - p[:, 0]).min(),
                     (p[:, 1] - BASIN[2]).min(), (BASIN[3] - p[:, 1]).min())
        clear.append(min(c_obs, c_wall))
    clear = np.array(clear)
    t = d["t"]
    v = np.hypot(np.gradient(d["x"], t), np.gradient(d["y"], t))
    bx, by = (BERTH[0] + BERTH[1]) / 2, (BERTH[2] + BERTH[3]) / 2
    in_berth = (np.abs(d["x"] - bx) < 0.5) & (np.abs(d["y"] - by) < 0.5)
    k = np.argmax(in_berth) if in_berth.any() else None
    return {
        "clear": clear,
        "min_clear": clear.min(),
        "collision": bool((clear < 0).any()),
        "t_end": t[-1] - t[0],
        "v_entry": v[k] if k is not None else np.nan,
        "err_final": np.hypot(d["x"][-1] - bx, d["y"][-1] - by),
        "t": t - t[0],
    }


def main():
    files = sys.argv[1:]
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1])
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    ax.add_patch(Rectangle((BASIN[0], BASIN[2]), BASIN[1] - BASIN[0], BASIN[3] - BASIN[2],
                           fill=False, lw=2, color="black"))
    for o in OBSTACLES:
        ax.add_patch(Rectangle((o[0], o[2]), o[1] - o[0], o[3] - o[2], color="#f2c14e", alpha=0.8))
    ax.add_patch(Rectangle((BERTH[0], BERTH[2]), BERTH[1] - BERTH[0], BERTH[3] - BERTH[2],
                           color="#2e8b57", alpha=0.35))
    ax.text(-3.5, 2.15, "Kaiplass", ha="center", fontsize=9)

    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        name = "veipunkter" if "veipunkter" in os.path.basename(f) else "direkte"
        col = COLORS[name]
        r = analyse(d)
        status = "KOLLISJON" if r["collision"] else "ok"
        print(f"{name:11s} varighet {r['t_end']:5.0f} s | min. klaring {r['min_clear']:+.2f} m ({status}) | "
              f"fart inn i kaiplass {r['v_entry']:.2f} m/s | sluttfeil {r['err_final']:.3f} m")
        ax.plot(d["x"], d["y"], color=col, lw=1.6,
                label=f"{name}: min. klaring {r['min_clear']:+.2f} m, {status}")
        for i in range(0, len(d), max(1, len(d) // 45)):
            h = hull_points(d["x"][i], d["y"][i], d["psi"][i], n=2)
            ax.fill(h[[0, 1, 3, 2], 0], h[[0, 1, 3, 2], 1], color=col, alpha=0.12, lw=0)
        bad = r["clear"] < 0
        if bad.any():
            ax.plot(d["x"][bad], d["y"][bad], ".", color="black", ms=3)
        ax2.plot(r["t"], r["clear"], color=col, lw=1.4, label=name)

    ax.plot(3.0, -2.3, "o", color="black", ms=6)
    ax.text(3.15, -2.55, "Start", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-3.3, 3.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Docking med dagens DP-regulator (C/S Voyager, simulator)")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("Klaring skrog–hinder [m]")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    out = os.path.join(os.path.dirname(files[0]), "docking_sammenligning.png")
    fig.savefig(out, dpi=150)
    print(f"Figur: {out}")


if __name__ == "__main__":
    main()
