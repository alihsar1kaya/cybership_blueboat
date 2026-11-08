#!/usr/bin/env python3
"""Plotter 4-hjørnetest (som Fig. 6 i Gezer et al.) og skriver RMSE.

Bruk: python3 plot_four_corner.py <csv> [<png>]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".csv", ".png")
    d = np.genfromtxt(path, delimiter=",", names=True)
    d = d[~np.isnan(d["x_d"])]
    t = d["t"] - d["t"][0]

    ex, ey = d["x"] - d["x_d"], d["y"] - d["y_d"]
    epsi = np.arctan2(np.sin(d["psi"] - d["psi_d"]), np.cos(d["psi"] - d["psi_d"]))
    rmse_pos = np.sqrt(np.mean(ex ** 2 + ey ** 2))
    rmse_yaw = np.degrees(np.sqrt(np.mean(epsi ** 2)))
    print(f"RMSE posisjon: {rmse_pos:.3f} m   RMSE yaw: {rmse_yaw:.3f}°   varighet: {t[-1]:.0f} s")

    fig = plt.figure(figsize=(7, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    sq = np.array([[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5]])
    ax.plot(sq[:, 0], sq[:, 1], color="#2a5bd7", lw=1.5, label="Hjørner (mål)")
    ax.plot(d["x_d"], d["y_d"], color="#888888", lw=1, ls=":", label="Referanse (filter)")
    ax.plot(d["x"], d["y"], color="#d1495b", lw=1.5, ls="--", label="Simulert båt")
    L = 0.5  # skisse av skroget hver ~3 s
    for i in range(0, len(d), max(1, len(d) // 120)):
        c, s = np.cos(d["psi"][i]), np.sin(d["psi"][i])
        hull = np.array([[L / 2, 0], [L / 4, 0.08], [-L / 2, 0.08], [-L / 2, -0.08], [L / 4, -0.08], [L / 2, 0]])
        xy = hull @ np.array([[c, s], [-s, c]]) + [d["x"][i], d["y"][i]]
        ax.plot(xy[:, 0], xy[:, 1], color="black", lw=0.4, alpha=0.35)
    ax.plot(d["x"][0], d["y"][0], "o", color="#2e8b57", label="Start")
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)
    ax.legend(loc="center", fontsize=8)
    ax.set_title(f"C/S Voyager – 4-hjørnetest i simulator\n"
                 f"RMSE pos {rmse_pos:.3f} m, yaw {rmse_yaw:.2f}°")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, ex, label=r"$\tilde{x}$ [m]", color="#2a5bd7", lw=1)
    ax2.plot(t, ey, label=r"$\tilde{y}$ [m]", color="#d1495b", lw=1)
    ax2.plot(t, epsi, label=r"$\tilde{\psi}$ [rad]", color="#2e8b57", lw=1)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("Følgefeil")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, ncol=3)
    fig.savefig(out, dpi=150)
    print(f"Figur: {out}")


if __name__ == "__main__":
    main()
