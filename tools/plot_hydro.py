#!/usr/bin/env python3
"""
Plot Capytaine A(w), B(w) (diagonal + couplings) with power-based A_eq for several w_p, and the PM
weighting spectra. Optional second file (e.g. without lid) drawn dashed for comparison.

    python3 tools/plot_hydro.py src/.../microampere/hydro_capytaine.npz [other.npz] --wp 1.5 2.5 3 4
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cybership_utilities import hydro

TERMS = [("11", 0, 0), ("22", 1, 1), ("33", 2, 2), ("44", 3, 3), ("55", 4, 4), ("66", 5, 5),
         ("26", 1, 5), ("24", 1, 3), ("35", 2, 4), ("15", 0, 4), ("46", 3, 5), ("13", 0, 2)]

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--wp", type=float, nargs="+", default=[1.5, 2.5, 3.0, 4.0])
ap.add_argument("--out", default=os.path.expanduser("~/ros_ws/results/hydro_microampere.png"))
a = ap.parse_args()

data = [hydro.load_hydro(f) for f in a.files]
fig, axes = plt.subplots(5, 6, figsize=(22, 16))
for k, (name, i, j) in enumerate(TERMS):
    for row, var in ((0, "A"), (2, "B")):
        ax = axes[row + k // 6, k % 6]
        for n, h in enumerate(data):
            f = h["freqs"]
            ok = np.isfinite(f)
            ax.plot(f[ok], h[var][i, j, ok], "-" if n == 0 else "--", lw=1.5 if n == 0 else 1,
                    label=os.path.basename(a.files[n]))
        if var == "A":
            h = data[0]
            ax.axhline(h["A"][i, j, -1], color="gray", ls=":", lw=1)
            for wp in a.wp:
                A_eq, _ = hydro.power_based_matrices(h["freqs"], h["A"], h["B"], wp)
                ax.plot(0.88 * wp, A_eq[i, j], "o", ms=5)
        ax.set_title(f"{var}{name}(ω)")
        ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=7)
ax = axes[4, 0]
w = np.linspace(0.05, 12, 400)
for wp in a.wp:
    S = hydro.pm_spectrum(w, wp)
    ax.plot(w, S / np.trapz(S, w), label=f"ω_p {wp}")
ax.set_title("S_N(ω), PM weighting")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
for ax in axes[4, 1:]:
    ax.axis("off")
for ax in axes[3, :]:
    ax.set_xlabel("ω [rad/s]")
fig.suptitle("Microampere, Capytaine on the hull CAD (BODY frame, CO). Dots: power-based A_eq plotted at the spectral peak 0.88 ω_p; dotted: A(∞)")
fig.tight_layout()
fig.savefig(a.out, dpi=90)
print(a.out)
