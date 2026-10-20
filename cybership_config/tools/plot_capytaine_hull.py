#!/usr/bin/env python3
"""
Figure of what capytaine_from_stl.py does, for one hull STL:

  1. the panel mesh Capytaine solves on (immersed part, both hulls)
  2. cross sections: CAD outer skin vs. the offset table, with the waterline
  3. added mass A(omega) for surge, sway and yaw, with A(0) and A(inf)
  4. box hulls (capytaine_catamaran.py) vs. the CAD hull

    python3 plot_capytaine_hull.py ~/ros_ws/cad/hull/PUB-BR-101381-001.STL \
        --mass 16.9 --hull-offset 0.3765 --out capytaine_helga.png
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import capytaine as cpt

from capytaine_from_stl import OffsetTable, panel_mesh, RHO, G
from capytaine_catamaran import catamaran_body


def solve(body, dofs, omegas):
    problems = [cpt.RadiationProblem(body=body, radiating_dof=d, omega=w, rho=RHO)
                for d in dofs for w in omegas]
    return cpt.assemble_dataset(cpt.BEMSolver().solve_all(problems, progress_bar=False))["added_mass"]


def diag(A, dof, omega):
    return float(A.sel(omega=omega, radiating_dof=dof, influenced_dof=dof))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stl")
    p.add_argument("--mass", type=float, default=16.9)
    p.add_argument("--hull-offset", type=float, default=0.3765)
    p.add_argument("--box-width", type=float, default=0.2, help="hull width of the box comparison [m]")
    p.add_argument("--out", default="capytaine_hull.png")
    a = p.parse_args()

    # --- CAD hull -> offset table -> panel mesh (same steps as capytaine_from_stl.py)
    hull = OffsetTable(a.stl)
    z_wl = hull.waterline_for_volume(a.mass / RHO / 2)
    stations = hull.sample(z_wl, 36, 12)
    meshes = [panel_mesh(stations, hull.x_mid, hull.y_mid, z_wl, s * a.hull_offset, 24, mirror=s < 0)
              for s in (1, -1)]
    body = cpt.FloatingBody(mesh=meshes[0].join_meshes(meshes[1]),
                            dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)),
                            center_of_mass=(0, 0, 0), mass=a.mass)

    dofs = ["Surge", "Sway", "Yaw"]
    omegas = np.r_[0.0, np.linspace(0.5, 9.5, 14), np.inf]
    A = solve(body, dofs, omegas)
    C = body.compute_hydrostatic_stiffness(rho=RHO, g=G)
    gm = float(C.sel(influenced_dof="Roll", radiating_dof="Roll")) / (a.mass * G)

    box_draft = a.mass / (RHO * 2 * 1.2 * a.box_width)
    box = catamaran_body(1.2, a.box_width, a.hull_offset, box_draft, 0.025, mass=a.mass)
    A_box = solve(box, dofs, [0.0])
    C_box = box.compute_hydrostatic_stiffness(rho=RHO, g=G)
    gm_box = float(C_box.sel(influenced_dof="Roll", radiating_dof="Roll")) / (a.mass * G)

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(f"Capytaine on the BlueBoat hull CAD: {a.mass} kg, hulls at ±{a.hull_offset} m, "
                 f"draft {1000 * (z_wl - hull.z_keel):.0f} mm, {body.mesh.nb_faces} panels", fontsize=13)

    # 1. panel mesh
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    v, f = body.mesh.vertices, body.mesh.faces
    ax.add_collection3d(Poly3DCollection(v[f], facecolor="#3366CC", edgecolor="k", linewidth=0.15, alpha=0.9))
    span = 0.65
    xx, yy = np.meshgrid([-span, span], [-span, span])
    ax.plot_surface(xx, yy, 0 * xx, color="#88bbee", alpha=0.15)
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(-0.4, 0.1)
    ax.set_box_aspect((1, 1, 0.45))
    ax.set_xlabel("x forward [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=-20, azim=-130)
    ax.set_title("1. Panel mesh Capytaine solves on (below the waterline)")

    # 2. cross sections
    ax = fig.add_subplot(2, 2, 2)
    L = hull.x_max - hull.x_min
    for frac, col in [(0.15, "C0"), (0.5, "C1"), (0.8, "C2")]:
        x = hull.x_min + frac * L
        seg = hull.section(x)
        pts = seg.reshape(-1, 3)
        ax.plot((pts[:, 1] - hull.y_mid) * 1000, (pts[:, 2] - z_wl) * 1000, ".", ms=0.6, color=col, alpha=0.4)
        st = min(stations, key=lambda s: abs(s[0] - x))
        r = np.array(st[1])
        ax.plot((np.r_[r[::-1, 1], r[:, 2]] - hull.y_mid) * 1000, (np.r_[r[::-1, 0], r[:, 0]] - z_wl) * 1000,
                "-o", ms=3, color=col, label=f"{frac:.0%} of length from stern")
    ax.axhline(0, color="#3366CC", lw=2, label="waterline")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("y [mm]"); ax.set_ylabel("z [mm] (0 = waterline)")
    ax.set_title("2. CAD skin (dots, outer + inner) vs. offset table (lines)")
    ax.legend(fontsize=8, loc="upper right")

    # 3. A(omega)
    ax = fig.add_subplot(2, 2, 3)
    w = omegas[1:-1]
    for dof, col, unit in [("Surge", "C0", "kg"), ("Sway", "C1", "kg"), ("Yaw", "C2", "kg m²")]:
        scale = 10 if dof == "Yaw" else 1
        ax.plot(w, [scale * diag(A, dof, x) for x in w], "-o", ms=3, color=col,
                label=f"{dof}{' ×10' if scale > 1 else ''} [{unit}]")
        ax.axhline(scale * diag(A, dof, 0.0), color=col, ls="--", lw=1)
        ax.axhline(scale * diag(A, dof, np.inf), color=col, ls=":", lw=1)
    ax.axvspan(0, 1.0, color="0.85", zorder=0)
    ax.text(0.08, ax.get_ylim()[1] * 0.92, "DP motion\n(slow)", fontsize=8)
    # Short waves: panels too large (Capytaine warns) and irregular frequencies (no lid) -> not reliable
    ax.axvspan(7.0, w[-1], color="#f4cccc", zorder=0)
    ax.text(7.1, ax.get_ylim()[1] * 0.92, "not reliable\n(mesh too coarse\nfor short waves)", fontsize=8)
    ax.set_xlabel("ω [rad/s]"); ax.set_ylabel("added mass")
    ax.set_title("3. Added mass vs. frequency (-- ω = 0 used in vessel.yaml, ··· ω = ∞)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 4. box vs CAD
    ax = fig.add_subplot(2, 2, 4)
    labels = ["Surge [kg]", "Sway [kg]", "Yaw ×10 [kg m²]", "GM roll ×10 [m]"]
    cad = [diag(A, "Surge", 0), diag(A, "Sway", 0), 10 * diag(A, "Yaw", 0), 10 * gm]
    bx = [diag(A_box, "Surge", 0), diag(A_box, "Sway", 0), 10 * diag(A_box, "Yaw", 0), 10 * gm_box]
    i = np.arange(len(labels))
    b1 = ax.bar(i - 0.2, bx, 0.4, label=f"box hulls {a.box_width} m wide, {1000 * box_draft:.0f} mm draft",
                color="0.6")
    b2 = ax.bar(i + 0.2, cad, 0.4, label="BlueBoat hull CAD", color="#3366CC")
    ax.bar_label(b1, fmt="%.1f", fontsize=8); ax.bar_label(b2, fmt="%.1f", fontsize=8)
    ax.set_xticks(i, labels); ax.grid(alpha=0.3, axis="y")
    ax.set_title("4. Guessed box shape vs. real hull shape (ω = 0)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
