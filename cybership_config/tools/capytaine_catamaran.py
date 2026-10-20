#!/usr/bin/env python3
"""
Added mass and hydrostatics of a catamaran with Capytaine (linear potential flow, BEM).

Each hull is approximated by a box. The draft is chosen so the displaced water equals the
vessel mass (Archimedes), unless --draft is given.

    python3 capytaine_catamaran.py --mass 16.9 --length 1.2 --hull-width 0.2 --hull-offset 0.365

Prints the added mass at omega = 0 (slow motion, relevant for DP) and omega = inf,
in the order surge, sway, heave, roll, pitch, yaw, about CO = (0, 0, 0) on the waterline.
Only the diagonal is used by vessel.yaml at the moment.

Needs: pip install capytaine (in ~/ros_ws/venv).
"""

import argparse

import numpy as np
import capytaine as cpt

RHO = 1000.0
G = 9.81
DOFS = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]


def catamaran_body(length, hull_width, hull_offset, draft, panel_size, cog_z=0.0, mass=None):
    """Two box hulls at y = +-hull_offset, clipped at the waterline (z = 0, z up)."""
    nx = max(4, int(round(length / panel_size)))
    ny = max(2, int(round(hull_width / panel_size)))
    nz = max(2, int(round(2 * draft / panel_size)))
    hulls = [
        cpt.mesh_parallelepiped(
            size=(length, hull_width, 2 * draft),  # half above water, removed by immersed_part()
            center=(0.0, y, 0.0),
            resolution=(nx, ny, nz),
            name=f"hull_{'port' if y > 0 else 'starboard'}",
        )
        for y in (hull_offset, -hull_offset)
    ]
    mesh = hulls[0].join_meshes(hulls[1])
    body = cpt.FloatingBody(
        mesh=mesh,
        dofs=cpt.rigid_body_dofs(rotation_center=(0.0, 0.0, 0.0)),
        center_of_mass=(0.0, 0.0, cog_z),
        mass=mass,
        name="catamaran",
    )
    return body.immersed_part()


def added_mass(body, omegas):
    solver = cpt.BEMSolver()
    problems = [
        cpt.RadiationProblem(body=body, radiating_dof=dof, omega=w, rho=RHO)
        for dof in DOFS
        for w in omegas
    ]
    results = solver.solve_all(problems, progress_bar=False)
    return cpt.assemble_dataset(results)["added_mass"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mass", type=float, default=16.9, help="vessel mass [kg]")
    p.add_argument("--length", type=float, default=1.2, help="hull length [m]")
    p.add_argument("--hull-width", type=float, default=0.2, help="width of one hull [m]")
    p.add_argument("--hull-offset", type=float, default=0.365, help="hull centerline to boat centerline [m]")
    p.add_argument("--draft", type=float, default=None, help="draft [m] (default: from mass)")
    p.add_argument("--cog-z", type=float, default=0.0, help="CG height above waterline [m], z up")
    p.add_argument("--panel-size", type=float, default=0.025, help="mesh panel size [m]")
    a = p.parse_args()

    draft = a.draft or a.mass / (RHO * 2 * a.length * a.hull_width)
    body = catamaran_body(a.length, a.hull_width, a.hull_offset, draft, a.panel_size,
                          cog_z=a.cog_z, mass=a.mass)
    print(f"Hull {a.length} x {a.hull_width} m, offset +-{a.hull_offset} m, draft {draft*1000:.1f} mm, "
          f"{body.mesh.nb_faces} panels")
    print(f"Displaced mass: {RHO * body.volume:.2f} kg (vessel {a.mass} kg)")

    A = added_mass(body, [0.0, np.inf])
    print("\nAdded mass, diagonal [kg, kg m^2]:")
    print(f"{'':>8} {'omega=0':>10} {'omega=inf':>10}")
    for i, dof in enumerate(DOFS):
        a0 = float(A.sel(omega=0.0, radiating_dof=dof, influenced_dof=dof))
        ainf = float(A.sel(omega=np.inf, radiating_dof=dof, influenced_dof=dof))
        print(f"{dof:>8} {a0:10.3f} {ainf:10.3f}")
    print("\nvessel.yaml (omega = 0):")
    diag = [float(A.sel(omega=0.0, radiating_dof=d, influenced_dof=d)) for d in DOFS]
    print("  added_mass: [" + ", ".join(f"{v:.2f}" for v in diag) + "]")

    C = body.compute_hydrostatic_stiffness(rho=RHO, g=G)
    c44 = float(C.sel(influenced_dof="Roll", radiating_dof="Roll"))
    c55 = float(C.sel(influenced_dof="Pitch", radiating_dof="Pitch"))
    print(f"\nHydrostatics (CG {a.cog_z} m above waterline):")
    print(f"  GM_phi:   {c44 / (a.mass * G):.3f} m")
    print(f"  GM_theta: {c55 / (a.mass * G):.3f} m")


if __name__ == "__main__":
    main()
