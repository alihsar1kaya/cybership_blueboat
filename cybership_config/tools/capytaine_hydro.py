#!/usr/bin/env python3
"""
Frequency-dependent hydrodynamic data from a hull CAD file (STL) with Capytaine, as MSS stores it
for ShipX/WAMIT/MSS-Capytaine vessels (vessel.A, vessel.B, vessel.C, vessel.freqs):

    A(omega)   6x6 added mass                    [kg, kg m, kg m^2]
    B(omega)   6x6 potential (radiation) damping [N s/m, ...]
    C          6x6 hydrostatic restoring about CO
    force_rao  6 x n_w x n_beta complex first-order wave excitation (Froude-Krylov + diffraction) per
               metre wave amplitude, MSS convention: elevation zeta = A cos(w t + eps) at CO gives
               tau = A Re(H e^{i(w t + eps)}) (Capytaine uses e^{-i w t}, so H = conj(F_capytaine)).
               Headings beta = wave propagation direction relative to the bow, clockwise (0 following
               sea, 90 waves travelling towards starboard, 180 head sea), 0-350 deg in 10 deg steps,
               computed for 0-180 deg and mirrored (port-starboard symmetric hull), as MSS-Capytaine.

Used by Fossen's power-based averaging (MSS computeManeuveringModel.m, Fossen 2025) to get the
constant M_A and B_eq for the simulator; see cybership_utilities/hydro.py.

The panel mesh is the same offset-table mesh as capytaine_from_stl.py. A lid on the waterplane
removes the irregular frequencies of the BEM solution (Capytaine lid_mesh).

Frames: Capytaine solves in x forward, y port, z up with the bow at +x. The results are rotated to
Fossen's BODY frame (x forward, y starboard, z down): A_b = T A T with T = diag(1,-1,-1,1,-1,-1).
Coordinate origin CO: on the waterline, midships (middle of the hull length), on the centerline.

    python3 capytaine_hydro.py ~/ros_ws/cad/hull/PUB-BR-101381-001.STL --mass 30.2 \
        --hull-offset 0.374 --swap-sides --out ../config/microampere/hydro_capytaine.npz

Needs (in ~/ros_ws/venv): pip install capytaine trimesh
"""

import argparse
import time

import numpy as np
import capytaine as cpt

from capytaine_from_stl import DOFS, G, RHO, OffsetTable, panel_mesh

# Capytaine (z up, y port) -> Fossen BODY (z down, y starboard): rotation of pi about x
T_BODY = np.diag([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])


def bow_is_positive_x(hull):
    """
    The bow is the end where the keel rises (rocker / forefoot). The BlueBoat stern has a flat keel
    and a low nacelle for the thruster, so the waterline breadth is not a reliable test there.
    Returns (bow at +x, keel rise near the -x end, keel rise near the +x end) [m].
    """
    L = hull.x_max - hull.x_min

    def keel_rise(x):
        seg = hull.section(x)
        return seg[:, :, 2].min() - hull.z_keel if len(seg) else 0.0

    aft, fwd = keel_rise(hull.x_min + 0.05 * L), keel_rise(hull.x_max - 0.05 * L)
    return fwd > aft, aft, fwd


def to_body(M):
    """6x6 (or 6x6xN) matrix from Capytaine axes to Fossen BODY axes."""
    if M.ndim == 2:
        return T_BODY @ M @ T_BODY
    return np.einsum("ij,jkn,kl->iln", T_BODY, M, T_BODY)


def matrix(ds, var, omega):
    """6x6 numpy matrix [influenced, radiating] from a Capytaine dataset at one frequency."""
    return ds[var].sel(omega=omega).transpose("influenced_dof", "radiating_dof").sel(
        influenced_dof=DOFS, radiating_dof=DOFS).values


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stl", help="hull STL")
    p.add_argument("--mass", type=float, required=True, help="vessel mass [kg]")
    p.add_argument("--hull-offset", type=float, default=0.0, help="catamaran: hull centre from centerline [m]")
    p.add_argument("--axes", default="zxy")
    p.add_argument("--scale", type=float, default=1e-3)
    p.add_argument("--swap-sides", action="store_true")
    p.add_argument("--cog-z", type=float, default=0.0, help="CG height above waterline [m] (hydrostatics)")
    p.add_argument("--stations", type=int, default=36)
    p.add_argument("--heights", type=int, default=12)
    p.add_argument("--cap-rings", type=int, default=4, help="rings in the end caps (smaller panels)")
    p.add_argument("--omega-max", type=float, default=15.0, help="[rad/s]")
    p.add_argument("--n-omega", type=int, default=60)
    p.add_argument("--no-lid", action="store_true", help="no irregular-frequency removal (for comparison)")
    p.add_argument("--rao-omega-max", type=float, default=7.0, help="[rad/s] highest force-RAO frequency")
    p.add_argument("--rao-n-omega", type=int, default=28)
    p.add_argument("--no-rao", action="store_true", help="skip the diffraction (force RAO) problems")
    p.add_argument("--out", required=True, help="output .npz")
    a = p.parse_args()

    n_hulls = 2 if a.hull_offset > 0 else 1
    hull = OffsetTable(a.stl, axes=a.axes, scale=a.scale)
    z_wl = hull.waterline_for_volume(a.mass / RHO / n_hulls)
    bow_fwd, k_aft, k_fwd = bow_is_positive_x(hull)
    if not bow_fwd:
        hull = OffsetTable(a.stl, axes=a.axes, scale=a.scale, flip_length=True)
        z_wl = hull.waterline_for_volume(a.mass / RHO / n_hulls)
    print(f"Keel rise near the ends (-x / +x): {1000 * k_aft:.0f} / {1000 * k_fwd:.0f} mm -> "
          f"bow at {'+x' if bow_fwd else '-x, flipped to +x'}")
    print(f"Draft {1000 * (z_wl - hull.z_keel):.0f} mm")

    stations = hull.sample(z_wl, a.stations, a.heights)
    side = -1.0 if a.swap_sides else 1.0
    n_contour = 2 * a.heights
    meshes = [panel_mesh(stations, hull.x_mid, hull.y_mid, z_wl, side * a.hull_offset, n_contour,
                         cap_rings=a.cap_rings)]
    if n_hulls == 2:
        meshes.append(panel_mesh(stations, hull.x_mid, hull.y_mid, z_wl, -side * a.hull_offset, n_contour,
                                 mirror=True, cap_rings=a.cap_rings))
    mesh = meshes[0] if n_hulls == 1 else meshes[0].join_meshes(meshes[1])
    lid = None if a.no_lid else mesh.generate_lid(z=mesh.lowest_lid_position(a.omega_max))
    body = cpt.FloatingBody(mesh=mesh, lid_mesh=lid, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)),
                            center_of_mass=(0.0, 0.0, a.cog_z), mass=a.mass, name="hull")
    awp = float(body.compute_hydrostatics(rho=RHO, g=G)["waterplane_area"])
    print(f"{mesh.nb_faces} hull panels, {0 if lid is None else lid.nb_faces} lid panels, "
          f"displaced mass {RHO * body.volume:.2f} kg, waterplane area {awp:.3f} m^2")

    omegas = np.r_[0.0, np.linspace(a.omega_max / a.n_omega, a.omega_max, a.n_omega), np.inf]
    problems = [cpt.RadiationProblem(body=body, radiating_dof=d, omega=w, rho=RHO) for d in DOFS for w in omegas]
    t0 = time.time()
    ds = cpt.assemble_dataset(cpt.BEMSolver().solve_all(problems, progress_bar=False))
    print(f"Solved {len(problems)} radiation problems in {time.time() - t0:.0f} s")

    A = np.stack([matrix(ds, "added_mass", w) for w in omegas], axis=2)
    B = np.stack([matrix(ds, "radiation_damping", w) for w in omegas], axis=2)
    B[:, :, 0] = 0.0  # no radiation at omega = 0
    B[:, :, -1] = 0.0  # nor at omega = inf
    C_cpt = body.compute_hydrostatic_stiffness(rho=RHO, g=G).transpose(
        "influenced_dof", "radiating_dof").sel(influenced_dof=DOFS, radiating_dof=DOFS).values
    A, B, C = to_body(A), to_body(B), to_body(C_cpt)

    rao = {}
    if not a.no_rao:
        w_rao = np.linspace(a.rao_omega_max / a.rao_n_omega, a.rao_omega_max, a.rao_n_omega)
        beta_half = np.radians(np.arange(0, 181, 10))
        # BODY heading beta (clockwise, towards starboard) = Capytaine wave_direction -beta (z up, y port)
        problems = [cpt.DiffractionProblem(body=body, wave_direction=-b, omega=w, rho=RHO)
                    for w in w_rao for b in beta_half]
        t0 = time.time()
        dsd = cpt.assemble_dataset(cpt.BEMSolver().solve_all(problems, progress_bar=False))
        print(f"Solved {len(problems)} diffraction problems in {time.time() - t0:.0f} s")
        F = (dsd["diffraction_force"] + dsd["Froude_Krylov_force"]).transpose(
            "influenced_dof", "omega", "wave_direction").sel(influenced_dof=DOFS)
        F = F.sel(omega=w_rao, wave_direction=-beta_half).values      # 6 x n_w x n_half, Capytaine axes
        F = np.conj(np.einsum("ij,jkl->ikl", T_BODY, F))                # BODY axes, MSS phase convention
        # mirror 190-350 deg from 170-10 deg: port-starboard symmetry flips sway, roll and yaw
        mirror = np.array([1, -1, 1, -1, 1, -1])[:, None, None]
        F_full = np.concatenate([F, mirror * F[:, :, -2:0:-1]], axis=2)
        rao = {"rao_freqs": w_rao, "rao_headings": np.radians(np.arange(0, 360, 10)), "force_rao": F_full}

    np.savez(a.out, freqs=omegas, A=A, B=B, C=C, mass=a.mass, rho=RHO, g=G, **rao,
             draft=z_wl - hull.z_keel, waterplane_area=awp, lid=not a.no_lid,
             source=f"capytaine {cpt.__version__}: {a.stl} hull_offset {a.hull_offset} swap_sides {a.swap_sides}")
    np.set_printoptions(precision=2, suppress=True, linewidth=140)
    print("\nA(0) [BODY, CO]:\n", A[:, :, 0])
    print("A(inf):\n", A[:, :, -1])
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
