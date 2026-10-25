#!/usr/bin/env python3
"""
Added mass and hydrostatics from a hull CAD file (STL) with Capytaine.

The CAD hull is usually hollow and far too detailed for a BEM solver (Capytaine needs ~1000-3000
panels, a CAD STL has ~10^5 triangles). So the hull is first turned into an offset table, like the
classic ship lines plan: the hull is cut at stations along its length, and at each station the outer
skin is read off at a number of heights (outermost point on each side; inner skins lie inside and are
ignored). A clean panel mesh is built from that table.

The waterline is found so that the displaced water equals the vessel mass (Archimedes).
For a catamaran the hull is mirrored to y = +-hull_offset.

    python3 capytaine_from_stl.py ~/ros_ws/cad/hull/PUB-BR-101381-001.STL --mass 16.9 --hull-offset 0.3765

Output: added mass at omega = 0 and inf (surge, sway, heave, roll, pitch, yaw about CO on the
waterline, midships), GM_phi/GM_theta, and the lines for vessel.yaml.

Needs (in ~/ros_ws/venv): pip install capytaine trimesh
"""

import argparse

import numpy as np
import trimesh
from trimesh.intersections import mesh_plane
import capytaine as cpt

RHO = 1000.0
G = 9.81
DOFS = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
AXES = {"x": 0, "y": 1, "z": 2}


class OffsetTable:
    """Outer skin of a hull, sampled from an STL. Units: m, x forward, y sideways, z up."""

    def __init__(self, stl_path, axes="zxy", scale=1e-3, flip_length=False):
        mesh = trimesh.load(stl_path)
        # Reorder CAD axes to (length, width, height)
        v = mesh.vertices[:, [AXES[a] for a in axes]] * scale
        if flip_length:
            v[:, 0] = -v[:, 0]
        self.mesh = trimesh.Trimesh(v, mesh.faces, process=False)
        lo, hi = self.mesh.bounds
        self.x_min, self.x_max = lo[0], hi[0]
        self.z_keel = lo[2]
        self.x_mid = 0.5 * (lo[0] + hi[0])
        self.y_mid = 0.5 * (lo[1] + hi[1])
        self.height = hi[2] - lo[2]

    def section(self, x):
        return mesh_plane(self.mesh, [1.0, 0.0, 0.0], [x, 0.0, 0.0])

    @staticmethod
    def breadth_at(seg, z):
        """(y_left, y_right) of the outer skin at height z in a section, or None."""
        a, b = seg[:, 0, :], seg[:, 1, :]
        hit = (a[:, 2] - z) * (b[:, 2] - z) <= 0
        a, b = a[hit], b[hit]
        if len(a) == 0:
            return None
        dz = b[:, 2] - a[:, 2]
        dz[dz == 0] = 1e-12
        y = a[:, 1] + (z - a[:, 2]) / dz * (b[:, 1] - a[:, 1])
        return y.min(), y.max()

    def sample(self, z_waterline, n_stations, n_heights):
        """List of stations (x, [(z, y_left, y_right), ...]) below the waterline (absolute CAD z)."""
        stations = []
        eps = 1e-3 * (self.x_max - self.x_min)
        for x in np.linspace(self.x_min + eps, self.x_max - eps, n_stations):
            seg = self.section(x)
            if len(seg) == 0:
                continue
            z_keel = seg[:, :, 2].min()
            # skip sections that do not reach the waterline (e.g. a skeg sticking out behind the hull)
            if z_keel >= z_waterline - 1e-3 or self.breadth_at(seg, z_waterline) is None:
                continue
            rows = []
            for z in np.linspace(z_keel + 1e-4, z_waterline, n_heights):
                br = self.breadth_at(seg, z)
                if br is not None:
                    rows.append((z, br[0], br[1]))
            if len(rows) >= 2:
                stations.append((x, rows))
        return stations

    def volume(self, z_waterline, n_stations=60, n_heights=40):
        """Displaced volume [m^3] below z_waterline from the offset table."""
        xs, areas = [], []
        for x, rows in self.sample(z_waterline, n_stations, n_heights):
            r = np.array(rows)
            xs.append(x)
            areas.append(np.trapz(r[:, 2] - r[:, 1], r[:, 0]))
        return np.trapz(areas, xs) if len(xs) > 1 else 0.0

    def waterline_for_volume(self, target):
        lo, hi = self.z_keel, self.z_keel + self.height
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.volume(mid) < target else (lo, mid)
        return 0.5 * (lo + hi)


def panel_mesh(stations, x0, y0, z0, y_shift, n_contour, mirror=False, cap_rings=1):
    """
    Quad panels from the offset table, translated so CO = (x0, y0, z0) maps to (0, y_shift, 0).
    The end caps are rings shrinking towards the section centre: cap_rings = 1 is a single fan of
    triangles, more rings give smaller panels (needed for short waves, Capytaine's lambda/8 rule).
    """
    contours = []
    for x, rows in stations:
        r = np.array(rows)
        # contour: left side from waterline down to keel, then right side up to waterline
        left = [(x, yl, z) for z, yl, _ in r[::-1]]
        right = [(x, yr, z) for z, _, yr in r]
        pts = np.array(left + right)
        pts[:, 0] -= x0
        pts[:, 1] -= y0
        pts[:, 2] -= z0
        # resample to a fixed number of points so neighbouring stations can be joined
        contours.append(_resample(pts, n_contour))
    n = n_contour

    vertices, faces = [], []
    for c in contours:
        vertices.extend(c)
    for i in range(len(contours) - 1):
        for k in range(n - 1):
            a, b = i * n + k, i * n + k + 1
            faces.append([a, b, b + n, a + n])
    # end caps (transom / bow): rings shrinking to the contour's centroid, a fan in the middle
    for i in (0, len(contours) - 1):
        c = contours[i]
        centre = np.array([c[:, 0].mean(), c[:, 1].mean(), 0.0])
        outer = [i * n + k for k in range(n)]
        for r in range(1, cap_rings):
            ring = centre + (1.0 - r / cap_rings) * (c - centre)
            start = len(vertices)
            vertices.extend(ring)
            inner = [start + k for k in range(n)]
            for k in range(n - 1):
                a, b, bi, ai = outer[k], outer[k + 1], inner[k + 1], inner[k]
                faces.append([b, a, ai, bi] if i == 0 else [a, b, bi, ai])
            outer = inner
        ci = len(vertices)
        vertices.append(centre)
        for k in range(n - 1):
            a, b = outer[k], outer[k + 1]
            faces.append([ci, b, a, a] if i == 0 else [ci, a, b, b])
    v = np.array(vertices)
    if mirror:
        v[:, 1] = -v[:, 1]
        faces = [f[::-1] for f in faces]
    v[:, 1] += y_shift
    return cpt.Mesh(vertices=v, faces=np.array(faces))


def _resample(pts, n):
    """n points equally spaced along a polyline."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0.0, d[-1], n)
    return np.column_stack([np.interp(s, d, pts[:, j]) for j in range(3)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stl", help="hull STL")
    p.add_argument("--mass", type=float, default=16.9, help="vessel mass [kg]")
    p.add_argument("--hull-offset", type=float, default=0.0,
                   help="hull centerline to boat centerline [m]; > 0 mirrors the hull (catamaran)")
    p.add_argument("--axes", default="zxy", help="CAD axes for (length, width, height), default zxy")
    p.add_argument("--scale", type=float, default=1e-3, help="CAD units to m (default mm)")
    p.add_argument("--swap-sides", action="store_true",
                   help="put the mirrored hull on +y and the STL hull on -y (turns asymmetric hulls around)")
    p.add_argument("--cog-z", type=float, default=0.0, help="CG height above waterline [m]")
    p.add_argument("--stations", type=int, default=36)
    p.add_argument("--heights", type=int, default=12)
    a = p.parse_args()

    n_hulls = 2 if a.hull_offset > 0 else 1
    hull = OffsetTable(a.stl, axes=a.axes, scale=a.scale)
    target = a.mass / RHO / n_hulls
    z_wl = hull.waterline_for_volume(target)
    print(f"Hull {hull.x_max - hull.x_min:.3f} m long, {hull.height:.3f} m high")
    print(f"Waterline {1000 * (z_wl - hull.z_keel):.0f} mm above keel "
          f"(draft), freeboard {1000 * (hull.z_keel + hull.height - z_wl):.0f} mm, "
          f"{1000 * target:.2f} L per hull")

    stations = hull.sample(z_wl, a.stations, a.heights)
    y0 = hull.y_mid
    n_contour = 2 * a.heights
    side = -1.0 if a.swap_sides else 1.0
    meshes = [panel_mesh(stations, hull.x_mid, y0, z_wl, side * a.hull_offset, n_contour)]
    if n_hulls == 2:
        meshes.append(panel_mesh(stations, hull.x_mid, y0, z_wl, -side * a.hull_offset, n_contour, mirror=True))
    mesh = meshes[0] if n_hulls == 1 else meshes[0].join_meshes(meshes[1])
    body = cpt.FloatingBody(
        mesh=mesh,
        dofs=cpt.rigid_body_dofs(rotation_center=(0.0, 0.0, 0.0)),
        center_of_mass=(0.0, 0.0, a.cog_z),
        mass=a.mass,
        name="hull",
    )
    print(f"{body.mesh.nb_faces} panels, displaced mass from panel mesh: {RHO * body.volume:.2f} kg")

    solver = cpt.BEMSolver()
    problems = [cpt.RadiationProblem(body=body, radiating_dof=d, omega=w, rho=RHO)
                for d in DOFS for w in (0.0, np.inf)]
    A = cpt.assemble_dataset(solver.solve_all(problems, progress_bar=False))["added_mass"]

    print("\nAdded mass, diagonal [kg, kg m^2]:")
    print(f"{'':>8} {'omega=0':>10} {'omega=inf':>10}")
    for d in DOFS:
        print(f"{d:>8} {float(A.sel(omega=0.0, radiating_dof=d, influenced_dof=d)):10.3f} "
              f"{float(A.sel(omega=np.inf, radiating_dof=d, influenced_dof=d)):10.3f}")
    diag = [float(A.sel(omega=0.0, radiating_dof=d, influenced_dof=d)) for d in DOFS]

    C = body.compute_hydrostatic_stiffness(rho=RHO, g=G)
    gm_phi = float(C.sel(influenced_dof="Roll", radiating_dof="Roll")) / (a.mass * G)
    gm_theta = float(C.sel(influenced_dof="Pitch", radiating_dof="Pitch")) / (a.mass * G)
    print(f"\nHydrostatics (CG {a.cog_z} m above waterline): GM_phi {gm_phi:.3f} m, GM_theta {gm_theta:.3f} m")
    print("\nvessel.yaml (omega = 0):")
    print("  added_mass: [" + ", ".join(f"{v:.2f}" for v in diag) + "]")


if __name__ == "__main__":
    main()
