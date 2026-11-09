#!/usr/bin/env python3
"""
Projected wind areas from a CAD mesh (Fossen 2021 Sec. 10.1, TTK4190 L7 slides 5 and 10):

    A_Fw  frontal projected area above the waterline [m^2]  (seen from ahead, y-z plane)
    A_Lw  lateral projected area above the waterline [m^2]  (seen from the side, x-z plane)
    H_Fw, H_Lw  height of the area centroids above the waterline [m]
    s_L   longitudinal centroid of A_Lw from the main section (L_oa / 2), positive forward [m]
    L_oa  length overall [m]

The mesh is cut at the waterline and every triangle above it is projected onto the plane; the union of
the projected triangles is the silhouette (shapely), so overlapping parts are counted once.

    python3 cybership_config/tools/wind_areas.py cybership_description/meshes/microampere/microampere.stl \
        --keel-z -0.376 --draft 0.274
"""
import argparse

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union


def silhouette(tri2d):
    polys = [Polygon(t) for t in tri2d]
    polys = [p for p in polys if p.is_valid and p.area > 1e-10]
    return unary_union(polys)


def wind_areas(mesh, z_wl):
    """mesh in metres with x forward and z up; z_wl = waterline height in mesh coordinates."""
    above = mesh.slice_plane(plane_origin=[0, 0, z_wl], plane_normal=[0, 0, 1])
    tri = above.triangles                         # (n, 3, 3)
    front = silhouette(tri[:, :, [1, 2]])          # (y, z)
    side = silhouette(tri[:, :, [0, 2]])           # (x, z)
    x_min, x_max = mesh.bounds[0][0], mesh.bounds[1][0]
    return {
        "A_Fw": front.area,
        "A_Lw": side.area,
        "H_Fw": front.centroid.y - z_wl,
        "H_Lw": side.centroid.y - z_wl,
        "s_L": side.centroid.x - 0.5 * (x_min + x_max),
        "L_oa": x_max - x_min,
        "top_above_wl": mesh.bounds[1][2] - z_wl,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--keel-z", type=float, required=True, help="z of the keel in the mesh [m]")
    ap.add_argument("--draft", type=float, required=True, help="draft below the waterline [m]")
    ap.add_argument("--scale", type=float, default=1.0, help="mesh units to metres (e.g. 0.001 for mm)")
    a = ap.parse_args()
    m = trimesh.load(a.stl)
    m.apply_scale(a.scale)
    r = wind_areas(m, a.keel_z + a.draft)
    for k, v in r.items():
        print(f"{k:13s} {v:8.4f}")
    print("\nvessel.yaml (model.wind):")
    print(f"  areas: {{A_Fw: {r['A_Fw']:.3f}, A_Lw: {r['A_Lw']:.3f}, H_Fw: {r['H_Fw']:.3f}, "
          f"H_Lw: {r['H_Lw']:.3f}, s_L: {r['s_L']:.3f}, L_oa: {r['L_oa']:.3f}}}")
