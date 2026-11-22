"""
Docking geometry, without ROS: a quay as a convex polygon in NED, the berth alongside one of its faces, the free
water in front of that face as half-planes for an optimal-control problem, and the true clearance between the
hull and the quay for tests.

The hull is a rectangle (length x beam, centred on the CO, vessel.yaml hull). Its corners are what can touch a
straight quay face first, so the docking problems keep the corners inside the half-planes (Martinsen et al. 2020,
Bitar et al. 2020: a convex safe set, the vessel's corners inside it).

    quay = Quay([(8.0, -3.0), (9.5, -3.0), (9.5, 3.0), (8.0, 3.0)])
    berth = quay.berth(face=3, side="port", gap=0.10, hull=(1.195, 0.93))   # alongside the face (8,3)-(8,-3)
"""
import math

import numpy as np


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def hull_corners(eta, length, beam):
    """The hull rectangle's corners in NED [4 x 2]: bow starboard, bow port, stern port, stern starboard."""
    x, y, psi = eta[0], eta[1], eta[2]
    c, s = math.cos(psi), math.sin(psi)
    body = np.array([[length / 2, beam / 2], [length / 2, -beam / 2], [-length / 2, -beam / 2],
                     [-length / 2, beam / 2]])
    return np.column_stack([x + c * body[:, 0] - s * body[:, 1], y + s * body[:, 0] + c * body[:, 1]])


class Quay:
    """A convex quay or pier [(north, east), ...] in NED, corners in either order."""

    def __init__(self, corners):
        p = np.asarray(corners, float)
        c = p.mean(axis=0)
        order = np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))   # counter-clockwise in (x, y)
        self.corners = p[order]
        self.centre = c
        # face i runs from corner i to corner i+1; its outward unit normal and offset (n . p <= d inside)
        self.normals, self.offsets = [], []
        for i in range(len(self.corners)):
            a, b = self.corners[i], self.corners[(i + 1) % len(self.corners)]
            t = (b - a) / np.linalg.norm(b - a)
            n = np.array([t[1], -t[0]])
            if n @ (c - a) > 0:
                n = -n
            self.normals.append(n)
            self.offsets.append(float(n @ a))
        self.normals, self.offsets = np.array(self.normals), np.array(self.offsets)

    def face(self, i):
        """(start corner, end corner, outward normal, length) of face i."""
        a, b = self.corners[i], self.corners[(i + 1) % len(self.corners)]
        return a, b, self.normals[i], float(np.linalg.norm(b - a))

    def berth(self, face, side="port", gap=0.10, hull=(1.195, 0.93), along=0.5):
        """
        Berth pose (north, east, heading) alongside face `face`: the hull's port or starboard side parallel to the
        face at `gap` [m] (the fenders), its centre at the fraction `along` of the face.
        """
        a, b, n, _ = self.face(face)
        t = (b - a) / np.linalg.norm(b - a)
        p = a + along * (b - a) + n * (hull[1] / 2 + gap)
        # port side to the quay: the quay is on the boat's left, so the boat heads along t or -t with the
        # normal pointing to its starboard... port = -y_body; the quay lies at -n from the boat
        heading = math.atan2(t[1], t[0])
        starboard = np.array([-math.sin(heading), math.cos(heading)])   # the boat's y axis in NED
        if (side == "port") == (starboard @ -n > 0):     # the quay (at -n from the boat) is on the wrong side
            heading = wrap(heading + math.pi)
        return np.array([p[0], p[1], heading])

    def halfplane(self, face, margin=0.0):
        """The free water in front of face `face` as one half-plane a . p <= b (a = -normal)."""
        n, d = self.normals[face], self.offsets[face]
        return -n, -(d + margin)

    def clearance(self, eta, length, beam):
        """Smallest distance [m] between the hull rectangle and the quay; negative = how deep they overlap (the
        deepest hull corner inside the quay or quay corner inside the hull: with fenders, their compression)."""
        from shapely.geometry import Polygon
        corners = hull_corners(eta, length, beam)
        hull = Polygon(corners)
        quay = Polygon(self.corners)
        if not hull.intersects(quay):
            return hull.distance(quay)
        depth = 0.0
        for p in corners:                                  # hull corners inside the quay
            d = self.offsets - self.normals @ p
            if np.all(d > 0):
                depth = max(depth, float(np.min(d)))
        hull_q = Quay(corners)
        for p in self.corners:                             # quay corners inside the hull
            d = hull_q.offsets - hull_q.normals @ p
            if np.all(d > 0):
                depth = max(depth, float(np.min(d)))
        return -depth

    def face_gap(self, eta, length, beam, face):
        """Distance [m] from the hull's corners to face `face`'s line (the smallest; < 0: through the line)."""
        n, d = self.normals[face], self.offsets[face]
        return float(np.min(hull_corners(eta, length, beam) @ n - d))


DIRECTIONS = {"north": (1.0, 0.0), "south": (-1.0, 0.0), "east": (0.0, 1.0), "west": (0.0, -1.0)}


def face_facing(quay, facing):
    """Index of the quay face whose outward normal is closest to `facing` (a name or a NED vector)."""
    v = np.asarray(DIRECTIONS[facing] if isinstance(facing, str) else facing, float)
    return int(np.argmax(quay.normals @ (v / np.linalg.norm(v))))


def load_harbour(path, hull=(1.195, 0.93), shift=(0.0, 0.0)):
    """
    Quays and berths from a harbour YAML (cybership_config/config/<vessel>/harbour.yaml):
    ({quay name: Quay}, {berth name: dict(quay, face, pose (north, east, heading), halfplane (a, b), gap, side)}).
    shift [m, NED]: the map moved by this much, to test what a map error does (the simulator keeps the true quay).
    """
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    dn, de = float(shift[0]), float(shift[1])
    quays = {name: Quay([(c[0] + dn, c[1] + de) for c in q["corners"]]) for name, q in (cfg.get("quays") or {}).items()}
    berths = {}
    for name, b in (cfg.get("berths") or {}).items():
        quay = quays[b["quay"]]
        face = face_facing(quay, b.get("facing", "south"))
        gap = float(b.get("gap", 0.10))
        pose = quay.berth(face, b.get("side", "port"), gap, hull, along=float(b.get("along", 0.5)))
        berths[name] = dict(quay=b["quay"], face=face, pose=pose, halfplane=quay.halfplane(face), gap=gap,
                            side=b.get("side", "port"))
    return quays, berths
