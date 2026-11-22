"""
Contact between the hull and quays: fenders on the hull's sides, quays as convex polygons (harbour.yaml).

Points along the hull's outline (the fender line, a rectangle length x beam about the CO) are checked against each
quay. A point inside a quay is pushed out along the normal of the face it is least deep behind, with a spring-damper
that only pushes (a fender cannot pull):

    F_n = max(0, k delta + c delta_dot),      delta = depth behind the face [m], delta_dot its rate

and Coulomb friction along the face, F_t = -mu F_n tanh(v_t / v_s) (smooth, so the integrator does not chatter).
The forces act at the points, so a corner touching the quay also turns the boat. Without this, a quay was only a
constraint in the planner and a boat pushed into it passed through.

Parameters (vessel.yaml 'contact'): stiffness 3000 N/m per point is a soft fender (a 30 kg boat at 0.3 m/s stops in
~3 cm), damping 400 N s/m gives a coefficient of restitution of roughly 0.3-0.5, mu 0.3 for a rubber fender on wood
or concrete. All ESTIMATES: measure the fender (force against compression) before trusting peak forces.
"""
import math

import numpy as np


class QuayContact:
    def __init__(self, quays, length, beam, stiffness=3000.0, damping=400.0, friction=0.3, points_per_side=7,
                 slip_speed=0.02):
        """quays: list of corner lists [(north, east), ...] (convex polygons in NED)."""
        self.k, self.c, self.mu, self.v_s = float(stiffness), float(damping), float(friction), float(slip_speed)
        self.quays = []
        for corners in quays:
            p = np.asarray(corners, float)
            centre = p.mean(axis=0)
            p = p[np.argsort(np.arctan2(p[:, 1] - centre[1], p[:, 0] - centre[0]))]
            normals, offsets = [], []
            for i in range(len(p)):
                a, b = p[i], p[(i + 1) % len(p)]
                t = (b - a) / np.linalg.norm(b - a)
                n = np.array([t[1], -t[0]])
                if n @ (centre - a) > 0:
                    n = -n
                normals.append(n)
                offsets.append(float(n @ a))
            self.quays.append((np.array(normals), np.array(offsets)))
        # the fender line: points along all four sides of the hull rectangle, corners included (BODY, FRD)
        xs = np.linspace(-length / 2, length / 2, points_per_side)
        ys = np.linspace(-beam / 2, beam / 2, max(3, points_per_side // 2))
        pts = [(x, -beam / 2) for x in xs] + [(x, beam / 2) for x in xs]
        pts += [(-length / 2, y) for y in ys[1:-1]] + [(length / 2, y) for y in ys[1:-1]]
        self.points = np.array(pts)
        self.last = []                                 # [(point NED, force NED)] of the last call, for display

    def forces(self, eta, nu):
        """
        eta = (north, east, psi), nu = (u, v, r) BODY. Returns tau = [X, Y, 0, 0, 0, N] in BODY (6DOF, for
        MarineCraft.tau_ext) and keeps the contact points and forces in self.last.
        """
        x, y, psi = eta
        u, v, r = nu
        c, s = math.cos(psi), math.sin(psi)
        R = np.array([[c, -s], [s, c]])
        v_ned = R @ np.array([u, v])
        tau = np.zeros(6)
        self.last = []
        for normals, offsets in self.quays:
            for rb in self.points:
                rn = R @ rb
                p = np.array([x, y]) + rn
                depth = offsets - normals @ p            # > 0 for every face: the point is inside the quay
                if np.any(depth <= 0.0):
                    continue
                j = int(np.argmin(depth))                 # the face it is least deep behind: push out through it
                n, delta = normals[j], float(depth[j])
                vp = v_ned + r * np.array([-rn[1], rn[0]])
                fn = max(0.0, self.k * delta - self.c * float(n @ vp))
                if fn == 0.0:
                    continue
                t = np.array([-n[1], n[0]])
                ft = -self.mu * fn * math.tanh(float(t @ vp) / self.v_s)
                f_ned = fn * n + ft * t
                f_b = R.T @ f_ned
                tau[0] += f_b[0]
                tau[1] += f_b[1]
                tau[5] += rb[0] * f_b[1] - rb[1] * f_b[0]
                self.last.append((p, f_ned))
        return tau


def from_config(harbour_cfg, contact_cfg, length, beam):
    """QuayContact from a harbour YAML dict ('quays') and vessel.yaml's 'contact' section; None without quays."""
    quays = [q["corners"] for q in (harbour_cfg.get("quays") or {}).values()]
    if not quays:
        return None
    c = contact_cfg or {}
    return QuayContact(quays, length, beam, stiffness=c.get("stiffness", 3000.0), damping=c.get("damping", 400.0),
                       friction=c.get("friction", 0.3), points_per_side=int(c.get("points_per_side", 7)))
