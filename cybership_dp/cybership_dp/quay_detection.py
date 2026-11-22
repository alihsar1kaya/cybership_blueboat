"""
Quay face from a lidar point cloud, without ROS (cybership_dp/nodes/quay_detector_node.py runs it on /ouster/points).

A quay face seen from the water is a near-vertical wall; seen from above it is a straight line. The fit:

1. points into the boat's level frame: the lidar's mounting (vessel.yaml 'lidar', FLU), then BODY FRD, then roll
   and pitch removed (Rzyx without the yaw), so heights are heights above the water;
2. keep what can be a quay: between z_low above the water (the water plane and calm-water mirror images below it
   go, Benham & Mangelson 2026) and z_high, between r_min (the Ouster's 0.5 m) and r_max from the lidar, and not
   on the boat's own hull;
3. RANSAC for the line with the most points within `threshold` (Wang et al. 2025 fit the berthing plane the same
   way, 2.3 cm / 0.26 deg on a ship), then total least squares (PCA) on its inliers.

Result in BODY FRD from the CO: the unit normal n pointing from the boat to the quay and the distance d, so the
face is n . p = d; its angle alpha = atan2(n_y, n_x); the residual RMS, number of inliers and their span along the
face; and 1-sigma uncertainties of d and alpha from the scatter.
"""
import math

import numpy as np


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


class QuayDetector:
    def __init__(self, lidar_position_flu, lidar_rpy_flu_deg=(0.0, 0.0, 0.0), hull=(1.195, 0.93), z_low=0.05,
                 z_high=2.0, r_min=0.5, r_max=15.0, threshold=0.03, iterations=150, min_inliers=25, min_span=0.8,
                 seed=0):
        r, p, y = (math.radians(v) for v in lidar_rpy_flu_deg)
        self.R_l = rot_z(y) @ rot_y(p) @ rot_x(r)                   # lidar frame -> BODY FLU
        self.t_l = np.asarray(lidar_position_flu, float)
        self.hull = hull
        self.z_low, self.z_high, self.r_min, self.r_max = z_low, z_high, r_min, r_max
        self.threshold, self.iterations = threshold, iterations
        self.min_inliers, self.min_span = min_inliers, min_span
        self.rng = np.random.default_rng(seed)

    def level(self, points_lidar, roll=0.0, pitch=0.0):
        """Lidar points [N x 3] -> level BODY FRD (x forward, y starboard, z down from the CO; roll, pitch NED)."""
        p_flu = points_lidar @ self.R_l.T + self.t_l
        p_frd = p_flu @ FLU_TO_FRD.T
        return p_frd @ (rot_y(pitch) @ rot_x(roll)).T

    def select(self, points_level, points_lidar):
        """The points that can be a quay face (see the module docstring)."""
        rng = np.linalg.norm(points_lidar, axis=1)
        z = points_level[:, 2]
        on_hull = (np.abs(points_level[:, 0]) < self.hull[0] / 2 + 0.1) & (np.abs(points_level[:, 1]) < self.hull[1] / 2 + 0.1)
        keep = (np.isfinite(rng) & (rng >= self.r_min) & (rng <= self.r_max) & (z <= -self.z_low)
                & (z >= -self.z_high) & ~on_hull)
        return points_level[keep, :2]

    def fit(self, points_lidar, roll=0.0, pitch=0.0, prior=None, gate=0.5):
        """
        points_lidar: [N x 3] in the lidar's frame. prior: an expected line (n, d) in BODY FRD; then only points
        within `gate` of it are used (the map and the estimated pose say where the face should be). Returns a dict
        (ok, normal, d, alpha, rms, inliers, span, sigma_d, sigma_alpha, points) or ok False with a reason.
        """
        pts = np.asarray(points_lidar, float)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        xy = self.select(self.level(pts, roll, pitch), pts)
        if prior is not None:
            n_p, d_p = np.asarray(prior[0], float), float(prior[1])
            xy = xy[np.abs(xy @ n_p - d_p) < gate]
        if len(xy) < self.min_inliers:
            return dict(ok=False, reason=f"{len(xy)} points")
        best = None
        for _ in range(self.iterations):
            i, j = self.rng.choice(len(xy), 2, replace=False)
            t = xy[j] - xy[i]
            norm = np.linalg.norm(t)
            if norm < 0.2:
                continue
            n = np.array([t[1], -t[0]]) / norm
            inl = np.abs((xy - xy[i]) @ n) < self.threshold
            count = int(inl.sum())
            if best is None or count > best[0]:
                best = (count, inl)
        if best is None or best[0] < self.min_inliers:
            return dict(ok=False, reason="no line")
        p = xy[best[1]]
        centre = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - centre, full_matrices=False)
        t = vt[0]                                                  # along the face
        n = np.array([t[1], -t[0]])
        d = float(n @ centre)
        if d < 0:                                                   # normal from the boat to the quay
            n, d = -n, -d
        resid = (p - centre) @ n
        rms = float(np.sqrt(np.mean(resid ** 2)))
        along = (p - centre) @ t
        span = float(along.max() - along.min())
        if span < self.min_span:
            return dict(ok=False, reason=f"span {span:.2f} m")
        m = len(p)
        # 1-sigma: the line's offset from the scatter, its angle from the scatter over the span (and a floor for
        # what the points do not show: the lidar's range bias, the mounting)
        sigma_d = math.hypot(rms / math.sqrt(m), 0.01)
        sigma_alpha = math.hypot(rms / max(span, 1e-3) * math.sqrt(12.0 / m), math.radians(0.2))
        return dict(ok=True, normal=n, d=d, alpha=math.atan2(n[1], n[0]), rms=rms, inliers=m, span=span,
                    sigma_d=sigma_d, sigma_alpha=sigma_alpha, points=p)
