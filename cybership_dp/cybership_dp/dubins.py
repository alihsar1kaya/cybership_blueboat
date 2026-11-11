"""
Dubins paths: the shortest path from a pose to a pose for a craft that only moves forward with a minimum
turning radius R (Dubins 1957; Fossen 2021 Ch. 12.4 for path generation from circles and lines). The
path is two arcs joined by a line (LSL, RSR, LSR, RSL) or three arcs (LRL, RLR), and it ends with the
goal heading, so the boat arrives pointing the right way instead of turning on the spot at the goal.

Frame: NED, x north, y east, heading psi from north towards east. A turn with d = +1 increases psi
(to starboard), d = -1 decreases it. Cross-track error is positive to starboard of the path, as in los.py.

    path = DubinsPath.shortest((x0, y0, psi0), (x1, y1, psi1), radius)
    s, (x_p, y_p), pi_h, y_e, kappa = path.project(x, y)
"""

import math


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def mod2pi(a):
    return a % (2 * math.pi)


def unit(a):
    return (math.cos(a), math.sin(a))


class Segment:
    """A line (d = 0) or an arc of radius R turning d = +1 / -1, from point p0 with heading theta0."""

    def __init__(self, p0, theta0, length, d=0, radius=0.0):
        self.p0, self.theta0, self.length, self.d, self.radius = p0, theta0, length, d, radius
        if d:
            u = unit(theta0 + d * math.pi / 2)
            self.center = (p0[0] + radius * u[0], p0[1] + radius * u[1])
            self.phi0 = theta0 - d * math.pi / 2       # angle of p0 seen from the centre

    def at(self, s):
        """(x, y, heading) at arc length s along the segment."""
        if not self.d:
            u = unit(self.theta0)
            return self.p0[0] + s * u[0], self.p0[1] + s * u[1], self.theta0
        phi = self.phi0 + self.d * s / self.radius
        return (self.center[0] + self.radius * math.cos(phi), self.center[1] + self.radius * math.sin(phi),
                wrap(self.theta0 + self.d * s / self.radius))

    def local(self, x, y, s_hint):
        """(s, y_e): arc length of the closest point and cross-track error (positive to starboard)."""
        if not self.d:
            dx, dy = x - self.p0[0], y - self.p0[1]
            c, sn = math.cos(self.theta0), math.sin(self.theta0)
            return c * dx + sn * dy, -sn * dx + c * dy
        wx, wy = x - self.center[0], y - self.center[1]
        s = self.radius * wrap(self.d * (math.atan2(wy, wx) - self.phi0))
        # an arc can sweep more than half a turn: take the lap closest to where we were
        lap = 2 * math.pi * self.radius
        s += lap * round((s_hint - s) / lap)
        return s, self.d * (self.radius - math.hypot(wx, wy))


class DubinsPath:
    def __init__(self, segments, word):
        self.segments = [g for g in segments if g.length > 1e-9]
        self.word = word
        self.length = sum(g.length for g in self.segments)
        self.k = 0          # current segment
        self.s_seg = 0.0    # arc length along it

    @staticmethod
    def candidates(start, goal, R):
        """All Dubins words that exist between the two poses: [(length, word, segments)]."""
        x0, y0, t0 = start
        x1, y1, t1 = goal
        out = []

        def centre(x, y, t, d):
            u = unit(t + d * math.pi / 2)
            return x + R * u[0], y + R * u[1]

        name = {1: "R", -1: "L"}
        for d1 in (1, -1):
            for d2 in (1, -1):
                c1, c2 = centre(x0, y0, t0, d1), centre(x1, y1, t1, d2)
                D = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
                alpha = math.atan2(c2[1] - c1[1], c2[0] - c1[0])
                # circle - line - circle
                if d1 == d2:
                    tl, line = alpha, D
                elif D >= 2 * R:
                    tl, line = alpha + d1 * math.asin(2 * R / D), math.sqrt(D * D - 4 * R * R)
                else:
                    tl = None
                if tl is not None:
                    a1 = R * mod2pi(d1 * (tl - t0))
                    a2 = R * mod2pi(d2 * (t1 - tl))
                    s1 = Segment((x0, y0), t0, a1, d1, R)
                    p = s1.at(a1)
                    s2 = Segment((p[0], p[1]), tl, line)
                    q = s2.at(line)
                    s3 = Segment((q[0], q[1]), tl, a2, d2, R)
                    out.append((a1 + line + a2, name[d1] + "S" + name[d2], [s1, s2, s3]))
                # circle - circle - circle (same turn direction at both ends)
                if d1 == d2 and 1e-9 < D <= 4 * R:
                    for side in (1, -1):
                        a = alpha + side * math.acos(D / (4 * R))
                        c3 = (c1[0] + 2 * R * math.cos(a), c1[1] + 2 * R * math.sin(a))
                        p_phi = math.atan2(c3[1] - c1[1], c3[0] - c1[0])          # touch point seen from c1
                        tp = p_phi + d1 * math.pi / 2
                        q_phi = math.atan2(c2[1] - c3[1], c2[0] - c3[0])          # touch point seen from c3
                        tq = q_phi + math.pi + d1 * math.pi / 2
                        a1 = R * mod2pi(d1 * (tp - t0))
                        am = R * mod2pi(-d1 * (tq - tp))
                        a3 = R * mod2pi(d1 * (t1 - tq))
                        s1 = Segment((x0, y0), t0, a1, d1, R)
                        p = s1.at(a1)
                        s2 = Segment((p[0], p[1]), tp, am, -d1, R)
                        q = s2.at(am)
                        s3 = Segment((q[0], q[1]), tq, a3, d1, R)
                        out.append((a1 + am + a3, name[d1] + name[-d1] + name[d1], [s1, s2, s3]))
        return out

    @classmethod
    def shortest(cls, start, goal, radius):
        length, word, segs = min(cls.candidates(start, goal, radius), key=lambda c: c[0])
        return cls(segs, word)

    @classmethod
    def with_approach(cls, start, goal, radius, approach=0.0):
        """Shortest path to a point `approach` metres before the goal on the goal heading, then a straight
        final approach leg to the goal: the boat lines up on the goal heading before it arrives instead
        of ending the path half way through a turn."""
        x1, y1, t1 = goal
        if approach <= 0.0:
            return cls.shortest(start, goal, radius)
        pre = (x1 - approach * math.cos(t1), y1 - approach * math.sin(t1), t1)
        path = cls.shortest(start, pre, radius)
        path.segments.append(Segment((pre[0], pre[1]), t1, approach))
        path.length += approach
        path.word += "+S"
        return path

    def at(self, s):
        """(x, y, heading) at arc length s from the start (clamped to the path)."""
        s = max(0.0, min(s, self.length))
        for g in self.segments:
            if s <= g.length:
                return g.at(s)
            s -= g.length
        return self.segments[-1].at(self.segments[-1].length)

    def project(self, x, y):
        """
        Progress along the path for position (x, y). Moves forward through the segments as the boat
        passes their ends (never back, so a loop in the path is not cut short).
        Returns (s, (x_p, y_p), pi_h, y_e, kappa): arc length from the start, closest path point,
        path-tangential angle, cross-track error and signed curvature d/R of the current segment.
        """
        while True:
            g = self.segments[self.k]
            s, ye = g.local(x, y, self.s_seg)
            if s >= g.length and self.k < len(self.segments) - 1:
                self.k, self.s_seg = self.k + 1, 0.0
                continue
            break
        self.s_seg = s
        done = sum(h.length for h in self.segments[:self.k])
        px, py, th = g.at(min(max(s, 0.0), g.length)) if g.d else g.at(s)
        kappa = g.d / g.radius if g.d else 0.0
        return done + s, (px, py), th, ye, kappa

    @property
    def turning(self):
        """Total heading change along the path [rad] (sum over the arcs)."""
        return sum(g.length / g.radius for g in self.segments if g.d)

    def remaining(self):
        return self.length - (sum(h.length for h in self.segments[:self.k]) + self.s_seg)
