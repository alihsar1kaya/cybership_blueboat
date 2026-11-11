"""
Constrained thrust allocation with slack (Fossen 2021, Sec. 11.2.3), in two stages per time step:

  1. Direction:  unconstrained extended-thrust solution u_e = T_e^+ tau gives the direction each
                 azimuth should point, alpha* = atan2(F_y, F_x) (eq. 11.23-11.24). The propeller turns
                 both ways, so alpha* + pi with reverse thrust gives the same force: the set-point is
                 whichever is closer to the actual angle (no half turn when the demand changes sign).
                 The azimuth turns towards it at its own rate, so alpha* is only a set-point.

  2. Forces:     with every azimuth at the angle it ACTUALLY has, solve the convex QP

                     min  f' W f + s' Q s
                     s.t. T(alpha) f = tau + s
                          f_min <= f <= f_max

                 s is the slack: the part of tau the thrusters cannot give. Q >> W, and Q per DOF
                 sets the priority when the demand is too large (e.g. keep heading, give up sway).

Thrust loss at speed (compensate_inflow): a propeller gives less thrust when the water already flows
into it (actuators.py, linear K_T(J)). The forces from stage 2 are what the thrusters should deliver; the
command is raised (or lowered) so that they do at the boat's speed through the water. Without it the
loss acts as a drag the controller does not know about, about 10 N at 0.6 m/s with the stern thrusters
working against each other, and the boat overshoots when it stops.

Plain Python (numpy/scipy), only surge, sway and yaw (X, Y, N). Thruster data from vessel.yaml.
"""

import math

import numpy as np
from scipy.optimize import lsq_linear

from cybership_utilities.marine_craft.actuators import bollard_thrust_for


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _column(direction, x, y):
    """Generalized force [X, Y, N] from a unit thrust along 'direction' [rad] at (x, y) (NED body)."""
    c, s = math.cos(direction), math.sin(direction)
    return np.array([c, s, x * s - y * c])


class ConstrainedAllocator:
    def __init__(self, thrusters, slack_weights=(1e4, 1e4, 1e5), angle_threshold=1.0, zero_thrust_speed=None):
        """
        thrusters: vessel.yaml thruster entries (type, position, max_force, reverse_ratio,
                   direction_deg for fixed thrusters, allocation.weight).
        slack_weights: Q = diag(qX, qY, qN).
        angle_threshold [N]: only move an azimuth set-point when it is asked for at least this much
                   force, so it does not swing around when the demand is near zero; or when the angle
                   it holds leaves force undelivered (a small sway demand only the azimuth can give).
        zero_thrust_speed [m/s]: u_0 of the thrusters (thruster_model in vessel.yaml) for
                   compensate_inflow; None = thrust independent of speed.
        """
        self.zero_thrust_speed = zero_thrust_speed
        self.thrusters = list(thrusters)
        self.Q = np.asarray(slack_weights, dtype=float)
        self.angle_threshold = float(angle_threshold)
        self._short = False      # the last allocation could not deliver the demand (slack)

        cols_e, w_e = [], []
        self.lb, self.ub, self.w = [], [], []
        self.alpha_cmd = {}
        for t in self.thrusters:
            x, y = t["position"][0], t["position"][1]
            f_max = float(t.get("max_force", np.inf))
            weight = float((t.get("allocation") or {}).get("weight", 1.0))
            self.w.append(weight)
            if t["type"] == "fixed":
                d = math.radians(t.get("direction_deg", 0.0))
                cols_e.append(_column(d, x, y)); w_e.append(weight)
                self.lb.append(-float(t.get("reverse_ratio", 1.0)) * f_max)
            else:
                cols_e += [_column(0.0, x, y), _column(math.pi / 2, x, y)]; w_e += [weight, weight]
                # the azimuth propeller turns both ways: negative = reverse along its angle
                self.lb.append(-float(t.get("reverse_ratio", 1.0)) * f_max)
                self.alpha_cmd[t["name"]] = 0.0
            self.ub.append(f_max)
        self.T_e = np.column_stack(cols_e)                    # extended thrust configuration, 3 x n_e
        W_inv = np.diag(1.0 / np.asarray(w_e))
        self.T_e_pinv = W_inv @ self.T_e.T @ np.linalg.pinv(self.T_e @ W_inv @ self.T_e.T)  # weighted pinv (pinv: also for fewer than 3 DOF, e.g. no bow thruster)
        self.lb, self.ub, self.w = map(np.asarray, (self.lb, self.ub, self.w))

    def T(self, angles):
        """Configuration matrix T(alpha), one column per thruster, azimuths at the given angles."""
        cols = []
        for t in self.thrusters:
            x, y = t["position"][0], t["position"][1]
            d = math.radians(t.get("direction_deg", 0.0)) if t["type"] == "fixed" else angles[t["name"]]
            cols.append(_column(d, x, y))
        return np.column_stack(cols)

    def desired_angles(self, tau, angles=None):
        """
        Stage 1: azimuth set-points from the unconstrained extended-thrust solution, alpha* or
        alpha* + pi (reverse thrust), whichever is closer to the actual angle in 'angles'.
        """
        u_e = self.T_e_pinv @ tau
        k = 0
        for t in self.thrusters:
            if t["type"] == "fixed":
                k += 1
                continue
            fx, fy = u_e[k], u_e[k + 1]
            if math.hypot(fx, fy) >= self.angle_threshold or self._short:
                a = math.atan2(fy, fx)
                current = (angles or {}).get(t["name"], self.alpha_cmd[t["name"]])
                flipped = _wrap(a + math.pi)
                if abs(_wrap(flipped - current)) < abs(_wrap(a - current)):
                    a = flipped
                self.alpha_cmd[t["name"]] = a
            k += 2
        return dict(self.alpha_cmd)

    def allocate(self, tau, angles):
        """
        tau:    desired [X, Y, N]
        angles: actual azimuth angles {name: alpha [rad]}
        Returns (forces f [one per thruster, along its actual direction], alpha set-points, slack s).
        """
        tau = np.asarray(tau, dtype=float).reshape(3)
        alpha_cmd = self.desired_angles(tau, angles)
        T = self.T(angles)
        # min |sqrt(Q) (T f - tau)|^2 + |sqrt(W) f|^2  subject to bounds  (= the QP above)
        A = np.vstack([np.sqrt(self.Q)[:, None] * T, np.diag(np.sqrt(self.w))])
        b = np.concatenate([np.sqrt(self.Q) * tau, np.zeros(len(self.thrusters))])
        f = lsq_linear(A, b, bounds=(self.lb, self.ub), method="bvls").x
        slack = T @ f - tau
        self._short = bool(np.any(np.abs(slack) > 0.5))
        return f, alpha_cmd, slack

    def compensate_inflow(self, f, angles, nu_r):
        """
        Thrust to command (bollard, what the driver maps to PWM) so that each thruster delivers f at the
        velocity through the water nu_r = [u_r, v_r, r]. The inflow of a thruster at (x, y) pointing along
        alpha is (u_r - r y) cos(alpha) + (v_r + r x) sin(alpha). Clipped to the thruster limits.
        """
        if self.zero_thrust_speed is None:
            return np.asarray(f, dtype=float)
        u, v, r = nu_r
        out = []
        for t, f_i in zip(self.thrusters, f):
            x, y = t["position"][0], t["position"][1]
            a = math.radians(t.get("direction_deg", 0.0)) if t["type"] == "fixed" else angles[t["name"]]
            inflow = (u - r * y) * math.cos(a) + (v + r * x) * math.sin(a)
            out.append(bollard_thrust_for(float(f_i), inflow, float(t.get("max_force", np.inf)),
                                          float(t.get("reverse_ratio", 1.0)), self.zero_thrust_speed))
        return np.clip(out, self.lb, self.ub)
