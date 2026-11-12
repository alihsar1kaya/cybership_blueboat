"""
Optimal-control trajectory planner for a fully actuated surface vessel (Fossen 2021 Ch. 12.1: trajectory
tracking for three or more controls; NTNU milliAmpere: Bitar et al. 2020, Martinsen et al. 2020).

One nonlinear program per goal: from the current state to the goal pose at rest, with the thruster forces
as inputs and the vessel's 3-DOF model as the dynamics. The optimizer decides how to move (turning while
moving, sideways, backing up or turning on the spot) from one cost instead of a set of manoeuvre rules:

    min  w_T T + sum_k dt (w_v v_k^2 + w_r r_k^2 + w_rev min(u_k, 0)^2 + w_xte e_k^2 + w_f |f_k / f_max|^2
                           + w_df |(f_k - f_k-1) / f_max|^2)
         e_k: distance from the straight line start -> goal, weight w_xte; for a turn on the spot the distance
              from the start point, weight w_hold (else it slid 16-22 cm forward while turning)
    s.t. eta_dot = R(psi) nu,   M nu_dot + C(nu) nu + D(nu) nu = B f       (RK4, multiple shooting)
         stern thrusters  -reverse_ratio f_max <= f_i <= f_max,  azimuth  f_x^2 + f_y^2 <= f_max^2
         speed limits on u, v, r;  start = current state;  end = goal pose with nu = 0;  T free

Model (NED, x north, y east, psi from north): M = diag(M_eff), D(nu) = diag(a_i + b_i |nu_i|) fitted to the
vessel's own damping, C(nu) nu = [-m22 v r, m11 u r, (m22 - m11) u v]. Thrust lost at speed is left to the
allocation (it compensates for it), so the planner works with the force the thrusters deliver.

    planner = OCPPlanner.from_vessel_config(cfg, params)
    plan = planner.plan([x, y, psi, u, v, r], [x_g, y_g, psi_g])
    plan["t"], plan["eta"], plan["nu"], plan["nu_dot"], plan["f"]
"""
import math
import time

import casadi as ca
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class OCPPlanner:
    def __init__(self, M, damping_coeffs, thrusters, *, N=40, substeps=2,
                 u_max=0.6, u_min=-0.35, v_max=0.3, r_max_deg=25.0,
                 w_T=1.0, w_v=30.0, w_r=5.0, w_rev=8.0, w_xte=0.0, w_hold=40.0, w_f=2.0, w_df=20.0, max_iter=300):
        """
        M: (m11, m22, m33) incl. added mass. damping_coeffs: ((a_u, b_u), (a_v, b_v), (a_r, b_r)) for
        d_i(x) = a_i x + b_i x|x|. thrusters: vessel.yaml thruster entries (position, type, max_force,
        reverse_ratio, direction_deg).
        """
        self.M = np.asarray(M, float)
        self.dc = np.asarray(damping_coeffs, float)
        self.N, self.substeps = int(N), int(substeps)
        self.bounds = dict(u_max=u_max, u_min=u_min, v_max=v_max, r_max=math.radians(r_max_deg))
        self.w = dict(T=w_T, v=w_v, r=w_r, rev=w_rev, xte=w_xte, hold=w_hold, f=w_f, df=w_df)
        self.max_iter = max_iter
        self._thrusters(thrusters)
        self._build()

    @classmethod
    def from_vessel_config(cls, cfg, params, **kw):
        """Mass and damping from the same model the simulator uses (vessel_model), thrusters from vessel.yaml."""
        from cybership_utilities import vessel_model
        M = np.diag(vessel_model.M_eff_3dof(params))
        craft = vessel_model.make_marine_craft(params)
        coeffs = []
        for i, idx in enumerate((0, 1, 5)):
            xs = np.linspace(0.05, (0.8, 0.4, 0.6)[i], 12)
            ys = []
            for x in xs:
                nu = np.zeros(6)
                nu[idx] = x
                ys.append(-craft.damping(nu)[idx])
            A = np.column_stack([xs, xs * xs])
            coeffs.append(np.linalg.lstsq(A, np.array(ys), rcond=None)[0])
        return cls(M, coeffs, cfg["thrusters"], **kw)

    def _thrusters(self, thrusters):
        """Columns of B (tau = B f) and bounds: one force per fixed thruster, (f_x, f_y) per azimuth."""
        cols, lb, ub, self.azimuth_idx, self.fmax = [], [], [], [], []
        for t in thrusters:
            x, y = float(t["position"][0]), float(t["position"][1])
            fmax, rr = float(t.get("max_force", 67.0)), float(t.get("reverse_ratio", 1.0))
            if t["type"] == "fixed":
                a = math.radians(t.get("direction_deg", 0.0))
                cols.append([math.cos(a), math.sin(a), x * math.sin(a) - y * math.cos(a)])
                lb.append(-rr * fmax); ub.append(fmax); self.fmax.append(fmax)
            else:
                self.azimuth_idx.append((len(cols), fmax))
                cols += [[1.0, 0.0, -y], [0.0, 1.0, x]]
                lb += [-fmax, -fmax]; ub += [fmax, fmax]; self.fmax += [fmax, fmax]
        self.B = np.array(cols).T
        self.f_lb, self.f_ub, self.fmax = np.array(lb), np.array(ub), np.array(self.fmax)
        self.nf = self.B.shape[1]

    def _dynamics(self, x, f):
        psi, u, v, r = x[2], x[3], x[4], x[5]
        m11, m22, m33 = self.M
        sabs = lambda z: ca.sqrt(z * z + 1e-4)            # smooth |z| for the solver
        d = [self.dc[0, 0] * u + self.dc[0, 1] * u * sabs(u),
             self.dc[1, 0] * v + self.dc[1, 1] * v * sabs(v),
             self.dc[2, 0] * r + self.dc[2, 1] * r * sabs(r)]
        cor = [-m22 * v * r, m11 * u * r, (m22 - m11) * u * v]
        tau = ca.mtimes(ca.DM(self.B), f)
        nu_dot = ca.vertcat(*[(tau[i] - cor[i] - d[i]) / self.M[i] for i in range(3)])
        eta_dot = ca.vertcat(ca.cos(psi) * u - ca.sin(psi) * v, ca.sin(psi) * u + ca.cos(psi) * v, r)
        return ca.vertcat(eta_dot, nu_dot)

    def _build(self):
        N, ns = self.N, self.substeps
        opti = ca.Opti()
        X = opti.variable(6, N + 1)
        F = opti.variable(self.nf, N)
        T = opti.variable()
        x0 = opti.parameter(6)
        goal = opti.parameter(3)
        line = opti.parameter(4)        # unit direction start -> goal (n_x, n_y), 1 for a turn on the spot, weight
        dt = T / N
        h = dt / ns
        for k in range(N):
            xk = X[:, k]
            for _ in range(ns):                              # RK4 sub-steps, input held over the interval
                k1 = self._dynamics(xk, F[:, k])
                k2 = self._dynamics(xk + h / 2 * k1, F[:, k])
                k3 = self._dynamics(xk + h / 2 * k2, F[:, k])
                k4 = self._dynamics(xk + h * k3, F[:, k])
                xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == xk)
        b = self.bounds
        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(opti.bounded(b["u_min"], X[3, :], b["u_max"]))
        opti.subject_to(opti.bounded(-b["v_max"], X[4, :], b["v_max"]))
        opti.subject_to(opti.bounded(-b["r_max"], X[5, :], b["r_max"]))
        for i in range(self.nf):
            opti.subject_to(opti.bounded(self.f_lb[i], F[i, :], self.f_ub[i]))
        for j, fmax in self.azimuth_idx:
            opti.subject_to(F[j, :] ** 2 + F[j + 1, :] ** 2 <= fmax ** 2)
        # end at the goal, at rest; heading by its cosine and sine (no wrap-around)
        opti.subject_to(X[0, N] == goal[0])
        opti.subject_to(X[1, N] == goal[1])
        opti.subject_to(ca.cos(X[2, N]) == ca.cos(goal[2]))
        opti.subject_to(ca.sin(X[2, N]) == ca.sin(goal[2]))
        opti.subject_to(X[3:, N] == 0)
        opti.subject_to(opti.bounded(1.0, T, 300.0))
        w, fs = self.w, ca.DM(self.fmax)
        J = w["T"] * T
        for k in range(N):
            fn = F[:, k] / fs
            back = 0.5 * (X[3, k] - ca.sqrt(X[3, k] ** 2 + 1e-4))     # smooth min(u, 0): backing up
            dx, dy = X[0, k] - x0[0], X[1, k] - x0[1]
            xte2 = (1 - line[2]) * (line[0] * dy - line[1] * dx) ** 2 + line[2] * (dx * dx + dy * dy)
            J += dt * line[3] * xte2
            J += dt * (w["v"] * X[4, k] ** 2 + w["r"] * X[5, k] ** 2 + w["rev"] * back ** 2 + w["f"] * ca.sumsqr(fn))
            if k > 0:
                J += dt * w["df"] * ca.sumsqr(fn - F[:, k - 1] / fs)
        opti.minimize(J)
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"print_level": 0, "max_iter": self.max_iter, "sb": "yes", "tol": 1e-4})
        self.opti, self.X, self.F, self.T, self.x0, self.goal, self.J, self.line = opti, X, F, T, x0, goal, J, line

    def _guess(self, start, goal, mode):
        """Initial guesses: 'direct' (heading turns evenly, straight line) or 'forward' (bow first)."""
        N = self.N
        s = np.linspace(0.0, 1.0, N + 1)
        p0, p1 = np.asarray(start[:2], float), np.asarray(goal[:2], float)
        xy = p0[:, None] + (p1 - p0)[:, None] * s
        dist = float(np.hypot(*(p1 - p0)))
        psi0 = float(start[2])
        dpsi_goal = wrap(goal[2] - psi0)
        if mode == 'direct_alt':                            # turn the other way round
            dpsi_goal = dpsi_goal - 2 * np.pi * np.sign(dpsi_goal or 1.0)
            mode = 'direct'
        if mode == 'forward' and dist > 0.5:
            bearing = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            a = psi0 + wrap(bearing - psi0)
            psi = np.where(s < 0.2, psi0 + (a - psi0) * s / 0.2,
                           np.where(s < 0.8, a, a + wrap(goal[2] - a) * (s - 0.8) / 0.2))
            T = dist / 0.4 + (abs(wrap(bearing - psi0)) + abs(wrap(goal[2] - bearing))) / 0.3 + 4.0
        else:
            psi = psi0 + dpsi_goal * s
            T = dist / 0.2 + abs(dpsi_goal) / 0.3 + 4.0
        X = np.zeros((6, N + 1))
        X[0], X[1], X[2] = xy[0], xy[1], psi
        X[:3, 0] = start[:3]
        X[3:, 0] = start[3:]
        return X, T

    def plan(self, start, goal, guesses=None):
        """Best plan over the initial guesses. Returns a dict with t, eta, nu, nu_dot, f, T, cost, ok, solve_time."""
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        best, t0 = None, time.time()
        d = np.asarray(goal[:2]) - np.asarray(start[:2])
        dist = float(np.hypot(*d))
        line = [d[0] / dist, d[1] / dist, 0.0, self.w['xte']] if dist > 0.5 else [1.0, 0.0, 1.0, self.w['hold']]
        if guesses is None:
            guesses = ['forward', 'direct']
            if abs(wrap(goal[2] - start[2])) > np.radians(150):      # near 180 deg: try both ways round
                guesses.append('direct_alt')
        for mode in guesses:
            self.opti.set_value(self.line, line)
            X0, T0 = self._guess(start, goal, mode)
            self.opti.set_value(self.x0, start)
            self.opti.set_value(self.goal, goal)
            self.opti.set_initial(self.X, X0)
            self.opti.set_initial(self.F, 0.0)
            self.opti.set_initial(self.T, T0)
            try:
                sol = self.opti.solve()
                st = sol
                ok = True
            except RuntimeError:
                st = self.opti.debug
                ok = False
            X, F, T, J = np.array(st.value(self.X)), np.array(st.value(self.F)), float(st.value(self.T)), float(st.value(self.J))
            if ok and (best is None or J < best["cost"]):
                best = dict(X=X, F=F, T=T, cost=J, ok=True, guess=mode)
        if best is None:
            return dict(ok=False, solve_time=time.time() - t0)
        N, X, F, T = self.N, best["X"], best["F"], best["T"]
        t = np.linspace(0.0, T, N + 1)
        nu_dot = np.gradient(X[3:], t, axis=1)
        return dict(ok=True, t=t, eta=X[:3].T, nu=X[3:].T, nu_dot=nu_dot.T, f=F.T, T=T, cost=best["cost"],
                    guess=best["guess"], solve_time=time.time() - t0)


def sample(plan, t):
    """
    Reference at time t [s] from the start of a plan, for the DP trajectory-tracking law: pose eta_d
    (heading unwrapped along the plan), eta_d_dot = R(psi) nu and eta_d_ddot = R(psi) (nu_dot + r S nu).
    After the end it holds the final pose at rest. Returns (eta_d, eta_d_dot, eta_d_ddot, finished).
    """
    ts = plan["t"]
    if t >= ts[-1]:
        e = plan["eta"][-1].copy()
        return e, np.zeros(3), np.zeros(3), True
    psi = np.unwrap(plan["eta"][:, 2])
    e = np.array([np.interp(t, ts, plan["eta"][:, 0]), np.interp(t, ts, plan["eta"][:, 1]), np.interp(t, ts, psi)])
    nu = np.array([np.interp(t, ts, plan["nu"][:, i]) for i in range(3)])
    nd = np.array([np.interp(t, ts, plan["nu_dot"][:, i]) for i in range(3)])
    c, s = math.cos(e[2]), math.sin(e[2])
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    S = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    e[2] = wrap(e[2])
    return e, R @ nu, R @ (nd + nu[2] * S @ nu), False
