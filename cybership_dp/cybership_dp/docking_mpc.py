"""
Docking by optimal control near a quay, without ROS (the candidates tools/docking_bench.py compares).

One model, two kinds of problem:

    free_time=True    from the current state to the berth, at rest over ground, with the final time free: a
                      plan to the berth. Solved once (plan, then DP tracking), or again every few seconds from
                      the estimated state with the previous plan as the initial guess (a shrinking-horizon MPC)
    free_time=False   a fixed horizon (NMPC): a cost on the distance to the berth, only the first input is used

Model, 3DOF with the velocity through the water (Fossen 2021, Ch. 10.3), x = [north, east, psi, u_r, v_r, r]:

    eta_dot = R(psi) [u_r, v_r, r] + [V_N, V_E, 0]
    M nu_r_dot + C(nu_r) nu_r + D(nu_r) nu_r = B f + R(psi)^T b

M and D are the OCP planner's (trajectory_planner.OCPPlanner.model_from_params); the current (V_N, V_E) and the
bias b (NED) come from the observer, so the plan already allows for what pushes the boat (offset-free).

Quay (Martinsen et al. 2020, Bitar et al. 2020): the hull's four corners stay in the free water in front of the
berth face, a half-plane a.p <= b, at least gap_min from it; the speed of each corner towards the face is limited
by its distance to the face, a.v <= v0 + k_gap * gap, so the boat slows down as it closes in. Both are soft (one
slack per step with an L1 cost), so a disturbance that pushes the boat over a limit does not make the problem
infeasible (and the value function stays continuous: inherent robustness, Krishnamoorthy L7).

What the benchmark settled (results/docking_mpc_plan.md):
    terminal="soft"     the berth is a stiff cost, not an equality (L4: the terminal cost is what gives stability):
                        60/60 first plans against 50/60
    substeps=4, T_max   the yaw damping is stiff at speed; RK4 with 2 substeps went unstable in long plans
    corridor            approach funnel: in front of the berth when close
    heading_funnel      lined up before close: the DP lags in yaw, and turning at the quay let the corners hit it
    gap_min             the plan's distance to the face, = the berth gap minus the DP's tracking tube (L8)
The plan is solved with plain nlpsol so T, the slack and the thrust are variable bounds, and from a geometric guess
(short turn, long turn, straight in, then a two-stage solve without the quay first) or the last plan shifted.

Speed (tools/docking_ocp_speed.py, 12 starts, laptop): compiled to C (compiled=True, cached in ~/.cache/cybership_dp)
a cold plan takes 0.19 s (0.33 s at most) instead of 0.72 (1.26) s; a replan warm with the multipliers too
(dual_warm) 0.04 (0.13) s instead of 0.47 (0.83) s.
"""
import math
import time
from types import SimpleNamespace

import casadi as ca
import numpy as np

from cybership_dp.trajectory_planner import OCPPlanner

DEFAULT_WEIGHTS = {
    # free final time: as the OCP planner, plus the slack
    True: dict(T=1.0, v=30.0, r=5.0, rev=8.0, f=2.0, df=20.0, slack=1e3),
    # fixed horizon: distance to the berth instead of time
    False: dict(p=10.0, p_delta=0.5, psi=20.0, u=5.0, v=30.0, r=5.0, rev=8.0, f=2.0, df=20.0,
                p_N=50.0, psi_N=50.0, vel_N=20.0, slack=1e3),
}


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class DockingOCP:
    def __init__(self, M, coeffs, thrusters, hull=(1.195, 0.93), *, free_time=True, N=40, horizon=10.0,
                 substeps=4, u_max=0.6, u_min=-0.35, v_max=0.3, r_max_deg=25.0, gap_min=0.03, v0=0.05,
                 k_gap=0.25, weights=None, max_iter=300, max_cpu_time=3.0, relax_first=False, terminal="soft",
                 acceptable=False, T_max=60.0, corridor=None, dual_warm=False, heading_funnel=None,
                 speed_funnel=None, compiled=True, mapped=False, compile_flags="-O1"):
        self.M, self.dc = np.asarray(M, float), np.asarray(coeffs, float)
        self.L, self.Bh = float(hull[0]), float(hull[1])
        self.free_time, self.N, self.horizon, self.substeps = bool(free_time), int(N), float(horizon), int(substeps)
        self.bounds = dict(u_max=u_max, u_min=u_min, v_max=v_max, r_max=math.radians(r_max_deg))
        self.gap_min, self.v0, self.k_gap = float(gap_min), float(v0), float(k_gap)
        self.w = {**DEFAULT_WEIGHTS[self.free_time], **(weights or {})}
        self.max_iter, self.max_cpu_time = int(max_iter), float(max_cpu_time)
        self.relax_first = bool(relax_first)
        self.terminal, self.acceptable = terminal, bool(acceptable)
        self.T_max = float(T_max)
        # approach funnel (w0, k1, k2): close to the face the CO must be in front of the berth, |offset along the
        # face| <= w0 + k1 gap + k2 gap^2, so the boat comes in from the front instead of sliding along the quay at
        # gap_min. Quadratic: tight in the last half metre, wide further out (a linear cone was violated by starts
        # far to the side, which then paid slack)
        self.corridor = corridor
        # warm start with the last solution's multipliers too, and a small barrier parameter (Bitar et al. 2020:
        # mu_init 1e-6); a primal-only warm start needed as many iterations as a cold one
        self.dual_warm = bool(dual_warm)
        # heading funnel (h0 [rad], kh [rad/m]): |psi - psi_berth| <= h0 + kh gap, so the boat has turned to the berth
        # heading before it is close. Without it the plan could still be turning in the last half metre; the DP lags
        # in yaw, and the hull corners swung into the quay in the benchmark (docking_bench_v1)
        self.heading_funnel = heading_funnel
        # speed funnel (va0 [m/s], ka [1/s], r0 [rad/s], kr [rad/(s m)]): close to the face the speed ALONG it and
        # the yaw rate are limited too, |v_along| <= va0 + ka gap, |r| <= r0 + kr gap. The DP's tracking tube is
        # largest while the boat moves and turns (Krishnamoorthy L8 S28); sliding in stern first at 0.3 m/s while
        # still turning took a hull corner 3 cm into the quay in a ROS test. Off by default: in the benchmark
        # (docking_bench_v4) it did not move the boat's real corner speed or yaw rate near the quay (0.21 m/s,
        # 7-8 deg/s, the DP's tracking) and gave 2 touches instead of 0 with the team's thrust limits
        self.speed_funnel = speed_funnel
        # compiled: the problem from three small SX functions (one interval's integration, one stage's constraints,
        # one stage's cost) mapped over the N stages in MX, compiled to C and cached (_compiled). The same NLP as
        # inlining it all in one SX graph, but evaluating that graph took 83 % of the solve time and its C code was
        # 53 MB (gcc ran out of patience); mapped it is 3.6 MB and the solves are ~4x faster (0.72 -> 0.19 s cold,
        # tools/docking_ocp_speed.py). Without a C compiler: the inlined SX graph. mapped=True without compiling is
        # only for comparison: the MX evaluation is then 2x slower than the inlined graph.
        self.compiled, self.mapped, self.compile_flags = bool(compiled), bool(mapped), str(compile_flags)
        OCPPlanner._thrusters(self, thrusters)            # B, f_lb, f_ub, fmax, azimuth_idx, nf
        self.corners = np.array([[self.L / 2, self.Bh / 2], [self.L / 2, -self.Bh / 2],
                                 [-self.L / 2, -self.Bh / 2], [-self.L / 2, self.Bh / 2]])
        self._pr = self._build()
        self.last = None                                    # the last solution, for the next warm start

    @classmethod
    def from_vessel_config(cls, cfg, params, **kw):
        M, coeffs = OCPPlanner.model_from_params(params)
        hull = (float(cfg["hull"]["length"]), float(cfg["hull"]["beam"]))
        return cls(M, coeffs, cfg["thrusters"], hull=hull, **kw)

    # ---- model ------------------------------------------------------------------------------------------------

    def _dynamics(self, x, f, vc, bias):
        psi, u, v, r = x[2], x[3], x[4], x[5]
        m11, m22, m33 = self.M
        sabs = lambda z: ca.sqrt(z * z + 1e-4)
        d = [self.dc[0, 0] * u + self.dc[0, 1] * u * sabs(u),
             self.dc[1, 0] * v + self.dc[1, 1] * v * sabs(v),
             self.dc[2, 0] * r + self.dc[2, 1] * r * sabs(r)]
        cor = [-m22 * v * r, m11 * u * r, (m22 - m11) * u * v]
        c, s = ca.cos(psi), ca.sin(psi)
        b_body = [c * bias[0] + s * bias[1], -s * bias[0] + c * bias[1], bias[2]]
        tau = ca.mtimes(ca.DM(self.B), f)
        nu_dot = ca.vertcat(*[(tau[i] + b_body[i] - cor[i] - d[i]) / self.M[i] for i in range(3)])
        eta_dot = ca.vertcat(c * u - s * v + vc[0], s * u + c * v + vc[1], r)
        return ca.vertcat(eta_dot, nu_dot)

    @staticmethod
    def _ground(x, vc):
        """Velocity over ground in BODY (u_g, v_g) of a state with the velocity through the water."""
        c, s = ca.cos(x[2]), ca.sin(x[2])
        return x[3] + c * vc[0] + s * vc[1], x[4] - s * vc[0] + c * vc[1]

    def _build(self):
        """
        The NLP with plain CasADi (nlpsol), so the thrust limits, the slack (>= 0) and the final time are VARIABLE
        bounds: IPOPT keeps every iterate inside them. With Opti they were general constraints, and an iterate
        with T < 0 made the integration blow up (NaN).
        """
        if self.compiled:
            pr = self._build_mapped(compiled=True)
            if pr is not None:
                return pr
        elif self.mapped:
            return self._build_mapped()
        N, ns, w, b, nf = self.N, self.substeps, self.w, self.bounds, self.nf
        X = ca.SX.sym("X", 6, N + 1)
        F = ca.SX.sym("F", nf, N)
        S = ca.SX.sym("S", N + 1)
        T = ca.SX.sym("T")
        P = ca.SX.sym("P", 6 + 3 + 2 + 3 + 3 + nf)
        x0, berth, vc, bias, hp, f_prev = (P[0:6], P[6:9], P[9:11], P[11:14], P[14:17], P[17:17 + nf])
        dt = T / N
        h = dt / ns
        g, lbg, ubg = [], [], []

        def con(expr, lo, hi):
            g.append(expr)
            n = expr.shape[0]
            lbg.extend([lo] * n)
            ubg.extend([hi] * n)

        con(X[:, 0] - x0, 0.0, 0.0)
        for k in range(N):
            xk = X[:, k]
            for _ in range(ns):
                k1 = self._dynamics(xk, F[:, k], vc, bias)
                k2 = self._dynamics(xk + h / 2 * k1, F[:, k], vc, bias)
                k3 = self._dynamics(xk + h / 2 * k2, F[:, k], vc, bias)
                k4 = self._dynamics(xk + h * k3, F[:, k], vc, bias)
                xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            con(X[:, k + 1] - xk, 0.0, 0.0)
        for j, fmax in self.azimuth_idx:
            con((F[j, :] ** 2 + F[j + 1, :] ** 2).T, -ca.inf, fmax ** 2)
        a = hp[:2]
        for k in range(1, N + 1):
            xk = X[:, k]
            ug, vg = self._ground(xk, vc)
            con(ca.vertcat(ug), b["u_min"], b["u_max"])
            con(ca.vertcat(vg), -b["v_max"], b["v_max"])
            c, s_ = ca.cos(xk[2]), ca.sin(xk[2])
            vn, ve = c * ug - s_ * vg, s_ * ug + c * vg     # velocity over ground in NED
            for cx, cy in self.corners:
                rx, ry = c * cx - s_ * cy, s_ * cx + c * cy  # corner from the CO in NED
                gap = hp[2] - (a[0] * (xk[0] + rx) + a[1] * (xk[1] + ry))
                towards = a[0] * (vn - xk[5] * ry) + a[1] * (ve + xk[5] * rx)
                con(ca.vertcat(gap + S[k]), self.gap_min, ca.inf)
                con(ca.vertcat(towards - self.k_gap * gap - S[k]), -ca.inf, self.v0)
            if self.corridor is not None:
                w0, k1, k2 = self.corridor
                gap_co = hp[2] - (a[0] * xk[0] + a[1] * xk[1]) - self.Bh / 2
                along = -a[1] * (xk[0] - berth[0]) + a[0] * (xk[1] - berth[1])
                width = k1 * gap_co + k2 * gap_co ** 2
                con(ca.vertcat(along - width - S[k], -along - width - S[k]), -ca.inf, w0)
            if self.speed_funnel is not None:
                va0, ka, r0, kr = self.speed_funnel
                gap_co = hp[2] - (a[0] * xk[0] + a[1] * xk[1]) - self.Bh / 2
                v_along = -a[1] * vn + a[0] * ve
                con(ca.vertcat(v_along - ka * gap_co - S[k], -v_along - ka * gap_co - S[k]), -ca.inf, va0)
                con(ca.vertcat(xk[5] - kr * gap_co - S[k], -xk[5] - kr * gap_co - S[k]), -ca.inf, r0)
            if self.heading_funnel is not None:
                h0, kh = self.heading_funnel
                gap_co = hp[2] - (a[0] * xk[0] + a[1] * xk[1]) - self.Bh / 2
                dpsi = xk[2] - berth[2]                     # the berth heading is on the start's branch
                con(ca.vertcat(dpsi - kh * gap_co - S[k], -dpsi - kh * gap_co - S[k]), -ca.inf, h0)
        fs = ca.DM(self.fmax)
        # L1 (exact penalty) plus a small quadratic term on the slack
        J = w["slack"] * (ca.sum1(S) + ca.sumsqr(S))
        for k in range(N):
            fn = F[:, k] / fs
            fp = (F[:, k - 1] if k > 0 else f_prev) / fs
            ug, vg = self._ground(X[:, k], vc)
            back = 0.5 * (ug - ca.sqrt(ug ** 2 + 1e-4))     # smooth min(u, 0)
            J += dt * (w["v"] * vg ** 2 + w["r"] * X[5, k] ** 2 + w["rev"] * back ** 2
                       + w["f"] * ca.sumsqr(fn) + w["df"] * ca.sumsqr(fn - fp))
        if self.free_time:
            J += w["T"] * T
            ug, vg = self._ground(X[:, N], vc)
            # the berth heading is given on the branch to turn towards (solve()): one equality, not cos and sin
            # (two equalities for one degree of freedom made the constraint Jacobian rank deficient)
            if self.terminal == "hard":
                con(X[:3, N] - berth, 0.0, 0.0)
                con(ca.vertcat(ug, vg, X[5, N]), 0.0, 0.0)
            else:
                # the berth as a stiff cost (NTNU's planners: no terminal equality, so a disturbed start that
                # cannot reach the berth exactly still has a solution close to it)
                J += w.get("term", 1e3) * ((X[0, N] - berth[0]) ** 2 + (X[1, N] - berth[1]) ** 2
                                           + 2 * (1 - ca.cos(X[2, N] - berth[2])) + ug ** 2 + vg ** 2 + X[5, N] ** 2)
        else:
            dl = w["p_delta"]
            for k in range(1, N + 1):
                d2 = (X[0, k] - berth[0]) ** 2 + (X[1, k] - berth[1]) ** 2
                huber = dl ** 2 * (ca.sqrt(1 + d2 / dl ** 2) - 1)      # quadratic near, linear far
                head = 1 - ca.cos(X[2, k] - berth[2])
                ug, vg = self._ground(X[:, k], vc)
                if k < N:
                    J += dt * (w["p"] * huber + w["psi"] * head + w["u"] * ug ** 2)
                else:
                    J += w["p_N"] * d2 + w["psi_N"] * head + w["vel_N"] * (ug ** 2 + vg ** 2 + X[5, k] ** 2)
        if self.free_time:
            wvar = ca.vertcat(ca.vec(X), ca.vec(F), S, T)
            p_all = P
        else:
            wvar = ca.vertcat(ca.vec(X), ca.vec(F), S)
            p_all = ca.vertcat(P, T)                       # the horizon is a parameter
        return self._finish(wvar, p_all, J, ca.vertcat(*g), lbg, ubg)

    def _finish(self, wvar, p_all, J, g, lbg, ubg, compiled=False):
        """Variable bounds and the IPOPT solvers for the NLP (wvar = [vec X, vec F, S, T]); compiled: from the cached
        shared library, None when it cannot be made."""
        N, b, nf = self.N, self.bounds, self.nf
        # velocity through the water: loose boxes (the limits over ground are the constraints above), yaw rate
        lbx = np.concatenate([np.tile([-np.inf, -np.inf, -np.inf, -1.0, -0.6, -b["r_max"]], N + 1),
                              np.tile(self.f_lb, N), np.zeros(N + 1)])
        ubx = np.concatenate([np.tile([np.inf, np.inf, np.inf, 1.0, 0.6, b["r_max"]], N + 1),
                              np.tile(self.f_ub, N), np.full(N + 1, np.inf)])
        if self.free_time:
            lbx, ubx = np.append(lbx, 1.0), np.append(ubx, self.T_max)
        ipopt = {"print_level": 0, "sb": "yes", "max_iter": self.max_iter, "tol": 1e-4,
                 "max_wall_time": self.max_cpu_time,
                 **({"acceptable_tol": 1e-2, "acceptable_iter": 5, "acceptable_constr_viol_tol": 1e-3}
                    if self.acceptable else {})}
        nlp = {"x": wvar, "p": p_all, "f": J, "g": g}
        if compiled:
            nlp = self._compiled(nlp)
            if nlp is None:
                return None
        # show_eval_warnings off: IPOPT's trial points far out make the integration overflow (Inf); it steps back
        # by itself, and the warnings only hid the real output
        base = {"print_time": False, "show_eval_warnings": False}
        solver = ca.nlpsol("dock", "ipopt", nlp, {**base, "ipopt": ipopt})
        warm = None
        if self.dual_warm:
            warm = ca.nlpsol("dock_warm", "ipopt", nlp, {**base, "ipopt": {
                **ipopt, "warm_start_init_point": "yes", "mu_init": 1e-4, "warm_start_bound_push": 1e-6,
                "warm_start_mult_bound_push": 1e-6, "warm_start_slack_bound_push": 1e-6}})
        return SimpleNamespace(solver=solver, warm=warm, lbx=lbx, ubx=ubx, lbg=np.array(lbg), ubg=np.array(ubg),
                               nx=6 * (N + 1), nF=nf * N)

    def _compiled(self, nlp):
        """
        The NLP's functions as a shared library (the path, for nlpsol), compiled once and cached by a hash of the
        problem in ~/.cache/cybership_dp (CYBERSHIP_OCP_CACHE): 30 s with gcc -O1 on a laptop, several times that on
        the boat's computer, and a moment to load after that. None without a C compiler or if it fails.
        """
        import hashlib
        import os
        import shutil
        import subprocess
        import tempfile
        f = ca.Function("nlp", [nlp["x"], nlp["p"]], [nlp["f"], nlp["g"]])
        key = hashlib.sha1((f.serialize() + self.compile_flags + ca.__version__).encode()).hexdigest()[:16]
        cache = os.path.expanduser(os.environ.get("CYBERSHIP_OCP_CACHE", "~/.cache/cybership_dp"))
        so = os.path.join(cache, f"docking_ocp_{key}.so")
        if os.path.exists(so):
            return so
        cc = shutil.which("gcc") or shutil.which("cc")
        if cc is None:
            return None
        os.makedirs(cache, exist_ok=True)
        print(f"Docking OCP: compiling the problem once ({cc} {self.compile_flags}, ~30-120 s) to {so}", flush=True)
        here = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)                                 # generate_dependencies writes in the working directory
                ca.nlpsol("gen", "ipopt", nlp, {"print_time": False}).generate_dependencies("nlp.c")
                os.chdir(here)
                part = f"{so}.{os.getpid()}.part"
                subprocess.run([cc, "-fPIC", "-shared", self.compile_flags, os.path.join(tmp, "nlp.c"), "-o", part,
                                "-lm"], check=True, capture_output=True)
                os.replace(part, so)                          # atomic: parallel processes may compile at once
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"Docking OCP: compiling failed ({e}); using the uncompiled problem", flush=True)
            return None
        finally:
            os.chdir(here)
        return so

    def _build_mapped(self, compiled=False):
        """The same NLP as the inlined build, from three small SX functions mapped over the N stages."""
        N, ns, w, b, nf = self.N, self.substeps, self.w, self.bounds, self.nf
        x, f, fp = ca.SX.sym("x", 6), ca.SX.sym("f", nf), ca.SX.sym("fp", nf)
        h, vc, bias = ca.SX.sym("h"), ca.SX.sym("vc", 2), ca.SX.sym("bias", 3)
        s, berth, hp = ca.SX.sym("s"), ca.SX.sym("berth", 3), ca.SX.sym("hp", 3)
        # one interval: ns RK4 substeps of length h
        xk = x
        for _ in range(ns):
            k1 = self._dynamics(xk, f, vc, bias)
            k2 = self._dynamics(xk + h / 2 * k1, f, vc, bias)
            k3 = self._dynamics(xk + h / 2 * k2, f, vc, bias)
            k4 = self._dynamics(xk + h * k3, f, vc, bias)
            xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        step = ca.Function("step", [x, f, h, vc, bias], [xk]).map(N)
        # one stage's constraints (k = 1..N), in the inlined build's order
        gs, lo, hi = [], [], []

        def con(expr, l_, u_):
            gs.append(expr)
            lo.append(l_)
            hi.append(u_)
        a = hp[:2]
        ug, vg = self._ground(x, vc)
        con(ug, b["u_min"], b["u_max"])
        con(vg, -b["v_max"], b["v_max"])
        c, s_ = ca.cos(x[2]), ca.sin(x[2])
        vn, ve = c * ug - s_ * vg, s_ * ug + c * vg
        for cx, cy in self.corners:
            rx, ry = c * cx - s_ * cy, s_ * cx + c * cy
            gap = hp[2] - (a[0] * (x[0] + rx) + a[1] * (x[1] + ry))
            towards = a[0] * (vn - x[5] * ry) + a[1] * (ve + x[5] * rx)
            con(gap + s, self.gap_min, ca.inf)
            con(towards - self.k_gap * gap - s, -ca.inf, self.v0)
        gap_co = hp[2] - (a[0] * x[0] + a[1] * x[1]) - self.Bh / 2
        if self.corridor is not None:
            w0, k1c, k2c = self.corridor
            along = -a[1] * (x[0] - berth[0]) + a[0] * (x[1] - berth[1])
            width = k1c * gap_co + k2c * gap_co ** 2
            con(along - width - s, -ca.inf, w0)
            con(-along - width - s, -ca.inf, w0)
        if self.speed_funnel is not None:
            va0, ka, r0, kr = self.speed_funnel
            v_along = -a[1] * vn + a[0] * ve
            con(v_along - ka * gap_co - s, -ca.inf, va0)
            con(-v_along - ka * gap_co - s, -ca.inf, va0)
            con(x[5] - kr * gap_co - s, -ca.inf, r0)
            con(-x[5] - kr * gap_co - s, -ca.inf, r0)
        if self.heading_funnel is not None:
            h0, kh = self.heading_funnel
            dpsi = x[2] - berth[2]
            con(dpsi - kh * gap_co - s, -ca.inf, h0)
            con(-dpsi - kh * gap_co - s, -ca.inf, h0)
        stage_g = ca.Function("stage_g", [x, s, berth, hp, vc], [ca.vertcat(*gs)]).map(N)
        # one interval's running cost (k = 0..N-1), without dt
        fs = ca.DM(self.fmax)
        fn, fpn = f / fs, fp / fs
        back = 0.5 * (ug - ca.sqrt(ug ** 2 + 1e-4))
        l = (w["v"] * vg ** 2 + w["r"] * x[5] ** 2 + w["rev"] * back ** 2 + w["f"] * ca.sumsqr(fn)
             + w["df"] * ca.sumsqr(fn - fpn))
        stage_l = ca.Function("stage_l", [x, f, fp, vc], [l]).map(N)

        X, F, S, T = ca.MX.sym("X", 6, N + 1), ca.MX.sym("F", nf, N), ca.MX.sym("S", N + 1), ca.MX.sym("T")
        P = ca.MX.sym("P", 6 + 3 + 2 + 3 + 3 + nf)
        x0, bP, vcP, biasP, hpP, f_prev = (P[0:6], P[6:9], P[9:11], P[11:14], P[14:17], P[17:17 + nf])
        dt = T / N
        g = [X[:, 0] - x0, ca.vec(X[:, 1:] - step(X[:, :-1], F, dt / ns, vcP, biasP))]
        lbg, ubg = [0.0] * (6 * (N + 1)), [0.0] * (6 * (N + 1))
        for j, fmax in self.azimuth_idx:
            g.append((F[j, :] ** 2 + F[j + 1, :] ** 2).T)
            lbg += [-ca.inf] * N
            ubg += [fmax ** 2] * N
        g.append(ca.vec(stage_g(X[:, 1:], S[1:].T, bP, hpP, vcP)))
        lbg += lo * N
        ubg += hi * N
        J = w["slack"] * (ca.sum1(S) + ca.sumsqr(S))
        J += dt * ca.sum2(stage_l(X[:, :-1], F, ca.horzcat(f_prev, F[:, :-1]), vcP))
        if self.free_time:
            J += w["T"] * T
            ugN, vgN = self._ground(X[:, N], vcP)
            if self.terminal == "hard":
                g += [X[:3, N] - bP, ca.vertcat(ugN, vgN, X[5, N])]
                lbg += [0.0] * 6
                ubg += [0.0] * 6
            else:
                J += w.get("term", 1e3) * ((X[0, N] - bP[0]) ** 2 + (X[1, N] - bP[1]) ** 2
                                           + 2 * (1 - ca.cos(X[2, N] - bP[2])) + ugN ** 2 + vgN ** 2 + X[5, N] ** 2)
            wvar, p_all = ca.vertcat(ca.vec(X), ca.vec(F), S, T), P
        else:
            dl = w["p_delta"]
            for k in range(1, N + 1):
                d2 = (X[0, k] - bP[0]) ** 2 + (X[1, k] - bP[1]) ** 2
                huber = dl ** 2 * (ca.sqrt(1 + d2 / dl ** 2) - 1)
                head = 1 - ca.cos(X[2, k] - bP[2])
                ugk, vgk = self._ground(X[:, k], vcP)
                if k < N:
                    J += dt * (w["p"] * huber + w["psi"] * head + w["u"] * ugk ** 2)
                else:
                    J += w["p_N"] * d2 + w["psi_N"] * head + w["vel_N"] * (ugk ** 2 + vgk ** 2 + X[5, k] ** 2)
            wvar, p_all = ca.vertcat(ca.vec(X), ca.vec(F), S), ca.vertcat(P, T)
        return self._finish(wvar, p_all, J, ca.vertcat(*g), lbg, ubg, compiled=compiled)

    # ---- initial guesses --------------------------------------------------------------------------------------

    def _guess(self, x0, berth, halfplane, vc, standoff=1.0):
        """Straight to a stand-off point `standoff` m out from the berth, turning to the berth heading (berth[2],
        on the branch to turn towards), then in sideways. standoff 0: straight to the berth."""
        a, bb = np.asarray(halfplane[0], float), float(halfplane[1])
        out = -a / np.linalg.norm(a)                       # away from the quay
        stand = np.asarray(berth[:2]) + standoff * out
        p0 = np.asarray(x0[:2])
        d1, d2 = float(np.linalg.norm(stand - p0)), max(standoff, 0.3)
        dpsi = abs(berth[2] - x0[2])
        T = max(4.0, d1 / 0.35 + dpsi / 0.35 + d2 / 0.12)
        N = self.N
        if not self.free_time:
            T = self.horizon
        s = np.linspace(0.0, 1.0, N + 1)
        split = d1 / 0.35 / max(T, 1e-6) if self.free_time else 0.7
        X = np.zeros((6, N + 1))
        for i, si in enumerate(s):
            if si <= split:
                q = si / max(split, 1e-6)
                p = p0 + q * (stand - p0)
            else:
                q = (si - split) / max(1 - split, 1e-6)
                p = stand + q * (np.asarray(berth[:2]) - stand)
            X[0, i], X[1, i] = p
            X[2, i] = x0[2] + min(1.0, si / max(split, 1e-6)) * (berth[2] - x0[2])
        dt = T / N
        vel = np.gradient(X[:2], dt, axis=1) - np.asarray(vc, float)[:, None]
        for i in range(N + 1):
            c, sn = math.cos(X[2, i]), math.sin(X[2, i])
            X[3, i], X[4, i] = c * vel[0, i] + sn * vel[1, i], -sn * vel[0, i] + c * vel[1, i]
        X[5] = np.gradient(np.unwrap(X[2]), dt)
        X[:, 0] = x0
        return X, np.zeros((self.nf, N)), T

    def _shift(self, elapsed, x0):
        """The last solution moved on by `elapsed` seconds, as the initial guess for this one."""
        X, F, T = self.last["X"], self.last["F"], self.last["T"]
        N = self.N
        t_old = np.linspace(0.0, T, N + 1)
        T_new = max(1.0, T - elapsed) if self.free_time else T
        t_new = elapsed + np.linspace(0.0, T_new, N + 1)
        Xs = np.array([np.interp(t_new, t_old, row) for row in X])
        Fs = np.array([np.interp(t_new[:-1], t_old[:-1], row) for row in F])
        Xs[:, 0] = x0
        return Xs, Fs, T_new

    # ---- solve ------------------------------------------------------------------------------------------------

    def solve(self, x0, berth, halfplane, vc=(0.0, 0.0), bias=(0.0, 0.0, 0.0), f_prev=None, elapsed=None):
        """
        x0 = [north, east, psi, u_r, v_r, r] (velocity THROUGH the water), berth = (north, east, psi), halfplane =
        (a, b) of the free water, vc = current NED [m/s], bias = observer bias NED [N, N, N m]. elapsed: seconds
        since the last solve; then that solution, moved on, is the initial guess. Without one, or if it fails, a
        geometric guess turning the short way round and, if that fails too, one turning the long way round.
        Returns a dict: ok, X, F, T, t, slack (largest), cost, solve_time, iterations, guess, and the plan as the
        DP tracking law takes it (eta, nu over ground, nu_dot).
        """
        t0 = time.perf_counter()
        x0 = np.asarray(x0, float)
        berth = np.asarray(berth, float)
        short = x0[2] + wrap(berth[2] - x0[2])
        args = (x0, halfplane, vc, bias, f_prev)
        tries, res, iters, t_eval = [], None, 0, 0.0
        if elapsed is not None and self.last is not None and self.last["ok"]:
            X0, F0, T0 = self._shift(elapsed, x0)
            psi_end = self.last["X"][2, -1]
            psi_end = x0[2] + (psi_end - x0[2]) + 2 * math.pi * round((short - psi_end) / (2 * math.pi))
            tries.append(("warm", psi_end, (X0, F0, T0, self.last.get("lam"))))
        long_ = short - math.copysign(2 * math.pi, short - x0[2]) if abs(short - x0[2]) > 1e-3 else short + 2 * math.pi
        tries += [("short", short, None), ("long", long_, None), ("direct", short, None), ("relaxed", short, None)]
        for name, psi_b, guess in tries:
            b = np.array([berth[0], berth[1], psi_b])
            if guess is None:
                guess = self._guess(x0, b, halfplane, vc, standoff=0.0 if name == "direct" else 1.0)
                if self.relax_first or name == "relaxed":
                    # two stages (as Bitar et al. 2020): first without the quay (the half-plane moved 100 m away),
                    # which converges from a rough guess, then with it, from that solution
                    far = (halfplane[0], halfplane[1] + 100.0 * float(np.linalg.norm(halfplane[0])))
                    pre = self._solve_once(b, guess, x0, far, vc, bias, f_prev)
                    iters += pre["iterations"]
                    if pre["ok"]:
                        guess = (pre["X"], pre["F"], pre["T"])
            res = self._solve_once(b, guess, *args)
            iters += res["iterations"]
            t_eval += res["t_eval"]
            res["guess"] = name
            if res["ok"]:
                break
        res["iterations"], res["solve_time"], res["t_eval"] = iters, time.perf_counter() - t0, t_eval
        if res["ok"]:
            self.last = res
        res.update(self.plan(res, vc))
        return res

    def _solve_once(self, berth, guess, x0, halfplane, vc, bias, f_prev):
        pr, N, nf = self._pr, self.N, self.nf
        X0, F0, T0 = guess[:3]
        lam = guess[3] if len(guess) > 3 else None
        X0 = X0.copy()
        X0[2] = x0[2] + np.unwrap(X0[2] - x0[2])           # the guess's heading on the start's branch
        p = np.concatenate([x0, berth, vc, bias, [halfplane[0][0], halfplane[0][1], halfplane[1]],
                            np.zeros(nf) if f_prev is None else f_prev])
        w0 = np.concatenate([X0.T.ravel(), np.asarray(F0).T.ravel(), np.zeros(N + 1)])
        if self.free_time:
            w0 = np.append(w0, T0)
        else:
            p = np.append(p, self.horizon)
        if lam is not None and pr.warm is not None:
            solver = pr.warm
            sol = solver(x0=w0, p=p, lbx=pr.lbx, ubx=pr.ubx, lbg=pr.lbg, ubg=pr.ubg, lam_x0=lam[0], lam_g0=lam[1])
        else:
            solver = pr.solver
            sol = solver(x0=w0, p=p, lbx=pr.lbx, ubx=pr.ubx, lbg=pr.lbg, ubg=pr.ubg)
        st = solver.stats()
        wopt = np.array(sol["x"]).ravel()
        X = wopt[:pr.nx].reshape(N + 1, 6).T
        F = wopt[pr.nx:pr.nx + pr.nF].reshape(N, nf).T
        S = wopt[pr.nx + pr.nF:pr.nx + pr.nF + N + 1]
        T = float(wopt[-1]) if self.free_time else self.horizon
        ok = bool(st["success"]) or (self.acceptable and st.get("return_status") == "Solved_To_Acceptable_Level")
        t_eval = sum(float(v) for k, v in st.items() if k.startswith("t_wall_nlp_"))   # CasADi's function evaluations
        return dict(ok=ok, X=X, F=F, T=T, t=np.linspace(0.0, T, N + 1), slack=float(np.max(S)),
                    t_eval=t_eval, t_wall=float(st.get("t_wall_total", 0.0)),
                    lam=(np.array(sol["lam_x"]).ravel(), np.array(sol["lam_g"]).ravel()),
                    cost=float(sol["f"]), iterations=int(st.get("iter_count", 0)), status=st.get("return_status", ""))

    def plan(self, res, vc):
        """eta, nu over ground (BODY) and nu_dot on the solution's grid, for trajectory_planner.sample()."""
        X, t = res["X"], res["t"]
        c, s = np.cos(X[2]), np.sin(X[2])
        nu = np.vstack([X[3] + c * vc[0] + s * vc[1], X[4] - s * vc[0] + c * vc[1], X[5]])
        nu_dot = np.gradient(nu, t, axis=1) if res["T"] > 0 else np.zeros_like(nu)
        return dict(eta=X[:3].T, nu=nu.T, nu_dot=nu_dot.T)

    def first_tau(self, res):
        """tau = B f of the first step [X, Y, N] (the NMPC's command)."""
        return self.B @ res["F"][:, 0]
