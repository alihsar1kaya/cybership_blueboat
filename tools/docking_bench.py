#!/usr/bin/env python3
"""
Docking benchmark: candidate docking strategies against the same plant, scenarios and metrics, without ROS.

Plant: the simulator's 6DOF model (vessel_model.make_marine_craft) with mass and damping scaled by random factors
(model error), the thruster models (time constant, azimuth rate, thrust loss at speed, 0.05 s command latency),
ocean current and wind (Blendermann). Navigation: our DP Kalman filter (AsyncObserver) on RTK GNSS (5 Hz, 2 cm,
0.1 s late) and a compass (50 Hz, 0.5 deg). Control: the strategies give either a reference for our model-based DP
law (the position controller's, 10 Hz) or a force directly; our constrained allocation (20 Hz) turns it into
thruster commands. Solve times are measured and the result is applied that much later (times --cpu-factor), as on
the boat, where the solver runs while the boat moves.

    python3 tools/docking_bench.py                          # plan_once and replan_event, 36 runs each (~2 min)
    python3 tools/docking_bench.py --quick                  # 2 environments x 2 starts, to try a change (~30 s)
    python3 tools/docking_bench.py --strategies plan_once,replan,nmpc --out ~/ros_ws/results/docking_bench_v1
    python3 tools/docking_bench.py --strategies replan --thrusters team --cpu-factor 3 --map-err -0.1 \
        --multipath --lidar --out ...                       # the real boat's limits, a slow CPU, a quay map 10 cm
                                                            # off, RTK multipath near the quay, lidar quay fit

Variants (the known problems at a quay, from the literature and the TURLab inventory):
    --thrusters team   the team's limits: aft +52.8 / -26.7 N, azimuth 51.5 N forward only (thrust >= 0)
    --cpu-factor 3     the LattePanda: solves take 3x as long before they reach the boat
    --map-err E        the controller's quay is E m further out than the real one (E < 0: behind its face)
    --multipath        within 3 m of the quay the GNSS error grows (0.1 m noise + 0.15 m slow bias, reported
                       as 0.05 m): what RTK does near structures
    --lidar            the quay face measured relative to the boat (a RANSAC line from the Ouster: 2 cm, 0.26 deg,
                       10 Hz, only beyond its 0.5 m minimum range), turned into NED with the estimated pose: the
                       berth then moves with the boat's own position error
    --lidar-obs        the same measurement into the observer instead (DPKalmanFilter.correct_line against the
                       mapped face, and the face angle as a heading): the position across the quay then comes from
                       the lidar and only the position along it from GNSS

Writes <out>/runs.json (one record per run) and prints a summary table per strategy and environment.
"""
import os

# one thread per run: 12 parallel IPOPT/BLAS runs with a thread pool each made a 0.5 s solve take 15 s
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, f"{REPO}/cybership_dp")                  # the docking modules, also before a colcon build
from cybership_dp import trajectory_planner  # noqa: E402
from cybership_dp.constrained_allocation import ConstrainedAllocator  # noqa: E402
from cybership_dp.docking import Quay, hull_corners, wrap  # noqa: E402
from cybership_dp.docking_mpc import DockingOCP  # noqa: E402
from cybership_utilities import vessel_model  # noqa: E402
from cybership_utilities.marine_craft.actuators import make_thruster  # noqa: E402
from cybership_utilities.marine_craft.contact import QuayContact  # noqa: E402
from cybership_utilities.marine_craft.environment import OceanCurrent, Wind  # noqa: E402
from cybership_utilities.observer import AsyncObserver, DPKalmanFilter  # noqa: E402

DT = 0.01                       # plant step [s]
QUAY = [(8.0, -3.0), (9.5, -3.0), (9.5, 3.0), (8.0, 3.0)]   # small_harbour's quay, 8 m north
FACE, SIDE, GAP = 3, "port", 0.10                          # alongside the south face, port side, 10 cm fenders
ENVIRONMENTS = {
    "calm": {},
    "current_on": dict(current=(0.10, 0.0)),               # (speed [m/s], towards [deg]): onto the quay
    "current_along": dict(current=(0.10, 90.0)),
    "current_off": dict(current=(0.10, 180.0)),
    "wind_on": dict(wind=(6.0, 0.0)),                      # (speed [m/s], towards [deg]), with gusts
    "wind_along_current_on": dict(wind=(6.0, 270.0), current=(0.05, 0.0)),
}


# ---------------------------------------------------------------------------------------------------------------
# plant, sensors, observer
# ---------------------------------------------------------------------------------------------------------------

class Plant:
    """The simulator's physics: 6DOF craft, thrusters with latency, current and wind."""

    def __init__(self, cfg, params, eta0, env, rng, k_mass=1.0, k_damp=1.0, latency=0.05):
        p = dict(params)
        p["MA"] = params["MA"] * k_mass
        p["D"] = params["D"] * k_damp
        p["B_eq"] = params["B_eq"] * k_damp
        nl = dict(params["nonlinear"])
        for k in ("surge_quadratic", "yaw_quadratic"):
            nl[k] = nl[k] * k_damp
        p["nonlinear"] = nl
        self.craft = vessel_model.make_marine_craft(p, eta0=[eta0[0], eta0[1], 0, 0, 0, eta0[2]])
        tm = cfg.get("thruster_model") or {}
        self.thrusters = cfg["thrusters"]
        self.act = {t["name"]: make_thruster({"zero_thrust_speed": tm.get("zero_thrust_speed"), **t})
                    for t in self.thrusters}
        speed, direction = env.get("current", (0.0, 0.0))
        self.current = OceanCurrent(speed=speed, direction_deg=direction)
        self.wind = None
        if "wind" in env:
            w = cfg["model"]["wind"]
            self.wind = Wind(w["areas"], w.get("coefficients", "speed_boat"), w.get("rho_a", 1.225),
                             speed=env["wind"][0], direction_deg=env["wind"][1], mu=0.05, sigma=0.5,
                             seed=int(rng.integers(1 << 30)))
        self.latency, self.queue, self.t = latency, deque(), 0.0
        # fenders against the quay (as the simulator, vessel.yaml 'contact'); the force in self.contact_force
        cc = cfg.get("contact") or {}
        self.contact = QuayContact([QUAY], float(cfg["hull"]["length"]), float(cfg["hull"]["beam"]),
                                   stiffness=cc.get("stiffness", 3000.0), damping=cc.get("damping", 400.0),
                                   friction=cc.get("friction", 0.3))
        self.contact_force = 0.0

    def command(self, cmds):
        """cmds {name: (fx, fy)} as the allocation sends them; they reach the thrusters `latency` s later."""
        self.queue.append((self.t + self.latency, cmds))

    def step(self):
        while self.queue and self.queue[0][0] <= self.t + 1e-9:
            for name, (fx, fy) in self.queue.popleft()[1].items():
                self.act[name].set_command(fx, fy)
        self.current.step(DT)
        c = self.craft
        c.current_ned = self.current.velocity_ned()
        tau_ext = np.zeros(6)
        if self.wind is not None:
            self.wind.step(DT)
            tau_ext = tau_ext + self.wind.loads(c.eta, c.nu)
        tau_c = self.contact.forces([c.eta[0], c.eta[1], c.eta[5]], [c.nu[0], c.nu[1], c.nu[5]])
        self.contact_force = math.hypot(tau_c[0], tau_c[1])
        c.tau_ext = tau_ext + tau_c
        psi = c.eta[5]
        cs, sn = math.cos(psi), math.sin(psi)
        vc = c.current_ned
        u_r, v_r, r = c.nu[0] - (cs * vc[0] + sn * vc[1]), c.nu[1] - (-sn * vc[0] + cs * vc[1]), c.nu[5]
        tau = np.zeros(6)
        self.forces = {}
        for t in self.thrusters:
            a = self.act[t["name"]]
            a.step(DT)
            ang = getattr(a, "angle", math.radians(t.get("direction_deg", 0.0)))
            x, y = t["position"][0], t["position"][1]
            a.inflow = (u_r - r * y) * math.cos(ang) + (v_r + r * x) * math.sin(ang)
            fx, fy, _ = a.output()
            if t["type"] == "fixed":
                d = math.radians(t.get("direction_deg", 0.0))
                fx, fy = fx * math.cos(d), fx * math.sin(d)
            tau[0] += fx
            tau[1] += fy
            tau[5] += x * fy - y * fx
            self.forces[t["name"]] = math.hypot(fx, fy)
        c.step(tau=tau, dt=DT)
        self.t += DT

    @property
    def eta3(self):
        e = self.craft.eta
        return np.array([e[0], e[1], e[5]])

    @property
    def nu3(self):
        n = self.craft.nu
        return np.array([n[0], n[1], n[5]])

    def azimuth_angles(self):
        return {t["name"]: self.act[t["name"]].angle for t in self.thrusters if t["type"] == "azimuth"}


class Navigation:
    """RTK GNSS + compass into our DP Kalman filter, as dp_observer_node runs it."""

    def __init__(self, params, rng, sigma_gnss=0.02, gnss_rate=5.0, gnss_delay=0.1, sigma_psi_deg=0.5,
                 multipath=None):
        craft = vessel_model.make_marine_craft(params)
        idx = [0, 1, 5]
        damping = lambda nu: -craft.damping(np.array([nu[0], nu[1], 0, 0, 0, nu[2]]))[idx]
        kf = DPKalmanFilter(vessel_model.M_eff_3dof(params), vessel_model.D_3dof(params),
                            q_bias=[0.2, 0.2, 0.02], q_nu=[0.05, 0.05, 0.005], damping=damping,
                            current={"q": 1e-5, "p0": 0.04})
        self.kf = kf
        self.obs = AsyncObserver(kf, 0.02, math.radians(sigma_psi_deg))
        self.rng, self.sigma_gnss, self.sigma_psi = rng, sigma_gnss, math.radians(sigma_psi_deg)
        self.gnss_period, self.gnss_delay = 1.0 / gnss_rate, gnss_delay
        self.pending = deque()
        self.next_gnss, self.next_step = 0.0, 0.0
        # multipath near the quay: (quay, face, hull) -> extra noise and a slow Gauss-Markov bias within 3 m
        self.multipath, self.mp_bias = multipath, np.zeros(2)

    def update(self, t, plant, tau_allocated):
        if t + 1e-9 >= self.next_gnss:
            e = plant.eta3
            sigma, extra = self.sigma_gnss, np.zeros(2)
            if self.multipath is not None:
                quay, face, hull = self.multipath
                a = math.exp(-self.gnss_period / 20.0)               # bias time constant 20 s, 0.15 m
                self.mp_bias = a * self.mp_bias + math.sqrt(1 - a * a) * 0.15 * self.rng.normal(size=2)
                if quay.face_gap(e, hull[0], hull[1], face) < 3.0:
                    extra = self.mp_bias + self.rng.normal(0, 0.10, 2)
                    sigma = 0.05                                     # what the receiver claims
            self.pending.append((t + self.gnss_delay, t, e[0] + self.rng.normal(0, self.sigma_gnss) + extra[0],
                                 e[1] + self.rng.normal(0, self.sigma_gnss) + extra[1], sigma))
            self.next_gnss += self.gnss_period
        while self.pending and self.pending[0][0] <= t + 1e-9:
            t_rx, t_meas, n, e, sigma = self.pending.popleft()
            self.obs.gnss(t, t_meas, n, e, sigma)
        if t + 1e-9 >= self.next_step:
            self.obs.heading(t, wrap(plant.eta3[2] + self.rng.normal(0, self.sigma_psi)))
            self.obs.step(t, tau_allocated)
            self.next_step += 0.02

    @property
    def ready(self):
        return self.obs.ready

    def lidar_line(self, t, plant, n_map, d_map, sigma_rho=0.02, sigma_ang=math.radians(0.26)):
        """10 Hz: the lidar's distance to the face and the face's angle in BODY, as observer measurements."""
        if not self.ready or t + 1e-9 < getattr(self, "next_lidar", 0.0):
            return
        self.next_lidar = t + 0.1
        e = plant.eta3
        c, s = math.cos(e[2]), math.sin(e[2])
        p_l = e[:2] + np.array([c * LIDAR_POS[0] - s * LIDAR_POS[1], s * LIDAR_POS[0] + c * LIDAR_POS[1]])
        n_true, d_true = self.true_face
        rho = float(n_true @ p_l - d_true)
        # the Ouster's 0.5 m minimum range is per point: closer than that it still sees the face obliquely, fore and
        # aft and below it (the lidar is ~0.5 m above the water), so the line can be fitted down to ~0.2 m, with
        # fewer points (more noise). To be checked with the Gazebo lidar and real data (plan step 3-4)
        if not 0.2 <= rho <= 15.0:
            return
        sigma_rho = sigma_rho * (1.0 + max(0.0, 0.5 - rho) / 0.3)
        rho_m = rho + self.rng.normal(0, sigma_rho)
        ang_m = math.atan2(n_true[1], n_true[0]) - e[2] + self.rng.normal(0, sigma_ang)
        psi_m = wrap(math.atan2(n_map[1], n_map[0]) - ang_m)       # heading relative to the mapped face
        self.kf.correct_compass(psi_m, sigma_ang)
        psi = self.kf.eta[2]
        c, s = math.cos(psi), math.sin(psi)
        lever = np.array([c * LIDAR_POS[0] - s * LIDAR_POS[1], s * LIDAR_POS[0] + c * LIDAR_POS[1]])
        self.kf.correct_line(n_map, rho_m + d_map - float(n_map @ lever), sigma_rho * 1.5)

    def estimate(self):
        kf = self.kf
        cur = np.asarray(kf.current, float) if kf.nc else np.zeros(2)
        return dict(eta=np.array(kf.eta, float), nu=np.array(kf.nu, float), bias=np.array(kf.bias, float), vc=cur)


# ---------------------------------------------------------------------------------------------------------------
# DP law and allocation (as position_control_node and vessel_force_controller)
# ---------------------------------------------------------------------------------------------------------------

class DPLaw:
    """position_control_node._model_based_tau with bias feedforward and the current in the feedforward."""

    def __init__(self, params, omega_n=(0.45, 0.6, 1.0), zeta=(0.9, 0.9, 0.9)):
        self.M = vessel_model.M_eff_3dof(params)
        m, d = np.diag(self.M), np.diag(vessel_model.D_3dof(params))
        w, z = np.asarray(omega_n), np.asarray(zeta)
        self.Kp, self.Kd = m * w ** 2, np.maximum(2 * z * m * w - d, 0.0)
        self.craft = vessel_model.make_marine_craft(params)

    def tau(self, est, eta_d, eta_d_dot, eta_d_ddot):
        psi = est["eta"][2]
        c, s = math.cos(psi), math.sin(psi)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        e = eta_d - est["eta"]
        e[2] = wrap(e[2])
        nu_d, nu_d_dot = R.T @ eta_d_dot, R.T @ eta_d_ddot
        nu_c = R.T @ np.array([est["vc"][0], est["vc"][1], 0.0])
        nu_c_dot = -nu_d[2] * np.array([-nu_c[1], nu_c[0], 0.0])
        nu_rd, nu_rd_dot = nu_d - nu_c, nu_d_dot - nu_c_dot
        damping_ff = -self.craft.damping(np.array([nu_rd[0], nu_rd[1], 0, 0, 0, nu_rd[2]]))[[0, 1, 5]]
        return self.M @ nu_rd_dot + damping_ff + self.Kp * (R.T @ e) + self.Kd * (nu_d - est["nu"]) - R.T @ est["bias"]


class Allocation:
    def __init__(self, cfg):
        alloc = cfg.get("allocation", {}) or {}
        u0 = (cfg.get("thruster_model", {}) or {}).get("zero_thrust_speed")
        self.thrusters = cfg["thrusters"]
        self.al = ConstrainedAllocator(self.thrusters, slack_weights=alloc.get("slack_weights", (1e4, 1e4, 1e5)),
                                       angle_threshold=alloc.get("angle_threshold", 1.0),
                                       zero_thrust_speed=u0 if alloc.get("inflow_compensation", False) else None)

    def __call__(self, tau, angles, nu_r):
        f, alpha_cmd, slack = self.al.allocate(tau, angles)
        tau_alloc = self.al.T(angles) @ f
        if self.al.zero_thrust_speed is not None:
            f = self.al.compensate_inflow(f, angles, nu_r)
        cmds = {}
        for t, f_i in zip(self.thrusters, f):
            if t["type"] == "fixed":
                cmds[t["name"]] = (float(f_i), 0.0)
            else:
                a = alpha_cmd[t["name"]]
                mag = float(f_i) if abs(f_i) > 1e-3 else 1e-3
                cmds[t["name"]] = (mag * math.cos(a), mag * math.sin(a))
        return cmds, tau_alloc


# ---------------------------------------------------------------------------------------------------------------
# what the controller believes about the quay: a map (possibly off), or the lidar's line fit
# ---------------------------------------------------------------------------------------------------------------

LIDAR_POS = np.array([0.011, 0.039, -0.523])     # Ouster in BODY (FRD) from the CO (sensor_extrinsics.yaml + IMU)


class QuayBelief:
    """
    The berth face as the controller knows it: n . p = d in NED (n out of the quay). From the map: the true face
    moved out by map_err. With the lidar: the face measured from the boat (distance and normal in BODY, noise
    2 cm / 0.26 deg, only at >= 0.5 m), put into NED with the ESTIMATED pose and smoothed (0.5 s), so an error in
    the boat's own position moves the believed quay with it.
    """

    def __init__(self, quay, face, hull, rng, map_err=0.0, lidar=False):
        self.quay, self.face, self.hull, self.rng, self.lidar = quay, face, hull, rng, lidar
        self.n = quay.normals[face].copy()
        self.d = float(quay.offsets[face]) + map_err
        a, b, _, _ = quay.face(face)
        self.mid = 0.5 * (a + b)                      # the berth's place along the face (from the map)
        self.berth_map = quay.berth(face, SIDE, GAP, hull)
        self.n_map = self.n.copy()
        self.lidar_seen = False

    def update(self, true_eta, est_eta):
        if not self.lidar:
            return
        c, s = math.cos(true_eta[2]), math.sin(true_eta[2])
        p_l = true_eta[:2] + np.array([c * LIDAR_POS[0] - s * LIDAR_POS[1], s * LIDAR_POS[0] + c * LIDAR_POS[1]])
        n_true, d_true = self.quay.normals[self.face], self.quay.offsets[self.face]
        rho = float(n_true @ p_l - d_true)                        # lidar's distance to the face
        if not 0.5 <= rho <= 15.0:
            return                                                # too close (min range) or too far: keep
        ang = math.atan2(n_true[1], n_true[0]) - true_eta[2]       # the face normal in BODY
        rho_m = rho + self.rng.normal(0, 0.02)
        ang_m = ang + self.rng.normal(0, math.radians(0.26))
        ce, se = math.cos(est_eta[2]), math.sin(est_eta[2])
        p_le = est_eta[:2] + np.array([ce * LIDAR_POS[0] - se * LIDAR_POS[1], se * LIDAR_POS[0] + ce * LIDAR_POS[1]])
        n_m = np.array([math.cos(ang_m + est_eta[2]), math.sin(ang_m + est_eta[2])])
        d_m = float(n_m @ p_le - rho_m)
        k = 0.2 if self.lidar_seen else 1.0                       # 10 Hz, ~0.5 s smoothing
        self.lidar_seen = True
        n_new = (1 - k) * self.n + k * n_m
        self.n = n_new / np.linalg.norm(n_new)
        self.d = (1 - k) * self.d + k * d_m

    def halfplane(self):
        return -self.n, -self.d

    def berth(self):
        q = self.mid - (self.n @ self.mid - self.d) * self.n
        p = q + self.n * (self.hull[1] / 2 + GAP)
        turn = math.atan2(self.n[1], self.n[0]) - math.atan2(self.n_map[1], self.n_map[0])
        return np.array([p[0], p[1], wrap(self.berth_map[2] + turn)])


# ---------------------------------------------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------------------------------------------

def x0_from(est):
    """[n, e, psi, u_r, v_r, r] (velocity through the water) from the observer's estimate."""
    psi = est["eta"][2]
    c, s = math.cos(psi), math.sin(psi)
    vn, ve = est["vc"]
    nu = est["nu"]
    return np.array([*est["eta"], nu[0] - (c * vn + s * ve), nu[1] - (-s * vn + c * ve), nu[2]])


class PlanStrategy:
    """
    Free-final-time OCP to the berth, tracked by the DP law. replan=0: planned once. replan>0: planned again every
    `replan` s from the estimate (warm start), until `stop_replan` s before the plan ends. Without a plan (the first
    solve failed) the boat holds where it is and tries again every 2 s, as the node would.
    """

    def __init__(self, ocp, belief, replan=0.0, stop_replan=4.0, cpu_factor=1.0, from_plan=False, event=False,
                 lock_gap=0.0):
        self.ocp, self.belief, self.from_plan, self.event, self.lock_gap = ocp, belief, from_plan, event, lock_gap
        self.plan_berth = None
        self.replan, self.stop_replan, self.cpu_factor = replan, stop_replan, cpu_factor
        self.plan, self.t_plan, self.pending, self.next_solve = None, None, None, 0.0
        self.solves, self.failures, self.solve_times = 0, 0, []
        self.hold = None

    def _start_state(self, t, est, warm):
        """
        The measured state; with from_plan (Bitar et al. 2020) the previous plan's state at this time while the
        boat follows it within 0.15 m and 5 deg, so the DP's reference does not jump at every new plan.
        """
        x_meas = x0_from(est)
        if not (self.from_plan and warm and self.plan is not None):
            return x_meas
        eta_p, eta_dot_p, _, done = trajectory_planner.sample(self.plan, t - self.t_plan)
        if done or math.hypot(*(eta_p[:2] - est["eta"][:2])) > 0.15 or abs(wrap(eta_p[2] - est["eta"][2])) > math.radians(5):
            return x_meas
        c, s = math.cos(eta_p[2]), math.sin(eta_p[2])
        vn, ve = eta_dot_p[0] - est["vc"][0], eta_dot_p[1] - est["vc"][1]
        return np.array([*eta_p, c * vn + s * ve, -s * vn + c * ve, eta_dot_p[2]])

    def _solve(self, t, est, warm):
        elapsed = (t - self.t_plan) if (warm and self.plan is not None) else None
        res = self.ocp.solve(self._start_state(t, est, warm), self.belief.berth(), self.belief.halfplane(), vc=est["vc"],
                             bias=est["bias"], elapsed=elapsed)
        self.solves += 1
        self.solve_times.append(res["solve_time"])
        if not res["ok"]:
            self.failures += 1
            return
        plan = dict(t=res["t"], eta=res["eta"], nu=res["nu"], nu_dot=res["nu_dot"], T=res["T"],
                    berth=self.belief.berth(), hp=self.belief.halfplane())
        self.pending = (t + res["solve_time"] * self.cpu_factor, t, plan)

    def _needs_replan(self, t, est):
        """
        Event-triggered replanning: only when the plan no longer fits, i.e. the boat is off it (0.25 m, 10 deg),
        the quay as it is known now moved (5 cm, 2 deg; a lidar update or a map correction), or the plan crosses
        the quay half-plane as it is known now. Replanning every 2 s regardless reset the manoeuvre so often that
        the boat drifted about in benchmark v2.
        """
        eta_d, _, _, done = trajectory_planner.sample(self.plan, t - self.t_plan)
        if done:
            return False
        a, b = self.belief.halfplane()
        if self.lock_gap > 0 and b - float(a @ est["eta"][:2]) - self.belief.hull[1] / 2 < self.lock_gap:
            return False                     # the last metre is the DP's (NTNU: replanning reset the tracker)
        e = eta_d - est["eta"]
        if math.hypot(e[0], e[1]) > 0.25 or abs(wrap(e[2])) > math.radians(10):
            return True
        b_old, b_new = self.plan["berth"], self.belief.berth()
        if math.hypot(*(b_new[:2] - b_old[:2])) > 0.05 or abs(wrap(b_new[2] - b_old[2])) > math.radians(2):
            return True
        a, b = self.belief.halfplane()
        return bool(np.any(self.plan["eta"][:, :2] @ a > b - self.belief.hull[1] / 2 - 0.02))

    def reference(self, t, est):
        if self.plan is None and self.pending is None and t >= self.next_solve:
            self._solve(t, est, warm=False)
            self.next_solve = t + (self.replan if self.replan > 0 else 2.0)
            if self.pending is None:
                self.next_solve = t + 2.0
        if self.pending is not None and t >= self.pending[0]:
            _, self.t_plan, self.plan = self.pending
            self.pending = None
        if (self.replan > 0 and self.plan is not None and self.pending is None and t >= self.next_solve
                and self.plan["T"] - (t - self.t_plan) > self.stop_replan
                and (not self.event or self._needs_replan(t, est))):
            self._solve(t, est, warm=True)
            self.next_solve = t + self.replan
            self.replans = getattr(self, "replans", 0) + 1
        if self.plan is None:
            if self.hold is None:
                self.hold = est["eta"].copy()
            return ("ref", self.hold, np.zeros(3), np.zeros(3))   # still planning: hold where it was
        eta_d, eta_d_dot, eta_d_ddot, done = trajectory_planner.sample(self.plan, t - self.t_plan)
        if done:
            eta_d = self.belief.berth()                          # at the end: the berth as it is known now
        return ("ref", eta_d, eta_d_dot, eta_d_ddot)


class NMPCStrategy:
    """Fixed-horizon NMPC every `period` s; the first input's tau goes straight to the allocation (held)."""

    def __init__(self, ocp, belief, period=0.5, cpu_factor=1.0):
        self.ocp, self.belief, self.period, self.cpu_factor = ocp, belief, period, cpu_factor
        self.tau, self.pending, self.next_solve, self.t_last = np.zeros(3), None, 0.0, None
        self.solves, self.failures, self.solve_times = 0, 0, []
        self.f_prev = None

    def reference(self, t, est):
        if self.pending is not None and t >= self.pending[0]:
            self.tau, self.f_prev = self.pending[1], self.pending[2]
            self.pending = None
        if self.pending is None and t >= self.next_solve:
            elapsed = None if self.t_last is None else t - self.t_last
            res = self.ocp.solve(x0_from(est), self.belief.berth(), self.belief.halfplane(), vc=est["vc"],
                                 bias=est["bias"], f_prev=self.f_prev, elapsed=elapsed)
            self.solves += 1
            self.solve_times.append(res["solve_time"])
            self.t_last = t
            if res["ok"]:
                self.pending = (t + res["solve_time"] * self.cpu_factor, self.ocp.first_tau(res), res["F"][:, 0])
            else:
                self.failures += 1
            self.next_solve = t + self.period
        return ("tau", self.tau)


# ---------------------------------------------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------------------------------------------

CORRIDOR = (0.15, 0.5, 1.5)
HEADING_FUNNEL = (math.radians(5.0), math.radians(30.0))     # 5 deg + 30 deg per metre of gap
SPEED_FUNNEL = (0.05, 0.3, math.radians(2.0), math.radians(10.0))   # along the face and yaw rate near it


def make_strategy(name, cfg, params, belief, cpu_factor):
    """
    plan_once / replan: free-time OCP (soft berth, 4 RK4 substeps, T <= 60 s) with the approach funnel, tracked by
    the DP law; replan_pred: replanned from the previous plan's state while the boat follows it; *_nofunnel: the
    same without the funnel. nmpc: fixed 10 s horizon straight to the allocation.
    """
    speed = SPEED_FUNNEL if name.endswith("_speed") else None
    name = name.replace("_speed", "")
    base = name.replace("_nofunnel", "").replace("_noheading", "")
    corridor = None if name.endswith("_nofunnel") else CORRIDOR
    heading = None if name.endswith("_nofunnel") or name.endswith("_noheading") else HEADING_FUNNEL
    if base in ("plan_once", "replan", "replan_pred", "replan_event", "replan_event_lock"):
        ocp = DockingOCP.from_vessel_config(cfg, params, free_time=True, gap_min=GAP - 0.02, max_cpu_time=8.0,
                                            corridor=corridor, heading_funnel=heading, speed_funnel=speed)
        return PlanStrategy(ocp, belief, replan=(0.0 if base == "plan_once" else 2.0), cpu_factor=cpu_factor,
                            from_plan=(base == "replan_pred"), event=base.startswith("replan_event"),
                            lock_gap=(1.0 if base == "replan_event_lock" else 0.0))
    if base == "nmpc":
        ocp = DockingOCP.from_vessel_config(cfg, params, free_time=False, N=25, horizon=10.0, substeps=3,
                                            gap_min=GAP - 0.02, max_cpu_time=2.0, max_iter=150, corridor=corridor,
                                            heading_funnel=heading)
        return NMPCStrategy(ocp, belief, period=0.5, cpu_factor=cpu_factor)
    raise ValueError(name)


def team_thrusters(cfg, azimuth_reverse=0.0):
    """The team's limits (blueboat_core_control, 2026-11): aft +52.8 / -26.72 N, azimuth 51.5 N, forward only
    (azimuth_reverse > 0: what it would be if the Pico ran the azimuth's T200 in reverse too)."""
    import copy
    cfg = copy.deepcopy(cfg)
    for t in cfg["thrusters"]:
        if t["type"] == "fixed":
            t["max_force"], t["reverse_ratio"] = 52.8, 26.72 / 52.8
        else:
            t["max_force"], t["reverse_ratio"] = 51.5, azimuth_reverse
    return cfg


def run(job):
    strategy, env_name, start, seed, cpu_factor, t_max, opts = job
    rng = np.random.default_rng(seed)
    cfg, _ = vessel_model.load_vessel_config("microampere")
    if opts.get("thrusters") == "team":
        cfg = team_thrusters(cfg)
    elif opts.get("thrusters") == "team_rev":
        cfg = team_thrusters(cfg, azimuth_reverse=26.72 / 52.8)
    params = vessel_model.model_parameters(cfg)
    L, B = float(cfg["hull"]["length"]), float(cfg["hull"]["beam"])
    quay = Quay(QUAY)
    berth = quay.berth(FACE, SIDE, GAP, (L, B))             # the TRUE berth: what success is measured against
    k_mass, k_damp = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
    plant = Plant(cfg, params, start, ENVIRONMENTS[env_name], rng, k_mass, k_damp)
    nav = Navigation(params, rng, multipath=(quay, FACE, (L, B)) if opts.get("multipath") else None)
    law, alloc = DPLaw(params), Allocation(cfg)
    belief = QuayBelief(quay, FACE, (L, B), rng, map_err=opts.get("map_err", 0.0), lidar=opts.get("lidar", False))
    nav.true_face = (quay.normals[FACE], quay.offsets[FACE])
    strat = make_strategy(strategy, cfg, params, belief, cpu_factor)
    tau_alloc, tau_cmd = np.zeros(3), np.zeros(3)
    t_start, t = 3.0, 0.0                          # the observer settles first (hold position)
    rec = dict(min_clear=np.inf, max_towards_near=0.0, co_towards_near=0.0, yaw_rate_near=0.0, energy=0.0,
               t_dock=None, tube_n=[], tube_psi=[], contact_peak=0.0, contact_speed=0.0)
    ref = None
    in_tol_since, safe_since, hold_eta, log = None, None, None, []
    n_steps = int(t_max / DT)
    for i in range(n_steps):
        t = i * DT
        nav.update(t, plant, tau_alloc)
        if opts.get("lidar_obs"):
            nav.lidar_line(t, plant, belief.n, belief.d)
        if i % 10 == 0 and nav.ready:                       # 10 Hz control
            est = nav.estimate()
            belief.update(plant.eta3, est["eta"])
            if t < t_start:
                hold_eta = est["eta"].copy() if hold_eta is None else hold_eta
                tau_cmd = law.tau(est, hold_eta, np.zeros(3), np.zeros(3))
            else:
                out = strat.reference(t, est)
                tau_cmd = out[1] if out[0] == "tau" else law.tau(est, *out[1:])
                ref = out[1] if out[0] == "ref" else None
        if i % 5 == 0:                                       # 20 Hz allocation
            psi = plant.eta3[2] if not nav.ready else nav.estimate()["eta"][2]
            est_nu = nav.estimate()["nu"] if nav.ready else np.zeros(3)
            vc = nav.estimate()["vc"] if nav.ready else np.zeros(2)
            c, s = math.cos(psi), math.sin(psi)
            nu_r = est_nu - np.array([c * vc[0] + s * vc[1], -s * vc[0] + c * vc[1], 0.0])
            cmds, tau_alloc = alloc(tau_cmd, plant.azimuth_angles(), nu_r)
            plant.command(cmds)
        before = plant.nu3.copy(), plant.eta3.copy()           # velocity and pose before this step
        plant.step()
        rec["energy"] += DT * sum(plant.forces.values())
        if plant.contact_force > 0.0:
            if rec["contact_peak"] == 0.0:                   # the first touch: how fast the hull came in
                n_f = Quay(QUAY).normals[FACE]
                cs, sn = math.cos(before[1][2]), math.sin(before[1][2])
                v_ned = np.array([cs * before[0][0] - sn * before[0][1],
                                  sn * before[0][0] + cs * before[0][1]])
                rec["contact_speed"] = max(0.0, float(-n_f @ v_ned))
            rec["contact_peak"] = max(rec["contact_peak"], plant.contact_force)
        if i % 10 == 0 and t >= t_start:
            eta, nu = plant.eta3, plant.nu3
            clear = quay.clearance(eta, L, B)
            rec["min_clear"] = min(rec["min_clear"], clear)
            gap = quay.face_gap(eta, L, B, FACE)
            n = quay.normals[FACE]
            c, s = math.cos(eta[2]), math.sin(eta[2])
            v_ned = np.array([c * nu[0] - s * nu[1], s * nu[0] + c * nu[1]])
            towards = 0.0
            for corner in hull_corners(eta, L, B):
                rel = corner - eta[:2]
                towards = max(towards, -n @ (v_ned + nu[2] * np.array([-rel[1], rel[0]])))
            if gap < 0.5 and ref is not None:
                # tube (Krishnamoorthy L8): how far the boat is from the reference, along the face normal and in
                # heading, close to the quay; the quay constraint has to be tightened by this much
                e = eta - ref
                rec["tube_n"].append(float(-n @ e[:2]))         # > 0: the boat is closer to the quay than planned
                rec["tube_psi"].append(float(abs(wrap(e[2]))))
            if gap < 0.3:
                rec["max_towards_near"] = max(rec["max_towards_near"], towards)
                rec["co_towards_near"] = max(rec["co_towards_near"], float(-n @ v_ned))
                rec["yaw_rate_near"] = max(rec["yaw_rate_near"], abs(float(nu[2])))
            err = eta - berth
            err[2] = wrap(err[2])
            speed = math.hypot(*nu[:2])
            ok = math.hypot(err[0], err[1]) < 0.10 and abs(err[2]) < math.radians(5) and speed < 0.05
            # safe alongside: the right distance to the face and heading; along the face 0.3 m is fine
            gap_err = gap - GAP
            along = abs(float(np.array([-n[1], n[0]]) @ (eta[:2] - berth[:2])))
            safe = abs(gap_err) < 0.05 and abs(err[2]) < math.radians(5) and speed < 0.05 and along < 0.3
            if safe and safe_since is None:
                safe_since = t
            elif not safe:
                safe_since = None
            if safe_since is not None and t - safe_since >= 5.0 and rec.get("t_safe") is None:
                rec["t_safe"] = safe_since - t_start
            if ok and in_tol_since is None:
                in_tol_since = t
            elif not ok:
                in_tol_since = None
            if in_tol_since is not None and t - in_tol_since >= 5.0 and rec["t_dock"] is None:
                rec["t_dock"] = in_tol_since - t_start
            if i % 50 == 0:
                log.append([round(t, 2), *np.round(eta, 3).tolist(), round(gap, 3)])
            if rec["t_dock"] is not None and t - t_start - rec["t_dock"] > 15.0:
                break
            if rec.get("t_safe") is not None and t - t_start - rec["t_safe"] > 40.0:
                break
    eta = plant.eta3
    err = eta - berth
    err[2] = wrap(err[2])
    st = np.array(strat.solve_times) if strat.solve_times else np.zeros(1)
    return dict(strategy=strategy, env=env_name, start=list(map(float, start)), seed=seed, opts=opts,
                k_mass=round(k_mass, 3), k_damp=round(k_damp, 3),
                docked=rec["t_dock"] is not None and rec["min_clear"] > 0.0, t_dock=rec["t_dock"],
                safe=rec.get("t_safe") is not None and rec["min_clear"] > 0.0, t_safe=rec.get("t_safe"),
                collision=bool(rec["min_clear"] <= 0.0), min_clear=float(rec["min_clear"]),
                max_towards_near=float(rec["max_towards_near"]), co_towards_near=float(rec["co_towards_near"]),
                yaw_rate_near_deg=float(math.degrees(rec["yaw_rate_near"])), energy=float(rec["energy"]),
                contact_peak=float(rec["contact_peak"]), contact_speed=float(rec["contact_speed"]),
                tube_n_max=float(max(rec["tube_n"])) if rec["tube_n"] else 0.0,
                tube_psi_max_deg=float(math.degrees(max(rec["tube_psi"]))) if rec["tube_psi"] else 0.0,
                final_pos_err=float(math.hypot(err[0], err[1])), final_psi_err_deg=float(math.degrees(err[2])),
                solves=strat.solves, failures=strat.failures, solve_median=float(np.median(st)),
                solve_p95=float(np.percentile(st, 95)), solve_max=float(np.max(st)), log=log)


def starts_for(berth, n, seed):
    """Start poses 4-7 m from the berth on the open-water side (bearing 120-240 deg from it), any heading."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        b = math.radians(rng.uniform(120, 240))
        d = rng.uniform(4.0, 7.0)
        out.append([berth[0] + d * math.cos(b), berth[1] + d * math.sin(b), rng.uniform(-math.pi, math.pi)])
    return out


def summarize(runs):
    rows = []
    for s in sorted({r["strategy"] for r in runs}):
        for env in list(ENVIRONMENTS) + ["ALL"]:
            rr = [r for r in runs if r["strategy"] == s and (env == "ALL" or r["env"] == env)]
            if not rr:
                continue
            docked = [r for r in rr if r["docked"]]
            rows.append(dict(
                strategy=s, env=env, n=len(rr), docked=len(docked), collisions=sum(r["collision"] for r in rr),
                safe=sum(r.get("safe", False) for r in rr),
                t_dock_median=float(np.median([r["t_dock"] for r in docked])) if docked else None,
                min_clear=float(min(r["min_clear"] for r in rr)),
                towards_p95=float(np.percentile([r["max_towards_near"] for r in rr], 95)),
                co_towards_p95=float(np.percentile([r.get("co_towards_near", 0.0) for r in rr], 95)),
                yaw_near_p95=float(np.percentile([r.get("yaw_rate_near_deg", 0.0) for r in rr], 95)),
                tube_n_p95=float(np.percentile([r.get("tube_n_max", 0.0) for r in rr], 95)),
                tube_psi_p95=float(np.percentile([r.get("tube_psi_max_deg", 0.0) for r in rr], 95)),
                final_err_median=float(np.median([r["final_pos_err"] for r in rr])),
                solve_p95=float(np.percentile([r["solve_p95"] for r in rr], 95)),
                solve_max=float(max(r["solve_max"] for r in rr)),
                failures=sum(r["failures"] for r in rr), energy_median=float(np.median([r["energy"] for r in rr])),
                contacts=sum(1 for r in rr if r.get("contact_peak", 0.0) > 0.0),
                contact_peak_max=float(max(r.get("contact_peak", 0.0) for r in rr))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="plan_once,replan_event",
                    help="plan_once, replan, replan_event, replan_event_lock, replan_pred, nmpc (slow), "
                         "and *_nofunnel / *_noheading")
    ap.add_argument("--envs", default=",".join(ENVIRONMENTS))
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--cpu-factor", type=float, default=1.0, help="solve time on the boat / here")
    ap.add_argument("--t-max", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--thrusters", default="sim", choices=["sim", "team", "team_rev"])
    ap.add_argument("--map-err", type=float, default=0.0)
    ap.add_argument("--multipath", action="store_true")
    ap.add_argument("--lidar", action="store_true")
    ap.add_argument("--lidar-obs", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/ros_ws/results/docking_bench"))
    a = ap.parse_args()
    cfg, _ = vessel_model.load_vessel_config("microampere")
    berth = Quay(QUAY).berth(FACE, SIDE, GAP, (float(cfg["hull"]["length"]), float(cfg["hull"]["beam"])))
    envs = a.envs.split(",")
    n_starts = a.starts
    if a.quick:
        envs, n_starts = ["calm", "current_on"], 2
    starts = starts_for(berth, n_starts, seed=7)
    opts = dict(thrusters=a.thrusters, map_err=a.map_err, multipath=a.multipath, lidar=a.lidar, lidar_obs=a.lidar_obs)
    jobs = [(s, e, st, 1000 * i + j, a.cpu_factor, a.t_max, opts)
            for s in a.strategies.split(",") for i, e in enumerate(envs) for j, st in enumerate(starts)]
    t0 = time.time()
    print(f"{len(jobs)} runs on {min(a.workers, len(jobs))} processes ...", flush=True)
    runs = []
    with Pool(min(a.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(run, jobs, chunksize=1):
            runs.append(r)
            print(f"  {len(runs):3d}/{len(jobs)}  {r['strategy']:16s} {r['env']:22s} "
                  f"{'safe' if r.get('safe') else 'not safe':8s} {'TOUCH' if r['collision'] else '':5s} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rows = summarize(runs)
    with open(os.path.join(a.out, "runs.json"), "w") as f:
        json.dump(dict(args=vars(a), berth=berth.tolist(), runs=runs, summary=rows), f, indent=1)
    print(f"{len(runs)} runs in {time.time() - t0:.0f} s -> {a.out}/runs.json")
    hdr = f"{'strategy':16s} {'env':22s} {'n':>3s} {'safe':>4s} {'dock':>4s} {'touch':>5s} {'F_max':>5s} {'t_dock':>6s} {'minclr':>6s} " \
          f"{'v_in95':>6s} {'vco95':>6s} {'r95':>5s} {'tube_n':>6s} {'tube_p':>6s} {'err':>5s} {'s_p95':>5s} " \
          f"{'s_max':>5s} {'fail':>4s} {'energy':>7s}"
    print(hdr)
    for r in rows:
        td = f"{r['t_dock_median']:.1f}" if r["t_dock_median"] is not None else "-"
        print(f"{r['strategy']:16s} {r['env']:22s} {r['n']:3d} {r['safe']:4d} {r['docked']:4d} {r['contacts']:5d} "
              f"{r['contact_peak_max']:5.0f} {td:>6s} "
              f"{r['min_clear']:6.3f} {r['towards_p95']:6.3f} {r['co_towards_p95']:6.3f} {r['yaw_near_p95']:5.1f} "
              f"{r['tube_n_p95']:6.3f} {r['tube_psi_p95']:6.1f} "
              f"{r['final_err_median']:5.3f} {r['solve_p95']:5.2f} "
              f"{r['solve_max']:5.2f} {r['failures']:4d} {r['energy_median']:7.0f}")


if __name__ == "__main__":
    main()
