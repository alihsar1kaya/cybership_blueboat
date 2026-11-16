"""
A towed load behind the vessel (vessel.yaml 'tow'): TowedPointMass (default, body.type: point) or TowedBody
(body.type: hull), on a rigid towbar or a rope (line.rigid). towed_from_config picks one.

TowedBody is a 3-DOF (surge, sway, yaw) craft of its own, NED frame like the vessel:

    M nu_dot + C(nu_r) nu_r + D(nu_r) = tau_line,     M = diag(m + A11, m + A22, Iz + A66)
    C(nu) nu = [-m22 v r, m11 u r, (m22 - m11) u v]   (the Munk moment that makes towed bodies yaw)

Drag is quadratic and integrated over strips along the hull, so a yaw rate gives sway drag at the ends and
the body lines up behind the boat by itself: surge 1/2 rho Cd_x B T |u_r| u_r, each strip of length dx
1/2 rho Cd_y T dx |v_r + x r| (v_r + x r) sideways. A small linear damping keeps it still at rest.

The towline joins the vessel's tow point to the body's bow. It only pulls, as a spring with damping:

    T = max(0, k (d - L) + c d_dot)  when d > L, else 0       (d = distance between the two points)

The vessel feels -T along the line at its tow point. The body is integrated with sub-steps (the line is
stiff), with the vessel's tow point moving linearly over the step.

Waves (step_with_vessel, optional): the body moves in the water particle velocity u_w (relative velocity
in the drag) and is pushed by the water's acceleration (Froude-Krylov plus added mass, as in Morison's
equation): (rho V + A_ii) a_w along surge and sway. The tow point follows the vessel's total motion (LF +
WF), so the line jerks when vessel and body move out of phase, e.g. half a wavelength apart.

Contact (step_with_vessel, optional): vessel and body are a few circles along their centre lines; where
they overlap a fender-like force k_c p + c_c p_dot (compression only) pushes them apart, so the body bumps
into the stern instead of gliding through the vessel when the line goes slack.

TowedPointMass is a floating mass without heading or shape: see its docstring.

All defaults are ESTIMATES (30 kg load) until the towing team says what they tow and how it is connected.
"""
import math

import numpy as np

RHO = 1025.0


class TowedBody:
    def __init__(self, length=1.2, beam=0.5, mass=30.0, cd_x=0.6, cd_y=1.0, added_mass=(0.1, 1.0, 0.5),
                 line_length=3.0, line_stiffness=2000.0, line_damping=100.0, linear_damping=(2.0, 5.0, 1.0),
                 strips=8, substeps=10, block_coefficient=0.6, contact_stiffness=5000.0, contact_damping=150.0,
                 rigid=False):
        """
        Dimensions [m], mass [kg]; draft from the displacement, T = m / (rho L B C_b). added_mass: factors
        (surge of m, sway of the 2-D strip value rho pi T^2 / 2 per metre, yaw of the sway one times L^2/12).
        line_stiffness [N/m] (EA / L of the rope, or less for a stretchy line), line_damping [N s/m].
        """
        self.L, self.B, self.m = float(length), float(beam), float(mass)
        self.T = self.m / (RHO * self.L * self.B * block_coefficient)
        self.cd_x, self.cd_y = float(cd_x), float(cd_y)
        Iz = self.m * (self.L ** 2 + self.B ** 2) / 12.0
        a22 = added_mass[1] * RHO * math.pi * self.T ** 2 / 2.0 * self.L
        self.M = np.array([self.m * (1 + added_mass[0]), self.m + a22, Iz + added_mass[2] * a22 * self.L ** 2 / 12.0])
        self.line_length, self.k, self.c = float(line_length), float(line_stiffness), float(line_damping)
        self.rigid = bool(rigid)        # a towbar: pushes as well as pulls (see TowedPointMass)
        self.d_lin = np.asarray(linear_damping, float)
        self.xs = (np.arange(strips) + 0.5) / strips * self.L - self.L / 2.0     # strip centres along x
        self.dx = self.L / strips
        self.substeps = int(substeps)
        self.Vol = self.m / RHO
        self.FK = np.array([RHO * self.Vol, RHO * self.Vol]) + (self.M[:2] - self.m)   # rho V + A11, A22
        self.k_c, self.c_c = float(contact_stiffness), float(contact_damping)
        self.circles = np.array([[x, 0.0] for x in np.linspace(-self.L / 2 + self.B / 2, self.L / 2 - self.B / 2, 3)])
        self.radius = self.B / 2
        self.eta = np.zeros(3)          # x, y, psi (NED)
        self.nu = np.zeros(3)           # u, v, r (body)
        self.tension = 0.0
        self.contact_force = 0.0

    @classmethod
    def from_config(cls, cfg):
        """From the 'tow' section of vessel.yaml (body and line); missing keys keep the defaults."""
        b, line = cfg.get("body", {}), cfg.get("line", {})
        kw = dict(length=b.get("length"), beam=b.get("beam"), mass=b.get("mass"), cd_x=b.get("cd_x"),
                  cd_y=b.get("cd_y"), line_length=line.get("length"), line_stiffness=line.get("stiffness"),
                  line_damping=line.get("damping"))
        return cls(rigid=bool(line.get("rigid", False)), **{k: float(v) for k, v in kw.items() if v is not None})

    def surge_model(self):
        """(mass incl. added mass, a, b) of the body in a straight tow, drag a u + b u|u|."""
        return self.M[0], self.d_lin[0], 0.5 * RHO * self.cd_x * self.B * self.T

    def describe(self):
        return (f"{self.L} x {self.B} m body, {self.m} kg (draft {self.T * 100:.0f} cm), "
                f"{self.line_length} m {'towbar' if self.rigid else 'line'}")

    def bow(self, eta=None):
        """Attachment point on the body (its bow) in NED."""
        x, y, psi = self.eta if eta is None else eta
        return np.array([x + self.L / 2 * math.cos(psi), y + self.L / 2 * math.sin(psi)])

    def place_behind(self, point, heading):
        """Put the body on the line astern of `point` (NED) along `heading`, the line just slack, at rest."""
        back = self.line_length * 0.99 + self.L / 2
        self.eta = np.array([point[0] - back * math.cos(heading), point[1] - back * math.sin(heading), heading])
        self.nu = np.zeros(3)

    def _line_force(self, p_tow, v_tow, eta, nu):
        """Tension and the unit vector from the body's bow to the vessel's tow point (NED)."""
        psi = eta[2]
        bow = self.bow(eta)
        rel = p_tow - bow
        d = float(np.hypot(*rel))
        if d < 1e-9 or (d <= self.line_length and not self.rigid):
            return 0.0, np.zeros(2)
        e = rel / d
        c, s = math.cos(psi), math.sin(psi)
        # velocity of the bow point in NED: R (u, v) + r x (L/2 along the body)
        v_bow = np.array([c * nu[0] - s * nu[1], s * nu[0] + c * nu[1]]) + nu[2] * self.L / 2 * np.array([-s, c])
        d_dot = float(e @ (np.asarray(v_tow) - v_bow))
        f = self.k * (d - self.line_length) + self.c * d_dot
        return (f if self.rigid else max(0.0, f)), e

    def _accel(self, p_tow, v_tow, eta, nu, current, wave=None, contact=None):
        psi = eta[2]
        c, s = math.cos(psi), math.sin(psi)
        water = np.asarray(current[:2], float) + (wave[0] if wave is not None else 0.0)
        u_r = nu[0] - (c * water[0] + s * water[1])
        v_r = nu[1] - (-s * water[0] + c * water[1])
        r = nu[2]
        T, e = self._line_force(p_tow, v_tow, eta, nu)
        F = T * e                                                   # on the body, NED
        Fb = np.array([c * F[0] + s * F[1], -s * F[0] + c * F[1]])  # body frame
        tau = np.array([Fb[0], Fb[1], self.L / 2 * Fb[1]])          # applied at the bow (x = L/2)
        if wave is not None:                                        # water acceleration: Froude-Krylov + added mass
            aw = wave[1]
            tau[:2] += self.FK * np.array([c * aw[0] + s * aw[1], -s * aw[0] + c * aw[1]])
        if contact is not None:                                     # (force on the body NED, point NED)
            Fc, pc = contact
            Fcb = np.array([c * Fc[0] + s * Fc[1], -s * Fc[0] + c * Fc[1]])
            rb = np.array([c * (pc[0] - eta[0]) + s * (pc[1] - eta[1]), -s * (pc[0] - eta[0]) + c * (pc[1] - eta[1])])
            tau += np.array([Fcb[0], Fcb[1], rb[0] * Fcb[1] - rb[1] * Fcb[0]])
        vl = v_r + self.xs * r                                      # sway velocity of each strip
        dY = -0.5 * RHO * self.cd_y * self.T * self.dx * np.abs(vl) * vl
        drag = np.array([-0.5 * RHO * self.cd_x * self.B * self.T * abs(u_r) * u_r, dY.sum(), (self.xs * dY).sum()])
        m11, m22, _ = self.M
        cor = np.array([-m22 * v_r * r, m11 * u_r * r, (m22 - m11) * u_r * v_r])
        nu_dot = (tau + drag - cor - self.d_lin * np.array([u_r, v_r, r])) / self.M
        eta_dot = np.array([c * nu[0] - s * nu[1], s * nu[0] + c * nu[1], r])
        return eta_dot, nu_dot, T, e

    def step(self, dt, p_tow0, p_tow1, current=(0.0, 0.0)):
        """
        Advance dt with the vessel's tow point moving from p_tow0 to p_tow1 (NED). Returns the force on the
        vessel (NED, along the line towards the body) at the end of the step.
        """
        p0, p1 = np.asarray(p_tow0, float), np.asarray(p_tow1, float)
        v_tow = (p1 - p0) / dt
        h = dt / self.substeps
        force = np.zeros(2)
        for i in range(self.substeps):
            p = p0 + (p1 - p0) * i / self.substeps        # tow point at the start of the sub-step
            # semi-implicit Euler: velocities from the forces at time t, then positions with the new velocities
            _, nu_dot, T, e = self._accel(p, v_tow, self.eta, self.nu, current)
            force -= T * e / self.substeps                # mean force on the vessel over the step
            self.nu = self.nu + h * nu_dot
            c, s = math.cos(self.eta[2]), math.sin(self.eta[2])
            self.eta = self.eta + h * np.array([c * self.nu[0] - s * self.nu[1], s * self.nu[0] + c * self.nu[1], self.nu[2]])
            self.eta[2] = (self.eta[2] + math.pi) % (2 * math.pi) - math.pi
        self.tension = float(np.hypot(*force))
        return force


    def _contact(self, eta, nu, vessel, vessel_circles):
        """Fender force between the body's and the vessel's circles: (force on the body NED, point NED) or None.
        vessel = (x, y, psi, vx, vy, r) in NED; vessel_circles = [(x, y, radius)] in its BODY frame."""
        c, s = math.cos(eta[2]), math.sin(eta[2])
        cv, sv = math.cos(vessel[2]), math.sin(vessel[2])
        vb = np.array([c * nu[0] - s * nu[1], s * nu[0] + c * nu[1]])
        total, point, n_hit = np.zeros(2), np.zeros(2), 0
        for bx, by in self.circles:
            pb = np.array([eta[0] + c * bx - s * by, eta[1] + s * bx + c * by])
            vel_b = vb + nu[2] * np.array([-(s * bx + c * by), c * bx - s * by])
            for vx, vy, vr in vessel_circles:
                pv = np.array([vessel[0] + cv * vx - sv * vy, vessel[1] + sv * vx + cv * vy])
                rel = pb - pv
                d = float(np.hypot(*rel))
                pen = self.radius + vr - d
                if pen <= 0.0 or d < 1e-9:
                    continue
                n = rel / d
                vel_v = np.array([vessel[3], vessel[4]]) + vessel[5] * np.array([-(sv * vx + cv * vy), cv * vx - sv * vy])
                pen_dot = -float(n @ (vel_b - vel_v))
                f = max(0.0, self.k_c * pen + self.c_c * pen_dot)
                total += f * n
                point += pv + vr * n
                n_hit += 1
        if n_hit == 0:
            return None
        return total, point / n_hit

    def step_with_vessel(self, dt, vessel0, vessel_vel, tow_point_body, current=(0.0, 0.0), waves=None,
                         vessel_circles=None, wf=None):
        """
        Advance dt next to a moving vessel. vessel0 = (x, y, psi) NED at the start of the step,
        vessel_vel = (vx, vy, r) NED, held over the step; tow_point_body = (x, y) in its BODY frame.
        waves(dx, dy) -> (velocity, acceleration) of the water at that offset from the vessel's CO, NED.
        wf = (dx, dy, dpsi, dvx, dvy, dr): the vessel's wave-frequency motion (NED) added at the tow point.
        Returns (force NED, yaw moment about the vessel's CO) on the vessel, means over the step.
        """
        h = dt / self.substeps
        tx, ty = tow_point_body
        F_line, F_cont, M_cont = np.zeros(2), np.zeros(2), 0.0
        wave = None
        if waves is not None:
            wave = waves(self.eta[0] - vessel0[0], self.eta[1] - vessel0[1])
        wf = np.zeros(6) if wf is None else np.asarray(wf, float)
        for i in range(self.substeps):
            t = i * h
            x, y = vessel0[0] + vessel_vel[0] * t + wf[0], vessel0[1] + vessel_vel[1] * t + wf[1]
            psi = vessel0[2] + vessel_vel[2] * t + wf[2]
            r = vessel_vel[2] + wf[5]
            c, s = math.cos(psi), math.sin(psi)
            p = np.array([x + c * tx - s * ty, y + s * tx + c * ty])
            v_tow = np.array([vessel_vel[0] + wf[3], vessel_vel[1] + wf[4]]) + r * np.array([-(s * tx + c * ty), c * tx - s * ty])
            contact = None
            if vessel_circles is not None:
                contact = self._contact(self.eta, self.nu, (x, y, psi, vessel_vel[0] + wf[3], vessel_vel[1] + wf[4], r),
                                        vessel_circles)
            _, nu_dot, T, e = self._accel(p, v_tow, self.eta, self.nu, current, wave, contact)
            F_line -= T * e / self.substeps
            if contact is not None:
                Fc, pc = contact
                F_cont -= Fc / self.substeps
                M_cont -= ((pc[0] - x) * Fc[1] - (pc[1] - y) * Fc[0]) / self.substeps
            self.nu = self.nu + h * nu_dot
            cb, sb = math.cos(self.eta[2]), math.sin(self.eta[2])
            self.eta = self.eta + h * np.array([cb * self.nu[0] - sb * self.nu[1], sb * self.nu[0] + cb * self.nu[1], self.nu[2]])
            self.eta[2] = (self.eta[2] + math.pi) % (2 * math.pi) - math.pi
        self.tension = float(np.hypot(*F_line))
        self.contact_force = float(np.hypot(*F_cont))
        return F_line, F_cont, M_cont


class TowedPointMass:
    """
    The load as a point mass: a float without heading or shape of its own, on a towbar or a rope.

        (m + A) p_ddot = F_bar + F_drag + F_waves + F_contact                    (NED, horizontal)
        F_drag = -1/2 rho Cd A_f |v_r| v_r - d v_r,   v_r = p_dot - (current + wave particle velocity)

    Its size follows from the displacement, as a half-submerged sphere of volume 2 m / rho: radius r, frontal
    area pi r^2 / 2, added mass a rho V. Drag and added mass are the same in every direction. Waves push it
    with (rho V + A) a_w as TowedBody, and it bumps into the vessel's circles as one circle of radius r.

    rigid=True is a towbar pinned at both ends. It pulls and pushes, as a stiff spring with damping,

        f = k (d - L) + c d_dot          (d = distance tow point - load; f > 0 pulls the load towards the boat)

    so the load stays L from the tow point and swings freely around it. An exact constraint needs the vessel's
    acceleration, which the simulator knows one step late, and that coupling is unstable when the load is
    heavier than what the stern feels sideways (~22 kg for Microampere). The stiff spring is stable at the
    simulator's 0.01 s step and stretches 2.5 mm per 50 N with the defaults. rigid=False is a rope: f = max(0,
    ...) and only when d > L.

    tension is the mean bar force over the last step, signed: negative while the bar pushes.
    """

    def __init__(self, mass=30.0, cd=0.5, added_mass=0.5, line_length=2.0, line_stiffness=20000.0,
                 line_damping=1000.0, rigid=True, linear_damping=2.0, substeps=10, contact_stiffness=5000.0,
                 contact_damping=150.0):
        self.m = float(mass)
        self.Vol = self.m / RHO
        self.radius = (3.0 * 2.0 * self.Vol / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.drag_area = float(cd) * math.pi * self.radius ** 2 / 2.0
        self.A = float(added_mass) * RHO * self.Vol
        self.M = self.m + self.A
        self.FK = RHO * self.Vol + self.A
        self.line_length, self.k, self.c = float(line_length), float(line_stiffness), float(line_damping)
        self.rigid = bool(rigid)
        self.d_lin = float(linear_damping)
        self.substeps = int(substeps)
        self.k_c, self.c_c = float(contact_stiffness), float(contact_damping)
        self.p = np.zeros(2)            # position, NED
        self.v = np.zeros(2)            # velocity, NED
        self._dir = np.array([1.0, 0.0])    # unit vector from the load to the tow point
        self.tension = 0.0
        self.contact_force = 0.0

    @classmethod
    def from_config(cls, cfg):
        """From the 'tow' section of vessel.yaml; missing keys keep the defaults."""
        b, line = cfg.get("body", {}), cfg.get("line", {})
        kw = dict(mass=b.get("mass"), cd=b.get("cd"), added_mass=b.get("added_mass"), line_length=line.get("length"),
                  line_stiffness=line.get("stiffness"), line_damping=line.get("damping"))
        return cls(rigid=bool(line.get("rigid", True)), **{k: float(v) for k, v in kw.items() if v is not None})

    def surge_model(self):
        """(mass incl. added mass, a, b) of the load in a straight tow, drag a u + b u|u|: what the vessel's
        controller and planner add in surge. The bar carries the load along itself only, so sway and yaw
        stay the vessel's own."""
        return self.M, self.d_lin, 0.5 * RHO * self.drag_area

    def describe(self):
        return (f"point mass {self.m} kg (radius {self.radius:.2f} m), "
                f"{self.line_length} m {'towbar' if self.rigid else 'line'}")

    @property
    def eta(self):
        """x, y and the direction from the load to the tow point (NED), like TowedBody.eta."""
        return np.array([self.p[0], self.p[1], math.atan2(self._dir[1], self._dir[0])])

    @property
    def nu(self):
        """Velocity along and across that direction; no yaw rate of its own."""
        c, s = self._dir
        return np.array([c * self.v[0] + s * self.v[1], -s * self.v[0] + c * self.v[1], 0.0])

    def bow(self, eta=None):
        """Attachment point (NED): the load itself."""
        return self.p.copy() if eta is None else np.asarray(eta[:2], float)

    def place_behind(self, point, heading):
        """Put the load L astern of `point` (NED) along `heading`, at rest."""
        self._dir = np.array([math.cos(heading), math.sin(heading)])
        self.p = np.asarray(point, float)[:2] - self.line_length * self._dir
        self.v = np.zeros(2)

    def _bar_force(self, p_tow, v_tow, p, v):
        """Bar force on the load along e (unit vector load -> tow point), and e."""
        rel = p_tow - p
        d = float(np.hypot(*rel))
        if d < 1e-9:
            return 0.0, self._dir
        e = rel / d
        if d <= self.line_length and not self.rigid:
            return 0.0, e
        f = self.k * (d - self.line_length) + self.c * float(e @ (v_tow - v))
        return (f if self.rigid else max(0.0, f)), e

    def _contact(self, p, v, vessel, vessel_circles):
        """Fender force on the load (NED) and the contact point, or None; arguments as TowedBody._contact."""
        cv, sv = math.cos(vessel[2]), math.sin(vessel[2])
        total, point, n_hit = np.zeros(2), np.zeros(2), 0
        for vx, vy, vr in vessel_circles:
            pv = np.array([vessel[0] + cv * vx - sv * vy, vessel[1] + sv * vx + cv * vy])
            rel = p - pv
            d = float(np.hypot(*rel))
            pen = self.radius + vr - d
            if pen <= 0.0 or d < 1e-9:
                continue
            n = rel / d
            vel_v = np.array([vessel[3], vessel[4]]) + vessel[5] * np.array([-(sv * vx + cv * vy), cv * vx - sv * vy])
            f = max(0.0, self.k_c * pen - self.c_c * float(n @ (v - vel_v)))
            total += f * n
            point += pv + vr * n
            n_hit += 1
        if n_hit == 0:
            return None
        return total, point / n_hit

    def step(self, dt, p_tow0, p_tow1, current=(0.0, 0.0)):
        """Advance dt with the tow point moving from p_tow0 to p_tow1 (NED); the force on the vessel (NED)."""
        p0 = np.asarray(p_tow0, float)
        v = (np.asarray(p_tow1, float) - p0) / dt
        return self.step_with_vessel(dt, (p0[0], p0[1], 0.0), (v[0], v[1], 0.0), (0.0, 0.0), current)[0]

    def step_with_vessel(self, dt, vessel0, vessel_vel, tow_point_body, current=(0.0, 0.0), waves=None,
                         vessel_circles=None, wf=None):
        """Same arguments and return as TowedBody.step_with_vessel: (force NED, contact force NED, contact
        yaw moment about the vessel's CO), means over the step."""
        h = dt / self.substeps
        tx, ty = tow_point_body
        F_line, F_cont, M_cont, f_mean = np.zeros(2), np.zeros(2), 0.0, 0.0
        water_v, water_a = np.asarray(current[:2], float), np.zeros(2)
        if waves is not None:
            u_w, a_w = waves(self.p[0] - vessel0[0], self.p[1] - vessel0[1])
            water_v, water_a = water_v + u_w, np.asarray(a_w, float)
        wf = np.zeros(6) if wf is None else np.asarray(wf, float)
        for i in range(self.substeps):
            t = i * h
            x, y = vessel0[0] + vessel_vel[0] * t + wf[0], vessel0[1] + vessel_vel[1] * t + wf[1]
            psi = vessel0[2] + vessel_vel[2] * t + wf[2]
            r = vessel_vel[2] + wf[5]
            c, s = math.cos(psi), math.sin(psi)
            p_tow = np.array([x + c * tx - s * ty, y + s * tx + c * ty])
            vx, vy = vessel_vel[0] + wf[3], vessel_vel[1] + wf[4]
            v_tow = np.array([vx, vy]) + r * np.array([-(s * tx + c * ty), c * tx - s * ty])
            f, e = self._bar_force(p_tow, v_tow, self.p, self.v)
            vr = self.v - water_v
            F = f * e - (0.5 * RHO * self.drag_area * float(np.hypot(*vr)) + self.d_lin) * vr + self.FK * water_a
            if vessel_circles is not None:
                contact = self._contact(self.p, self.v, (x, y, psi, vx, vy, r), vessel_circles)
                if contact is not None:
                    Fc, pc = contact
                    F = F + Fc
                    F_cont -= Fc / self.substeps
                    M_cont -= ((pc[0] - x) * Fc[1] - (pc[1] - y) * Fc[0]) / self.substeps
            F_line -= f * e / self.substeps
            f_mean += f / self.substeps
            self._dir = e
            # semi-implicit Euler: velocity from the forces at time t, then position with the new velocity
            self.v = self.v + h * F / self.M
            self.p = self.p + h * self.v
        self.tension = f_mean
        self.contact_force = float(np.hypot(*F_cont))
        return F_line, F_cont, M_cont


def towed_from_config(cfg):
    """The towed load from vessel.yaml 'tow': body.type point (default) or hull."""
    if (cfg.get("body") or {}).get("type", "point") == "hull":
        return TowedBody.from_config(cfg)
    return TowedPointMass.from_config(cfg)


class LoadCell:
    """
    The measured towbar force for the vessel's observer: a 2-axis load cell at the tow point (sensor/tow_force,
    BODY axes, the force on the vessel), lightly low-pass filtered against the noise, as a wrench [X, Y, N] at
    the CO (N = x F_y - y F_x). The observer adds it to its input: the load's pull, drag and inertia, at any
    bar angle, are then a known force instead of a model error in the bias.

    Not for the controller: cancelling the measured force there was unstable in yaw. It holds the load's
    inertia (45 kg x 0.6^2 m^2 at the stern, more than Microampere's 9.4 kg m^2), and any filter delays it.
    """

    def __init__(self, tow_point, filter_time=0.1):
        self.x, self.y = float(tow_point[0]), float(tow_point[1])
        self.T = float(filter_time)
        self.f = np.zeros(2)
        self._t = None

    @classmethod
    def from_config(cls, cfg):
        """From the 'tow' section of vessel.yaml (tow_point, load_cell.filter)."""
        return cls(cfg.get("tow_point", (-0.6, 0.0, -0.1)), (cfg.get("load_cell") or {}).get("filter", 0.1))

    def update(self, t, fx, fy):
        """New measurement [N] at time t [s]."""
        z = np.array([fx, fy], float)
        if self._t is None or self.T <= 0.0:
            self.f = z
        else:
            a = min(1.0, max(0.0, t - self._t) / self.T)
            self.f = self.f + a * (z - self.f)
        self._t = t

    def wrench(self):
        fx, fy = self.f
        return np.array([fx, fy, self.x * fy - self.y * fx])


def tow_wrench(force_ned, psi, tow_point):
    """6-DOF wrench in BODY on the vessel from a horizontal force (NED) at tow_point = (x, y, z) in BODY."""
    c, s = math.cos(psi), math.sin(psi)
    fx = c * force_ned[0] + s * force_ned[1]
    fy = -s * force_ned[0] + c * force_ned[1]
    x, y, z = tow_point
    return np.array([fx, fy, 0.0, -z * fy, z * fx, x * fy - y * fx])
