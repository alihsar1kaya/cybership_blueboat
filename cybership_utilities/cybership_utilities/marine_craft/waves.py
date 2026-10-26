"""
Wave loads (Fossen 2021, Ch. 10.2; TTK4190 "How to simulate ocean waves?"): a Python port of MSS
waveSpectrum.m (JONSWAP), waveDirectionalSpectrum.m and waveForceRAO.m.

    S(w)       JONSWAP spectrum (Hs, w0, gamma), DNV normalization
    M(mu)      spreading (2/pi) cos^2(mu), mu in [-pi/2, pi/2] about the main direction
    components one random frequency per interval dw (and direction per dmu), random phase eps,
               amplitude A = sqrt(2 S(w) M(mu) dw dmu)
    zeta(t)    = sum A cos(w_e t + eps)                             wave elevation at CO
    tau_wave1  = sum A [Re H(w, beta) cos(w_e t + eps) - Im H sin(w_e t + eps)]   first-order loads

H are the force RAOs from Capytaine (capytaine_hydro.py: force_rao, rao_freqs, rao_headings), BODY
axes at CO, per metre wave amplitude; Re and Im are interpolated linearly in frequency and direction
(as MSS, avoids phase unwrapping). beta = beta_wave - psi + mu is the propagation direction relative
to the bow (0 following, 180 head sea), beta_wave the main direction the waves travel towards in NED
(0 north, 90 east). w_e = |w - w^2 U cos(beta) / g| is the frequency of encounter.

Motion RAOs (optional, hydro.motion_raos): the wave-frequency (WF) motion
    eta_WF(t) = sum A Re(X(w, beta) e^{i(w_e t + eps)}),   nu_WF = d/dt eta_WF
for the classic DP-simulator split total motion = LF + WF (Fossen 10.2.4, 13.2): the LF (maneuvering)
model gets no first-order forces, the WF motion is added on top of it for the sensors and the display.
Putting first-order forces into the nonlinear 6-DOF model instead rectifies the heave/pitch motion
(<q w>, <w sin theta>) into a spurious surge drift of metres per minute against the waves.

Second-order mean drift (slide 11) is not modelled: MSS-Capytaine gives no drift coefficients, and
Microampere's fjord waves (wavelength ~8-13 m) are long compared with the hull (1.2 m), where the
drift force is small; the observer's bias state takes slowly varying forces.
"""

import math

import numpy as np

G = 9.81


def jonswap(w, Hs, w0, gamma=3.3, g=G):
    """JONSWAP spectrum S(w) [m^2 s] as MSS waveSpectrum.m (case 7), DNV normalization."""
    w = np.asarray(w, dtype=float)
    alpha = (5.0 / 16.0) * Hs**2 * w0**4 / g**2
    sigma = np.where(w < w0, 0.07, 0.09)
    S1 = alpha * g**2 * w**-5 * np.exp(-1.25 * (w0 / w) ** 4)
    S2 = gamma ** np.exp(-((w - w0) ** 2) / (2 * (sigma * w0) ** 2))
    return (1 - 0.287 * math.log(gamma)) * S1 * S2


class WaveModel:
    def __init__(self, Hs, w0, beta_wave, rao_freqs, rao_headings, force_rao, gamma=3.3,
                 n_freq=100, n_dir=24, spreading=True, omega_max=None, seed=None, g=G, motion_rao=None):
        """
        Hs [m], w0 peak frequency [rad/s], beta_wave [rad] NED direction the waves travel towards.
        rao_*: Capytaine force RAOs (rad/s, rad, 6 x n_w x n_beta complex), headings 0..2pi excl. 2pi.
        """
        rng = np.random.default_rng(seed)
        self.g = g
        self.Hs, self.w0, self.beta_wave = float(Hs), float(w0), float(beta_wave)
        w_max = float(np.max(rao_freqs)) if omega_max is None else float(omega_max)
        dw = w_max / n_freq
        self.Omega = (np.arange(n_freq) + 0.5) * dw + (rng.random(n_freq) - 0.5) * dw
        if spreading:
            dmu = math.pi / (n_dir - 1)
            mu = -math.pi / 2 + np.arange(n_dir) * dmu + (rng.random(n_dir) - 0.5) * dmu
            self.mu = np.clip(mu, -math.pi / 2, math.pi / 2)
            M = (2 / math.pi) * np.cos(self.mu) ** 2
            M = M / np.trapz(M, self.mu)
        else:
            dmu, self.mu, M = 1.0, np.zeros(1), np.ones(1)
        S = jonswap(self.Omega, Hs, w0, gamma, g)
        self.Amp = np.sqrt(2.0 * np.outer(S, M) * dw * dmu)            # n_freq x n_dir
        self.phase = 2 * math.pi * rng.random(self.Amp.shape)
        # RAO table, closed in direction (append 2 pi) for interpolation
        self.rao_headings = np.r_[np.asarray(rao_headings, dtype=float), 2 * math.pi]
        F = np.asarray(force_rao)
        F = np.concatenate([F, F[:, :, :1]], axis=2)
        # interpolate in frequency once (linear, extrapolated as MSS): 6 x n_freq x n_beta
        w_rao = np.asarray(rao_freqs, dtype=float)
        self.H = self._table(F, w_rao)
        self.X = None
        if motion_rao is not None:
            X = np.asarray(motion_rao)
            # motions: no extrapolation below the table (would blow up), zero above (damped out)
            self.X = self._table(np.concatenate([X, X[:, :, :1]], axis=2), w_rao, extrapolate=False)

    def _table(self, F, w_rao, extrapolate=True):
        T = np.empty((6, len(self.Omega), F.shape[2]), dtype=complex)
        for d in range(6):
            for j in range(F.shape[2]):
                if extrapolate:
                    T[d, :, j] = _interp_extrap(self.Omega, w_rao, F[d, :, j])
                else:
                    T[d, :, j] = (np.interp(self.Omega, w_rao, F[d, :, j].real, left=0, right=0)
                                  + 1j * np.interp(self.Omega, w_rao, F[d, :, j].imag, left=0, right=0))
        return T

    def _rao(self, beta, table=None):
        """RAO(w_k, beta_kj) for all components: 6 x n_freq x n_dir, linear in direction."""
        T = self.H if table is None else table
        b = np.mod(beta, 2 * math.pi)
        hdg = self.rao_headings
        idx = np.clip(np.searchsorted(hdg, b) - 1, 0, len(hdg) - 2)
        f = (b - hdg[idx]) / (hdg[idx + 1] - hdg[idx])
        cols = np.arange(T.shape[1])[:, None]
        return T[:, cols, idx] * (1 - f) + T[:, cols, idx + 1] * f

    def motion(self, t, psi, U=0.0):
        """WF motion (eta_WF, nu_WF), 6 each, BODY axes (positions along the body axes) at CO."""
        beta = np.broadcast_to(self.beta_wave - psi + self.mu[None, :], self.Amp.shape)
        w_e = np.abs(self.Omega[:, None] - self.Omega[:, None] ** 2 / self.g * U * np.cos(beta))
        ph = w_e * t + self.phase
        c, s = np.cos(ph), np.sin(ph)
        X = self._rao(beta, self.X)
        eta = np.sum(self.Amp * (X.real * c - X.imag * s), axis=(1, 2))
        nu = np.sum(self.Amp * w_e * (-X.real * s - X.imag * c), axis=(1, 2))
        return eta, nu

    def loads(self, t, psi, U=0.0):
        """First-order wave loads tau_wave1 (6, BODY at CO) and wave elevation at CO, at time t."""
        beta = self.beta_wave - psi + self.mu[None, :]                # n_freq x n_dir (broadcast)
        beta = np.broadcast_to(beta, self.Amp.shape)
        w_e = np.abs(self.Omega[:, None] - self.Omega[:, None] ** 2 / self.g * U * np.cos(beta))
        ph = w_e * t + self.phase
        c, s = np.cos(ph), np.sin(ph)
        H = self._rao(beta)
        tau = np.sum(self.Amp * (H.real * c - H.imag * s), axis=(1, 2))
        return tau, float(np.sum(self.Amp * c))


def _interp_extrap(x, xp, fp):
    """Linear interpolation with linear extrapolation (MSS interp1 'linear', 'extrap'), complex."""
    re = np.interp(x, xp, fp.real)
    im = np.interp(x, xp, fp.imag)
    for arr, f in ((re, fp.real), (im, fp.imag)):
        lo, hi = x < xp[0], x > xp[-1]
        arr[lo] = f[0] + (x[lo] - xp[0]) * (f[1] - f[0]) / (xp[1] - xp[0])
        arr[hi] = f[-1] + (x[hi] - xp[-1]) * (f[-1] - f[-2]) / (xp[-1] - xp[-2])
    return re + 1j * im


def make_waves(env, hydro_data, seed=None):
    """WaveModel from vessel/sim parameters: env = dict(Hs, w0 or Tp, direction_deg, spreading, ...)."""
    w0 = env.get("w0") or 2 * math.pi / env["Tp"]
    return WaveModel(env["Hs"], w0, math.radians(env.get("direction_deg", 0.0)),
                     hydro_data["rao_freqs"], hydro_data["rao_headings"], hydro_data["force_rao"],
                     gamma=env.get("gamma", 3.3), n_freq=int(env.get("n_freq", 100)),
                     n_dir=int(env.get("n_dir", 24)), spreading=bool(env.get("spreading", True)),
                     seed=seed, motion_rao=hydro_data.get("motion_rao"))
