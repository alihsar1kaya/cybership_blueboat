"""
Constant maneuvering matrices from frequency-dependent hydrodynamic data, by Fossen's power-based
averaging (Fossen 2025, Ocean Engineering 341, 122494; MSS computeManeuveringModel.m; L3a/L3b):

    A_eq = int A(w) S_N(w) dw        equivalent added mass        -> M_A
    B_eq = int B(w) S_N(w) dw        equivalent potential damping -> part of D (with Bv)

    S(w)   = alpha / w^5 exp(-beta (w_p / w)^4)    PM spectrum, alpha = 8.1e-3 g^2, beta = 0.74
    S_N(w) = S(w) / m_0,  m_0 = int S(w) dw        normalized, so only the spectral shape matters

The spectrum is evaluated for U = 0 (as in MSS); the peak-frequency parameter w_p describes the
sea state. Note that with this parametrization S(w) itself peaks at (4 beta / 5)^(1/4) w_p = 0.88 w_p.

The frequency-dependent data (A, B, C, freqs) come from cybership_config/tools/capytaine_hydro.py.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

G = 9.81

# numpy >= 2 renamed trapz to trapezoid (ROS Jazzy ships numpy 1.26)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def load_hydro(path):
    """Hydro data saved by capytaine_hydro.py: dict with freqs, A, B (6x6xN) and C (6x6), BODY axes at CO."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def pm_spectrum(w, w_p, g=G):
    """Pierson-Moskowitz weighting spectrum as in MSS computeManeuveringModel.m."""
    w = np.asarray(w, dtype=float)
    return 8.1e-3 * g**2 / w**5 * np.exp(-0.74 * (w_p / w) ** 4)


def power_based_matrices(freqs, A, B, w_p, n_fine=100, w_max=None):
    """
    A_eq and B_eq (6x6) by power-based averaging, a line-by-line port of MSS computeManeuveringModel.m:
    infinite frequency excluded, w = 0 replaced by 1e-6, PCHIP interpolation onto 100 points,
    trapezoidal integration. w_max (optional) excludes higher frequencies, e.g. a resonance that
    potential theory overpredicts.
    """
    freqs = np.asarray(freqs, dtype=float)
    keep = np.isfinite(freqs) & (freqs <= (np.inf if w_max is None else w_max))
    freqs, A, B = freqs[keep], A[:, :, keep], B[:, :, keep]
    w_min = max(freqs.min(), 1e-6)
    w = np.linspace(w_min, freqs.max(), n_fine)
    S = pm_spectrum(w, w_p)
    S_N = S / _trapz(S, w)
    A_eq = _trapz(PchipInterpolator(freqs, A, axis=2)(w) * S_N, w, axis=2)
    B_eq = _trapz(PchipInterpolator(freqs, B, axis=2)(w) * S_N, w, axis=2)
    return A_eq, B_eq


def symmetrize(M):
    """BEM results are symmetric only up to discretization errors; Fossen's m2c also uses (M + M')/2."""
    return 0.5 * (M + M.T)


def viscous_damping(M, G_mat, B_eq, T_126, delta_zeta_345=None, zeta_345=None):
    """
    Diagonal viscous damping Bv on top of B_eq (MSS computeManeuveringModel.m, floating vessel):

        Bv_ii = M_ii / T_i - B_eq_ii                  i = 1, 2, 6  (total time constant T_i)
        Bv_ii = 2 delta_zeta_i sqrt(M_ii G_ii)        i = 3, 4, 5  (extra relative damping, MSS)
     or Bv_ii = 2 zeta_i sqrt(M_ii G_ii) - B_eq_ii    i = 3, 4, 5  (total relative damping zeta_345)

    The total form is for small craft, whose heave/roll/pitch natural frequencies lie above the wave
    spectrum, so B_eq (weighted by the spectrum) holds almost no damping at resonance.
    Raises if a T_i would need negative viscous damping (T_i > M_ii / B_eq_ii).
    """
    Bv = np.zeros((6, 6))
    for T, i in zip(T_126, (0, 1, 5)):
        target = M[i, i] / T
        if target - B_eq[i, i] < -1e-10 * max(1.0, abs(target)):
            raise ValueError(f"T_{i + 1} = {T} s needs negative viscous damping; "
                             f"choose T <= M/B_eq = {M[i, i] / B_eq[i, i]:.4g} s")
        Bv[i, i] = max(0.0, target - B_eq[i, i])
    for k, i in enumerate((2, 3, 4)):
        critical = 2.0 * np.sqrt(M[i, i] * G_mat[i, i])
        if zeta_345 is not None:
            Bv[i, i] = max(0.0, zeta_345[k] * critical - B_eq[i, i])
        else:
            Bv[i, i] = delta_zeta_345[k] * critical
    return Bv


def motion_raos(freqs, A, B, F, w_rao, M_RB, Bv, G):
    """
    First-order motion RAOs (Fossen 2021, Ch. 10.2.4; MSS waveMotionRAO) from the force RAOs F
    (6 x n_w x n_beta, MSS convention) and the frequency-dependent A(w), B(w):

        X(w, beta) = [-w^2 (M_RB + A(w)) + i w (B(w) + Bv) + G]^-1 F(w, beta)

    Bv: viscous damping on top of the potential damping (limits the roll/pitch resonances).
    Returns 6 x n_w x n_beta complex: displacement [m] and angle [rad] per metre wave amplitude.
    """
    keep = np.isfinite(freqs)
    Ai = PchipInterpolator(freqs[keep], A[:, :, keep], axis=2)(w_rao)
    Bi = PchipInterpolator(freqs[keep], B[:, :, keep], axis=2)(w_rao)
    X = np.empty_like(F)
    for k, w in enumerate(w_rao):
        H = -w**2 * (M_RB + Ai[:, :, k]) + 1j * w * (Bi[:, :, k] + Bv) + G
        X[:, k, :] = np.linalg.solve(H, F[:, k, :])
    return X
