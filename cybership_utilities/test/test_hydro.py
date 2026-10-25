"""
Unit tests for the power-based averaging (Fossen 2025, MSS computeManeuveringModel.m), no ROS needed.
"""

import os

import numpy as np
import pytest

from cybership_utilities import hydro

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HYDRO = os.path.join(REPO, "cybership_config", "config", "microampere", "hydro_capytaine.npz")


def test_normalized_spectrum_integrates_to_one_and_peaks_at_0_88_wp():
    w = np.linspace(1e-3, 30, 20000)
    S = hydro.pm_spectrum(w, 2.5)
    assert np.trapz(S / np.trapz(S, w), w) == pytest.approx(1.0)
    assert w[np.argmax(S)] == pytest.approx((4 * 0.74 / 5) ** 0.25 * 2.5, rel=1e-3)


def test_constant_coefficients_are_unchanged():
    """If A and B do not depend on frequency, the weighted average is the constant itself."""
    freqs = np.r_[0.0, np.linspace(0.25, 12, 48), np.inf]
    A0 = np.arange(36.0).reshape(6, 6)
    B0 = 2.0 * A0
    A = np.repeat(A0[:, :, None], len(freqs), axis=2)
    B = np.repeat(B0[:, :, None], len(freqs), axis=2)
    A_eq, B_eq = hydro.power_based_matrices(freqs, A, B, 2.5)
    assert A_eq == pytest.approx(A0, rel=1e-6)
    assert B_eq == pytest.approx(B0, rel=1e-6)


def test_low_peak_frequency_gives_the_low_frequency_added_mass():
    """A narrow spectrum at low frequency weights A(w) near w_p: A_eq -> A(0.88 w_p)."""
    freqs = np.r_[0.0, np.linspace(0.05, 12, 240), np.inf]
    a = 10.0 + freqs[:-1]                     # A11 rises linearly with frequency
    A = np.zeros((6, 6, len(freqs)))
    A[0, 0, :-1] = a
    A[0, 0, -1] = a[-1]
    A_eq, _ = hydro.power_based_matrices(freqs, A, np.zeros_like(A), 0.5)
    w = np.linspace(1e-6, 12, 100)
    S = hydro.pm_spectrum(w, 0.5)
    mean_w = np.trapz(w * S, w) / np.trapz(S, w)  # linear A: A_eq = A(mean frequency of S_N)
    assert A_eq[0, 0] == pytest.approx(10.0 + mean_w, rel=1e-3)


def test_viscous_damping_gives_the_requested_time_constants():
    M = np.diag([33.0, 170.0, 66.0, 7.0, 3.6, 9.4])
    Gm = np.diag([0, 0, 2581.0, 317.0, 158.0, 0])
    B_eq = np.diag([1.0, 30.0, 40.0, 0.8, 0.2, 0.5])
    Bv = hydro.viscous_damping(M, Gm, B_eq, T_126=[5.0, 1.0, 1.0], delta_zeta_345=[0.0, 0.1, 0.0])
    D = B_eq + Bv
    assert M[0, 0] / D[0, 0] == pytest.approx(5.0)
    assert M[1, 1] / D[1, 1] == pytest.approx(1.0)
    assert Bv[3, 3] == pytest.approx(2 * 0.1 * np.sqrt(7.0 * 317.0))
    # total relative damping in heave/roll/pitch: D_ii = 2 zeta sqrt(M G) whatever B_eq holds
    D = B_eq + hydro.viscous_damping(M, Gm, B_eq, [5.0, 1.0, 1.0], zeta_345=[0.3, 0.2, 0.4])
    for zeta, i in zip([0.3, 0.2, 0.4], (2, 3, 4)):
        assert D[i, i] / (2 * np.sqrt(M[i, i] * Gm[i, i])) == pytest.approx(zeta)
    with pytest.raises(ValueError):     # T_2 longer than M/B_eq needs negative viscous damping
        hydro.viscous_damping(M, Gm, B_eq, T_126=[5.0, 10.0, 1.0], delta_zeta_345=[0, 0, 0])


@pytest.mark.skipif(not os.path.exists(HYDRO), reason="no Capytaine data")
def test_microampere_hydro_data_is_physical():
    """Capytaine data in BODY axes: positive diagonal, symmetric up to BEM error, M_A > 0, bow at +x."""
    h = hydro.load_hydro(HYDRO)
    A0 = h["A"][:, :, 0]
    assert np.all(np.diag(A0) > 0)
    scale = np.sqrt(np.outer(np.diag(A0), np.diag(A0)))
    assert np.max(np.abs(A0 - A0.T) / scale) < 0.05
    A_eq, B_eq = hydro.power_based_matrices(h["freqs"], h["A"], h["B"], 2.5)
    assert np.all(np.linalg.eigvalsh(hydro.symmetrize(A_eq)) > 0)
    assert np.all(np.diag(B_eq) >= 0)
    # more lateral area aft (flat keel) than forward (rocker): sway-yaw coupling A26 < 0
    assert A_eq[1, 5] < 0
    # port-starboard symmetry: no coupling between longitudinal (1, 3, 5) and lateral (2, 4, 6) DOFs
    lon, lat = [0, 2, 4], [1, 3, 5]
    assert np.max(np.abs(A_eq[np.ix_(lon, lat)])) < 0.01 * A_eq[1, 1]
