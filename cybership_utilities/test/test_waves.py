"""
Unit tests for the wave model (port of MSS waveSpectrum/waveDirectionalSpectrum/waveForceRAO) and
the Capytaine force RAOs, no ROS needed.
"""

import math
import os

import numpy as np
import pytest

from cybership_utilities import hydro
from cybership_utilities.marine_craft.waves import WaveModel, jonswap

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HYDRO = os.path.join(REPO, "cybership_config", "config", "microampere", "hydro_capytaine.npz")


def flat_rao(value=1.0):
    """Frequency- and direction-independent RAO table (for testing the summation)."""
    w = np.linspace(0.1, 8.0, 20)
    hdg = np.radians(np.arange(0, 360, 10))
    return w, hdg, np.full((6, len(w), len(hdg)), value, dtype=complex)


def test_jonswap_gives_the_significant_wave_height():
    """Hs = 4 sqrt(m0) (DNV normalization of JONSWAP, gamma 3.3)."""
    w = np.linspace(0.05, 30, 20000)
    for Hs, w0 in ((0.3, 2.7), (3.0, 0.8)):
        m0 = np.trapz(jonswap(w, Hs, w0), w)
        assert 4 * math.sqrt(m0) == pytest.approx(Hs, rel=0.03)


def test_wave_elevation_statistics():
    """
    Long-crested sea: std of the elevation = Hs / 4. Short-crested (MSS waveDirectionalSpectrum): the
    directions in a frequency bin share the frequency, so one realization's Hs varies (~+-15 %);
    the mean variance over realizations is Hs^2 / 16.
    """
    w, hdg, F = flat_rao()
    waves = WaveModel(0.3, 2.7, 0.0, w, hdg, F, spreading=False, seed=1)
    z = np.array([waves.loads(t, 0.0)[1] for t in np.arange(0, 2000, 0.25)])
    assert z.std() == pytest.approx(0.3 / 4, rel=0.05)
    var = []
    for seed in range(40):
        waves = WaveModel(0.3, 2.7, 0.0, w, hdg, F, spreading=True, seed=seed)
        var.append(np.sum(np.abs(np.sum(waves.Amp * np.exp(1j * waves.phase), axis=1)) ** 2) / 2)
    assert math.sqrt(np.mean(var)) == pytest.approx(0.3 / 4, rel=0.07)


def test_regular_wave_force_is_the_rao():
    """One component of amplitude A: tau(t) = A Re(H e^{i(w t + eps)}) (MSS phase convention)."""
    w, hdg = np.linspace(0.5, 5.0, 10), np.radians(np.arange(0, 360, 10))
    H = 100.0 * np.exp(1j * 0.7)
    F = np.zeros((6, len(w), len(hdg)), dtype=complex)
    F[2] = H
    waves = WaveModel(0.3, 2.7, 0.0, w, hdg, F, spreading=False, n_freq=1, omega_max=2.0, seed=0)
    waves.Amp[:] = 0.1
    for t in (0.0, 0.3, 1.1):
        tau, zeta = waves.loads(t, 0.0)
        ph = waves.Omega[0] * t + waves.phase[0, 0]
        assert zeta == pytest.approx(0.1 * math.cos(ph))
        assert tau[2] == pytest.approx(0.1 * (H * np.exp(1j * ph)).real)


def test_direction_is_relative_to_the_heading():
    """Waves towards east seen from a boat heading east are a following sea (beta = 0)."""
    w, hdg = np.linspace(0.5, 5.0, 10), np.radians(np.arange(0, 360, 10))
    F = np.zeros((6, len(w), len(hdg)), dtype=complex)
    F[1, :, 9] = 50.0                                     # sway load only for beta = 90 deg
    waves = WaveModel(0.3, 2.7, math.radians(90), w, hdg, F, spreading=False, seed=0)
    beam = np.abs([waves.loads(t, 0.0)[0][1] for t in np.arange(0, 20, 0.1)]).max()
    follow = np.abs([waves.loads(t, math.radians(90))[0][1] for t in np.arange(0, 20, 0.1)]).max()
    assert beam > 1.0 and follow < 1e-9


@pytest.mark.skipif(not os.path.exists(HYDRO), reason="no Capytaine data")
def test_capytaine_force_raos_are_physical():
    h = hydro.load_hydro(HYDRO)
    F, w, hdg = h["force_rao"], h["rao_freqs"], np.degrees(h["rao_headings"])
    k0, k90, k180 = [int(np.argmin(np.abs(hdg - b))) for b in (0, 90, 180)]
    # long waves: heave force = hydrostatic, rho g A_wp zeta upwards (BODY z down -> negative)
    C33 = h["C"][2, 2]
    assert F[2, 0, k0].real == pytest.approx(-C33, rel=0.1)
    assert abs(F[2, 0, k0].imag) < 0.1 * C33
    # head / following seas on a port-starboard symmetric hull: no sway, roll or yaw load
    for k in (k0, k180):
        assert np.abs(F[[1, 3, 5], :, k]).max() < 0.01 * np.abs(F[2, :, k]).max()
    # following sea (travelling towards +x): pitch moment rho g k I_L A sin(w t), i.e. Im H55 < 0
    assert F[4, 0, k0].imag < 0 < F[4, 0, k180].imag
    # mirrored headings: beta and 360 - beta have the same heave, opposite sway
    k270 = int(np.argmin(np.abs(hdg - 270)))
    assert F[2, :, k270] == pytest.approx(F[2, :, k90])
    assert F[1, :, k270] == pytest.approx(-F[1, :, k90])


@pytest.mark.skipif(not os.path.exists(HYDRO), reason="no Capytaine data")
def test_motion_raos_follow_long_waves():
    """
    Motion RAOs X = [-w^2 (M_RB + A) + i w (B + Bv) + G]^-1 F (Fossen 10.2.4): in long waves the boat
    follows the surface: heave amplitude -> wave amplitude, pitch -> wave slope k = w^2 / g.
    """
    from cybership_utilities import vessel_model
    path = os.path.join(REPO, "cybership_config", "config", "microampere", "vessel.yaml")
    params = vessel_model.model_parameters(vessel_model.load_vessel_config(path=path)[0])
    h = hydro.load_hydro(HYDRO)
    X = hydro.motion_raos(h["freqs"], h["A"], h["B"], h["force_rao"], h["rao_freqs"], params["MRB"],
                          params["D"] - params["B_eq"], params["G"])
    w = h["rao_freqs"]
    k0 = 0                                                   # following sea
    assert abs(X[2, 0, k0]) == pytest.approx(1.0, rel=0.1)   # heave, w = 0.25 rad/s
    # pitch at 2 rad/s (Microampere's waves): the CG above CO needs the surge motion to tilt the boat
    # along the surface (m z_g w^2 X1), which the surge damping slightly reduces at low frequency
    i = int(np.argmin(np.abs(w - 2.0)))
    assert abs(X[4, i, k0]) == pytest.approx(w[i] ** 2 / 9.81, rel=0.1)
    assert np.all(np.isfinite(X)) and np.abs(X[3]).max() < 5.0    # roll resonance limited by damping
