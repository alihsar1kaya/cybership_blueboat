"""
Kinematics (Fossen 2021, Ch. 2): how body-frame velocities move the vessel in the NED frame.

    eta = [x, y, z, phi, theta, psi]   position (NED) and Euler angles (zyx convention)
    nu  = [u, v, w, p, q, r]           linear and angular velocity in the BODY frame

    eta_dot = J(eta) nu,   J(eta) = blockdiag(R(Theta), T(Theta))
"""

import math

import numpy as np


def Smtrx(a):
    """Skew-symmetric matrix S(a), so that a x b = S(a) b."""
    return np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ])


def Rzyx(phi, theta, psi):
    """Rotation matrix from BODY to NED, zyx Euler angles."""
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    return np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
        [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth, cth * sphi, cth * cphi],
    ])


def Tzyx(phi, theta):
    """Angular velocity transformation: Theta_dot = T(Theta) [p, q, r]. Singular at theta = +-90 deg."""
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    if abs(cth) < 1e-9:
        raise ValueError("Tzyx is singular for theta = +-90 degrees")
    return np.array([
        [1.0, sphi * sth / cth, cphi * sth / cth],
        [0.0, cphi, -sphi],
        [0.0, sphi / cth, cphi / cth],
    ])


def J(eta):
    """6x6 transformation J(eta) = blockdiag(Rzyx, Tzyx)."""
    phi, theta, psi = eta[3], eta[4], eta[5]
    out = np.zeros((6, 6))
    out[0:3, 0:3] = Rzyx(phi, theta, psi)
    out[3:6, 3:6] = Tzyx(phi, theta)
    return out


def eta_dot(eta, nu):
    """Kinematic equation eta_dot = J(eta) nu."""
    return J(eta) @ nu
