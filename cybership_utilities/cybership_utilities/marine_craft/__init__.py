"""
Marine craft model after Fossen (2021), Handbook of Marine Craft Hydrodynamics and Motion Control:

    kinematics:  eta_dot = J(eta) nu
    dynamics:    M nu_r_dot + C(nu_r) nu_r + D(nu_r) nu_r + g(eta) = tau + tau_ext
    actuators:   commanded thrust -> saturation, deadband, thrust dynamics, azimuth rate -> actual thrust

Plain Python (no ROS), so the same code can be used by the simulator, an observer and unit tests.
"""

from cybership_utilities.marine_craft.kinematics import Smtrx, Rzyx, Tzyx, J, eta_dot
from cybership_utilities.marine_craft.dynamics import m2c, hoerner, cross_flow_drag, MarineCraft
from cybership_utilities.marine_craft.actuators import FixedThruster, AzimuthThruster, make_thruster
