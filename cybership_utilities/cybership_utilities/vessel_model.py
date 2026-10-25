"""
Vessel model parameters from cybership_config/config/<vessel>/vessel.yaml.

Shared by the simulator and the controllers, so both use the same mass, inertia,
added mass and damping for a vessel.

If vessel.yaml has a 'model' section, its values are used directly:

    model:
      mass: 16.9                       # [kg]
      inertia: [Ix, Iy, Iz]            # [kg m^2] about CO, diagonal
      added_mass: [Xu, Yv, Zw, Kp, Mq, Nr]      # positive values, diagonal
      heave_stiffness: 2581            # [N/m] rho*g*A_wp (optional, default 0)
      # either explicit linear damping ...
      damping_linear: [Xu, Yv, Zw, Kp, Mq, Nr]  # positive values, diagonal
      # ... or Fossen's method (as in his Otter USV model), which also adds the nonlinear terms:
      damping:
        # surge as in Fossen's Otter model (MSS otter.m, 2026-10-14):
        surge_time_constant: 5.0       # X_u = M11 / T_surge (low-speed, DP)
        max_speed: 3.15                # X_|u|u = max_thrust / max_speed^2
        max_thrust: 134.0
        surge_linear_decay: 3.0        # X_u fades as e^(-3|u|), default 3
        # ... or the coefficients directly (no fading unless surge_linear_decay is given):
        # surge_linear: 2.0            # X_u [N s/m]
        # surge_quadratic: 13.5        # X_|u|u [N s^2/m^2]
        sway_time_constant: 1.0        # Y_v = M22 / T_sway
        yaw_time_constant: 1.0         # N_r = M66 / T_yaw
        yaw_quadratic_factor: 10.0     # N_|r|r = factor * N_r
        zeta: [0.3, 0.2, 0.4]          # heave, roll, pitch: d = 2 zeta omega_n M
        cross_flow: {length: 1.056, beam: 0.161, draft: 0.274}   # one hull, Hoerner Cd
      cancel_munk: [yaw]               # cancel destabilising Munk moments (roll/pitch/yaw), default none
      thruster_roll_pitch_moments: true  # false: thrusters only give X, Y, N (as in Fossen's Otter)

Otherwise the values are computed from the 'hull' section like shoeboxpy's 6DOF box
(mass = rho*L*B*T, uniform-box inertia, default added mass and damping factors).
Roll/pitch restoring always comes from hull.GM_phi / hull.GM_theta.
"""

import os

import numpy as np
import yaml

# shoeboxpy.model6dof.Shoebox defaults, used when vessel.yaml has no 'model' section
_SHOEBOX_ALPHA = np.array([0.1, 0.2, 1.0, 0.1, 0.1, 0.1])  # added mass factors
_SHOEBOX_BETA = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])  # damping factors


def vessel_config_path(vessel_model):
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(
        get_package_share_directory("cybership_config"), "config", vessel_model, "vessel.yaml"
    )


def load_vessel_config(vessel_model="voyager", path=""):
    """Return (config dict, path). An explicit path overrides vessel_model."""
    path = path or vessel_config_path(vessel_model)
    with open(path, "r") as f:
        return yaml.safe_load(f), path


def model_parameters(config, rho=1000.0, g=9.81):
    """
    Return the 6DOF model as diagonal vectors plus the nonlinear terms:
    {'mass', 'inertia' (3), 'MRB', 'MA', 'D' (linear), 'G' (6), 'nonlinear' (dict), 'source'}.
    """
    model = config.get("model")
    hull = config["hull"]
    nonlinear = {"surge_quadratic": 0.0, "surge_linear_decay": 0.0, "yaw_quadratic": 0.0,
                 "cross_flow": None, "cancel_munk": [],
                 "thruster_roll_pitch_moments": True}
    if model:
        mass = float(model["mass"])
        inertia = np.asarray(model["inertia"], dtype=float)
        MA = np.asarray(model["added_mass"], dtype=float)
        heave_stiffness = float(model.get("heave_stiffness", 0.0))
        nonlinear["cancel_munk"] = list(model.get("cancel_munk", []))
        nonlinear["thruster_roll_pitch_moments"] = bool(model.get("thruster_roll_pitch_moments", True))
        source = "model"
    else:
        L, B, T = hull["length"], hull["beam"], hull["draft"]
        mass = rho * L * B * T
        inertia = mass / 12.0 * np.array([B**2 + T**2, L**2 + T**2, L**2 + B**2])
        MA = _SHOEBOX_ALPHA * np.concatenate([[mass] * 3, inertia])
        heave_stiffness = 0.0
        source = "hull"
    MRB = np.concatenate([[mass] * 3, inertia])
    G = np.array([0.0, 0.0, heave_stiffness,
                  mass * g * hull.get("GM_phi", 0.0), mass * g * hull.get("GM_theta", 0.0), 0.0])

    if model and "damping" in model:
        D, extra = _fossen_damping(model["damping"], MRB + MA, G)
        nonlinear.update(extra)
        source += ", Fossen damping"
    elif model:
        D = np.asarray(model["damping_linear"], dtype=float)
    else:
        D = _SHOEBOX_BETA * MRB

    for name, vec, n in (("inertia", inertia, 3), ("added_mass", MA, 6), ("damping_linear", D, 6)):
        if vec.shape != (n,):
            raise ValueError(f"vessel.yaml model.{name} must have {n} values, got {vec.tolist()}")
    return {"mass": mass, "inertia": inertia, "MRB": MRB, "MA": MA, "D": D, "G": G,
            "nonlinear": nonlinear, "source": source}


def _fossen_damping(d, M, G):
    """
    Damping as in Fossen's Otter USV model (MSS otter.m, 2026-10-14, Fossen 2021):
    linear terms from time constants (surge, sway, yaw) and relative damping (heave, roll, pitch),
    plus quadratic surge resistance, nonlinear yaw damping and cross-flow drag.

    Surge, as Fossen calibrates the Otter:
        X_u    = M11 / T_surge                    linear, dominates at low speed
        X_|u|u = max_thrust / max_speed^2         quadratic, gives the top speed
        X = -(X_u e^(-k|u|) u + X_|u|u |u| u)     blending, k = surge_linear_decay (3)
    M and G are diagonal vectors.
    """
    zeta = np.asarray(d.get("zeta", [0.3, 0.2, 0.4]), dtype=float)
    omega = np.sqrt(G[2:5] / M[2:5])  # natural frequencies in heave, roll, pitch
    Nr = M[5] / d["yaw_time_constant"]
    if "surge_time_constant" in d:
        Xu = M[0] / d["surge_time_constant"]
        Xuu = float(d["max_thrust"]) / float(d["max_speed"]) ** 2
        decay = float(d.get("surge_linear_decay", 3.0))
    else:
        Xu = d["surge_linear"]
        Xuu = float(d.get("surge_quadratic", 0.0))
        decay = float(d.get("surge_linear_decay", 0.0))
    D = np.array([
        Xu,
        M[1] / d["sway_time_constant"],
        *(2.0 * zeta * omega * M[2:5]),
        Nr,
    ])
    extra = {
        "surge_quadratic": Xuu,
        "surge_linear_decay": decay,
        "yaw_quadratic": float(d.get("yaw_quadratic_factor", 0.0)) * Nr,
        "cross_flow": d.get("cross_flow"),
    }
    return D, extra


def make_marine_craft(params, eta0=None, nu0=None, rho=1000.0):
    """Build the Fossen marine craft model (cybership_utilities.marine_craft) from model_parameters()."""
    from cybership_utilities.marine_craft import MarineCraft
    nl = params["nonlinear"]
    return MarineCraft(
        np.diag(params["MRB"]), np.diag(params["MA"]), np.diag(params["D"]), np.diag(params["G"]),
        surge_quadratic=nl["surge_quadratic"], yaw_quadratic=nl["yaw_quadratic"],
        cross_flow=nl["cross_flow"], cancel_munk=nl["cancel_munk"], rho=rho, eta0=eta0, nu0=nu0,
        surge_linear_decay=nl["surge_linear_decay"],
    )


def M_eff_3dof(params):
    """(M_RB + M_A) for surge, sway, yaw as a 3x3 matrix, as used by the DP controllers."""
    idx = [0, 1, 5]
    return np.diag(params["MRB"][idx] + params["MA"][idx])


def D_3dof(params):
    """Linear damping for surge, sway, yaw as a 3x3 matrix."""
    return np.diag(params["D"][[0, 1, 5]])
