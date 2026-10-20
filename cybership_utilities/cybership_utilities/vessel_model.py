"""
Vessel model parameters from cybership_config/config/<vessel>/vessel.yaml.

Shared by the simulator and the controllers, so both use the same mass, inertia,
added mass and damping for a vessel.

If vessel.yaml has a 'model' section, its values are used directly:

    model:
      mass: 16.9                       # [kg]
      inertia: [Ix, Iy, Iz]            # [kg m^2] about CO, diagonal
      added_mass: [Xu, Yv, Zw, Kp, Mq, Nr]      # positive values, diagonal
      damping_linear: [Xu, Yv, Zw, Kp, Mq, Nr]  # positive values, diagonal

Otherwise the values are computed from the 'hull' section like shoeboxpy's 6DOF box
(mass = rho*L*B*T, uniform-box inertia, default added mass and damping factors).
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


def model_parameters(config, rho=1000.0):
    """
    Return the 6DOF model as diagonal vectors:
    {'mass', 'inertia' (3), 'MRB' (6), 'MA' (6), 'D' (6), 'source'}.
    """
    model = config.get("model")
    if model:
        mass = float(model["mass"])
        inertia = np.asarray(model["inertia"], dtype=float)
        MA = np.asarray(model["added_mass"], dtype=float)
        D = np.asarray(model["damping_linear"], dtype=float)
        source = "model"
    else:
        hull = config["hull"]
        L, B, T = hull["length"], hull["beam"], hull["draft"]
        mass = rho * L * B * T
        inertia = mass / 12.0 * np.array([B**2 + T**2, L**2 + T**2, L**2 + B**2])
        base = np.concatenate([[mass] * 3, inertia])
        MA = _SHOEBOX_ALPHA * base
        D = _SHOEBOX_BETA * base
        source = "hull"
    for name, vec, n in (("inertia", inertia, 3), ("added_mass", MA, 6), ("damping_linear", D, 6)):
        if vec.shape != (n,):
            raise ValueError(f"vessel.yaml model.{name} must have {n} values, got {vec.tolist()}")
    MRB = np.concatenate([[mass] * 3, inertia])
    return {"mass": mass, "inertia": inertia, "MRB": MRB, "MA": MA, "D": D, "source": source}


def apply_to_shoebox(shoebox, params):
    """Overwrite a shoeboxpy.model6dof.Shoebox's mass, added mass and damping."""
    shoebox.m = params["mass"]
    shoebox.MRB = np.diag(params["MRB"])
    shoebox.MA = np.diag(params["MA"])
    shoebox.D = np.diag(params["D"])
    shoebox.M_eff = shoebox.MRB + shoebox.MA
    shoebox.invM_eff = np.linalg.inv(shoebox.M_eff)
    return shoebox


def M_eff_3dof(params):
    """(M_RB + M_A) for surge, sway, yaw as a 3x3 matrix, as used by the DP controllers."""
    idx = [0, 1, 5]
    return np.diag(params["MRB"][idx] + params["MA"][idx])
