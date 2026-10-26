"""
Vessel model parameters from cybership_config/config/<vessel>/vessel.yaml.

Shared by the simulator and the controllers, so both use the same mass, inertia,
added mass and damping for a vessel.

If vessel.yaml has a 'model' section, its values are used directly:

    model:
      mass: 16.9                       # [kg]
      inertia: [Ix, Iy, Iz]            # [kg m^2] about the CG, diagonal
      cg: [x_g, y_g, z_g]              # [m] CG from CO in BODY (z down), default [0, 0, 0];
                                       # M_RB = H(r_g)' diag(m I, I_g) H(r_g) as MSS rbody.m
      added_mass: [Xu, Yv, Zw, Kp, Mq, Nr]      # positive values, diagonal
      # ... or full 6x6 added mass from frequency-dependent Capytaine data by Fossen's power-based
      # averaging (cybership_utilities/hydro.py); overrides added_mass:
      hydro:
        file: hydro_capytaine.npz      # relative to vessel.yaml (cybership_config/tools/capytaine_hydro.py)
        wave_peak_frequency: 2.5       # [rad/s] w_p of the PM weighting spectrum
        max_frequency: 6.5             # [rad/s] optional, frequencies above are left out
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
        roll_quadratic_factor: 1.0     # kappa_phi: K = D44 (1 + kappa |p|) p, default 0
        zeta: [0.3, 0.2, 0.4]          # heave, roll, pitch: total relative damping d = 2 zeta omega_n M
        cross_flow: {length: 1.056, beam: 0.161, draft: 0.274}   # one hull, Hoerner Cd
      cancel_munk: [yaw]               # cancel destabilising Munk moments (roll/pitch/yaw), default none
      thruster_roll_pitch_moments: true  # false: thrusters only give X, Y, N (as in Fossen's Otter)

Otherwise the values are computed from the 'hull' section like shoeboxpy's 6DOF box
(mass = rho*L*B*T, uniform-box inertia, default added mass and damping factors).
Roll/pitch restoring always comes from hull.GM_phi / hull.GM_theta, given for a CG on the waterline
and lowered by the CG height above the waterline (GM = GM_wl + z_g, z_g < 0 above).
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
        config = yaml.safe_load(f)
    config["_path"] = path  # files in vessel.yaml (e.g. model.hydro.file) are relative to it
    return config, path


def added_mass_matrix(model, config_path=""):
    """
    6x6 added mass M_A about CO: power-based A_eq from model.hydro (symmetrized), else diag(added_mass).
    Also returns B_eq (6x6, zeros without hydro data), the hydrostatic restoring (6x6 for a CG on the
    waterline, None without hydro data) and a short description.
    """
    h = model.get("hydro")
    if not h:
        MA = np.diag(np.asarray(model["added_mass"], dtype=float))
        return MA, np.zeros((6, 6)), None, "diagonal added_mass"
    from cybership_utilities import hydro
    path = h["file"]
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(config_path), path)
    data = hydro.load_hydro(path)
    w_max = h.get("max_frequency")
    A_eq, B_eq = hydro.power_based_matrices(data["freqs"], data["A"], data["B"], float(h["wave_peak_frequency"]),
                                            w_max=None if w_max is None else float(w_max))
    # Restoring: heave/roll/pitch block of the hydrostatic C (as MSS computeManeuveringModel.m), CG on WL
    C = np.zeros((6, 6))
    C[2:5, 2:5] = hydro.symmetrize(data["C"][2:5, 2:5])
    return hydro.symmetrize(A_eq), hydro.symmetrize(B_eq), C, \
        f"power-based A_eq, B_eq, w_p {h['wave_peak_frequency']} rad/s, w <= {w_max}, {os.path.basename(path)}"


def model_parameters(config, rho=1000.0, g=9.81):
    """
    Return the 6DOF model plus the nonlinear terms:
    {'mass', 'inertia' (3), 'cg' (3), 'MRB' (6x6), 'MA' (6x6), 'B_eq' (6x6), 'D' (6x6, linear),
     'G' (6x6), 'nonlinear' (dict), 'source'}.
    """
    model = config.get("model")
    hull = config["hull"]
    nonlinear = {"surge_quadratic": 0.0, "surge_linear_decay": 0.0, "yaw_quadratic": 0.0,
                 "roll_quadratic_factor": 0.0, "cross_flow": None, "cancel_munk": [],
                 "thruster_roll_pitch_moments": True}
    if model:
        mass = float(model["mass"])
        inertia = np.asarray(model["inertia"], dtype=float)
        cg = np.asarray(model.get("cg", [0.0, 0.0, 0.0]), dtype=float)
        MA, B_eq, C_wl, ma_source = added_mass_matrix(model, config.get("_path", ""))
        heave_stiffness = float(model.get("heave_stiffness", 0.0))
        nonlinear["cancel_munk"] = list(model.get("cancel_munk", []))
        nonlinear["thruster_roll_pitch_moments"] = bool(model.get("thruster_roll_pitch_moments", True))
        source = f"model ({ma_source})"
    else:
        L, B, T = hull["length"], hull["beam"], hull["draft"]
        mass = rho * L * B * T
        inertia = mass / 12.0 * np.array([B**2 + T**2, L**2 + T**2, L**2 + B**2])
        cg = np.zeros(3)
        MA = np.diag(_SHOEBOX_ALPHA * np.concatenate([[mass] * 3, inertia]))
        B_eq = np.zeros((6, 6))
        C_wl = None
        heave_stiffness = 0.0
        source = "hull"
    from cybership_utilities.marine_craft import Hmtrx
    H = Hmtrx(cg)
    MRB = H.T @ np.diag(np.concatenate([[mass] * 3, inertia])) @ H
    if C_wl is None:
        G = np.diag([0.0, 0.0, heave_stiffness, mass * g * hull.get("GM_phi", 0.0),
                     mass * g * hull.get("GM_theta", 0.0), 0.0])
    else:
        G = C_wl.copy()  # full heave/roll/pitch block from Capytaine (heave-pitch coupling)
    # Both are for a CG on the waterline; a CG z_g below CO (z down) raises GM by z_g: G44, G55 += m g z_g
    G[3, 3] += mass * g * cg[2]
    G[4, 4] += mass * g * cg[2]

    if model and "damping" in model:
        D, extra = _fossen_damping(model["damping"], np.diag(MRB) + np.diag(MA), np.diag(G), B_eq)
        nonlinear.update(extra)
        source += ", Fossen damping"
    elif model:
        D = np.diag(np.asarray(model["damping_linear"], dtype=float))
    else:
        D = np.diag(_SHOEBOX_BETA * np.diag(MRB))

    for name, vec, shape in (("inertia", inertia, (3,)), ("added_mass", MA, (6, 6)),
                             ("damping_linear", D, (6, 6))):
        if vec.shape != shape:
            raise ValueError(f"vessel.yaml model.{name} must have shape {shape}, got {vec.tolist()}")
    return {"mass": mass, "inertia": inertia, "cg": cg, "MRB": MRB, "MA": MA, "B_eq": B_eq, "D": D, "G": G,
            "nonlinear": nonlinear, "source": source}


def _fossen_damping(d, M, G, B_eq=None):
    """
    Damping as in Fossen's Otter USV model (MSS otter.m, 2026-10-14, Fossen 2021):
    linear terms from time constants (surge, sway, yaw) and relative damping (heave, roll, pitch),
    plus quadratic surge resistance, nonlinear yaw damping and cross-flow drag.

    Surge, as Fossen calibrates the Otter:
        X_u    = M11 / T_surge                    linear, dominates at low speed
        X_|u|u = max_thrust / max_speed^2         quadratic, gives the top speed
        X = -(X_u e^(-k|u|) u + X_|u|u |u| u)     blending, k = surge_linear_decay (3)

    With hydro data (B_eq != 0), the linear D is Fossen's D = B_eq + Bv (MSS computeManeuveringModel.m):
    the potential damping B_eq (full 6x6) plus diagonal viscous Bv that gives the total time constants
    in surge/sway/yaw and the total relative damping zeta in heave/roll/pitch.
    M and G are diagonal vectors. Returns D as a 6x6 matrix.
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
    if B_eq is not None and np.any(B_eq):
        from cybership_utilities import hydro
        T_126 = [M[0] / Xu, d["sway_time_constant"], d["yaw_time_constant"]]
        D = B_eq + hydro.viscous_damping(np.diag(M), np.diag(G), B_eq, T_126, zeta_345=zeta)
    else:
        D = np.diag([
            Xu,
            M[1] / d["sway_time_constant"],
            *(2.0 * zeta * omega * M[2:5]),
            Nr,
        ])
    extra = {
        "surge_quadratic": Xuu,
        "surge_linear_decay": decay,
        "roll_quadratic_factor": float(d.get("roll_quadratic_factor", 0.0)),
        "yaw_quadratic": float(d.get("yaw_quadratic_factor", 0.0)) * Nr,
        "cross_flow": d.get("cross_flow"),
    }
    return D, extra


def make_marine_craft(params, eta0=None, nu0=None, rho=1000.0):
    """Build the Fossen marine craft model (cybership_utilities.marine_craft) from model_parameters()."""
    from cybership_utilities.marine_craft import MarineCraft
    nl = params["nonlinear"]
    return MarineCraft(
        params["MRB"], params["MA"], params["D"], params["G"],
        surge_quadratic=nl["surge_quadratic"], yaw_quadratic=nl["yaw_quadratic"],
        cross_flow=nl["cross_flow"], cancel_munk=nl["cancel_munk"], rho=rho, eta0=eta0, nu0=nu0,
        surge_linear_decay=nl["surge_linear_decay"], roll_quadratic_factor=nl["roll_quadratic_factor"],
    )


def M_eff_3dof(params):
    """(M_RB + M_A) for surge, sway, yaw as a 3x3 matrix (with the sway-yaw coupling), for the DP controllers."""
    idx = [0, 1, 5]
    return (params["MRB"] + params["MA"])[np.ix_(idx, idx)]


def D_3dof(params):
    """Linear damping for surge, sway, yaw as a 3x3 matrix."""
    idx = [0, 1, 5]
    return params["D"][np.ix_(idx, idx)]
