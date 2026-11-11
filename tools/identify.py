#!/usr/bin/env python3
"""
Identify vessel.yaml parameters from tests on the real boat (procedures: tools/identification.md).
The estimators are in cybership_utilities/identification.py and are checked against the simulator in
cybership_utilities/test/test_identification.py.

    python3 tools/identify.py bollard bollard.csv                     # columns: pwm, thrust (N)
    python3 tools/identify.py surge   <bag> [--start s --end s]       # speed after thrust steps
    python3 tools/identify.py roll    <bag> [--start s --end s]       # free roll decay
    python3 tools/identify.py latency <bag> [--topic sensor/gnss]     # header stamp vs. arrival

Topics are read under --ns (default /microampere). The result is printed as the vessel.yaml lines to change.
"""
import argparse
import csv
import math
import sys

import numpy as np

from cybership_utilities import identification as ident


def read_bag(path, topics):
    """{topic: [(t_received, msg), ...]} from a rosbag2 folder (sqlite3 or mcap)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    if missing:
        sys.exit(f"not in the bag: {missing}\nin the bag: {sorted(types)}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        out[topic].append((t_ns * 1e-9, deserialize_message(data, get_message(types[topic]))))
    return out


def window(t, *arrays, start=None, end=None):
    t = np.asarray(t)
    t0 = t[0]
    m = np.ones(t.size, bool)
    if start is not None:
        m &= t - t0 >= start
    if end is not None:
        m &= t - t0 <= end
    return (t[m],) + tuple(np.asarray(a)[m] for a in arrays)


def stamp(msg):
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def cmd_bollard(a):
    with open(a.file) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].strip().startswith("#")]
    if not rows[0][0].strip().lstrip("-").replace(".", "").isdigit():
        rows = rows[1:]                                    # header line
    pwm = np.array([float(r[0]) for r in rows])
    thrust = np.array([float(r[1]) for r in rows]) * (9.81 if a.kgf else 1.0)
    r = ident.bollard_curve(pwm, thrust, neutral=a.neutral, half_range=a.half_range)
    print(f"bollard pull, {len(pwm)} points:")
    print(f"    max_force: {r['max_force']:.1f}            # [N]")
    print(f"    reverse_ratio: {r['reverse_ratio']:.2f}")
    print(f"    pwm: {{neutral: {a.neutral}, deadzone: {round(r['deadzone'])}, half_range: {a.half_range}}}")
    if a.voltage:
        print(f"  measured at {a.voltage} V: thruster_model.rated_voltage: {a.voltage}")


def gnss_speed(fixes, smooth=1.0):
    """Speed over ground [m/s] from GNSS fixes: positions -> local metres, smoothed difference (straight runs)."""
    from scipy.signal import savgol_filter
    from cybership_utilities.marine_craft.sensors import llh2flat
    t = np.array([stamp(m) for _, m in fixes])
    lat0, lon0 = math.radians(fixes[0][1].latitude), math.radians(fixes[0][1].longitude)
    ne = np.array([llh2flat(math.radians(m.longitude), math.radians(m.latitude), 0.0, lon0, lat0)[:2]
                   for _, m in fixes])
    win = max(5, int(round(smooth / np.median(np.diff(t)))) | 1)
    v = savgol_filter(ne, win, 2, deriv=1, axis=0) / np.median(np.diff(t))
    return t, np.hypot(v[:, 0], v[:, 1])


def cmd_surge(a):
    force = f"{a.ns}/{a.force}"
    if a.gnss_speed:
        gnss = f"{a.ns}/{a.gnss}"
        d = read_bag(a.bag, [gnss, force])
        t, u = gnss_speed(d[gnss])
        print(f"speed from GNSS positions ({gnss}); drive straight, the speed is over ground")
    else:
        odom = f"{a.ns}/{a.odom}"
        d = read_bag(a.bag, [odom, force])
        t = np.array([s for s, _ in d[odom]])
        u = np.array([m.twist.twist.linear.x for _, m in d[odom]])
    tf = np.array([s for s, _ in d[force]])
    X = np.array([m.force.x for _, m in d[force]])
    F = X[np.clip(np.searchsorted(tf, t, side="right") - 1, 0, len(X) - 1)]   # hold the last command
    t, u, F = window(t, u, F, start=a.start, end=a.end)
    if a.u0:
        # the propellers give less than commanded when the boat moves (actuators.py): use what they give,
        # else the lost thrust shows up as extra hull drag. X is shared by the stern thrusters.
        n, fmax = a.thrusters, a.max_force * a.voltage_factor
        f = F / n * a.voltage_factor
        loss = np.sqrt(np.abs(f) * fmax) * u * np.sign(f) / (a.u0 * math.sqrt(a.voltage_factor))
        F = n * np.sign(f) * np.clip(np.abs(f) - loss, 0, 1.3 * np.abs(f))
        print(f"thrust corrected for speed: {n} thrusters, u_0 {a.u0} m/s, max {fmax:.0f} N each")
    # resample to 10 Hz for the fit
    tt = np.arange(t[0], t[-1], 0.1)
    r = ident.surge_response(tt, np.interp(tt, t, u), F[np.searchsorted(t, tt, side="right") - 1], decay=a.decay)
    print(f"surge, {tt[-1] - tt[0]:.0f} s of data:  M_11 = {r['M_11']:.1f} kg (incl. added mass), "
          f"X_u = {r['X_u']:.2f} N s/m, X_|u|u = {r['X_uu']:.2f} N s^2/m^2")
    print(f"    surge_time_constant: {r['surge_time_constant']:.2f}   # [s] in model.damping")
    print(f"    # X_|u|u = {r['X_uu']:.2f}: in vessel.yaml X_|u|u = max_thrust / max_speed^2, so keep max_thrust")
    print(f"    # and set max_speed = sqrt(max_thrust / {r['X_uu']:.2f}), e.g. {math.sqrt(134.0 / r['X_uu']):.2f} for 134 N")
    print("    # the force is the commanded one: calibrate the thrusters (bollard) first")


def cmd_roll(a):
    imu = f"{a.ns}/{a.imu}"
    d = read_bag(a.bag, [imu])
    from scipy.spatial.transform import Rotation
    t = np.array([stamp(m) for _, m in d[imu]])
    phi = np.array([Rotation.from_quat([m.orientation.x, m.orientation.y, m.orientation.z,
                                        m.orientation.w]).as_euler("xyz")[0] for _, m in d[imu]])
    t, phi = window(t, phi, start=a.start, end=a.end)
    r = ident.roll_decay(t, phi)
    print(f"roll decay: period {r['period']:.2f} s (omega_n {r['omega_n']:.2f} rad/s), zeta {r['zeta']:.3f}, "
          f"kappa {r['kappa']:.2f}")
    print(f"    zeta: [<heave>, {r['zeta']:.2f}, <pitch>]      # model.damping.zeta, roll is the middle value")
    print(f"    roll_quadratic_factor: {r['kappa']:.2f}")


def cmd_latency(a):
    topic = f"{a.ns}/{a.topic}"
    d = read_bag(a.bag, [topic])
    r = ident.latency([stamp(m) for _, m in d[topic]], [s for s, _ in d[topic]])
    print(f"{topic}: {r['n']} messages, delay mean {r['mean'] * 1000:.0f} ms, std {r['std'] * 1000:.0f} ms, "
          f"max {r['max'] * 1000:.0f} ms")
    key = "gnss" if "gnss" in a.topic else "imu" if "imu" in a.topic else a.topic
    print(f"    latency: {{{key}: {r['mean']:.3f}}}")
    print("    # only right if the sensor driver stamps with the measurement time (GNSS: the fix time)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="/microampere")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bollard"); b.add_argument("file"); b.add_argument("--kgf", action="store_true")
    b.add_argument("--neutral", type=int, default=1500); b.add_argument("--half-range", type=int, default=400)
    b.add_argument("--voltage", type=float)
    for name in ("surge", "roll", "latency"):
        p = sub.add_parser(name)
        p.add_argument("bag")
        p.add_argument("--start", type=float, help="[s] from the first message")
        p.add_argument("--end", type=float)
    sub.choices["surge"].add_argument("--odom", default="navigation/odom")
    sub.choices["surge"].add_argument("--force", default="control/force/command/mux")
    sub.choices["surge"].add_argument("--decay", type=float, default=3.0)
    sub.choices["surge"].add_argument("--gnss-speed", action="store_true",
                                      help="speed from the GNSS positions instead of an odometry topic")
    sub.choices["surge"].add_argument("--gnss", default="sensor/gnss")
    sub.choices["surge"].add_argument("--u0", type=float, help="zero-thrust inflow speed [m/s] (thruster_model)")
    sub.choices["surge"].add_argument("--thrusters", type=int, default=3,
                                      help="thrusters sharing X (the allocation also uses the bow azimuth)")
    sub.choices["surge"].add_argument("--max-force", type=float, default=67.0, help="[N] per thruster")
    sub.choices["surge"].add_argument("--voltage-factor", type=float, default=1.0,
                                      help="full thrust at the test voltage / at the rated voltage")
    sub.choices["roll"].add_argument("--imu", default="sensor/imu")
    sub.choices["latency"].add_argument("--topic", default="sensor/gnss")
    a = ap.parse_args()
    {"bollard": cmd_bollard, "surge": cmd_surge, "roll": cmd_roll, "latency": cmd_latency}[a.cmd](a)


if __name__ == "__main__":
    main()
