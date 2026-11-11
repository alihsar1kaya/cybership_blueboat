# Test and plot scripts (Microampere DP)

Run from `~/ros_ws` with the workspace sourced, e.g. `bash src/<repo>/tools/run_goal_motion_test.sh demo 0`. Headless tests use their own ROS_DOMAIN_ID and no web GUI, so they do not disturb a running simulator.

| Script | What it does |
|---|---|
| `run_dp_step_test.sh <vessel> <label> [launch args]` | DP step test (surge, sway, yaw), measures `sim/odom_true` |
| `run_dp_current_test.sh <speed> <dir> <label>` | DP in a cross current |
| `run_dp_wave_test.sh <Hs> <Tp> <dir> <label> [launch args]` | DP in waves |
| `GOAL="x y psi" [CURRENT="v dir"] [WIND="v dir"] [BATTERY=volts] [PF=true] run_goal_motion_test.sh <label> <Hs> [launch args]` | one goal manoeuvre (path, heading, hold), optional current, wind, battery voltage and ALOS on/off |
| `run_shake_test.sh` | roll/pitch and thruster activity while holding |
| `sweep_test.py --label <label>` | drives a running simulator through 14 goals (ahead, astern, sideways, diagonal, turns on the spot) and logs the state, every thruster and the allocation slack |
| `sweep_analysis.py <label> ... [--logs <dir>] [--out <name>]` | per goal: chosen manoeuvre, settle time, overshoot, cross-track error, thruster use; table and figures in `results/<name>/` (default `sweep`) |
| `model_report.py --label <label>` | time constants, natural periods, open-loop tests |
| `observer_fair_test.py` | our DP Kalman filter vs. cybership_observer Luenberger, same data, both tuned |
| `plot_hydro.py`, `plot_dp_step.py` | plots of Capytaine data and step tests |

Older Voyager scripts: `run_four_corner_sim.sh`, `run_docking_sim.sh`, `waypoints.py`, `record_dp.py`.
Results go to `~/ros_ws/results/`.

Measuring the real boat: [`identification.md`](identification.md) describes the tests (bollard pull, surge steps, top speed, roll decay, latency), and `identify.py` turns the logged rosbags into the values for `vessel.yaml`.
