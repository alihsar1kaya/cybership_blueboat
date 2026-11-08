# Test and plot scripts (Microampere DP)

Run from `~/ros_ws` (`~/ros_ws/tools` is a symlink to this folder). Headless tests use ROS_DOMAIN_ID 81.

| Script | What it does |
|---|---|
| `run_dp_step_test.sh <vessel> <label> [launch args]` | DP step test (surge, sway, yaw), measures `sim/odom_true` |
| `run_dp_current_test.sh <speed> <dir> <label>` | DP in a cross current |
| `run_dp_wave_test.sh <Hs> <Tp> <dir> <label> [launch args]` | DP in waves |
| `GOAL="x y psi" [CURRENT="v dir"] [PF=true] run_goal_motion_test.sh <label> <Hs> [launch args]` | one goal manoeuvre (path, heading, hold), optional current and ALOS on/off |
| `run_shake_test.sh` | roll/pitch and thruster activity while holding |
| `model_report.py --label <label>` | time constants, natural periods, open-loop tests |
| `observer_fair_test.py` | our DP Kalman filter vs. cybership_observer Luenberger, same data, both tuned |
| `plot_hydro.py`, `plot_dp_step.py` | plots of Capytaine data and step tests |

Older Voyager scripts: `run_four_corner_sim.sh`, `run_docking_sim.sh`, `waypoints.py`, `record_dp.py`.
Results go to `~/ros_ws/results/`.
