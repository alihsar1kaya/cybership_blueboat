<div align="center">

# Cybership BlueBoat DP simulator

Simulator and dynamic positioning (DP) controller for *Microampere*, a BlueBoat catamaran used in my master's thesis at NTNU.

![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-E95420?logo=ubuntu&logoColor=white)
![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-22314E?logo=ros&logoColor=white)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![License GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue)

[Install](#install) · [First run](#first-run) · [Examples](#examples) · [Reference](#reference) · [Change the boat](#change-the-boat) · [How it works](#how-it-works)

</div>

---

## What it does

You give the boat a goal in RViz. It drives there and holds its position and heading while waves, wind and current push it around.

The boat model follows Fossen's marine craft equations. The added mass and damping come from running Capytaine on the hull CAD, so they are computed rather than guessed. Every number about the boat sits in one YAML file, and each part of the system (simulator, observer, controller, thrust allocation) is a separate ROS 2 node. That makes it easy to change the boat, or to swap the simulator for the real boat later.

---

## Install

You need Ubuntu 24.04 and ROS 2 Jazzy ([install guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)). The rest takes about ten minutes.

```bash
# 1. system packages
sudo apt install ros-jazzy-desktop ros-dev-tools python3-venv python3-pip \
                 ros-jazzy-topic-tools ros-jazzy-nav2-msgs ros-jazzy-xacro

# 2. get the code
mkdir -p ~/ros_ws/src && cd ~/ros_ws/src
git clone --recursive https://github.com/alihsar1kaya/cybership_blueboat.git

# 3. Python packages, in a virtual environment that can still see ROS
cd ~/ros_ws
python3 -m venv venv --system-site-packages && touch venv/COLCON_IGNORE
source /opt/ros/jazzy/setup.bash && source venv/bin/activate
pip install -r src/cybership_blueboat/requirements.txt

# 4. build
sudo rosdep init 2>/dev/null; rosdep update
rosdep install --from-paths src -i -y --rosdistro jazzy
colcon build --symlink-install
```

Every terminal you open afterwards has to be sourced before ROS commands work:

```bash
cd ~/ros_ws && source /opt/ros/jazzy/setup.bash && source venv/bin/activate && source install/setup.bash
```

---

## First run

```bash
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true
```

RViz opens with the boat at the origin. Pick **2D Goal Pose** in the toolbar, then click on the map where the boat should go and drag in the direction it should point. Short moves of a metre or so are done sideways. For longer ones the boat turns, drives forwards or backwards (whichever is quicker) and then turns to the heading you chose. Once there it stays within a centimetre or two.

Stop everything with `Ctrl+C` in the launch terminal.

---

## Examples

Each example is a complete command you can paste. They all start the controller (`dp:=true`), so you can give goals in RViz as in the first run.

### 1. Calm water, perfect knowledge

```bash
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true
```

The controller is given the true position and velocity straight from the simulator. This is the best the boat can do and a good baseline when you compare other settings.

### 2. Realistic sensors

```bash
# accurate RTK GNSS (about 2 cm)
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true gnss:=rtk

# ordinary GNSS (about 20 cm noise and a slow drift of a few decimetres)
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true
```

With `observer:=true` the controller no longer sees the truth. It gets GNSS, compass and gyro readings at their real rates, and a Kalman filter turns them into an estimate. With RTK the boat holds 1–2 cm. With ordinary GNSS it holds 5–15 cm, but it slowly wanders with the GNSS error because it has no way of knowing the receiver is wrong. That is why the real boat should have RTK.

### 3. Waves

Pick a wave height with `wave_hs` (in metres). The BlueBoat is built for calm water, so the useful range is small:

| Sea | Command addition | What you see |
|---|---|---|
| Light ripples | `waves:=true wave_hs:=0.02` | barely moves |
| Default | `waves:=true` (0.05 m) | rolls and pitches about 0.5° |
| Choppy for this boat | `waves:=true wave_hs:=0.1` | rolls and pitches about 1°, heaves 2 cm |

For example:

```bash
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true gnss:=rtk waves:=true wave_hs:=0.1
```

You can also change the waves while the simulator runs. Open a second terminal and turn them up, down or off:

```bash
ros2 param set /microampere/sim environment.waves.Hs 0.03            # calmer
ros2 param set /microampere/sim environment.waves.Hs 0.1             # rougher
ros2 param set /microampere/sim environment.waves.Hs 0               # off
ros2 param set /microampere/sim environment.waves.direction_deg 90   # waves now travel towards east
ros2 param set /microampere/sim environment.waves.Tp 2.0             # shorter, steeper waves
```

The boat rocks with the waves, but the controller should not try to fight that motion. The Kalman filter separates the slow drift from the fast wave motion, so the thrusters stay calm. You can see this on `sim/path_lf` in RViz, the track without the wave motion.

### 4. Wind

```bash
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true gnss:=rtk wind_speed:=8 wind_dir:=90
```

This is an 8 m/s wind (a fresh breeze, Beaufort 5) blowing towards east, so it hits the boat from the side. The bow thruster pushes back with about 6 N and the boat still holds within a centimetre.

```bash
ros2 param set /microampere/sim environment.wind.speed 12           # stronger
ros2 param set /microampere/sim environment.wind.direction_deg 180  # now blowing towards south
ros2 param set /microampere/sim environment.wind.sigma 0.5          # gusts on
ros2 param set /microampere/sim environment.wind.mu 0.1             # how quickly gusts die out [1/s]
```

### 5. Current

```bash
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true gnss:=rtk
# then, in a second terminal:
ros2 param set /microampere/sim environment.current.speed 0.15
ros2 param set /microampere/sim environment.current.direction_deg 90
```

Current is the hardest load for this boat. Sideways it can only push back with the bow thruster, and a side current above roughly 0.25 m/s is more than that thruster can hold. Head-on the stern thrusters take it easily, so if the current gets strong, give the boat a goal heading that points the bow into it.

### 6. Everything at once

```bash
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true gnss:=rtk \
    waves:=true wave_hs:=0.05 wave_dir:=45 wind_speed:=6 wind_dir:=90
ros2 param set /microampere/sim environment.current.speed 0.1      # in a second terminal
```

### 7. Without RViz

```bash
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true gnss:=rtk use_gui:=false
```

Useful on a server or for automatic tests. Send goals from the terminal:

```bash
# 5 m north, 3 m east, heading 45° (quaternion z = sin(45°/2), w = cos(45°/2))
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: world}, pose: {position: {x: 5.0, y: 3.0}, orientation: {z: 0.383, w: 0.924}}}"
```

The scripts in [`tools/`](tools/README.md) do this for you and log the result. This one moves the boat 10 m north in a side current and prints the path length and how well it held position afterwards:

```bash
GOAL="10 0 0" CURRENT="0.15 90" bash src/cybership_blueboat/tools/run_goal_motion_test.sh demo 0 observer:=true gnss:=rtk
```

### 8. Sensor delays and GNSS problems

The simulator delays the sensors and the thruster commands the way real hardware does: GNSS arrives about 0.1 s after the measurement, and a thruster reacts about 0.05 s after the command. The values are under `latency` in `vessel.yaml`. The Kalman filter knows when each GNSS fix was taken and corrects for the delay.

You can also take the GNSS away while the boat holds position:

```bash
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true gnss:=rtk
# in a second terminal:
ros2 param set /microampere/sim sensors.gnss.available false   # GNSS gone
ros2 param set /microampere/sim sensors.gnss.available true    # back
ros2 param set /microampere/sim sensors.gnss.rtk_fix false     # RTK fix lost, accuracy drops to ~20-50 cm
ros2 param set /microampere/sim sensors.gnss.rtk_fix true      # fix back
ros2 param set /microampere/sim sensors.gnss.outage_rate 1     # random outages, about one per minute
```

Without GNSS the filter keeps going on the boat model and the compass (dead reckoning), and the observer prints a warning. In a 0.1 m/s current the boat drifted at most 8 cm during a 20 s outage, then returned once GNSS came back. Losing the RTK fix is worse: the boat follows the receiver's slow error and wanders 20–40 cm, because nothing tells it the position is wrong.

### 9. Battery voltage and thrust at speed

The thrusters lose force in two ways, as on the real boat. A lower battery voltage gives less thrust for the same command (the T200 gives 6.7 kgf at 20 V but 5.25 kgf at 16 V), and a propeller pushes less the faster water flows into it, so the thrust falls as the boat picks up speed.

```bash
ros2 launch cybership_simulator vessel.launch.py vessel:=microampere dp:=true observer:=true gnss:=rtk
# in a second terminal:
ros2 param set /microampere/sim battery.voltage 14.8                 # a 4S battery, about 70 % of full thrust
ros2 param set /microampere/sim environment.current.speed 0.2
ros2 param set /microampere/sim environment.current.direction_deg 90
```

The controller is not told about the voltage, so it has to make up for the weaker thrusters through its bias estimate. At 14.8 V in a 0.2 m/s side current the boat held within 6 cm, against 2 cm at full voltage, with the bow thruster close to its limit.

### Switching the controller off and on

```bash
ros2 service call /microampere/position_controller/change_state std_srvs/srv/SetBool "{data: false}"   # thrusters to zero
ros2 service call /microampere/position_controller/change_state std_srvs/srv/SetBool "{data: true}"    # back on
```

---

## Reference

### Launch arguments

Add them to the end of the `ros2 launch` line as `name:=value`. They are read once at start.

| Argument | Default | Unit | Meaning |
|---|---|---|---|
| `vessel` | `voyager` | | which boat; loads `cybership_config/config/<vessel>/vessel.yaml` |
| `dp` | `false` | | start the controller, guidance and thrust allocation |
| `observer` | `false` | | feed the controller from simulated sensors and a Kalman filter instead of the true state |
| `gnss` | `standard` | | `standard` or `rtk`; both profiles are defined in `vessel.yaml` |
| `noisy_odom` | `false` | | without the observer: give the controller the true state plus white noise |
| `waves` | `false` | | start with waves |
| `wave_hs` | `0.05` | m | significant wave height; keep it below about 0.15 m for this boat |
| `wave_tp` | `2.3` | s | peak period; 1.5–3 s fits short, wind-driven waves |
| `wave_dir` | `0` | deg | direction the waves travel towards |
| `wind_speed` | `0` | m/s | mean wind speed |
| `wind_dir` | `0` | deg | direction the wind blows towards |
| `use_gui` | `true` | | start RViz |

### Weather and sensors while running

Change these with `ros2 param set /microampere/sim <name> <value>`. They take effect at once.

| Parameter | Default | Unit | Meaning |
|---|---|---|---|
| `environment.waves.Hs` | from `wave_hs` | m | wave height, 0 turns the waves off |
| `environment.waves.Tp` | from `wave_tp` | s | peak period |
| `environment.waves.direction_deg` | from `wave_dir` | deg | travel direction |
| `environment.waves.spreading` | `1` | | 1 for short-crested (realistic) waves, 0 for long, parallel crests |
| `environment.wind.speed` | from `wind_speed` | m/s | mean wind speed, 0 turns the wind off |
| `environment.wind.direction_deg` | from `wind_dir` | deg | direction the wind blows towards |
| `environment.wind.sigma` | `0` | m/s/√s | gust strength, 0 for steady wind |
| `environment.wind.mu` | `0` | 1/s | how fast gusts return to the mean |
| `environment.current.speed` | `0` | m/s | current speed, 0 turns it off |
| `environment.current.direction_deg` | `0` | deg | direction the current flows towards |
| `battery.voltage` | `20` | V | battery voltage; lower gives less thrust (T200 thrust-voltage curve) |
| `sensors.gnss.available` | `true` | | false simulates a GNSS outage |
| `sensors.gnss.rtk_fix` | `true` | | false makes an RTK receiver fall back to normal accuracy (only with `gnss:=rtk`) |
| `sensors.gnss.outage_rate` | `0` | 1/min | random outages per minute, 0 for none |
| `sensors.gnss.outage_duration` | `10` | s | length of each random outage |

Directions use the marine convention: x is north, y is east, and 90° is east. Wind, waves and current are all given as the direction they move towards, so a wind coming from the north is `wind_dir:=180`.

---

## Change the boat

All boat data is in [`cybership_config/config/microampere/vessel.yaml`](cybership_config/config/microampere/vessel.yaml). The simulator, the controller and the observer read the same file, so you only change things in one place. Edit it and restart the launch; you don't need to rebuild.

| Section | Contents |
|---|---|
| `model` | mass, inertia, hydrodynamics, damping and the wind areas |
| `thrusters` | one entry per thruster with position, direction and strength |
| `sensors` | rate and noise for GNSS, compass and gyro |
| `thruster_model` | battery voltage, thrust-voltage curve and how fast thrust drops with speed |
| `latency` | delays of GNSS, IMU and thruster commands, in seconds |
| `allocation` | how the total force is split between the thrusters |

Each value has a comment saying where it came from. The ones marked ESTIMATE still need measuring on the real boat.

<details>
<summary>Changing a thruster</summary>

```yaml
thrusters:
  - name: port
    type: fixed                     # fixed or azimuth (rotating)
    frame: stern_port_thruster_link # link in the URDF, used by RViz
    position: [-0.55, -0.29, 0.177] # [m] x forward, y starboard, z down
    direction_deg: 0.0              # 0 pushes forward
    max_force: 67.0                 # [N]
    reverse_ratio: 0.8              # reverse thrust is 80 % of forward
```

To make it stronger, raise `max_force`. To move it, change `position`. You can delete a thruster or copy one with a new `name` and `position`; the allocation works with whatever thrusters are listed. Without the bow thruster the boat can no longer push sideways, so expect it to drift in a cross current.
</details>

<details>
<summary>Making the controller faster or calmer</summary>

The settings are in [`cybership_config/config/microampere/controller_position.yaml`](cybership_config/config/microampere/controller_position.yaml). The gains are computed from the boat model, so you mostly choose how fast it should react.

| Setting | Higher | Lower |
|---|---|---|
| `control.omega_n` | holds tighter and uses more thrust | softer |
| `filter.omega` | reaches a new goal sooner | smoother moves |
| `guidance.forward_speed` | faster transit | slower transit |

You can change them while it runs, for example `ros2 param set /microampere/position_controller guidance.forward_speed 0.4`.
</details>

<details>
<summary>Adding your own boat</summary>

1. Copy the folder: `cp -r cybership_config/config/microampere cybership_config/config/myboat`
2. Edit `myboat/vessel.yaml`. If you have no Capytaine results, delete `model.hydro` and set `added_mass` by hand.
3. Rebuild once so the new folder gets installed: `colcon build --symlink-install --packages-select cybership_config`
4. Start it with `vessel:=myboat`.

If the boat has no URDF yet it still runs, but RViz only shows the coordinate frames.
</details>

---

## How it works

The layout follows Fossen's guidance, navigation and control diagram. Each box below is its own node and they only talk through topics. To run on the real boat, you replace the simulator box with the sensor and thruster drivers and leave the rest alone.

```mermaid
flowchart LR
    GOAL["Goal<br/>(RViz)"] --> CTRL
    subgraph DP["DP stack"]
        OBS["Observer<br/>Kalman filter"] --> CTRL["Guidance and<br/>controller"]
        CTRL --> ALLOC["Thrust<br/>allocation"]
    end
    ALLOC --> THR
    subgraph SIM["Simulator"]
        THR["Thrusters"] --> CRAFT["Boat<br/>6-DOF model"]
        ENV["Waves, wind,<br/>current"] --> CRAFT
        CRAFT --> SENS["GNSS, compass,<br/>gyro"]
    end
    SENS --> OBS
```

<details>
<summary>Topics</summary>

All topics are under the vessel name, for example `/microampere/sensor/gnss`. A node that replaces a box has to publish and subscribe to the same ones.

| Topic | Type | From and to |
|---|---|---|
| `sensor/gnss` | `sensor_msgs/NavSatFix` | simulator or real GNSS to the observer (5 Hz) |
| `sensor/imu` | `sensor_msgs/Imu` | simulator or real IMU to the observer (50 Hz) |
| `navigation/odom` | `nav_msgs/Odometry` | observer to the controller |
| `control/force/command/position` | `geometry_msgs/Wrench` | controller to the force mux (body frame, N and Nm) |
| `control/force/command/manual` | `geometry_msgs/Wrench` | joystick or your own node to the force mux |
| `control/force/command/mux` | `geometry_msgs/Wrench` | force mux to the allocation |
| `thruster/<name>/command` | `geometry_msgs/Wrench` | allocation to the simulator or real thrusters |
| `sim/odom_true` | `nav_msgs/Odometry` | true state from the simulator, for evaluation |
| `sim/path_lf` | `nav_msgs/Path` | the track without wave motion, for RViz |

To drive manually through the mux: `ros2 service call /microampere/force_mux/select topic_tools_interfaces/srv/MuxSelect "{topic: control/force/command/manual}"`
</details>

<details>
<summary>Models and references</summary>

| Part | Code | Fossen (2021) |
|---|---|---|
| Rigid body, kinematics, Coriolis | `cybership_utilities/marine_craft/` | ch. 2–3 |
| Added mass and damping from Capytaine | `cybership_config/tools/`, `hydro_capytaine.npz` | ch. 5–6 |
| Current, wind (Blendermann) and waves (JONSWAP) | `marine_craft/environment.py`, `waves.py` | ch. 10 |
| Sensors | `marine_craft/sensors.py` | ch. 13–14 |
| Kalman filter with wave filter | `cybership_utilities/observer.py` | ch. 13.4.6 |
| DP controller and guidance | `cybership_dp/nodes/position_control_node.py` | ch. 15 |
| Thrust allocation (QP) | `cybership_dp/constrained_allocation.py` | ch. 11.2.3 |
| ALOS path following, off by default | `cybership_dp/los.py` | ch. 12.5 |
</details>

---

## Tests

```bash
python3 -m pytest -q src/cybership_blueboat/cybership_utilities/test src/cybership_blueboat/cybership_dp/test
```

The unit tests check the boat model, the weather, the sensors, the Kalman filter, the thrust allocation and the path following. They take about a minute and need no ROS nodes running.

---

## Known limitations

- Several boat values are still estimates (centre of gravity, inertia, surge time constant, thruster positions, the thrust at speed, and which voltage the 67 N thruster rating refers to). They are marked ESTIMATE in `vessel.yaml` and should be measured on the real boat.
- The slow drift force from waves (second order) is not modelled. The Kalman filter's bias estimate absorbs it, as it would on the real boat.
- The wind areas come from the CAD model. The real boat's antennas and electronics are not in it.
- There is no connection to the real boat yet. The topics are laid out for it, but the drivers still have to be written.
- Only one goal at a time; there are no waypoint routes and no obstacle avoidance.

---

## Troubleshooting

<details>
<summary>Nothing happens, or two terminals don't see each other</summary>

Check that both terminals ran the `source` line and use the same `ROS_DOMAIN_ID` (`echo $ROS_DOMAIN_ID`).
</details>

<details>
<summary>"Address already in use", or two boats fight each other</summary>

An older simulator is still running. Stop it with `Ctrl+C` in its terminal before starting a new one.
</details>

<details>
<summary>The build fails in a hardware package</summary>

You only need the simulator packages:
`colcon build --symlink-install --packages-up-to cybership_simulator cybership_dp cybership_viz`
</details>

<details>
<summary>NumPy errors</summary>

ROS 2 Jazzy is built against NumPy 1.x. Fix it with `pip install "numpy<2"`.
</details>

---

## Credits

This repository builds on the [Cybership Software Suite](https://github.com/NTNU-MCS/cybership_software_suite) from NTNU-MCS ([DOI](https://doi.org/10.5281/zenodo.17233653)). Their original README is in [`docs/`](docs/README_cybership.md). The models follow T. I. Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control* (2021), and the [MSS toolbox](https://github.com/cybergalactic/MSS). The hull geometry is from the [Blue Robotics BlueBoat](https://bluerobotics.com/store/boat/blueboat/blueboat/). Licensed under GPL-3.0, see [`LICENSE.md`](LICENSE.md).
