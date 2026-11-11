# Measuring the boat

Most numbers in `cybership_config/config/microampere/vessel.yaml` are still estimates, marked ESTIMATE. This page describes the tests that replace them with measured values, and how `tools/identify.py` turns the logs into the lines to paste into `vessel.yaml`.

The estimators were checked against the simulator first. On simulated test data they find the simulator's own values again (see `cybership_utilities/test/test_identification.py`), so if a real test gives something odd, the data is the first suspect.

## Before you go out

Weigh the boat with batteries and everything it carries, and write down the battery voltage at the start and end of each test. Thrust depends on voltage, and the T200 gives about 20 % less at 16 V than at 20 V.

Record everything with one command and keep it running during the test:

```bash
ros2 bag record -o test_<name> /microampere/sensor/gnss /microampere/sensor/imu \
    /microampere/navigation/odom /microampere/control/force/command/mux
```

## 1. Bollard pull (thrust curve)

Tie the boat to the dock with a spring scale or load cell in the line. Step one thruster through PWM values, for example 1100 to 1900 µs in 50 µs steps, and note the steady pull at each. Do forward and reverse. Write the results in a CSV file with one `pwm, thrust` pair per line (thrust in N, or in kg with `--kgf`).

```bash
python3 tools/identify.py bollard bollard.csv --voltage 16.2
```

This gives `max_force`, `reverse_ratio` and the ESC deadzone for the `thrusters` entries, and the voltage it applies to (`thruster_model.rated_voltage`). Do this test first, because the surge test needs the thrust.

## 2. Surge steps (mass and drag)

On a calm day with no current, put the controller in manual through the force mux and give the boat a few forward thrust steps on a straight line: a small one (a few N) held until the speed settles, then a medium and a large one, each for 30–40 s.

```bash
ros2 service call /microampere/force_mux/select topic_tools_interfaces/srv/MuxSelect "{topic: control/force/command/manual}"
ros2 topic pub -r 10 /microampere/control/force/command/manual geometry_msgs/msg/Wrench "{force: {x: 8.0}}"
```

Then:

```bash
python3 tools/identify.py surge test_surge --gnss-speed --u0 5.0
```

It fits the surge mass (including added mass), the linear drag `X_u` and the quadratic drag `X_|u|u`, and prints `surge_time_constant`. Use the GNSS speed rather than the observer's: the observer already assumes a mass, so it would partly give back its own guess.

`--u0` corrects the commanded force for the thrust the propellers lose at speed. Without it, the lost thrust looks like extra hull drag, and in the simulator test `X_|u|u` came out three times too high. With it, the simulator test gave the mass within 3 % and `X_|u|u` within 2 %. `X_u` came out about 20 % low, because at low throttle the propeller losses are large compared with the drag, so treat the time constant as the least certain number from this test.

## 3. Top speed (thrust lost at speed)

Drive straight at full throttle on the stern thrusters until the speed stops rising and note it from the GNSS. Do not put this number in `max_speed`: `max_speed` and `max_thrust` only describe the hull resistance, `X_|u|u = max_thrust / max_speed^2`, and that comes from the surge test above.

The top speed tells you how much thrust the propellers lose at speed. At the top speed `U` the resistance `X_|u|u U^2` equals the thrust that is left, `T_0 (1 - U / u_0)`, where `T_0` is the bollard pull of the two stern thrusters at the battery voltage of the test. Solve for `u_0` and set `thruster_model.zero_thrust_speed`:

```
u_0 = U / (1 - X_|u|u U^2 / T_0)
```

The surge fit needs `u_0` and this needs `X_|u|u`, so go around once: fit the surge test with the estimate (5.0), compute `u_0` from the top speed, and fit again. With the estimates in `vessel.yaml` the simulator reaches about 2.3 m/s.

## 4. Roll decay

With the boat at rest, push one side down by about 10° and let go. Keep the IMU logging until the roll has died out. Repeat a few times.

```bash
python3 tools/identify.py roll test_roll --start 3 --end 20
```

Pick `--start` just after you let go. It gives the roll period, the relative damping `zeta` (the middle value of `model.damping.zeta`) and `roll_quadratic_factor`. Pitch can be done the same way by pushing the bow down; use the pitch angle instead of roll.

## 5. Latency

Any bag from the boat will do:

```bash
python3 tools/identify.py latency test_surge --topic sensor/gnss
python3 tools/identify.py latency test_surge --topic sensor/imu
```

This compares when each message was measured (its header stamp) with when it arrived, and gives the values for the `latency` section. It is only correct if the driver stamps the message with the measurement time; most GNSS drivers use the fix time, but check yours. The thruster command delay cannot be read from the log without a thrust or RPM measurement, so leave it at the estimate unless the ESCs report RPM.

## Checking the result

After updating `vessel.yaml`, repeat one of the tests in the simulator (for example a 10 N surge step) and compare the speed curve with the logged one. If they match, the model is good for that motion.
