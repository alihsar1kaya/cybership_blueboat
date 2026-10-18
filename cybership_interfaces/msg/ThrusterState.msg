# Actual (measured or simulated) state of a single thruster.
# Published on thruster/<name>/state by both the simulator and the hardware driver,
# so the same tests can run against either.

std_msgs/Header header       # frame_id: the thruster's frame

geometry_msgs/Vector3 force  # actual force produced [N], in the thruster frame
float64 rpm                  # actual propeller speed [rev/min], NaN if unknown
float64 angle                # actual azimuth angle [rad], 0 for fixed thrusters
float64 angle_rate           # actual azimuth angle rate [rad/s], 0 for fixed thrusters
