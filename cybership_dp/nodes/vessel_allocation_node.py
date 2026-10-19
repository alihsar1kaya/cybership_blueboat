#!/usr/bin/env python3
"""
Thrust allocation for any vessel with a vessel.yaml.

    ros2 run cybership_dp vessel_allocation_node.py --ros-args -r __ns:=/voyager \
        -p vessel_model:=voyager -r control/force/command:=control/force/command/mux

Subscribes: control/force/command (geometry_msgs/Wrench)
Publishes:  thruster/<name>/command (geometry_msgs/Wrench), one per thruster in vessel.yaml
"""

import rclpy
import rclpy.executors

from cybership_dp.vessel_force_controller import VesselForceControllerROS


def main(args=None):
    rclpy.init(args=args)
    node = VesselForceControllerROS()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
