"""
Contract tests for thrusters.

These tests talk to the vessel only through ROS topics, so the same tests can run
against the simulator or against the real boat:

    thruster/<name>/command  (geometry_msgs/Wrench)            <- test publishes
    thruster/<name>/state    (cybership_interfaces/ThrusterState) -> test checks
    control/force/applied    (geometry_msgs/WrenchStamped)      -> test checks (sim)

Usage (simulator or boat must already be running):

    VESSEL=voyager pytest -v cybership_tests/test/test_thruster_contract.py

WARNING: on a real boat these tests spin the thrusters. Only run with the boat secured.
"""

import math
import os
import time

import pytest
import rclpy
from geometry_msgs.msg import Wrench, WrenchStamped
from cybership_interfaces.msg import ThrusterState

VESSEL = os.environ.get("VESSEL", "voyager")
TIMEOUT_S = float(os.environ.get("TEST_TIMEOUT_S", "3.0"))
FORCE_TOL = 0.05   # [N]
ANGLE_TOL = 0.05   # [rad]


@pytest.fixture(scope="module")
def node():
    rclpy.init()
    n = rclpy.create_node("thruster_contract_test", namespace=VESSEL)
    yield n
    n.destroy_node()
    rclpy.shutdown()


def command_and_wait(node, thruster, cmd: Wrench, msg_type, topic, accept):
    """Publish cmd on thruster/<thruster>/command until a message on `topic` satisfies `accept`."""
    pub = node.create_publisher(Wrench, f"thruster/{thruster}/command", 10)
    received = []
    sub = node.create_subscription(msg_type, topic, received.append, 10)
    try:
        t_end = time.monotonic() + TIMEOUT_S
        while time.monotonic() < t_end:
            pub.publish(cmd)
            rclpy.spin_once(node, timeout_sec=0.05)
            for m in received:
                if accept(m):
                    return m
            received.clear()
        return None
    finally:
        pub.publish(Wrench())  # stop the thruster again
        node.destroy_subscription(sub)
        node.destroy_publisher(pub)


def test_fixed_thruster_reports_commanded_force(node):
    """A fixed thruster commanded 0.5 N should report ~0.5 N and angle 0."""
    cmd = Wrench()
    cmd.force.x = 0.5
    state = command_and_wait(
        node, "tunnel", cmd, ThrusterState, "thruster/tunnel/state",
        lambda m: abs(m.force.x - 0.5) < FORCE_TOL,
    )
    assert state is not None, f"No matching thruster/tunnel/state within {TIMEOUT_S} s"
    assert abs(state.angle) < ANGLE_TOL


def test_azimuth_points_where_commanded(node):
    """An azimuth commanded straight sideways (0, 0.5) should report angle ~ +90 deg."""
    cmd = Wrench()
    cmd.force.y = 0.5
    state = command_and_wait(
        node, "port", cmd, ThrusterState, "thruster/port/state",
        lambda m: abs(m.force.y - 0.5) < FORCE_TOL,
    )
    assert state is not None, f"No matching thruster/port/state within {TIMEOUT_S} s"
    assert abs(state.angle - math.pi / 2) < ANGLE_TOL


def test_applied_force_includes_thruster(node):
    """Total applied force should contain the sideways force from the tunnel thruster."""
    cmd = Wrench()
    cmd.force.x = 0.5
    applied = command_and_wait(
        node, "tunnel", cmd, WrenchStamped, "control/force/applied",
        lambda m: abs(m.wrench.force.y - 0.5) < FORCE_TOL,
    )
    assert applied is not None, f"No matching control/force/applied within {TIMEOUT_S} s"
    assert applied.wrench.torque.z > 0.0  # bow thruster pushing to starboard turns the bow to starboard
