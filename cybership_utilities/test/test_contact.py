"""
Hull-quay contact (fenders), no ROS:

    cd ~/ros_ws/src/cybership_software_suite && python3 -m pytest -v cybership_utilities/test/test_contact.py
"""
import math

import numpy as np
import pytest

from cybership_utilities.marine_craft.contact import QuayContact

QUAY = [(7.25, -3.0), (8.75, -3.0), (8.75, 3.0), (7.25, 3.0)]     # the south face is x = 7.25
L, B = 1.195, 0.93


def test_no_force_clear_of_the_quay():
    c = QuayContact([QUAY], L, B)
    assert np.allclose(c.forces([6.0, 0.0, math.pi / 2], [0.0, 0.0, 0.0]), 0.0)
    assert c.last == []


def test_side_pressed_in_is_pushed_out_without_turning():
    c = QuayContact([QUAY], L, B, stiffness=3000.0, damping=0.0)
    # heading east, port side (-y BODY, north) 2 cm into the face along its whole length
    x = 7.25 - B / 2 + 0.02
    tau = c.forces([x, 0.0, math.pi / 2], [0.0, 0.0, 0.0])
    n_points = len(c.last)
    assert n_points == 7                                   # the port side's points
    assert tau[1] == pytest.approx(n_points * 3000.0 * 0.02)   # BODY +y (starboard) = NED south = out of the quay
    assert abs(tau[0]) < 1e-9 and abs(tau[5]) < 1e-6         # no surge (no slip), no yaw (symmetric)


def test_fender_only_pushes():
    c = QuayContact([QUAY], L, B, stiffness=3000.0, damping=400.0)
    x = 7.25 - B / 2 + 0.005
    # moving out fast: the damper would pull, the fender gives nothing
    tau = c.forces([x, 0.0, math.pi / 2], [0.0, 0.5, 0.0])         # sway +0.5 (starboard = south): out
    assert np.allclose(tau, 0.0)


def test_corner_touch_turns_the_boat():
    c = QuayContact([QUAY], L, B, damping=0.0)
    # heading north-east, the bow's port corner into the face
    psi = math.radians(60)
    corner = np.array([L / 2, -B / 2])
    R = np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])
    pos = np.array([7.25 + 0.02, 0.0]) - R @ corner
    tau = c.forces([pos[0], pos[1], psi], [0.0, 0.0, 0.0])
    assert len(c.last) >= 1
    assert abs(tau[5]) > 1.0                                # the push at the bow corner gives a yaw moment


def test_friction_opposes_sliding_along_the_face():
    c = QuayContact([QUAY], L, B, damping=0.0, friction=0.3)
    x = 7.25 - B / 2 + 0.02
    tau = c.forces([x, 0.0, math.pi / 2], [0.3, 0.0, 0.0])          # sliding east (surge) along the face
    assert tau[0] < 0.0 and abs(tau[0]) <= 0.3 * abs(tau[1]) + 1e-9
