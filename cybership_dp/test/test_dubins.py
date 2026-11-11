import math
import random

import pytest

from cybership_dp.dubins import DubinsPath, wrap


def _check(path, start, goal):
    """Starts at start, ends at goal, and every step moves along the heading (no jumps or kinks)."""
    x, y, h = path.at(0.0)
    assert math.hypot(x - start[0], y - start[1]) < 1e-9 and abs(wrap(h - start[2])) < 1e-9
    n = 400
    ds = path.length / n
    prev = (x, y, h)
    for k in range(1, n + 1):
        q = path.at(ds * k)
        mid = prev[2] + wrap(q[2] - prev[2]) / 2
        assert math.hypot(q[0] - prev[0] - ds * math.cos(mid), q[1] - prev[1] - ds * math.sin(mid)) < 0.02 * ds + 1e-9
        prev = q
    assert math.hypot(prev[0] - goal[0], prev[1] - goal[1]) < 1e-9 and abs(wrap(prev[2] - goal[2])) < 1e-9


def test_every_dubins_word_connects_the_poses():
    rng = random.Random(1)
    for _ in range(200):
        s = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-math.pi, math.pi))
        g = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-math.pi, math.pi))
        for length, word, segs in DubinsPath.candidates(s, g, 1.5):
            path = DubinsPath(segs, word)
            assert path.length == pytest.approx(length)
            _check(path, s, g)


def test_known_paths():
    # straight ahead: just the line
    p = DubinsPath.shortest((0, 0, 0), (10, 0, 0), 2.0)
    assert p.length == pytest.approx(10.0)
    # quarter turn to starboard (psi grows towards east): R pi/2
    p = DubinsPath.shortest((0, 0, 0), (2, 2, math.pi / 2), 2.0)
    assert p.word.startswith("R") and p.length == pytest.approx(math.pi)
    # 3 m sideways with R = 2 cannot be an S-bend (needs 2R between the circles): a long loop
    assert DubinsPath.shortest((0, 0, 0), (0, 3, 0), 2.0).length > 15


def test_projection_follows_progress_and_cross_track_sign():
    p = DubinsPath.shortest((0, 0, 0), (2, 2, math.pi / 2), 2.0)       # one starboard arc
    s, _, pi_h, ye, kappa = p.project(2 * math.sin(0.5), 2 - 2 * math.cos(0.5))
    assert s == pytest.approx(1.0) and ye == pytest.approx(0.0, abs=1e-9)
    assert pi_h == pytest.approx(0.5) and kappa == pytest.approx(0.5)
    # 0.1 m inside the turn = to starboard of the path = positive
    r = 1.9
    s, _, _, ye, _ = p.project(r * math.sin(0.6), 2 - r * math.cos(0.6))
    assert ye == pytest.approx(0.1)
