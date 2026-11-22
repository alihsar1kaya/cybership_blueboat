import os

import numpy as np
import pytest

from cybership_utilities import vessel_model
from cybership_utilities.marine_craft.tow import towed_from_config
from cybership_dp.trajectory_planner import OCPPlanner

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(scope="module")
def microampere():
    path = os.path.join(REPO, "cybership_config", "config", "microampere", "vessel.yaml")
    cfg = vessel_model.load_vessel_config(path=path)[0]
    return cfg, vessel_model.model_parameters(cfg)


def test_plan_reaches_the_goal_at_rest(microampere):
    cfg, P = microampere
    plan = OCPPlanner.from_vessel_config(cfg, P).plan([0, 0, 0, 0, 0, 0], [4.0, 0.0, 0.0])
    assert plan["ok"]
    assert np.allclose(plan["eta"][-1], [4.0, 0.0, 0.0], atol=1e-3)
    assert np.allclose(plan["nu"][-1], 0.0, atol=1e-3)
    assert plan["nu"][:, 0].max() <= 0.6 + 1e-3                      # forward speed limit


def test_towed_load_gives_a_gentler_plan(microampere):
    # the load on a towbar adds its mass and drag in surge: the same thrusters accelerate more mass
    cfg, P = microampere
    tow = towed_from_config(cfg["tow"]).surge_model()
    plain = OCPPlanner.from_vessel_config(cfg, P).plan([0, 0, 0, 0, 0, 0], [4.0, 0.0, 0.0])
    towed = OCPPlanner.from_vessel_config(cfg, P, tow=tow).plan([0, 0, 0, 0, 0, 0], [4.0, 0.0, 0.0])
    assert plain["ok"] and towed["ok"]
    assert towed["T"] > plain["T"]
    assert np.abs(towed["nu_dot"][:, 0]).max() < 0.8 * np.abs(plain["nu_dot"][:, 0]).max()


def _clearance(planner, plan, obstacles):
    """Smallest gap [m] between the footprint circles and the obstacles over the plan's steps."""
    worst = np.inf
    for x, y, psi in plan["eta"]:
        c, s = np.cos(psi), np.sin(psi)
        for cx, cy, rc in planner.footprint:
            px, py = x + c * cx - s * cy, y + s * cx + c * cy
            for ox, oy, r in obstacles:
                worst = min(worst, np.hypot(px - ox, py - oy) - r - rc)
    return worst


@pytest.mark.parametrize("obstacles", [[(3.0, 0.0, 0.5)], [(3.0, 0.3, 0.5)], [(2.5, 0.0, 0.4), (5.5, 0.2, 0.6)]])
def test_plan_goes_around_obstacles(microampere, obstacles):
    cfg, P = microampere
    planner = OCPPlanner.from_vessel_config(cfg, P)
    plan = planner.plan([0, 0, 0, 0, 0, 0], [8.0, 0.0, 0.0], obstacles=obstacles)
    assert plan["ok"] and plan["guess"] in ("port", "starboard")
    assert _clearance(planner, plan, obstacles) >= planner.margin - 0.01         # hull + margin stays clear
    assert np.allclose(plan["eta"][-1], [8.0, 0.0, 0.0], atol=1e-3)


def test_obstacle_away_from_the_line_changes_nothing(microampere):
    cfg, P = microampere
    planner = OCPPlanner.from_vessel_config(cfg, P)
    free = planner.plan([0, 0, 0, 0, 0, 0], [6.0, 0.0, 0.0])
    far = planner.plan([0, 0, 0, 0, 0, 0], [6.0, 0.0, 0.0], obstacles=[(3.0, 4.0, 0.5)])
    assert far["ok"] and far["T"] == pytest.approx(free["T"], rel=0.02)


def test_goal_inside_an_obstacle_is_refused(microampere):
    # the hull would overlap the obstacle at the goal: no plan (the controller holds and reports it)
    cfg, P = microampere
    plan = OCPPlanner.from_vessel_config(cfg, P).plan([0, 0, 0, 0, 0, 0], [4.0, 0.0, 0.0],
                                                      obstacles=[(4.2, 0.0, 0.5)])
    assert not plan["ok"] and plan["blocked"] == [(4.2, 0.0, 0.5)]


def test_obstacle_at_the_start_is_left_out(microampere):
    # the boat already overlaps it (it was placed on top of the boat): plan without it so the boat can leave
    cfg, P = microampere
    plan = OCPPlanner.from_vessel_config(cfg, P).plan([0, 0, 0, 0, 0, 0], [5.0, 0.0, 0.0],
                                                      obstacles=[(0.3, 0.2, 0.5)])
    assert plan["ok"] and plan["ignored"] == [(0.3, 0.2, 0.5)] and plan["obstacles"] == []


def test_obstacle_near_the_goal_is_kept_with_a_smaller_margin(microampere):
    # goal 0.8 m behind the edge of an obstacle: the plan still goes around it, closer than the margin only there
    cfg, P = microampere
    planner = OCPPlanner.from_vessel_config(cfg, P)
    obs = [(2.6, -1.6, 0.5)]
    plan = planner.plan([0, 0, 0, 0, 0, 0], [4.0, -1.6, 0.0], obstacles=obs)
    assert plan["ok"] and plan["obstacles"] == obs and plan["ignored"] == []
    assert 0.0 <= plan["margins"][0] < planner.margin
    assert _clearance(planner, plan, obs) >= plan["margins"][0] - 0.01
