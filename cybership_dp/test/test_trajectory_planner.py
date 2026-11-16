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
