import math

from cybership_dp.los import ALOS


def test_proportional_los_exam_2021_problem_4():
    """TTK4190 exam 2021 (LF), problem 4: ship (200, -100), WP1 (0, 0), WP2 (1200, 600), Delta 400 m."""
    los = ALOS(lookahead=400.0, gamma=0.0)
    los.set_path((0.0, 0.0), (1200.0, 600.0))
    _, ye = los.errors(200.0, -100.0)
    assert round(ye) in (-179, -178)
    assert round(math.degrees(los.update(200.0, -100.0, 0.1))) in (50, 51)


def _follow(gamma, crab_deg=10.0, U=0.6, h=0.05, seconds=200.0):
    """Perfect heading autopilot (psi = psi_d) and a constant crab angle beta_c from a cross current."""
    los = ALOS(lookahead=2.5, gamma=gamma)
    los.set_path((0.0, 0.0), (1000.0, 0.0))     # path due north
    x, y, beta = 0.0, 1.0, math.radians(crab_deg)
    for _ in range(int(seconds / h)):
        psi = los.update(x, y, h)
        x += h * U * math.cos(psi + beta)
        y += h * U * math.sin(psi + beta)
    return los, los.errors(x, y)[1]


def test_proportional_los_keeps_cross_track_error_in_current():
    _, ye = _follow(gamma=0.0)
    assert abs(ye) > 0.3        # y_e = Delta * tan(beta_c) = 0.44 m in steady state


def test_alos_removes_cross_track_error_and_estimates_crab_angle():
    los, ye = _follow(gamma=0.02)
    assert abs(ye) < 0.02
    assert abs(math.degrees(los.beta_hat) - 10.0) < 0.5


def test_along_track_and_remaining():
    los = ALOS(lookahead=2.0)
    los.set_path((1.0, 1.0), (1.0, 6.0))        # due east, 5 m
    xe, ye = los.errors(1.5, 4.0)
    assert abs(xe - 3.0) < 1e-9 and abs(ye + 0.5) < 1e-9      # 0.5 m north of an eastward path = left
    assert abs(los.remaining(1.5, 4.0) - 2.0) < 1e-9
