"""
Integration checks: scenarios.py / terrain.py (Person 2) running through simulation.py (Person 1).

Run from the project root:   python -m pytest -v
Needs terrain.py and scenarios.py in the project root, next to simulation.py.

What a *fair* comparison needs: each scenario changes ONE thing relative to "Heavy rain".
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scenarios as S  # noqa: E402
import terrain as T  # noqa: E402
from simulation import simulate, validate_city  # noqa: E402

NAMES = ["Normal rain", "Heavy rain", "Drainage failure", "Blocked channel"]
CRITICAL_M = 0.30


@pytest.fixture(scope="module")
def base():
    return T.make_dev_city()


@pytest.fixture(scope="module")
def made(base):
    return {n: S.make_scenario(base, n) for n in NAMES}


@pytest.fixture(scope="module")
def critical_cells(made):
    out = {}
    for n, (c, p) in made.items():
        depth = simulate(c, p)["depth"]
        out[n] = int((depth.max(axis=0) >= CRITICAL_M).sum())
    return out


@pytest.mark.parametrize("name", NAMES)
def test_scenario_runs_on_the_engine(base, name):
    c, p = S.make_scenario(base, name)
    validate_city(c)
    depth = simulate(c, p)["depth"]
    assert np.isfinite(depth).all() and depth.min() >= 0


def test_scenarios_do_not_modify_the_base_city(base):
    before = {k: v.copy() for k, v in base.items()}
    for n in NAMES:
        S.make_scenario(base, n)
    assert all(np.array_equal(before[k], base[k]) for k in before)


def test_failure_and_blocked_use_the_same_storm_as_heavy(made):
    heavy_c, heavy_p = made["Heavy rain"]
    for n in ("Drainage failure", "Blocked channel"):
        _, p = made[n]
        assert np.array_equal(p["rain_mm_h"], heavy_p["rain_mm_h"]), f"{n} uses a different storm from Heavy rain"
        assert p["k"] == heavy_p["k"] and p["dt_h"] == heavy_p["dt_h"] and p["sim_hours"] == heavy_p["sim_hours"]


def test_drainage_failure_changes_only_drainage_and_only_once(made):
    heavy_c, heavy_p = made["Heavy rain"]
    c, p = made["Drainage failure"]
    for key in heavy_c:
        if key != "drainage":
            assert np.array_equal(c[key], heavy_c[key]), f"'{key}' differs from Heavy rain"
    city_factor = float((c["drainage"] / heavy_c["drainage"]).mean())
    scale_factor = p["drainage_scale"] / heavy_p["drainage_scale"]
    assert city_factor * scale_factor < 1.0, "drainage is not reduced"
    assert city_factor == pytest.approx(1.0) or scale_factor == pytest.approx(1.0), \
        "drainage is cut twice (city data AND drainage_scale): use one lever"


def test_blocked_channel_changes_only_conductance_and_is_a_blockage_not_the_whole_river(made):
    heavy_c, _ = made["Heavy rain"]
    c, _ = made["Blocked channel"]
    for key in heavy_c:
        if key != "conductance":
            assert np.array_equal(c[key], heavy_c[key]), f"'{key}' differs from Heavy rain"
    blocked = int((c["conductance"] == 0).sum())
    assert blocked > 0, "nothing is blocked"
    assert blocked <= c["elevation"].shape[0], "blocked more than one row of cells: that walls off the whole river, not a blockage"


def test_each_scenario_is_worse_than_the_one_it_modifies(critical_cells):
    n = critical_cells
    assert n["Normal rain"] < n["Heavy rain"]
    assert n["Heavy rain"] < n["Drainage failure"]
    assert n["Heavy rain"] < n["Blocked channel"]


# ---------------------------------------------------------------- sidebar / custom scenarios
def test_scenario_names_list_matches_the_presets():
    assert S.SCENARIO_NAMES == NAMES


@pytest.mark.parametrize("shape", ["normal", "flat", "front"])
def test_custom_scenario_runs_for_every_storm_shape(base, shape):
    c, p = S.make_custom_scenario(base, peak_mm_h=40, duration_h=3, shape=shape)
    validate_city(c)
    assert p["rain_mm_h"].max() == pytest.approx(40.0)
    assert simulate(c, p)["depth"].max() > 0


def test_custom_scenario_stronger_rain_means_deeper_water(base):
    peaks = []
    for mm in (20, 40, 80):
        c, p = S.make_custom_scenario(base, peak_mm_h=mm)
        peaks.append(simulate(c, p)["depth"].max())
    assert peaks[0] < peaks[1] < peaks[2]


def test_custom_scenario_drainage_slider_passes_through(base):
    _, ok = S.make_custom_scenario(base, drainage_scale=1.0)
    _, failed = S.make_custom_scenario(base, drainage_scale=0.0)
    assert ok["drainage_scale"] == 1.0 and failed["drainage_scale"] == 0.0


def test_custom_scenario_long_storm_still_leaves_time_to_recede(base):
    _, p = S.make_custom_scenario(base, duration_h=9.0)
    assert p["sim_hours"] >= 9.0 + 3.0 - 1e-9


def test_custom_scenario_does_not_modify_the_base_city_and_rejects_bad_input(base):
    before = {k: v.copy() for k, v in base.items()}
    S.make_custom_scenario(base)
    assert all(np.array_equal(before[k], base[k]) for k in before)
    with pytest.raises(ValueError):
        S.make_custom_scenario(base, shape="hurricane")
    with pytest.raises(ValueError):
        S.make_custom_scenario(base, duration_h=0)