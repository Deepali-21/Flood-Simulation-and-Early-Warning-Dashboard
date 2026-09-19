"""
Sanity tests for simulation.py.  Run from the project root with:   pytest -v

If mass conservation (test 3) ever fails, do NOT trust any flood numbers.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation import simulate, validate_city  # noqa: E402

N = 20


# ---------------------------------------------------------------- helpers
def flat_city(size=N, drainage=0.0, depth0=0.0):
    return {
        "elevation": np.zeros((size, size)),
        "drainage": np.full((size, size), float(drainage)),
        "initial_depth": np.full((size, size), float(depth0)),
    }


def slope_city(size=N, drainage=0.0):
    """Ground drops from west (high) to east (low)."""
    x = np.tile(np.linspace(5.0, 0.0, size), (size, 1))
    return {"elevation": x, "drainage": np.full((size, size), float(drainage))}


def bowl_city(size=N, drainage=0.0):
    """A bowl: low in the middle, high at the rim."""
    y, x = np.mgrid[0:size, 0:size]
    r = np.hypot(x - size / 2, y - size / 2)
    return {"elevation": r * 0.3, "drainage": np.full((size, size), float(drainage))}


def rough_city(size=N, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "elevation": rng.uniform(0, 4, (size, size)),          # very rough ground
        "drainage": rng.uniform(0, 30, (size, size)),
    }


def params(rain=20.0, hours=3.0, **extra):
    return {"rain_mm_h": rain, "sim_hours": hours, **extra}


# ------------------------------------------------------------------ tests
def test_1_zero_rain_dry_city_stays_dry():
    r = simulate(bowl_city(drainage=10), params(rain=0.0))
    assert r["depth"].max() == 0.0


def test_2_flat_ground_uniform_water_does_not_move():
    r = simulate(flat_city(depth0=0.1), params(rain=0.0))
    assert np.allclose(r["depth"], 0.1)


def test_3_mass_conservation():
    """storage_end == storage_start + rain + source - drained - outlet (to round-off)."""
    city = rough_city()
    outlet = np.zeros((N, N), bool)
    outlet[:, -1] = True
    city["outlet_mask"] = outlet
    city["initial_depth"] = np.full((N, N), 0.05)
    src = np.zeros((N, N))
    src[0, 0] = 30.0
    city["inflow_mm_h"] = src
    rng = np.random.default_rng(3)
    city["conductance"] = rng.choice([0.0, 0.3, 1.0, 6.0], size=(N, N))   # blocked ... fast channel

    r = simulate(city, params(rain=60.0, hours=4.0))
    b = r["budget"]
    expected = b["storage"][0] + b["rain"][-1] + b["source"][-1] - b["drained"][-1] - b["outlet"][-1]
    assert b["storage"][-1] == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert b["outlet"][-1] > 0          # the outlet actually did something


def test_4_depths_never_negative_even_on_rough_ground():
    r = simulate(rough_city(seed=7), params(rain=100.0, hours=3.0))
    assert r["depth"].min() >= 0.0
    assert np.isfinite(r["depth"]).all()


def test_5_flat_ground_matches_hand_calculation():
    """No lateral flow on flat ground, so depth = (rain - drainage) * time, exactly."""
    r = simulate(flat_city(drainage=20.0), params(rain=50.0, hours=2.0))
    expected = (50.0 - 20.0) / 1000.0 * r["time_h"]            # metres
    assert np.allclose(r["depth"][:, 5, 5], expected)


def test_6_drainage_beats_rain_no_flooding():
    r = simulate(flat_city(drainage=30.0), params(rain=10.0, hours=3.0))
    assert r["depth"].max() < 1e-12


def test_7_water_runs_downhill_into_the_bowl():
    r = simulate(bowl_city(), params(rain=30.0, hours=3.0))
    final = r["depth"][-1]
    centre, rim = final[N // 2, N // 2], final[0, 0]
    assert centre > 5 * rim


def test_8_drainage_failure_makes_it_worse():
    city = bowl_city(drainage=25.0)
    ok = simulate(city, params(rain=60.0, hours=3.0))
    failed = simulate(city, params(rain=60.0, hours=3.0, drainage_scale=0.0))
    assert failed["depth"].max() > ok["depth"].max()


def test_9_blocked_channel_makes_water_back_up_upstream():
    city = slope_city()
    outlet = np.zeros((N, N), bool)
    outlet[:, -1] = True
    city["outlet_mask"] = outlet

    open_run = simulate(city, params(rain=40.0, hours=3.0))

    blocked = dict(city)
    cond = np.ones((N, N))
    cond[:, 12] = 0.0                                          # a wall across the whole map
    blocked["conductance"] = cond
    blocked_run = simulate(blocked, params(rain=40.0, hours=3.0))

    upstream = (slice(None), slice(8, 12))                     # columns just west of the wall
    assert blocked_run["depth"][-1][upstream].mean() > 2 * open_run["depth"][-1][upstream].mean()


def test_10_outlets_remove_water():
    closed = simulate(slope_city(), params(rain=40.0, hours=3.0))
    city = slope_city()
    outlet = np.zeros((N, N), bool)
    outlet[:, -1] = True
    city["outlet_mask"] = outlet
    with_outlet = simulate(city, params(rain=40.0, hours=3.0))
    assert with_outlet["budget"]["storage"][-1] < closed["budget"]["storage"][-1]


def test_11_result_does_not_depend_much_on_time_step():
    """The engine computes internally in fixed 1-minute steps, so the dashboard's
    recording step (1, 5 or 15 minutes) must not change the physics."""
    fine = simulate(bowl_city(drainage=10), params(rain=50.0, hours=2.0, dt_h=1 / 60))
    for dt in (5 / 60, 15 / 60):
        coarse = simulate(bowl_city(drainage=10), params(rain=50.0, hours=2.0, dt_h=dt))
        assert np.abs(coarse["depth"][-1] - fine["depth"][-1]).max() < 1e-9


def test_12_storm_ends_and_flood_recedes():
    rain = np.concatenate([np.full(12, 60.0)])                 # 1 hour of rain...
    r = simulate(bowl_city(drainage=20.0), {"rain_mm_h": rain, "sim_hours": 4.0})   # ...then 3 dry hours
    peak = r["depth"].max()
    assert r["depth"][-1].max() < peak
    assert r["rain_mm_h"][-1] == 0.0


def test_13_bad_input_gives_readable_errors():
    with pytest.raises(KeyError):
        validate_city({"elevation": np.zeros((5, 5))})                     # no drainage
    with pytest.raises(ValueError):
        validate_city({"elevation": np.zeros((5, 5)), "drainage": np.zeros((4, 5))})
    with pytest.raises(KeyError):
        simulate(flat_city(), {"sim_hours": 1.0})                          # no rain
    with pytest.raises(ValueError):
        simulate(flat_city(), {"rain_mm_h": 10.0})                         # scalar rain, no length


def test_14_fast_channel_conveys_water_out_faster():
    city = slope_city()
    outlet = np.zeros((N, N), bool)
    outlet[:, -1] = True
    city["outlet_mask"] = outlet
    normal = simulate(city, params(rain=40.0, hours=2.0))

    fast = dict(city)
    fast["conductance"] = np.full((N, N), 5.0)
    fast_run = simulate(fast, params(rain=40.0, hours=2.0))
    assert fast_run["budget"]["storage"][-1] < normal["budget"]["storage"][-1]


def test_15_negative_conductance_is_rejected():
    city = flat_city()
    city["conductance"] = np.full((N, N), -1.0)
    with pytest.raises(ValueError):
        validate_city(city)