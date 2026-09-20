"""
Checks for terrain.py against the simulation engine's contract.
Run from the project root:   python -m pytest -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import terrain as T  # noqa: E402
from simulation import simulate, validate_city  # noqa: E402


def heavy_storm(hours=8.0, dt=5 / 60, peak=60.0):
    t = np.arange(0, hours, dt)
    rain = peak * np.exp(-((t - 1.5) ** 2) / (2 * 0.5 ** 2))
    rain[t > 3.0] = 0.0
    return rain


def test_city_matches_the_engine_contract():
    city = T.make_dev_city()
    for key in ("elevation", "drainage", "initial_depth", "conductance", "outlet_mask", "population"):
        assert key in city, f"missing '{key}'"
    shape = validate_city(city)                               # raises a readable error if anything is wrong
    assert all(np.asarray(v).shape == shape for v in city.values())
    assert city["outlet_mask"].dtype == bool and city["outlet_mask"].any()
    assert (city["drainage"] >= 0).all() and (city["population"] >= 0).all()
    assert (city["conductance"] >= 0).all()


def test_same_seed_same_city_different_seed_different_terrain():
    a, b, c = T.make_dev_city(seed=1), T.make_dev_city(seed=1), T.make_dev_city(seed=2)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["elevation"], c["elevation"])


@pytest.mark.parametrize("size", [20, 30, 50])
def test_other_grid_sizes_work(size):
    city = T.make_dev_city(size=size)
    assert validate_city(city) == (size, size)
    assert city["outlet_mask"].any()
    assert T.make_channel_path(size=size).any()


def test_channel_mask_is_exactly_the_river_cells():
    city = T.make_dev_city()
    mask = T.make_channel_path()
    assert mask.dtype == bool and mask.shape == city["elevation"].shape
    assert np.array_equal(mask, city["conductance"] > 1)
    assert (city["outlet_mask"] & ~mask).sum() == 0            # the outlet is at the river mouth


def test_unknown_channel_kind_is_rejected():
    with pytest.raises(ValueError):
        T.make_channel_path(kind="canal")


def test_water_actually_leaves_the_city_through_the_outlet():
    r = simulate(T.make_dev_city(), {"rain_mm_h": heavy_storm(), "sim_hours": 8.0})
    assert r["budget"]["outlet"][-1] > 0


def test_basins_flood_and_the_hill_stays_dry():
    city = T.make_dev_city()
    z = city["elevation"]
    depth = simulate(city, {"rain_mm_h": heavy_storm(), "sim_hours": 8.0})["depth"].max(axis=0)

    def lowest_cell_depth(r0, r1, c0, c1):                     # deepest water at the lowest ground in a window
        window = z[r0:r1, c0:c1]
        r, c = np.unravel_index(window.argmin(), window.shape)
        return depth[r0 + r, c0 + c]

    assert lowest_cell_depth(7, 14, 6, 13) > 0.30              # West Basin
    assert lowest_cell_depth(19, 26, 10, 17) > 0.30            # South Basin
    assert depth[3:8, 2:7].max() < 0.05                        # the hill stays dry


def test_region_names_are_readable_and_unique():
    names = T.make_region_names()
    assert names.shape == (30, 30)
    assert len(set(names.ravel())) == 900
    assert names[22, 13].startswith("South Basin")
    assert names[12, 24].startswith("Riverside")