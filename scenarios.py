"""
scenarios.py -- Person 2 (Data and Scenarios)

Storm-shaped rainfall profiles and the four named scenario presets:
    "Normal rain", "Heavy rain", "Drainage failure", "Blocked channel"

Each preset returns (city, params) ready to pass straight into
simulation.simulate(city, params).

    make_scenario(city, "Heavy rain")          -> (city, params)   one of the four presets
    make_custom_scenario(city, peak_mm_h=40, ...)  -> (city, params)   built from the sidebar sliders
    SCENARIO_NAMES                              -> list for the scenario dropdown

Fair comparison rule: "Heavy rain", "Drainage failure" and "Blocked channel" share the SAME
storm, so the only difference between them is the failure being modelled.
"""

import numpy as np

from terrain import make_channel_path, make_dev_city

DEFAULT_DT_H = 5 / 60  # 5-minute steps
DEFAULT_SIM_HOURS = 8.0


# ---------------------------------------------------------------------------
# Rainfall profiles
# ---------------------------------------------------------------------------

def storm_profile(duration_h, dt_h=DEFAULT_DT_H, peak_mm_h=20.0, shape="normal",
                   peak_frac=0.35):
    """
    Builds a 1D rain_mm_h array shaped like a storm: ramps up, peaks, tapers.

    duration_h : total length of the storm itself (can be shorter than the
                 full simulated run -- simulate() pads the rest with zeros).
    dt_h        : step length, hours.
    peak_mm_h   : intensity at the storm's peak.
    shape       : "normal"  -- smooth ramp-up/taper (gamma-like, slightly
                                skewed so it peaks early and trails off)
                  "front"   -- intensity front-loaded, sharp start, long tail
                  "flat"    -- roughly constant intensity with a short
                                ramp in/out (useful for "Normal rain")
    peak_frac   : fraction of the storm's duration at which intensity peaks.
    """
    n_steps = max(1, int(round(duration_h / dt_h)))
    t = np.linspace(0, 1, n_steps)

    if shape == "flat":
        ramp = np.clip(t / 0.1, 0, 1) * np.clip((1 - t) / 0.1, 0, 1)
        profile = peak_mm_h * ramp

    elif shape == "front":
        # Sharp rise, slow exponential-style decay.
        rise = np.clip(t / max(peak_frac * 0.3, 1e-6), 0, 1)
        decay = np.exp(-np.clip((t - peak_frac), 0, None) / 0.35)
        profile = peak_mm_h * rise * decay

    else:  # "normal": skewed gamma-ish bump
        a, b = 2.2, 1.0 / peak_frac
        with np.errstate(divide="ignore"):
            shape_curve = (t ** (a - 1)) * np.exp(-b * t)
        shape_curve /= shape_curve.max() + 1e-12
        profile = peak_mm_h * shape_curve

    return np.clip(profile, 0, None).astype(float)


# ---------------------------------------------------------------------------
# Scenario presets
# ---------------------------------------------------------------------------

def _base_params(rain, sim_hours=DEFAULT_SIM_HOURS, dt_h=DEFAULT_DT_H,
                  k=2.4, drainage_scale=1.0):
    return {
        "rain_mm_h": rain,
        "dt_h": dt_h,
        "sim_hours": sim_hours,
        "k": k,
        "drainage_scale": drainage_scale,
    }


def _heavy_storm():
    """The one heavy storm shared by 'Heavy rain', 'Drainage failure' and 'Blocked channel',
    so the ONLY difference between those scenarios is the failure being modelled."""
    return storm_profile(duration_h=5.0, peak_mm_h=35.0, shape="normal", peak_frac=0.3)


def _cross_section(mask, block_at=0.65):
    """Reduce a channel mask to ONE row across the channel (a blockage, not the whole river).
    block_at = how far along the channel the blockage sits (0 = first row, 1 = last row).
    Pick it so the blockage is UPSTREAM of the outlet, so water backs up behind it."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise ValueError("make_channel_path() returned an empty mask")
    r = rows[int(round(block_at * (rows.size - 1)))]
    section = np.zeros_like(mask, dtype=bool)
    section[r] = mask[r]
    return section


def _fresh_city(seed):
    """A new, uncorrupted city dict (never mutate a caller's copy in place)."""
    return make_dev_city(seed=seed)


def scenario_normal_rain(city=None, seed=0):
    city = {k: v.copy() for k, v in (city or _fresh_city(seed)).items()}
    rain = storm_profile(duration_h=4.0, peak_mm_h=8.0, shape="flat")
    params = _base_params(rain, drainage_scale=1.0)
    return city, params


def scenario_heavy_rain(city=None, seed=0):
    city = {k: v.copy() for k, v in (city or _fresh_city(seed)).items()}
    rain = _heavy_storm()
    params = _base_params(rain, drainage_scale=1.0)
    return city, params


def scenario_drainage_failure(city=None, seed=0):
    city = {k: v.copy() for k, v in (city or _fresh_city(seed)).items()}
    rain = _heavy_storm()
    # ONE lever: 80% of drain capacity lost (pumps/inlets failing). drainage_scale stays 1.0
    # so the dashboard's drainage slider remains a separate, user-controlled multiplier.
    city["drainage"] = city["drainage"] * 0.2
    params = _base_params(rain, drainage_scale=1.0)
    return city, params


def scenario_blocked_channel(city=None, seed=0, kind="valley"):
    city = {k: v.copy() for k, v in (city or _fresh_city(seed)).items()}
    channel_mask = make_channel_path(kind=kind)
    if channel_mask.shape != city["elevation"].shape:
        raise ValueError(f"channel mask {channel_mask.shape} does not match city {city['elevation'].shape}")
    # Zero conductance across ONE cross-section of the channel -> water cannot pass,
    # so it backs up behind (upstream of) the blockage.
    blocked = city["conductance"].copy()
    blocked[_cross_section(channel_mask)] = 0.0
    city["conductance"] = blocked
    rain = _heavy_storm()
    params = _base_params(rain, drainage_scale=1.0)
    return city, params


_SCENARIOS = {
    "Normal rain": scenario_normal_rain,
    "Heavy rain": scenario_heavy_rain,
    "Drainage failure": scenario_drainage_failure,
    "Blocked channel": scenario_blocked_channel,
}


def make_scenario(city, name, seed=0):
    """
    Main entry point, matches the contract used elsewhere in the project:

        c, p = make_scenario(city, "Heavy rain")

    `city` may be None to build a fresh default city.
    """
    if name not in _SCENARIOS:
        raise ValueError(
            f"unknown scenario {name!r}; choose from {list(_SCENARIOS)}"
        )
    return _SCENARIOS[name](city=city, seed=seed)


SCENARIO_NAMES = list(_SCENARIOS)


def make_custom_scenario(city, peak_mm_h=35.0, duration_h=5.0, shape="normal",
                          drainage_scale=1.0, sim_hours=None, peak_frac=0.3):
    """
    Scenario built from the dashboard's sidebar controls.

    peak_mm_h       rainfall intensity at the storm's peak (mm/h)
    duration_h      how long the storm lasts (h)
    shape           "normal" (bell), "flat" (steady) or "front" (front-loaded)
    drainage_scale  user multiplier on drainage capacity (1.0 = as built, 0 = drains fully failed)
    sim_hours       total simulated time; by default the storm plus 3 hours for the flood to recede
    """
    if peak_mm_h < 0 or duration_h <= 0 or drainage_scale < 0:
        raise ValueError("peak_mm_h and drainage_scale must be >= 0 and duration_h must be > 0")
    if shape not in ("normal", "flat", "front"):
        raise ValueError(f"unknown storm shape {shape!r}; choose 'normal', 'flat' or 'front'")
    city = {k: v.copy() for k, v in city.items()}
    rain = storm_profile(duration_h=duration_h, peak_mm_h=peak_mm_h, shape=shape, peak_frac=peak_frac)
    if sim_hours is None:
        sim_hours = max(DEFAULT_SIM_HOURS, duration_h + 3.0)
    return city, _base_params(rain, sim_hours=sim_hours, drainage_scale=drainage_scale)


if __name__ == "__main__":
    base_city = make_dev_city()
    for name in _SCENARIOS:
        c, p = make_scenario(base_city, name)
        print(f"{name:20s} rain steps={len(p['rain_mm_h']):3d} "
              f"peak={p['rain_mm_h'].max():5.1f} mm/h "
              f"drainage_scale={p['drainage_scale']}")