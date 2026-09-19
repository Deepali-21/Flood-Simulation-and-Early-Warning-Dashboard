"""
simulation.py -- Flood simulation engine (Person 1)

Public API (this is the contract the rest of the team codes against):

    result = simulate(city, params)

INPUT  city   : dict of 2D NumPy arrays, all the same shape (H, W)
    required
        "elevation"      ground height                     [m]
        "drainage"       max drain removal rate            [mm/h]
    optional (defaults in brackets)
        "initial_depth"  water already on the ground       [m]      (zeros)
        "conductance"    how freely water crosses a cell boundary.
                         1 = ordinary ground, 0 = fully blocked,
                         0..1 = partial obstruction, 2..10 = fast
                         channel / canal / river          [-]      (ones)
        "outlet_mask"    True where water leaves the city  [bool]   (all False)
        "inflow_mm_h"    constant external source, e.g. a
                         swollen river feeding a cell      [mm/h]   (zeros)
        "population"     ignored here, used by analysis.py

INPUT  params : dict
    "rain_mm_h"       scalar or 1D array, one value per time step   [mm/h]   (required)
    "dt_h"            step length                                    [h]      (5/60)
    "sim_hours"       total simulated time; if omitted, the length of the
                      rain array decides (lets the storm end and the flood recede)
    "k"               lateral flow coefficient                       [1/h]    (2.4)
    "drainage_scale"  multiplier on the drainage map (dashboard slider) [-]    (1.0)
    "internal_dt_h"   largest internal computing step (advanced)     [h]      (1/60)

OUTPUT result : dict
    "depth"          (T, H, W) water depth at every recorded moment   [m]
    "time_h"         (T,)      time of each record, starting at 0     [h]
    "rain_mm_h"      (T-1,)    rain used in each step (rain[t] falls between
                               time_h[t] and time_h[t+1])
    "inflow_total"   (H, W)    total water each cell received from neighbours [m]
    "drained_total"  (H, W)    total water each cell's drains removed         [m]
    "budget"         dict of (T,) arrays, all in "metres x cells" (multiply by
                     cell area for m^3). Used by the tests: storage always equals
                     start + rain + source - drained - outlet.
    "meta"           dict: substeps, alpha, shape

Model in one line, per cell and step:
    depth += rain + source - drainage + inflow_from_neighbours - outflow_to_neighbours

Water moves between 4-connected neighbours according to the difference in
WATER SURFACE height  H = elevation + depth.  All flows are computed from the
old state and applied together, so water is conserved by construction.
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------
# Tunable constants
# --------------------------------------------------------------------------
DEFAULT_DT_H = 5 / 60     # 5 minutes
DEFAULT_K = 2.4           # 1/h; with dt = 5 min a cell moves ~20% of a height difference per step
MAX_ALPHA = 0.2           # stability limit: max fraction of a height difference moved per sub-step
INTERNAL_DT_H = 1 / 60    # the engine always computes in steps of at most 1 minute, whatever dt_h is.
                          # Why: a cell can't send more water than it holds, so water moves at most one
                          # cell per internal step. A fixed internal step makes the flood's SPEED
                          # independent of the step length the dashboard records at.

_GRID_KEYS = ("elevation", "drainage", "initial_depth", "conductance",
              "outlet_mask", "inflow_mm_h", "population")


# --------------------------------------------------------------------------
# Input checking
# --------------------------------------------------------------------------
def validate_city(city: dict) -> tuple[int, int]:
    """Check the city dict and return its (H, W). Raises a readable error if wrong."""
    for key in ("elevation", "drainage"):
        if key not in city:
            raise KeyError(f"city is missing required key '{key}'")

    shape = np.asarray(city["elevation"]).shape
    if len(shape) != 2:
        raise ValueError(f"'elevation' must be 2D, got shape {shape}")

    for key in _GRID_KEYS:
        if key in city and np.asarray(city[key]).shape != shape:
            raise ValueError(
                f"'{key}' has shape {np.asarray(city[key]).shape}, expected {shape} like 'elevation'"
            )

    if (np.asarray(city["drainage"]) < 0).any():
        raise ValueError("'drainage' contains negative values")
    if "initial_depth" in city and (np.asarray(city["initial_depth"]) < 0).any():
        raise ValueError("'initial_depth' contains negative values")
    if "conductance" in city:
        if (np.asarray(city["conductance"], dtype=float) < 0).any():
            raise ValueError("'conductance' must be >= 0 (0 = blocked, 1 = normal, >1 = fast channel)")
    if "inflow_mm_h" in city and (np.asarray(city["inflow_mm_h"]) < 0).any():
        raise ValueError("'inflow_mm_h' contains negative values")

    return shape


def _rain_series(params: dict, dt: float) -> np.ndarray:
    """Turn params into a 1D array of rain (mm/h), one value per time step."""
    if "rain_mm_h" not in params:
        raise KeyError(
            "params is missing 'rain_mm_h'. Pass a scalar (with 'sim_hours') or an "
            "array with one value per step, e.g. from scenarios.rainfall_profile()."
        )
    rain = np.atleast_1d(np.asarray(params["rain_mm_h"], dtype=float))
    if rain.ndim != 1:
        raise ValueError("'rain_mm_h' must be a scalar or a 1D array")
    if (rain < 0).any():
        raise ValueError("'rain_mm_h' contains negative values")

    if "sim_hours" in params:
        n_steps = int(round(float(params["sim_hours"]) / dt))
        if n_steps < 1:
            raise ValueError("'sim_hours' is shorter than one time step")
        if rain.size == 1:
            rain = np.full(n_steps, rain[0])                       # constant rain
        elif rain.size < n_steps:
            rain = np.concatenate([rain, np.zeros(n_steps - rain.size)])   # storm ends, water recedes
        else:
            rain = rain[:n_steps]
    elif rain.size == 1:
        raise ValueError("scalar 'rain_mm_h' needs 'sim_hours' so the run length is known")

    return rain


# --------------------------------------------------------------------------
# One sub-step of water moving between neighbouring cells
# --------------------------------------------------------------------------
def _lateral_flow(z, h, cond_v, cond_h, alpha):
    """
    Move water between 4-connected neighbours.

    Returns (net_change, inflow), both (H, W) arrays in metres.
    Edges between cells are handled explicitly, so what leaves one cell is exactly
    what arrives in its neighbour (mass conservation by construction).
    """
    surface = z + h                                   # water surface height H

    # signed candidate flow across every edge; positive = towards higher index (down / right)
    qv = alpha * cond_v * (surface[:-1, :] - surface[1:, :])     # (H-1, W) vertical edges
    qh = alpha * cond_h * (surface[:, :-1] - surface[:, 1:])     # (H, W-1) horizontal edges

    # how much each cell WOULD send away in total
    pot = np.zeros_like(h)
    pot[:-1, :] += np.maximum(qv, 0)
    pot[1:, :] += np.maximum(-qv, 0)
    pot[:, :-1] += np.maximum(qh, 0)
    pot[:, 1:] += np.maximum(-qh, 0)

    # a cell can never send more water than it holds -> scale its outflows down
    scale = np.where(pot > h, h / np.where(pot > 0, pot, 1.0), 1.0)

    # apply the SOURCE cell's scale to each edge flow
    qv = np.where(qv > 0, qv * scale[:-1, :], qv * scale[1:, :])
    qh = np.where(qh > 0, qh * scale[:, :-1], qh * scale[:, 1:])

    net = np.zeros_like(h)
    net[:-1, :] -= qv
    net[1:, :] += qv
    net[:, :-1] -= qh
    net[:, 1:] += qh

    inflow = np.zeros_like(h)
    inflow[1:, :] += np.maximum(qv, 0)
    inflow[:-1, :] += np.maximum(-qv, 0)
    inflow[:, 1:] += np.maximum(qh, 0)
    inflow[:, :-1] += np.maximum(-qh, 0)
    return net, inflow


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def simulate(city: dict, params: dict) -> dict:
    shape = validate_city(city)
    H, W = shape
    n_cells = H * W

    # ---- read inputs (with defaults) -------------------------------------
    z = np.asarray(city["elevation"], dtype=float)
    drain_mm_h = np.asarray(city["drainage"], dtype=float)
    h = np.asarray(city.get("initial_depth", np.zeros(shape)), dtype=float).copy()
    cond = np.asarray(city.get("conductance", np.ones(shape)), dtype=float)
    outlet = np.asarray(city.get("outlet_mask", np.zeros(shape)), dtype=bool)
    source_mm_h = np.asarray(city.get("inflow_mm_h", np.zeros(shape)), dtype=float)

    dt = float(params.get("dt_h", DEFAULT_DT_H))
    k = float(params.get("k", DEFAULT_K))
    drain_scale = float(params.get("drainage_scale", 1.0))
    if dt <= 0 or k < 0 or drain_scale < 0:
        raise ValueError("'dt_h' must be > 0, 'k' and 'drainage_scale' must be >= 0")

    rain = _rain_series(params, dt)
    n_steps = rain.size

    # ---- split each recorded step into fixed-size internal sub-steps ------
    # The internal step is at most 1 minute, and small enough that even the fastest
    # channel (largest conductance) moves <= MAX_ALPHA of a height difference per step.
    c_max = max(1.0, float(cond.max()))
    internal_dt = min(float(params.get("internal_dt_h", INTERNAL_DT_H)),
                      MAX_ALPHA / (max(k, 1e-12) * c_max))
    substeps = max(1, math.ceil(dt / internal_dt - 1e-9))
    dts = dt / substeps
    alpha = k * dts

    # ---- precompute things that never change -----------------------------
    drain_cap_m = drain_mm_h * drain_scale / 1000.0 * dts        # max removal per sub-step [m]
    source_m = source_mm_h / 1000.0 * dts                        # external source per sub-step [m]
    cond_v = np.minimum(cond[:-1, :], cond[1:, :])               # an edge is only as open as its tighter side
    cond_h = np.minimum(cond[:, :-1], cond[:, 1:])
    has_outlet = bool(outlet.any())

    # ---- storage for results ---------------------------------------------
    depth = np.empty((n_steps + 1, H, W))
    depth[0] = h
    inflow_total = np.zeros(shape)
    drained_total = np.zeros(shape)

    b_storage = np.empty(n_steps + 1)
    b_rain = np.zeros(n_steps + 1)
    b_source = np.zeros(n_steps + 1)
    b_drained = np.zeros(n_steps + 1)
    b_outlet = np.zeros(n_steps + 1)
    b_storage[0] = h.sum()
    source_cells_m = source_m.sum()

    cum_rain = cum_source = cum_drained = cum_outlet = 0.0

    # ---- time loop --------------------------------------------------------
    for t in range(n_steps):
        rain_m = rain[t] / 1000.0 * dts                          # rain per sub-step [m]
        for _ in range(substeps):
            # 1. rain and external source
            h += rain_m
            h += source_m
            cum_rain += rain_m * n_cells
            cum_source += source_cells_m

            # 2. drains (never remove more than is there)
            removed = np.minimum(h, drain_cap_m)
            h -= removed
            drained_total += removed
            cum_drained += removed.sum()

            # 3. water moves between neighbours
            net, inflow = _lateral_flow(z, h, cond_v, cond_h, alpha)
            h += net
            inflow_total += inflow

            # 4. outlets carry water out of the city
            if has_outlet:
                leaving = h[outlet].sum()
                h[outlet] = 0.0
                cum_outlet += leaving

            np.maximum(h, 0.0, out=h)                            # guard against 1e-17 round-off

        depth[t + 1] = h
        b_storage[t + 1] = h.sum()
        b_rain[t + 1] = cum_rain
        b_source[t + 1] = cum_source
        b_drained[t + 1] = cum_drained
        b_outlet[t + 1] = cum_outlet

    return {
        "depth": depth,
        "time_h": np.arange(n_steps + 1) * dt,
        "rain_mm_h": rain,
        "inflow_total": inflow_total,
        "drained_total": drained_total,
        "budget": {
            "storage": b_storage,
            "rain": b_rain,
            "source": b_source,
            "drained": b_drained,
            "outlet": b_outlet,
        },
        "meta": {"substeps": substeps, "alpha": alpha, "shape": shape, "dt_h": dt},
    }