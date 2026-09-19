"""
dev_run.py -- Person 1's playground for tuning the engine BEFORE terrain.py / scenarios.py exist.

    python dev_run.py

Builds a stand-in city, runs the four scenarios, prints a results table and saves dev_output.png.
Nothing here is imported by the dashboard. Once Person 2's files land, swap
`make_dev_city` / `make_scenario` for `terrain.build_city` / `scenarios.apply_scenario`.

Needs:  pip install numpy matplotlib
"""
import time

import numpy as np

from simulation import simulate

SIZE = 30
WARNING_M, CRITICAL_M = 0.15, 0.30       # stand-ins for config.THRESHOLDS
SIM_HOURS = 8.0                          # long enough to watch the flood recede
NORMAL_PEAK, HEAVY_PEAK = 12, 60         # peak rainfall in mm/h
DT_H = 5 / 60


# ---------------------------------------------------------------- stand-in city
def make_dev_city(size=SIZE, seed=42):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size

    def bump(cx, cy, r, amp):
        return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r ** 2))

    z = 8.0 - 3.0 * x - 2.0 * y                                   # ground slopes down to the south-east
    z += -1.5 * np.exp(-((x - 0.78) ** 2) / (2 * 0.04 ** 2))      # river valley running north -> south
    z += bump(0.30, 0.35, 0.09, -1.4)                             # basin 1
    z += bump(0.45, 0.75, 0.08, -1.2)                             # basin 2
    z += bump(0.15, 0.20, 0.12, +2.0)                             # hill
    z += rng.normal(0, 0.03, z.shape)

    drainage = 12 + 18 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.08)   # better drains downtown
    drainage[int(0.55 * size):int(0.85 * size), int(0.05 * size):int(0.30 * size)] = 6.0   # "old district"

    outlet = np.zeros((size, size), bool)
    valley_cols = (x[0] > 0.72) & (x[0] < 0.84)
    outlet[-1, valley_cols] = True                                # river mouth at the south edge

    conductance = np.ones((size, size))
    conductance[:, valley_cols] = 6.0                             # the river channel conveys water 6x faster

    population = (2000 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.1) + 200).astype(int)

    return {
        "elevation": z,
        "drainage": drainage,
        "initial_depth": np.zeros((size, size)),
        "conductance": conductance,
        "outlet_mask": outlet,
        "population": population,
    }


def rainfall_profile(peak_mm_h, storm_h=3.0, sim_hours=SIM_HOURS, dt_h=DT_H):
    """Bell-shaped storm centred in the storm window; zero rain afterwards."""
    t = np.arange(0, sim_hours, dt_h)
    centre, width = storm_h / 2, storm_h / 6
    rain = peak_mm_h * np.exp(-((t - centre) ** 2) / (2 * width ** 2))
    rain[t > storm_h] = 0.0
    return rain


def make_scenario(city, name):
    c = {key: v.copy() for key, v in city.items()}
    p = {"dt_h": DT_H, "sim_hours": SIM_HOURS}
    if name == "Normal rain":
        p["rain_mm_h"] = rainfall_profile(NORMAL_PEAK)
    elif name == "Heavy rain":
        p["rain_mm_h"] = rainfall_profile(HEAVY_PEAK)
    elif name == "Drainage failure":
        p["rain_mm_h"] = rainfall_profile(HEAVY_PEAK)
        p["drainage_scale"] = 0.2
    elif name == "Blocked channel":
        p["rain_mm_h"] = rainfall_profile(HEAVY_PEAK)
        valley = c["outlet_mask"][-1]                              # same columns as the river mouth
        cols = np.where(valley)[0]
        c["conductance"][20, cols] = 0.0                           # a blockage across the river
        c["drainage"][20, cols] = 0.0
    else:
        raise ValueError(name)
    return c, p


# ---------------------------------------------------------------- metrics
def summarise(name, city, res):
    d, t = res["depth"], res["time_h"]
    crit = d >= CRITICAL_M
    ever_crit = crit.any(axis=0)
    any_crit = crit.any(axis=(1, 2))
    first = t[any_crit.argmax()] if any_crit.any() else float("nan")
    warn_only = ((d >= WARNING_M) & ~crit).any(axis=0) & ~ever_crit
    return {
        "Scenario": name,
        "Peak depth (m)": round(float(d.max()), 2),
        "Critical cells": int(ever_crit.sum()),
        "Warning-only cells": int(warn_only.sum()),
        "First critical (h)": round(float(first), 2),
        "People in critical cells": int(city["population"][ever_crit].sum()),
        "Runtime (s)": round(res["meta"]["runtime"], 3),
    }


def main():
    city = make_dev_city()
    names = ["Normal rain", "Heavy rain", "Drainage failure", "Blocked channel"]
    results, rows = {}, []
    for name in names:
        c, p = make_scenario(city, name)
        t0 = time.perf_counter()
        res = simulate(c, p)
        res["meta"]["runtime"] = time.perf_counter() - t0
        results[name] = (c, res)
        rows.append(summarise(name, c, res))

    # ---- print a plain table -------------------------------------------------
    keys = list(rows[0])
    widths = [max(len(k), *(len(str(r[k])) for r in rows)) for k in keys]
    print("  ".join(k.ljust(w) for k, w in zip(keys, widths)))
    for r in rows:
        print("  ".join(str(r[k]).ljust(w) for k, w in zip(keys, widths)))

    # ---- water budget check on the heavy run -----------------------------------
    b = results["Heavy rain"][1]["budget"]
    err = b["storage"][-1] - (b["storage"][0] + b["rain"][-1] + b["source"][-1] - b["drained"][-1] - b["outlet"][-1])
    print(f"\nWater budget error on 'Heavy rain': {err:.2e} (should be ~0)")

    # ---- figure ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        print("matplotlib not installed -> skipping figure (pip install matplotlib)")
        return

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(city["elevation"], cmap="terrain")
    ax.set_title("Elevation (m)"); fig.colorbar(im, ax=ax, shrink=0.8)
    ax.contour(city["outlet_mask"], levels=[0.5], colors="k", linewidths=1)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(city["drainage"], cmap="viridis")
    ax.set_title("Drainage capacity (mm/h)"); fig.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(city["population"], cmap="magma")
    ax.set_title("Population per cell"); fig.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[0, 3])
    hours = np.arange(len(rainfall_profile(HEAVY_PEAK))) * DT_H
    ax.plot(hours, rainfall_profile(NORMAL_PEAK), label="Normal")
    ax.plot(hours, rainfall_profile(HEAVY_PEAK), label="Heavy")
    ax.set_title("Rainfall (mm/h)"); ax.set_xlabel("hours"); ax.legend()

    risk_cmap = ListedColormap(["#4caf50", "#ffb300", "#d32f2f"])
    norm = BoundaryNorm([0, WARNING_M, CRITICAL_M, 10], risk_cmap.N)
    for i, name in enumerate(names):
        c, res = results[name]
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(res["depth"].max(axis=0), cmap=risk_cmap, norm=norm)
        ax.set_title(f"{name}: worst-case risk")
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[2, :2])
    for name in names:
        res = results[name][1]
        ax.plot(res["time_h"], (res["depth"] >= CRITICAL_M).sum(axis=(1, 2)), label=name)
    ax.set_title("Critical cells over time"); ax.set_xlabel("hours"); ax.legend()

    ax = fig.add_subplot(gs[2, 2:])
    for name in names:
        res = results[name][1]
        ax.plot(res["time_h"], res["depth"].mean(axis=(1, 2)) * 1000, label=name)
    ax.set_title("Average depth over the city (mm)"); ax.set_xlabel("hours"); ax.legend()

    fig.savefig("dev_output.png", dpi=110)
    print("Saved dev_output.png")


if __name__ == "__main__":
    main()