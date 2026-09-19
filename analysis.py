"""
analysis.py  —  Person 3: Analytics and Early Warning

Turns raw simulation output (depth over time) into:
  - Safe / Warning / Critical classification
  - Time-to-critical estimates (exact crossing, or extrapolated ETA)
  - Affected population
  - A ranked early-warning table with alert levels and plain-language messages
  - Scenario comparison metrics

Contract (must not change without telling the rest of the team):

    classify(depth, thresholds) -> (T,H,W) int array, 0=Safe 1=Warning 2=Critical
    time_to_critical(depth, time_h, thresholds) -> (H,W) float array, hours, NaN if never
    summarize(result, city, thresholds) -> dict of KPIs + ranked warning list
    compare(results: dict) -> pandas.DataFrame

`result` is whatever simulation.py's simulate() returns:
    result = {"depth": (T,H,W) array, "time_h": (T,) array}

`city` is whatever terrain.py's build_city() returns:
    city = {"elevation": (H,W), "drainage": (H,W),
            "population": (H,W), "initial_depth": (H,W)}

All units: metres for depth/elevation, hours for time.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared constants — the About tab and the sidebar should read these too,
# so the documentation can never drift from the code.
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "warning": 0.15,   # m
    "critical": 0.30,  # m
}

# Weight applied to a region's population when summing "affected population"
POPULATION_WEIGHT = {
    0: 0.0,   # Safe
    1: 0.25,  # Warning
    2: 1.0,   # Critical
}

CLASS_NAMES = {0: "Safe", 1: "Warning", 2: "Critical"}

# Lead-time bands for the early-warning table
ALERT_LEVELS = [
    (1.0, "Immediate", "Evacuate, activate pumps"),
    (3.0, "Prepare", "Alert residents, clear drains"),
    (np.inf, "Watch", "Monitor, pre-position resources"),
]


# ---------------------------------------------------------------------------
# 1. Classification
# ---------------------------------------------------------------------------

def classify(depth, thresholds=THRESHOLDS):
    """
    Classify every cell at every time step into Safe (0) / Warning (1) / Critical (2).

    depth: (T, H, W) array of water depth in metres.
    Returns: (T, H, W) int array.
    """
    depth = np.asarray(depth)
    cls = np.zeros(depth.shape, dtype=int)
    cls[depth >= thresholds["warning"]] = 1
    cls[depth >= thresholds["critical"]] = 2
    return cls


# ---------------------------------------------------------------------------
# 2. Time to critical
# ---------------------------------------------------------------------------

def time_to_critical(depth, time_h, thresholds=THRESHOLDS, extrapolate_from=6):
    """
    Estimated time (in hours) until each cell reaches the Critical threshold.

    Two cases, per the team's design doc:
      - If the cell crosses the threshold during the run: exact crossing time.
      - If it hasn't crossed yet but is still rising: extrapolate from the
        recent rate of rise. If it's flat or falling, ETA is NaN
        ("not expected to reach critical").

    depth: (T, H, W)
    time_h: (T,)
    extrapolate_from: how many of the most recent steps to use for the rate.

    Returns: (H, W) float array of hours, NaN where never expected.
    """
    depth = np.asarray(depth)
    time_h = np.asarray(time_h)
    crit = thresholds["critical"]

    crossed = depth >= crit                 # (T, H, W) bool
    ever = crossed.any(axis=0)               # (H, W) bool
    first_idx = crossed.argmax(axis=0)       # index of first True (0 if never)
    eta = np.where(ever, time_h[first_idx], np.nan).astype(float)

    # For cells that never crossed, try extrapolating from the recent trend.
    not_ever = ~ever
    if not_ever.any() and depth.shape[0] > 1:
        n = min(extrapolate_from, depth.shape[0] - 1)
        recent_depth = depth[-1]                       # (H, W) current depth
        earlier_depth = depth[-1 - n]                   # n steps back
        dt = time_h[-1] - time_h[-1 - n]
        dt = dt if dt > 0 else np.nan

        rate = (recent_depth - earlier_depth) / dt       # m / h
        remaining = crit - recent_depth                  # m still needed

        with np.errstate(divide="ignore", invalid="ignore"):
            extrap = np.where(rate > 1e-6, remaining / rate, np.nan)

        # Only rising cells that haven't crossed yet get an extrapolated ETA,
        # and it's measured from "now" (the last simulated time step).
        extrap_time = np.where(
            (not_ever) & (rate > 1e-6) & (remaining > 0),
            time_h[-1] + extrap,
            np.nan,
        )
        eta = np.where(not_ever, extrap_time, eta)

    return eta


# ---------------------------------------------------------------------------
# 3. Affected population
# ---------------------------------------------------------------------------

def affected_population(depth, population, thresholds=THRESHOLDS,
                          weights=POPULATION_WEIGHT):
    """
    Estimated affected population, using each region's WORST class reached
    during the run (its peak risk), weighted by class.

    depth: (T, H, W)
    population: (H, W)
    Returns: float (total estimated affected population)
    """
    cls = classify(depth, thresholds)
    worst_class = cls.max(axis=0)  # (H, W) — worst class each cell ever reached
    weight_map = np.vectorize(weights.get)(worst_class)
    return float((weight_map * population).sum())


# ---------------------------------------------------------------------------
# 4. Alert levels + plain-language messages
# ---------------------------------------------------------------------------

def _alert_level(eta_hours):
    if eta_hours is None or (isinstance(eta_hours, float) and np.isnan(eta_hours)):
        return "Watch"
    for limit, label, _action in ALERT_LEVELS:
        if eta_hours <= limit:
            return label
    return "Watch"


def _reason(row, city, thresholds=THRESHOLDS):
    """Very simple rule-based explanation for why a region is at risk."""
    reasons = []
    if row["elevation"] <= np.percentile(city["elevation"], 20):
        reasons.append("low-lying basin")
    if row["drainage"] < 20:  # mm/h, rough "weak drainage" cutoff
        reasons.append("weak drainage")
    if row["now_depth"] >= thresholds["warning"]:
        reasons.append("already rising")
    return ", ".join(reasons) if reasons else "general rainfall accumulation"


def _message(region, eta_h, pop, reason):
    if eta_h is None or np.isnan(eta_h):
        return f"{region}: not expected to reach critical level. Cause: {reason}."
    mins = int(round(eta_h * 60))
    return (f"{region}: expected to reach critical level in about {mins} min. "
            f"About {int(pop):,} people affected. Cause: {reason}.")


# ---------------------------------------------------------------------------
# 5. Summarize — the main entry point app.py calls for one scenario's result
# ---------------------------------------------------------------------------

def summarize(result, city, thresholds=THRESHOLDS, top_n=15, region_names=None):
    """
    Build the full early-warning summary for one simulation result.

    result: {"depth": (T,H,W), "time_h": (T,)}
    city:   {"elevation", "drainage", "population", "initial_depth"} each (H,W)

    Returns a dict:
        {
          "kpis": {
              "critical_regions": int,
              "warning_regions": int,
              "first_critical_h": float or None,
              "peak_depth": float,
              "affected_population": float,
          },
          "classification": (T,H,W) int array,
          "eta": (H,W) float array (hours),
          "warning_table": pandas.DataFrame,   # ranked, top_n rows
        }
    """
    depth, time_h = result["depth"], result["time_h"]
    H, W = depth.shape[1], depth.shape[2]

    cls = classify(depth, thresholds)
    eta = time_to_critical(depth, time_h, thresholds)

    ever_critical = (cls == 2).any(axis=0)
    ever_warning = (cls >= 1).any(axis=0)

    # First moment ANY cell in the whole city goes critical
    any_crit_at = (cls == 2).any(axis=(1, 2))
    first_critical_h = float(time_h[any_crit_at.argmax()]) if any_crit_at.any() else None

    kpis = {
        "critical_regions": int(ever_critical.sum()),
        "warning_regions": int(ever_warning.sum() - ever_critical.sum()),
        "first_critical_h": first_critical_h,
        "peak_depth": float(depth.max()),
        "affected_population": affected_population(depth, city["population"], thresholds),
    }

    # Build region names if not supplied, e.g. "R0_0", "R0_1", ...
    if region_names is None:
        region_names = np.array([[f"R{y}_{x}" for x in range(W)] for y in range(H)])

    now_depth = depth[-1]
    now_class = cls[-1]

    df = pd.DataFrame({
        "Region": region_names.ravel(),
        "Now": [CLASS_NAMES[c] for c in now_class.ravel()],
        "now_depth": now_depth.ravel(),
        "Time to critical (h)": eta.ravel(),
        "Peak depth (m)": depth.max(axis=0).ravel(),
        "People at risk": city["population"].ravel(),
        "elevation": city["elevation"].ravel(),
        "drainage": city["drainage"].ravel(),
    })

    # Only show regions that are at least in Warning now, or have a finite ETA
    at_risk = df[(df["now_depth"] >= thresholds["warning"]) | df["Time to critical (h)"].notna()].copy()

    at_risk["Alert level"] = at_risk["Time to critical (h)"].apply(_alert_level)
    at_risk["Why"] = at_risk.apply(lambda r: _reason(r, city, thresholds), axis=1)
    at_risk["Message"] = at_risk.apply(
        lambda r: _message(r["Region"], r["Time to critical (h)"], r["People at risk"], r["Why"]),
        axis=1,
    )
    # Urgency score: population per hour of lead time (higher = more urgent).
    # Regions with no ETA (not currently trending to critical) rank lowest.
    safe_eta = at_risk["Time to critical (h)"].replace(0, np.nan)
    at_risk["Urgency"] = at_risk["People at risk"] / safe_eta
    at_risk["Urgency"] = at_risk["Urgency"].fillna(0)

    warning_table = (
        at_risk.sort_values(["Time to critical (h)", "Urgency"], ascending=[True, False])
        .drop(columns=["now_depth", "elevation", "drainage"])
        .head(top_n)
        .reset_index(drop=True)
    )

    return {
        "kpis": kpis,
        "classification": cls,
        "eta": eta,
        "warning_table": warning_table,
    }


def banner_message(summary):
    """One-line banner for the top of the Early Warning tab."""
    kpis = summary["kpis"]
    n_crit = kpis["critical_regions"]
    first_h = kpis["first_critical_h"]

    if n_crit == 0 and (summary["warning_table"].empty):
        return "SAFE: No regions currently trending toward critical conditions."

    if first_h is not None:
        mins = int(round(first_h * 60))
        top_region = summary["warning_table"].iloc[0]["Region"] if not summary["warning_table"].empty else "an area"
        return (f"CRITICAL ALERT: {n_crit} region(s) reached critical levels. "
                f"First impact in {mins} min in {top_region}.")

    if not summary["warning_table"].empty:
        soonest = summary["warning_table"].iloc[0]
        mins = int(round(soonest["Time to critical (h)"] * 60))
        return f"WARNING: {soonest['Region']} expected critical in about {mins} min."

    return "WATCH: Some regions rising; none expected critical within the simulated window."


# ---------------------------------------------------------------------------
# 6. Compare — scenario comparison table (feeds the Compare tab)
# ---------------------------------------------------------------------------

def compare(results, city, thresholds=THRESHOLDS):
    """
    results: dict of {scenario_name: {"depth":..., "time_h":...}}
    city: same city dict used for all scenarios (must be identical across runs
          for the comparison to be fair — same terrain/population/seed).

    Returns a pandas.DataFrame, one row per scenario, indexed by scenario name.
    """
    rows = []
    for name, result in results.items():
        s = summarize(result, city, thresholds)
        k = s["kpis"]
        rows.append({
            "Scenario": name,
            "Peak depth (m)": round(k["peak_depth"], 2),
            "Critical regions": k["critical_regions"],
            "Warning regions": k["warning_regions"],
            "First critical (h)": None if k["first_critical_h"] is None else round(k["first_critical_h"], 2),
            "People affected": int(k["affected_population"]),
        })

    table = pd.DataFrame(rows).set_index("Scenario")

    # Add a "vs first scenario" delta column for critical regions, handy for
    # st.metric(delta=...) in the Compare tab.
    if len(table) > 1:
        baseline = table["Critical regions"].iloc[0]
        table["Δ Critical vs " + table.index[0]] = table["Critical regions"] - baseline

    return table


# ---------------------------------------------------------------------------
# Self-test with fake data (matches Person 1's stub shape) — run directly:
#   python analysis.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    H, W, T = 10, 10, 40
    rng = np.random.default_rng(0)

    # Fake terrain: a basin in the middle, higher edges
    yy, xx = np.mgrid[0:H, 0:W]
    elevation = 5 + ((yy - H / 2) ** 2 + (xx - W / 2) ** 2) ** 0.5 * 0.3
    drainage = rng.uniform(10, 40, size=(H, W))
    population = rng.integers(100, 3000, size=(H, W)).astype(float)

    # Fake depth: basin fills up over time, edges stay drier
    basin_factor = 1 / (1 + elevation - elevation.min())
    depth = np.stack([
        0.02 * t * basin_factor + rng.normal(0, 0.002, size=(H, W)).clip(min=0)
        for t in range(T)
    ])
    depth = np.clip(depth, 0, None)
    time_h = np.arange(T) * (5 / 60)

    city = {"elevation": elevation, "drainage": drainage,
            "population": population, "initial_depth": np.zeros((H, W))}
    result = {"depth": depth, "time_h": time_h}

    summary = summarize(result, city)
    print("KPIs:", summary["kpis"])
    print()
    print(banner_message(summary))
    print()
    print(summary["warning_table"].head(5).to_string(index=False))
