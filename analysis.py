"""
analysis.py  —  Person 3: Analytics and Early Warning
"""

import numpy as np
import pandas as pd

THRESHOLDS = {
    "warning": 0.15,   # m
    "critical": 0.30,  # m
}

POPULATION_WEIGHT = {
    0: 0.0,
    1: 0.25,
    2: 1.0,
}

CLASS_NAMES = {0: "Safe", 1: "Warning", 2: "Critical"}

ALERT_LEVELS = [
    (1.0, "Immediate", "Evacuate, activate pumps"),
    (3.0, "Prepare", "Alert residents, clear drains"),
    (np.inf, "Watch", "Monitor, pre-position resources"),
]


def classify(depth, thresholds=THRESHOLDS):
    depth = np.asarray(depth)
    cls = np.zeros(depth.shape, dtype=int)
    cls[depth >= thresholds["warning"]] = 1
    cls[depth >= thresholds["critical"]] = 2
    return cls


def time_to_critical(depth, time_h, thresholds=THRESHOLDS, extrapolate_from=6, horizon_h=3.0):
    depth = np.asarray(depth)
    time_h = np.asarray(time_h)
    crit = thresholds["critical"]

    crossed = depth >= crit
    ever = crossed.any(axis=0)
    first_idx = crossed.argmax(axis=0)
    eta = np.where(ever, time_h[first_idx], np.nan).astype(float)

    not_ever = ~ever
    if not_ever.any() and depth.shape[0] > 1:
        n = min(extrapolate_from, depth.shape[0] - 1)
        recent_depth = depth[-1]
        earlier_depth = depth[-1 - n]
        dt = time_h[-1] - time_h[-1 - n]
        dt = dt if dt > 0 else np.nan

        rate = (recent_depth - earlier_depth) / dt
        remaining = crit - recent_depth

        with np.errstate(divide="ignore", invalid="ignore"):
            extrap = np.where(rate > 1e-6, remaining / rate, np.nan)

        plausible = ((not_ever) & (recent_depth >= thresholds["warning"])
                     & (rate > 1e-6) & (remaining > 0) & (extrap <= horizon_h))
        extrap_time = np.where(plausible, time_h[-1] + extrap, np.nan)
        eta = np.where(not_ever, extrap_time, eta)

    return eta


def affected_population(depth, population, thresholds=THRESHOLDS,
                          weights=POPULATION_WEIGHT):
    cls = classify(depth, thresholds)
    worst_class = cls.max(axis=0)
    weight_map = np.vectorize(weights.get)(worst_class)
    return float((weight_map * population).sum())


def _alert_level(lead_hours, now_class="Safe"):
    if now_class == "Critical":
        return "Critical now"
    if lead_hours is None or np.isnan(lead_hours):
        return "Watch"
    for limit, label, _action in ALERT_LEVELS:
        if lead_hours <= limit:
            return label
    return "Watch"


def _reason(row, city, thresholds=THRESHOLDS):
    reasons = []
    if row["elevation"] <= np.percentile(city["elevation"], 20):
        reasons.append("low-lying basin")
    if row["drainage"] < 20:
        reasons.append("weak drainage")
    if row["now_depth"] >= thresholds["warning"]:
        reasons.append("already rising")
    return ", ".join(reasons) if reasons else "general rainfall accumulation"


def _message(region, lead_h, pop, reason, now_class="Safe"):
    if now_class == "Critical":
        return (f"{region}: critical now. About {int(pop):,} people affected. Cause: {reason}.")
    if lead_h is None or np.isnan(lead_h):
        return f"{region}: in Warning, not expected to reach critical level. Cause: {reason}."
    mins = int(round(lead_h * 60))
    return (f"{region}: expected to reach critical level in about {mins} min. "
            f"About {int(pop):,} people affected. Cause: {reason}.")


def summarize(result, city, thresholds=THRESHOLDS, top_n=15, region_names=None):
    depth, time_h = result["depth"], result["time_h"]
    H, W = depth.shape[1], depth.shape[2]

    cls = classify(depth, thresholds)
    eta = time_to_critical(depth, time_h, thresholds)

    ever_critical = (cls == 2).any(axis=0)
    ever_warning = (cls >= 1).any(axis=0)

    any_crit_at = (cls == 2).any(axis=(1, 2))
    first_critical_h = float(time_h[any_crit_at.argmax()]) if any_crit_at.any() else None

    kpis = {
        "critical_regions": int(ever_critical.sum()),
        "warning_regions": int(ever_warning.sum() - ever_critical.sum()),
        "first_critical_h": first_critical_h,
        "peak_depth": float(depth.max()),
        "affected_population": affected_population(depth, city["population"], thresholds),
    }

    if region_names is None:
        region_names = np.array([[f"R{y}_{x}" for x in range(W)] for y in range(H)])

    now_h = float(time_h[-1])
    now_depth = depth[-1]
    now_class = cls[-1]

    df = pd.DataFrame({
        "Region": region_names.ravel(),
        "Now": [CLASS_NAMES[c] for c in now_class.ravel()],
        "now_depth": now_depth.ravel(),
        "Time to critical (h)": eta.ravel(),
        "Lead time (h)": np.maximum(eta.ravel() - now_h, 0.0),
        "Peak depth (m)": depth.max(axis=0).ravel(),
        "People at risk": city["population"].ravel(),
        "elevation": city["elevation"].ravel(),
        "drainage": city["drainage"].ravel(),
    })

    at_risk = df[(df["now_depth"] >= thresholds["warning"]) | df["Time to critical (h)"].notna()].copy()

    at_risk["Alert level"] = [_alert_level(l, n) for l, n in zip(at_risk["Lead time (h)"], at_risk["Now"])]
    at_risk["Why"] = at_risk.apply(lambda r: _reason(r, city, thresholds), axis=1)
    at_risk["Message"] = at_risk.apply(
        lambda r: _message(r["Region"], r["Lead time (h)"], r["People at risk"], r["Why"], r["Now"]),
        axis=1,
    )
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
        lead = soonest["Lead time (h)"]
        if np.isnan(lead):
            return (f"WATCH: {soonest['Region']} is in Warning; "
                    f"none expected to reach critical within the forecast window.")
        return f"WARNING: {soonest['Region']} expected critical in about {int(round(lead * 60))} min."

    return "WATCH: Some regions rising; none expected critical within the simulated window."


def compare(results, city, thresholds=THRESHOLDS):
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

    if len(table) > 1:
        baseline = table["Critical regions"].iloc[0]
        table["Delta Critical vs " + table.index[0]] = table["Critical regions"] - baseline

    return table
