"""
app.py — Person 4 (Dashboard, integration and demo)
FloodShield: Flood Simulation and Early Warning Dashboard

Integrates:
    terrain.py    (Person 2) -> make_dev_city()          [falls back to a mock if missing]
    scenarios.py  (Person 2) -> make_scenario(city, name) [falls back to a mock if missing]
    simulation.py (Person 1) -> simulate(city, params)    [falls back to a mock if missing]
    analysis.py   (Person 3) -> classify, time_to_critical, summarize,
                                 compare, banner_message   [falls back to a bug-fixed
                                 local copy if missing — see _FALLBACK_ANALYSIS below]

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Engine integration (terrain / scenarios / simulation), with a safe mock
# fallback so this file is runnable even before every module is pushed.
# ---------------------------------------------------------------------------
try:
    from terrain import make_dev_city
    from scenarios import make_scenario, SCENARIO_NAMES
    from simulation import simulate
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    SCENARIO_NAMES = ["Normal rain", "Heavy rain", "Drainage failure", "Blocked channel"]

    def make_dev_city():
        H, W = 30, 30
        yy, xx = np.mgrid[0:H, 0:W]
        elevation = (
            10
            + 0.15 * ((yy - H / 2) ** 2 + (xx - W / 2) ** 2) ** 0.5
            - 3 * np.exp(-(((yy - H * 0.7) ** 2 + (xx - W * 0.3) ** 2) / 30))
        )
        drainage = np.full((H, W), 20.0)
        conductance = np.ones((H, W))
        conductance[H // 2, :] = 3.0  # pretend river
        outlet_mask = np.zeros((H, W), dtype=bool)
        outlet_mask[0, :] = True
        population = np.random.default_rng(0).integers(20, 300, size=(H, W)).astype(float)
        return {
            "elevation": elevation,
            "drainage": drainage,
            "initial_depth": np.zeros((H, W)),
            "conductance": conductance,
            "outlet_mask": outlet_mask,
            "inflow_mm_h": np.zeros((H, W)),
            "population": population,
        }

    def _mock_storm(sim_hours, dt_h, peak):
        n = int(round(sim_hours / dt_h))
        t = np.linspace(0, 1, n)
        env = np.where(t < 0.35, np.sin((t / 0.35) * np.pi / 2), np.sin(((1 - t) / 0.65) * np.pi / 2))
        return np.clip(env, 0, 1) * peak

    def make_scenario(city, name, seed=None):
        dt_h, sim_hours = 5 / 60, 8.0
        c = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in city.items()}
        presets = {
            "Normal rain": dict(peak=15, drainage_scale=1.0),
            "Heavy rain": dict(peak=45, drainage_scale=1.0),
            "Drainage failure": dict(peak=15, drainage_scale=0.15),
            "Blocked channel": dict(peak=20, drainage_scale=1.0),
        }
        cfg = presets[name]
        if name == "Blocked channel":
            c["conductance"] = c["conductance"].copy()
            c["conductance"][15, 10:13] = 0.0
        rain = _mock_storm(sim_hours, dt_h, cfg["peak"])
        params = {"rain_mm_h": rain, "dt_h": dt_h, "sim_hours": sim_hours,
                  "drainage_scale": cfg["drainage_scale"]}
        return c, params

    def simulate(city, params):
        H, W = city["elevation"].shape
        dt_h = params.get("dt_h", 5 / 60)
        rain = np.atleast_1d(params["rain_mm_h"]).astype(float)
        n_steps = len(rain)
        dscale = params.get("drainage_scale", 1.0)
        depth = np.zeros((n_steps + 1, H, W))
        depth[0] = city.get("initial_depth", np.zeros((H, W)))
        drainage = city["drainage"] * dscale / 1000.0  # mm/h -> m/h
        low_spot = np.exp(-(((np.mgrid[0:H, 0:W][0] - H * 0.7) ** 2 +
                              (np.mgrid[0:H, 0:W][1] - W * 0.3) ** 2) / 30))
        for t in range(n_steps):
            rain_m_h = rain[t] / 1000.0
            d = depth[t]
            inflow = rain_m_h * dt_h * (0.5 + low_spot)
            outflow = np.minimum(d, drainage * dt_h)
            depth[t + 1] = np.clip(d + inflow - outflow, 0, None)
        time_h = np.arange(n_steps + 1) * dt_h
        zeros_hw = np.zeros((H, W))
        return {
            "depth": depth,
            "time_h": time_h,
            "rain_mm_h": rain,
            "inflow_total": zeros_hw,
            "drained_total": zeros_hw,
            "budget": {"time_h": time_h},
            "meta": {"substeps": 1, "alpha": 1.0, "shape": (H, W), "dt_h": dt_h},
        }
    # --- end mock engine

# ---------------------------------------------------------------------------
# Analytics integration (Person 3's analysis.py), with a fallback that
# mirrors the SAME bug-fixed logic, so app.py behaves identically whether
# analysis.py has landed in the repo yet or not. Delete the fallback block
# once analysis.py is confirmed present for everyone.
# ---------------------------------------------------------------------------
try:
    from analysis import (
        classify, time_to_critical, summarize, compare,
        banner_message, THRESHOLDS,
    )
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

    THRESHOLDS = {"warning": 0.15, "critical": 0.30}
    POPULATION_WEIGHT = {0: 0.0, 1: 0.25, 2: 1.0}
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

            # Only forecast cells already in Warning and due within horizon_h,
            # otherwise tiny drift produces false alarms (the "cries wolf" bug).
            plausible = ((not_ever) & (recent_depth >= thresholds["warning"])
                         & (rate > 1e-6) & (remaining > 0) & (extrap <= horizon_h))
            extrap_time = np.where(plausible, time_h[-1] + extrap, np.nan)
            eta = np.where(not_ever, extrap_time, eta)

        return eta

    def _affected_population(depth, population, thresholds=THRESHOLDS, weights=POPULATION_WEIGHT):
        cls = classify(depth, thresholds)
        worst_class = cls.max(axis=0)  # worst class EVER reached, not just final
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
            return f"{region}: critical now. About {int(pop):,} people affected. Cause: {reason}."
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
            "affected_population": _affected_population(depth, city["population"], thresholds),
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

        return {"kpis": kpis, "classification": cls, "eta": eta, "warning_table": warning_table}

    def banner_message(summary):
        kpis = summary["kpis"]
        n_crit = kpis["critical_regions"]
        first_h = kpis["first_critical_h"]

        if n_crit == 0 and summary["warning_table"].empty:
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
    # --- end analysis fallback

RISK_LABELS = {0: "Safe", 1: "Warning", 2: "Critical"}
RISK_COLORS = {0: "#2E8B57", 1: "#E1A100", 2: "#C0392B"}   # deeper, clearer semantic colors

# ---------------------------------------------------------------------------
# Warm formal theme palette (used by the CSS block and by the figures below)
# ---------------------------------------------------------------------------
THEME = {
    "bg": "#FBF3E7",            # warm ivory page background
    "panel": "#FFF9F0",         # card / panel background
    "sidebar": "#F3E3CE",       # warm tan sidebar
    "border": "#E3C79E",        # soft caramel border
    "heading": "#6B2E1F",       # deep brick / maroon for headings
    "text": "#3B2A20",          # dark coffee brown body text
    "accent": "#B5651D",        # warm burnt-orange accent (buttons, active tab)
    "accent_hover": "#8F4E17",  # darker accent on hover
}


# ---------------------------------------------------------------------------
# Region naming — every grid cell gets a stable, human-readable name so the
# heatmaps below can show "which grid this is" on hover.
# ---------------------------------------------------------------------------
def get_region_names(shape):
    """(H, W) array of readable cell names. Uses terrain.py's district-aware
    names when available, otherwise falls back to simple row/column labels."""
    H, W = shape
    try:
        from terrain import make_region_names
        names = make_region_names(size=H)
        if names.shape == shape:
            return names
    except Exception:
        pass
    return np.array([[f"Zone R{r}C{c}" for c in range(W)] for r in range(H)])


# ---------------------------------------------------------------------------
# Cached simulation + analysis runner
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_scenario(scenario_name: str, drainage_override, seed: int = 0):
    city = make_dev_city()
    c, p = make_scenario(city, scenario_name, seed=seed)
    if drainage_override is not None:
        p["drainage_scale"] = drainage_override
    result = simulate(c, p)
    summary = summarize(result, c, THRESHOLDS)
    return c, p, result, summary


# ---------------------------------------------------------------------------
# Plotly heatmap with time slider + play button
# Each cell is drawn as a clearly separated grid square with its own name,
# shown in the hover tooltip when the cursor moves over it.
# ---------------------------------------------------------------------------
def build_heatmap_figure(result: dict, view: str, cls: np.ndarray = None) -> go.Figure:
    depth = result["depth"]
    time_h = result["time_h"]
    T, H, W = depth.shape

    region_names = get_region_names((H, W))

    if view == "Water depth":
        z_stack = depth
        colorscale = [
            [0.0, "#FBF3E7"],   # dry cells match the page background
            [0.15, "#CFE8F0"],
            [0.40, "#7FC3DE"],
            [0.65, "#2E86AB"],
            [1.0, "#0B3D5C"],
        ]
        zmax = max(float(depth.max()), 0.01)
        colorbar_title = "Depth (m)"
        hover_value = "Depth: %{z:.2f} m"
    else:
        z_stack = cls.astype(float)
        colorscale = [[0.0, RISK_COLORS[0]], [0.5, RISK_COLORS[1]], [1.0, RISK_COLORS[2]]]
        zmax = 2
        colorbar_title = "Risk"
        hover_value = "Risk: %{customdata[1]}"

    # customdata carries [region name, risk label] per cell for the hover tooltip
    if view == "Water depth":
        customdata = np.dstack([region_names, region_names])
    else:
        risk_label_grid = np.vectorize(RISK_LABELS.get)(cls[-1] * 0 + 0)  # placeholder, replaced per-frame below
        customdata = np.dstack([region_names, region_names])

    def _customdata_for(i):
        if view == "Water depth":
            return np.dstack([region_names, region_names])
        labels = np.vectorize(RISK_LABELS.get)(cls[i])
        return np.dstack([region_names, labels])

    hovertemplate = "<b>%{customdata[0]}</b><br>" + hover_value + "<extra></extra>"

    common_kwargs = dict(
        colorscale=colorscale,
        zmin=0,
        zmax=zmax,
        xgap=1.5,
        ygap=1.5,
        hovertemplate=hovertemplate,
        colorbar=dict(
            title=dict(text=colorbar_title, font=dict(color=THEME["text"], size=13)),
            outlinewidth=1,
            outlinecolor=THEME["border"],
            tickfont=dict(color=THEME["text"]),
        ),
    )

    frames = [
        go.Frame(
            data=[go.Heatmap(z=z_stack[i], customdata=_customdata_for(i), **common_kwargs)],
            name=f"{time_h[i]:.2f}",
        )
        for i in range(T)
    ]

    fig = go.Figure(
        data=[go.Heatmap(z=z_stack[0], customdata=_customdata_for(0), **common_kwargs)],
        frames=frames,
    )

    slider_steps = [
        dict(
            method="animate",
            args=[[f.name], dict(mode="immediate",
                                  frame=dict(duration=150, redraw=True),
                                  transition=dict(duration=0))],
            label=f"{time_h[i]:.2f}h",
        )
        for i, f in enumerate(frames)
    ]

    fig.update_layout(
        title=dict(
            text=f"{view} over time — hover a cell to see its grid name",
            font=dict(color=THEME["heading"], size=18, family="Georgia, 'Times New Roman', serif"),
        ),
        height=580,
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["panel"],
        font=dict(color=THEME["text"]),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.14, x=0.0,
                bgcolor=THEME["panel"],
                bordercolor=THEME["border"],
                font=dict(color=THEME["heading"]),
                buttons=[
                    dict(label="▶️ Play", method="animate",
                         args=[None, dict(frame=dict(duration=150, redraw=True),
                                           fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
                ],
            )
        ],
        sliders=[dict(
            active=0, steps=slider_steps, x=0.05, len=0.9,
            currentvalue=dict(prefix="Time: ", font=dict(color=THEME["heading"])),
            font=dict(color=THEME["text"]),
            bgcolor=THEME["panel"],
            bordercolor=THEME["border"],
        )],
    )
    return fig


def build_final_risk_map(classification_last: np.ndarray, title: str) -> go.Figure:
    """A single clear, named-grid risk snapshot — used in the Compare tab."""
    H, W = classification_last.shape
    region_names = get_region_names((H, W))
    labels = np.vectorize(RISK_LABELS.get)(classification_last)
    customdata = np.dstack([region_names, labels])

    fig = go.Figure(
        go.Heatmap(
            z=classification_last,
            customdata=customdata,
            colorscale=[[0, RISK_COLORS[0]], [0.5, RISK_COLORS[1]], [1, RISK_COLORS[2]]],
            zmin=0, zmax=2,
            xgap=1.5, ygap=1.5,
            hovertemplate="<b>%{customdata[0]}</b><br>Risk: %{customdata[1]}<extra></extra>",
            colorbar=dict(
                title=dict(text="Risk", font=dict(color=THEME["text"], size=12)),
                outlinewidth=1, outlinecolor=THEME["border"],
                tickfont=dict(color=THEME["text"]),
            ),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color=THEME["heading"], size=15, family="Georgia, serif")),
        height=420,
        plot_bgcolor=THEME["bg"],
        paper_bgcolor=THEME["panel"],
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FloodShield", layout="wide", page_icon="🌊")

# ---- Warm formal theme (global CSS) ---------------------------------------
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {THEME['bg']};
            color: {THEME['text']};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {THEME['sidebar']};
            border-right: 1px solid {THEME['border']};
        }}
        section[data-testid="stSidebar"] * {{
            color: {THEME['text']};
        }}
        h1, h2, h3, h4 {{
            color: {THEME['heading']} !important;
            font-family: Georgia, 'Times New Roman', serif !important;
        }}
        p, li, span, label, div {{
            font-family: 'Segoe UI', Tahoma, Geneva, sans-serif;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background-color: {THEME['panel']};
            border: 1px solid {THEME['border']};
            border-radius: 10px;
            padding: 14px 10px;
            box-shadow: 0 1px 3px rgba(107, 46, 31, 0.12);
        }}
        div[data-testid="stMetricLabel"] {{
            color: {THEME['heading']};
        }}
        div[data-testid="stMetricValue"] {{
            color: {THEME['accent']};
        }}

        /* Buttons */
        .stButton>button {{
            background-color: {THEME['accent']};
            color: #FFF9F0;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}
        .stButton>button:hover {{
            background-color: {THEME['accent_hover']};
            color: #FFF9F0;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {THEME['panel']};
            border: 1px solid {THEME['border']};
            border-radius: 8px 8px 0 0;
            color: {THEME['text']};
            font-weight: 600;
            padding: 8px 16px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {THEME['accent']} !important;
            color: #FFF9F0 !important;
        }}

        /* Dataframes / tables */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {THEME['border']};
            border-radius: 8px;
            overflow: hidden;
        }}

        /* Alerts (info / warning / success) */
        div[data-testid="stAlert"] {{
            background-color: {THEME['panel']};
            border: 1px solid {THEME['border']};
            border-radius: 8px;
            color: {THEME['text']};
        }}

        /* Selectbox / slider labels */
        .stSelectbox label, .stSlider label, .stCheckbox label, .stRadio label {{
            color: {THEME['heading']} !important;
            font-weight: 600;
        }}

        hr {{
            border-color: {THEME['border']};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

if not ENGINE_AVAILABLE:
    st.warning(
        "Running with a local mock engine — terrain.py / scenarios.py / "
        "simulation.py were not found on the path. Drop the real modules "
        "next to app.py to switch over automatically.",
        icon="⚠️",
    )
if not ANALYSIS_AVAILABLE:
    st.warning(
        "analysis.py not found on the path — using a bundled fallback with "
        "identical logic. Drop analysis.py next to app.py to switch over "
        "automatically.",
        icon="⚠️",
    )

# ---- Formal, warm-styled page header --------------------------------------
st.markdown(
    f"""
    <div style="padding: 6px 0 14px 0; border-bottom: 2px solid {THEME['border']}; margin-bottom: 18px;">
        <h1 style="font-size: 2.6rem; margin-bottom: 0; color: {THEME['heading']};
                   font-family: Georgia, 'Times New Roman', serif; letter-spacing: 0.5px;">
            🌊 FloodShield
        </h1>
        <p style="font-size: 1.05rem; color: {THEME['text']}; margin-top: 4px;">
            Flood Simulation &amp; Early Warning Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.header("Scenario setup")
    scenario_name = st.selectbox("Preset scenario", SCENARIO_NAMES, index=1)

    st.subheader("Overrides (optional)")
    override_drainage = st.checkbox("Override drainage scale")
    drainage_scale = st.slider("Drainage capacity multiplier", 0.05, 1.5, 1.0, 0.05) \
        if override_drainage else None

    st.subheader("Risk thresholds (display only)")
    st.caption(
        f"Safe < {THRESHOLDS['warning']} m · "
        f"Warning {THRESHOLDS['warning']}–{THRESHOLDS['critical']} m · "
        f"Critical ≥ {THRESHOLDS['critical']} m"
    )

    run_clicked = st.button("▶️ Run simulation", type="primary", use_container_width=True)

if "state" not in st.session_state:
    st.session_state["state"] = None

if run_clicked:
    with st.spinner("Running simulation..."):
        city, params, result, summary = run_scenario(scenario_name, drainage_scale)
    st.session_state["state"] = dict(city=city, params=params, result=result, summary=summary)
    st.session_state["scenario_label"] = scenario_name

state = st.session_state["state"]

# ---- Tabs -------------------------------------------------------------
tab_sim, tab_warn, tab_compare, tab_about = st.tabs(
    ["Simulation", "Early Warning", "Compare", "About"]
)

# ---- Simulation tab ---------------------------------------------------
with tab_sim:
    if state is None:
        st.info("Set up a scenario in the sidebar and click **Run simulation**.")
    else:
        city, result, summary = state["city"], state["result"], state["summary"]
        kpis = summary["kpis"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak depth (m)", f"{kpis['peak_depth']:.2f}")
        c2.metric("Critical regions", kpis["critical_regions"])
        c3.metric("Warning regions", kpis["warning_regions"])
        first_h = kpis["first_critical_h"]
        c4.metric("First critical", f"{first_h:.2f} h" if first_h is not None else "—")

        st.write("")
        view = st.radio("View", ["Water depth", "Risk classification"], horizontal=True)
        fig = build_heatmap_figure(result, view, summary["classification"])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Move your cursor over any cell to see its grid name and value.")

        st.caption(f"Estimated affected population: {kpis['affected_population']:,.0f}")

# ---- Early Warning tab -------------------------------------------------
with tab_warn:
    if state is None:
        st.info("Run a simulation first.")
    else:
        summary = state["summary"]
        st.info(banner_message(summary))

        st.subheader("Ranked regions by urgency")
        df = summary["warning_table"]
        if df.empty:
            st.success("No regions currently at Warning or Critical risk in this scenario.")
        else:
            st.dataframe(
                df[["Region", "Now", "Alert level", "Lead time (h)", "Peak depth (m)",
                    "People at risk", "Why", "Message"]],
                use_container_width=True, hide_index=True,
            )

# ---- Compare tab --------------------------------------------------------
with tab_compare:
    st.subheader("Compare two scenarios")
    colA, colB = st.columns(2)
    with colA:
        scen_a = st.selectbox("Scenario A", SCENARIO_NAMES, index=0, key="scen_a")
    with colB:
        scen_b = st.selectbox("Scenario B", SCENARIO_NAMES, index=1, key="scen_b")

    if st.button("Compare"):
        with st.spinner("Running both scenarios..."):
            city_a, _, result_a, summary_a = run_scenario(scen_a, None)
            city_b, _, result_b, summary_b = run_scenario(scen_b, None)

        # compare() needs both scenarios to share the same city (terrain/population)
        # for the numbers to mean anything — true here since both use make_dev_city().
        comp_table = compare({scen_a: result_a, scen_b: result_b}, city_a, THRESHOLDS)
        st.dataframe(comp_table, use_container_width=True)

        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(
            x=result_a["time_h"], y=result_a["depth"].mean(axis=(1, 2)),
            name=f"{scen_a} (avg depth)", mode="lines",
            line=dict(color=THEME["accent"], width=3),
        ))
        line_fig.add_trace(go.Scatter(
            x=result_b["time_h"], y=result_b["depth"].mean(axis=(1, 2)),
            name=f"{scen_b} (avg depth)", mode="lines",
            line=dict(color=THEME["heading"], width=3, dash="dot"),
        ))
        line_fig.update_layout(
            title=dict(text="Average city-wide depth over time",
                       font=dict(color=THEME["heading"], family="Georgia, serif")),
            xaxis_title="Time (h)", yaxis_title="Depth (m)", height=400,
            plot_bgcolor=THEME["bg"], paper_bgcolor=THEME["panel"],
            font=dict(color=THEME["text"]),
            legend=dict(bgcolor=THEME["panel"], bordercolor=THEME["border"], borderwidth=1),
        )
        st.plotly_chart(line_fig, use_container_width=True)

        map_col1, map_col2 = st.columns(2)
        with map_col1:
            st.plotly_chart(
                build_final_risk_map(summary_a["classification"][-1], f"{scen_a} — final risk"),
                use_container_width=True,
            )
        with map_col2:
            st.plotly_chart(
                build_final_risk_map(summary_b["classification"][-1], f"{scen_b} — final risk"),
                use_container_width=True,
            )
        st.caption("Note: for a fair comparison, keep every parameter identical except the one "
                   "you're testing (e.g. same terrain/population, different rainfall). "
                   "Hover any cell above to see its grid name.")

# ---- About tab ------------------------------------------------------------
with tab_about:
    st.subheader("Core idea")
    st.latex(r"\Delta \text{depth} = \text{rain} - \text{drainage} + \text{inflow} - \text{outflow}")

    st.subheader("Parameters")
    param_df = pd.DataFrame([
        {"Parameter": "rain_mm_h", "Meaning": "Rainfall intensity per step", "Units": "mm/h", "Default": "scenario-defined"},
        {"Parameter": "drainage", "Meaning": "Base drainage capacity per cell", "Units": "mm/h", "Default": "terrain-defined"},
        {"Parameter": "drainage_scale", "Meaning": "Multiplier on drainage map", "Units": "—", "Default": "1.0"},
        {"Parameter": "elevation", "Meaning": "Terrain height", "Units": "m", "Default": "terrain-defined"},
        {"Parameter": "conductance", "Meaning": "Ease of flow to neighbours (0=blocked, 5-6=river)", "Units": "—", "Default": "1.0"},
        {"Parameter": "dt_h", "Meaning": "Simulation step length", "Units": "h", "Default": "5/60"},
        {"Parameter": "k", "Meaning": "Flow coefficient", "Units": "1/h", "Default": "2.4"},
    ])
    st.dataframe(param_df, use_container_width=True, hide_index=True)

    st.subheader("Assumptions")
    st.markdown(
        "- Rainfall is applied uniformly across the grid within a step.\n"
        "- Water does not soak into the ground (no infiltration term).\n"
        "- Each cell exchanges flow with its 4 orthogonal neighbours only.\n"
        "- Population figures are static per cell and used only for impact estimates, "
        "not by the physical simulation."
    )

    st.subheader("Risk classification & legend")
    leg1, leg2, leg3 = st.columns(3)
    leg1.markdown(f"🟩 **Safe** — depth < {THRESHOLDS['warning']} m")
    leg2.markdown(f"🟨 **Warning** — {THRESHOLDS['warning']}–{THRESHOLDS['critical']} m")
    leg3.markdown(f"🟥 **Critical** — depth ≥ {THRESHOLDS['critical']} m")

    st.subheader("How the early warning is produced")
    st.markdown(
        "Each cell's depth is classified every time step. The first time step where a cell "
        "reaches the Critical threshold is its **time-to-critical**. For cells not yet critical "
        "by the end of the run, we extrapolate from the recent trend to estimate when (if ever) "
        "it would cross — capped to a 3-hour forecast horizon, and only for cells already in the "
        "Warning band, to avoid false alarms from tiny drift. Regions are ranked soonest-first, "
        "alongside the population figure for that cell, in the Early Warning tab."
    )

    st.subheader("Limitations")
    st.markdown(
        "- No infiltration/soil absorption modelled.\n"
        "- Extrapolation for time-to-critical is a simple linear estimate over a 3-hour horizon, "
        "not a full forecast.\n"
        "- Terrain and population in the dev build are synthetic, not real city data.\n"
        "- Risk thresholds are placeholders pending team sign-off."
    )

    st.subheader("Team & tools")
    st.markdown(
        "Simulation: NumPy · Results & metrics: Pandas · Visuals: Plotly · UI: Streamlit · "
        "Deploy: Streamlit Community Cloud."
    )
    st.markdown(
        "| Person | Role | Owns |\n|---|---|---|\n"
        "| 1 | Simulation engine | simulation.py |\n"
        "| 2 | Data and scenarios | terrain.py, scenarios.py |\n"
        "| 3 | Analytics and early warning | analysis.py |\n"
        "| 4 | Dashboard, integration and demo | app.py, README.md |"
    )