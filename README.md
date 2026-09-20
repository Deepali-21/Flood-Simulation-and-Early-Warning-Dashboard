# Flood-Simulation-and-Early-Warning-Dashboard
# Urban Flood Simulation and Early Warning Dashboard

An interactive dashboard that simulates how rainfall accumulates and moves across a city, classifies every region as **Safe**, **Warning** or **Critical**, and tells decision-makers **which regions flood first, how long they have, and how many people are affected**.


**Live demo:** [ADD LINK] | **Demo video:** [ADD LINK]

---

## Overview

Whether a region floods depends on several things at once: how hard it rains, how fast the drains can clear water, how high the ground is, what water flows in from neighbours, and whether the connections between regions are open or blocked. This project models those interactions on a grid of city regions and turns the result into an early-warning output.

**Pipeline:** rainfall and terrain input → water-level simulation → flood progression → risk classification → early-warning output.

## Features

- **Configurable rainfall:** intensity, duration and storm shape (constant or bell-shaped).
- **Connected city grid:** each region exchanges water with its 4 neighbours; every region has its own elevation, drainage capacity, initial water level and population.
- **Water accumulation and movement:** rain, drainage, and flow downhill between regions, with outlets where water leaves the city.
- **Time-based visualization:** animated heatmap with a play button and an interactive time slider.
- **Risk classification:** Safe / Warning / Critical for every region at every time step.
- **Early warning:** ranked list of regions with estimated time to critical, alert level and a plain-language message.
- **Affected population estimate** for each scenario.
- **Scenarios:** normal rainfall, heavy rainfall, drainage failure and a blocked drainage channel, with a side-by-side **comparison** tab.

## How it works

### Simulation engine (`simulation.py`)

Each region holds a water depth `h`. Every step, for all regions at once (vectorised NumPy):

```
new depth = depth + rain - drainage + inflow from neighbours - outflow to neighbours
```

1. **Rain** is added to every region.
2. **Drains** remove water up to the region's drainage capacity, never more than is present.
3. **Flow between neighbours** follows the *water surface height* `elevation + depth`. Water moves from higher to lower surface, in proportion to the height difference and the connection strength.
4. **Outlets** (for example a river mouth) carry water out of the city.

Design decisions worth knowing:

- **Connection strength (`conductance`)** is set per region: `1` is normal ground, `0` is fully blocked, values between are partial obstructions, and `2` to `10` model fast river or canal channels. This is how a *blocked channel* is simulated.
- **Water is conserved by construction.** All flows are computed from the old state and applied together, and a region can never send more water than it holds. A water budget is returned so this can be checked.
- **Step-size independent.** The engine always computes internally in steps of 1 minute or less, and only records at the dashboard's step, so the flood's speed does not change with the recording interval.

### Analysis and early warning (`analysis.py`)

| Output | How it is computed |
|---|---|
| **Classification** | Safe below 0.15 m, Warning 0.15 to 0.30 m, Critical from 0.30 m (thresholds are configurable) |
| **Time to critical** | Exact crossing time if the region reaches 0.30 m; otherwise extrapolated from its recent rate of rise, for regions already in Warning that would cross within 3 hours |
| **Alert level** | By lead time: *Critical now*, *Immediate* (under 1 h), *Prepare* (1 to 3 h), *Watch* (beyond 3 h) |
| **Affected population** | Population of each region weighted by its worst class: 100% for Critical, 25% for Warning |
| **Reason** | Simple rules: low-lying basin, weak drainage, already at Warning depth |

### Scenarios

All scenarios use the same engine with different parameters, so comparisons are fair (same terrain, population and thresholds).

| Scenario | What changes |
|---|---|
| Normal rain | Light bell-shaped storm |
| Heavy rain | Much higher peak intensity |
| Drainage failure | Drainage capacity multiplied by 0.2 |
| Blocked channel | Drainage and conductance set to 0 across the river channel |

On the development demo city, critical regions rise from 0 (normal rain) to 24 (heavy rain), 40 (blocked channel) and 67 (drainage failure). The exact numbers change with the final terrain.

## Tech stack and open-source libraries

| Library | Purpose | License |
|---|---|---|
| [Python](https://www.python.org/) 3.10+ | Language | PSF |
| [NumPy](https://numpy.org/) | Vectorised simulation and array maths | BSD-3-Clause |
| [pandas](https://pandas.pydata.org/) | Warning tables and scenario comparison | BSD-3-Clause |
| [Plotly](https://plotly.com/python/) | Interactive heatmap, time slider, charts | MIT |
| [Streamlit](https://streamlit.io/) | Web dashboard and hosting | Apache-2.0 |
| [Matplotlib](https://matplotlib.org/) | Development plots in `dev_run.py` | Matplotlib License (BSD-compatible) |
| [pytest](https://pytest.org/) | Automated tests | MIT |

## Project structure

```
.
├── app.py                        # Streamlit dashboard
├── simulation.py                 # Flood simulation engine
├── analysis.py                   # Classification, time to critical, warnings, comparison
├── terrain.py                    # Synthetic city: elevation, drainage, population
├── scenarios.py                  # Scenario presets and rainfall profiles
├── dev_run.py                    # Stand-in city and scenarios for development
├── requirements.txt
├── LICENSE
├── tests/
│   ├── test_simulation.py             # Engine physics checks
│   ├── test_analysis_integration.py   # Analysis running on real engine output
│   ├── test_terrain.py                # City generator checks
│   └── test_scenarios_integration.py  # Scenarios running through the engine
└── README.md
```

## Installation

Requires Python 3.10 or newer.

```bash
git clone <https://github.com/Deepali-21/Flood-Simulation-and-Early-Warning-Dashboard.git>
cd <Flood-Simulation-and-Early-Warning-Dashboard>

python -m venv venv
# Windows (PowerShell):  .\venv\Scripts\Activate.ps1
# macOS / Linux:         source venv/bin/activate

pip install -r requirements.txt
```

## Usage

**Run the dashboard**

```bash
streamlit run app.py
```

Use the sidebar to set rainfall intensity, duration, storm shape, scenario and drainage scale, then run the simulation. Explore the tabs:

| Tab | Shows |
|---|---|
| Simulation | Animated heatmap with time slider and play button, KPI cards, risk counts over time |
| Early warning | Alert banner and ranked list of regions with time to critical, alert level and people at risk |
| Compare scenarios | Scenarios side by side: table and overlaid charts |
| About the model | Parameters, thresholds, assumptions and limitations |

**Try the engine without the dashboard**

```bash
python dev_run.py
```

Prints a scenario comparison table and saves `dev_output.png`.

**Use it as a library**

```python
from simulation import simulate
from analysis import summarize, banner_message
from dev_run import make_dev_city, make_scenario

city, params = make_scenario(make_dev_city(), "Heavy rain")
result = simulate(city, params)

summary = summarize(result, city)
print(banner_message(summary))
print(summary["warning_table"].head())
```

### Engine inputs and outputs

All units are metres and hours; rain and drainage are in mm/h. Grids are `(H, W)` NumPy arrays.

| `city` key | Meaning | Default |
|---|---|---|
| `elevation` | Ground height (m) | required |
| `drainage` | Maximum drain removal rate (mm/h) | required |
| `initial_depth` | Water present at the start (m) | zeros |
| `conductance` | Connection strength: 0 blocked, 1 normal, above 1 channel | ones |
| `outlet_mask` | True where water leaves the city | none |
| `inflow_mm_h` | Optional constant external source (mm/h) | zeros |
| `population` | People per region (used by analysis) | n/a |

`params`: `rain_mm_h` (required; an array with one value per step, or a number with `sim_hours`), `dt_h` (default 5/60), `k` (flow coefficient, default 2.4 per hour), `drainage_scale` (default 1.0).

`simulate()` returns `depth` `(T, H, W)` in metres, `time_h` `(T,)`, cumulative inflow and drainage per region, and a water `budget`.

## Testing

```bash
python -m pytest -v
```

The tests check that the engine:

- conserves water (start + rain + source - drained - outlet = end),
- never produces negative depths, even on rough terrain,
- matches a hand calculation on flat ground,
- makes water pool in basins, back up behind a blocked channel and flood worse when drains fail,
- gives the same result regardless of the recording step,

and that the analysis reports correct KPIs on real engine output, never crashes at any slider position, and does not forecast implausible times.

## Assumptions and limitations

This is a **simplified conceptual model**, not a hydrodynamic forecast. Results show relative risk and the order in which regions are affected, not exact depths.

- Rain is uniform across the city, and the ground does not absorb water.
- Water moves by a height-difference rule between 4-connected regions.
- Drainage is a fixed capacity per region, not a modelled pipe network.
- Terrain, drainage and population are synthetic, not measured.
- Warning lead times are short when a flood rises quickly, because forecasts extrapolate the recent trend.

**Future work:** real elevation data (DEM), spatially varying rainfall, soil infiltration, a pipe-network drainage model, 8-neighbour connectivity, and calibration against past flood records.

## Team

| Name | Role | Files |
|---|---|---|
| Deepali R Nayak | Simulation engine | `simulation.py`, `tests/test_simulation.py` |
| Dhanushree H S | Terrain, data and scenarios | `terrain.py`, `scenarios.py` |
| Daneshwari Shrishail Dolli | Analytics and early warning | `analysis.py` |
| Gayathri G B | Dashboard, integration and demo | `app.py`, `README.md` |

## Contributing

1. Work on your own branch and edit mainly your own files.
2. Run `python -m pytest -v` before you push; `main` should always run.
3. Pull the latest `main` before pushing, and merge small changes often.
4. Do not commit `venv/`, `__pycache__/` or generated images.
5. If you need to change another person's module or a function signature, tell them first.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Deepali R Nayak, Daneshwari Shrishail Dolli, Dhanushree H S, Gayathri G B.

## Acknowledgements

Built with the open-source libraries listed above. Developed for a 24-hour online hackathon.
