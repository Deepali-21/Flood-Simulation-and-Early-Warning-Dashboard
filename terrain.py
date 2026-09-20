"""
terrain.py -- Synthetic city generator (Person 2's module)

Builds the city that simulation.py runs on. Everything is generated from code, and
seeded, so the same seed always gives the same city (needed for fair scenario comparison).

Public API (what scenarios.py and app.py import):

    make_dev_city(size=30, seed=42) -> dict of (H, W) arrays   (alias: build_city)
    make_channel_path(kind="valley", size=30) -> bool (H, W) mask of the river channel
    make_conductance(size=30) -> (H, W) float array (1 = normal ground, 6 = river channel)
    make_region_names(size=30) -> (H, W) array of readable names, e.g. "Riverside R12C24"

City dict (the contract with simulation.py; units: metres, mm/h, people):
    "elevation"      ground height [m]
    "drainage"       drain capacity [mm/h]
    "initial_depth"  starting water [m]            (zeros; scenarios may change it)
    "conductance"    connection strength           (river channel = 6, elsewhere 1)
    "outlet_mask"    True where water leaves       (the river mouth, south edge)
    "population"     people per cell

The map (north = row 0, west = column 0):
    - ground slopes down towards the south-east
    - a river valley runs north -> south at ~78% of the map width, ending in an outlet
    - two low basins (West Basin, South Basin) that collect water and flood first
    - a hill in the north-west that stays dry
    - a well-drained city centre, and an "Old Town" with weak drainage
    - population densest downtown
"""
from __future__ import annotations

import numpy as np

DEFAULT_SIZE = 30
DEFAULT_SEED = 42

RIVER_CONDUCTANCE = 6.0      # the river channel conveys water 6x faster than ordinary ground


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _grid(size: int):
    """y, x coordinates as fractions 0..1 (y = row, x = column)."""
    y, x = np.mgrid[0:size, 0:size] / size
    return y, x


def river_columns(size: int = DEFAULT_SIZE) -> np.ndarray:
    """Column indices of the river channel (a band around 78% of the map width)."""
    x = np.arange(size) / size
    return np.flatnonzero((x > 0.72) & (x < 0.84))


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
def make_elevation(size: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = _grid(size)

    def bump(cx, cy, r, amp):
        return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r ** 2))

    z = 8.0 - 3.0 * x - 2.0 * y                                    # slopes down to the south-east
    z += -1.5 * np.exp(-((x - 0.78) ** 2) / (2 * 0.04 ** 2))       # river valley, north -> south
    z += bump(0.30, 0.35, 0.09, -1.4)                              # West Basin
    z += bump(0.45, 0.75, 0.08, -1.2)                              # South Basin
    z += bump(0.15, 0.20, 0.12, +2.0)                              # hill
    z += rng.normal(0, 0.03, z.shape)                              # small roughness (the only seeded part)
    return z


def make_drainage(size: int = DEFAULT_SIZE) -> np.ndarray:
    y, x = _grid(size)
    drainage = 12 + 18 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.08)   # 12 mm/h outskirts, 30 mm/h downtown
    drainage[int(0.55 * size):int(0.85 * size), int(0.05 * size):int(0.30 * size)] = 6.0   # Old Town: weak drains
    return drainage


def make_population(size: int = DEFAULT_SIZE) -> np.ndarray:
    y, x = _grid(size)
    return (2000 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.1) + 200).astype(int)


def make_outlet_mask(size: int = DEFAULT_SIZE) -> np.ndarray:
    """The river mouth: the channel cells on the south edge, where water leaves the city."""
    outlet = np.zeros((size, size), dtype=bool)
    outlet[-1, river_columns(size)] = True
    return outlet


def make_conductance(size: int = DEFAULT_SIZE, river_boost: float = RIVER_CONDUCTANCE) -> np.ndarray:
    """1 = ordinary ground; river channel cells convey water faster."""
    conductance = np.ones((size, size))
    conductance[:, river_columns(size)] = river_boost
    return conductance


def make_channel_path(kind: str = "valley", size: int = DEFAULT_SIZE) -> np.ndarray:
    """
    Boolean mask of the WHOLE river channel (every row).
    scenarios.py reduces this to a single cross-section when it models a blockage.
    """
    if kind != "valley":
        raise ValueError(f"unknown channel kind {kind!r}; only 'valley' is available")
    mask = np.zeros((size, size), dtype=bool)
    mask[:, river_columns(size)] = True
    return mask


# --------------------------------------------------------------------------
# the city
# --------------------------------------------------------------------------
def make_dev_city(size: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> dict:
    """Build the full city dict. Same (size, seed) always gives the same city."""
    if size < 12:
        raise ValueError("size must be at least 12 for the map features to fit")
    return {
        "elevation": make_elevation(size, seed),
        "drainage": make_drainage(size),
        "initial_depth": np.zeros((size, size)),
        "conductance": make_conductance(size),
        "outlet_mask": make_outlet_mask(size),
        "population": make_population(size),
    }


build_city = make_dev_city      # the name used in the original team contract


# --------------------------------------------------------------------------
# readable region names (pass to analysis.summarize(..., region_names=...))
# --------------------------------------------------------------------------
def make_district_map(size: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> np.ndarray:
    """District label for every cell. Earlier rules win where districts overlap."""
    y, x = _grid(size)
    z = make_elevation(size, seed)
    rows, cols = np.mgrid[0:size, 0:size]

    district = np.empty((size, size), dtype=object)
    # suburbs by quadrant (default)
    district[:] = "Suburb"
    district[(y < 0.5) & (x < 0.5)] = "North-West Suburb"
    district[(y < 0.5) & (x >= 0.5)] = "North-East Suburb"
    district[(y >= 0.5) & (x < 0.5)] = "South-West Suburb"
    district[(y >= 0.5) & (x >= 0.5)] = "South-East Suburb"

    downtown = np.hypot(x - 0.5, y - 0.5) < 0.18
    hillside = z > np.percentile(z, 88)
    west_basin = np.hypot(x - 0.30, y - 0.35) < 0.13
    south_basin = np.hypot(x - 0.45, y - 0.75) < 0.13
    old_town = np.zeros((size, size), dtype=bool)
    old_town[int(0.55 * size):int(0.85 * size), int(0.05 * size):int(0.30 * size)] = True
    riverside = np.isin(cols, np.concatenate([river_columns(size), river_columns(size) - 1, river_columns(size) + 1]))

    for mask, label in ((downtown, "Downtown"), (hillside, "Hillside"), (south_basin, "South Basin"),
                        (west_basin, "West Basin"), (old_town, "Old Town"), (riverside, "Riverside")):
        district[mask] = label                       # later entries override earlier ones
    return district


def make_region_names(size: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> np.ndarray:
    """(H, W) array of names like 'Riverside R12C24' (row 12, column 24, 0-indexed)."""
    district = make_district_map(size, seed)
    rows, cols = np.mgrid[0:size, 0:size]
    names = np.empty((size, size), dtype=object)
    for r in range(size):
        for c in range(size):
            names[r, c] = f"{district[r, c]} R{r}C{c}"
    return names


# --------------------------------------------------------------------------
# run directly to inspect the city:   python terrain.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    city = make_dev_city()
    print("Layer            shape     min      max")
    for key, arr in city.items():
        a = np.asarray(arr)
        print(f"{key:15s}  {str(a.shape):8s} {a.min():8.2f} {a.max():8.2f}   dtype={a.dtype}")
    print("\nRiver columns:", river_columns(), "| outlet cells:", int(city["outlet_mask"].sum()))
    print("Total population:", int(city["population"].sum()))
    d = make_district_map()
    labels, counts = np.unique(d, return_counts=True)
    print("\nDistricts:", dict(zip(labels, map(int, counts))))
    print("Example names:", make_region_names()[12, 24], "|", make_region_names()[22, 13])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: skipping terrain_preview.png)")
    else:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        for ax, (title, data, cmap) in zip(axes, [
            ("Elevation (m)", city["elevation"], "terrain"),
            ("Drainage (mm/h)", city["drainage"], "viridis"),
            ("Population", city["population"], "magma"),
            ("Conductance (river = 6)", city["conductance"], "Blues"),
        ]):
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.8)
        ax = axes[0]
        ax.contour(city["outlet_mask"], levels=[0.5], colors="k", linewidths=1)
        fig.savefig("terrain_preview.png", dpi=110)
        print("\nSaved terrain_preview.png")