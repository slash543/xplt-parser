# xplt-parser — FEBio Post-Processing Toolkit

Parse FEBio `.xplt` binary output and `.feb` mesh files to extract contact
surface geometry, compute per-facet contact pressure time-series, analyse
**Contact Surface Area Ratio (CSAR)**, and export training data for surrogate
models (see [surrogate-lab](../surrogate-lab/)).

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Class: SimulationCase](#core-class-simulationcase)
  - [Attributes](#attributes)
  - [Methods](#methods)
- [Surrogate-Lab Integration](#surrogate-lab-integration)
  - [Exporting the Surrogate CSV](#exporting-the-surrogate-csv)
  - [How Insertion Depth is Computed](#how-insertion-depth-is-computed)
  - [Adding New Variables](#adding-new-variables)
- [CSAR Analysis](#csar-analysis)
  - [Single-Region CSAR](#single-region-csar)
  - [Multi-Region CSAR](#multi-region-csar)
- [Plotting Functions](#plotting-functions)
- [VTP Export (ParaView)](#vtp-export-paraview)
- [File Structure](#file-structure)
- [Typical Workflow](#typical-workflow)

---

## Requirements

```
Python >= 3.10
numpy, pandas, matplotlib
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the interactive notebook:

```bash
pip install jupyter ipykernel
```

---

## Quick Start

```python
import xplt_core as xc

# Load one simulation (reads both .feb and .xplt)
case = xc.SimulationCase("my_run.feb", "my_run.xplt", label="Run A")
print(case.summary())

# Visualise geometry and contact
xc.plot_geometry(case)
xc.plot_contact_overview(case)

# Export surrogate-lab training CSV (one row per facet × timestep)
case.df_surrogate().to_csv("my_run_surrogate.csv", index=False)
```

---

## Core Class: `SimulationCase`

```python
case = xc.SimulationCase(
    feb_path             = "run.feb",
    xplt_path            = "run.xplt",
    label                = "Run A",          # display name (defaults to .feb stem)
    contact_surface_name = "contact_slave",  # substring match; None = largest surface
    tip_material_names   = {"catheter_tip"}, # materials that define the tip region
    tip_z_cutoff         = 50.0,             # fallback tip Z boundary [mm]
    cp_var_id            = 1,                # xplt variable ID for contact pressure
)
```

On construction the class:

1. Parses the `.feb` XML for material names, prescribed z-displacement boundary
   conditions, and load curves.
2. Reads the `.xplt` binary for node coordinates, surface facets, and the full
   contact-pressure time-series.
3. Computes facet centroids, areas, tip Z range, and **insertion depth** at
   every timestep directly from the `.feb` load curves — no manual scale factor
   needed.

### Attributes

| Attribute | Shape | Units | Description |
|---|---|---|---|
| `timesteps` | `(T,)` | s | Simulation time at each state |
| `cp_matrix` | `(T, N)` | MPa | Contact pressure per facet per timestep |
| `areas` | `(N,)` | mm² | Facet surface areas |
| `centroids` | `(N, 3)` | mm | Facet centroid XYZ coordinates |
| `facets` | `(N, 3)` | — | Triangle node indices (0-based) |
| `coords` | `(M, 3)` | mm | Global node coordinates |
| `insertion_depths` | `(T,)` | mm | Catheter insertion depth at each timestep |
| `n_facets` | int | — | Number of facets on the contact surface |
| `n_timesteps` | int | — | Number of parsed state blocks |
| `tip_z_min` | float | mm | Lower Z bound of the tip region |
| `tip_z_max` | float | mm | Upper Z bound of the tip region |
| `df_facets` | DataFrame | — | Per-facet geometry + peak contact pressure |

### Methods

| Method | Returns | Description |
|---|---|---|
| `summary()` | str | Human-readable overview of the loaded case |
| `df_surrogate()` | DataFrame | Tidy (facet × timestep) CSV for surrogate-lab |
| `compute_csar(zmin, zmax)` | `(ts, csar, area)` | Contact Surface Area Ratio over time |
| `compute_region_accumulation(z_bands)` | `(ts, band_stats, accumulated)` | Raw CSAR per Z band |
| `export_vtp()` | Path | Write per-timestep VTP + PVD for ParaView |

---

## Surrogate-Lab Integration

### Exporting the Surrogate CSV

`df_surrogate()` returns a **tidy long-format DataFrame** — one row per
(facet × timestep) — with exactly the column names that
[surrogate-lab](../surrogate-lab/) expects:

| Column | Units | Description |
|---|---|---|
| `centroid_x` | mm | Facet centroid X coordinate |
| `centroid_y` | mm | Facet centroid Y coordinate |
| `centroid_z` | mm | Facet centroid Z coordinate |
| `facet_area` | mm² | Facet surface area |
| `insertion_depth` | mm | Catheter insertion depth at this timestep |
| `contact_pressure` | MPa | Contact pressure at this facet at this timestep |

```python
df = case.df_surrogate()
print(df.shape)          # (n_timesteps * n_facets, 6)
print(df.columns.tolist())
# ['centroid_x', 'centroid_y', 'centroid_z', 'facet_area',
#  'insertion_depth', 'contact_pressure']

df.to_csv("my_run_surrogate.csv", index=False)
```

For multiple cases, drop all CSVs into `surrogate-lab/data/simulations/` and
the training pipeline will concatenate them automatically.

### How Insertion Depth is Computed

Insertion depth is read directly from the `.feb` file — **no manual scale
factor is needed**. When you load a new simulation the correct depths are
always used.

The `.feb` file defines one or more `<bc type="prescribed displacement">` on
the catheter with `<dof>z</dof>`. Each BC references a load controller that
maps simulation time to a scale factor (0 → 1).

`xplt_core` replays these BCs chronologically:

- **Absolute BCs** (`<relative>0</relative>`): set the running z-position
  directly as `magnitude × scale(t)`.
- **Relative BCs** (`<relative>1</relative>`): add an increment
  `magnitude × scale(t)` on top of the previous running position.

The result is an `insertion_depths` array (shape `(n_timesteps,)`, positive
mm) that reflects the exact catheter motion encoded in the `.feb` file.
Changing the prescribed displacement values in the `.feb` and re-loading
the case will automatically produce updated depths in the CSV.

### Adding New Variables

The variables in the surrogate CSV are driven by a **column registry** on
`SimulationCase`. Adding a new variable is a three-step process requiring no
changes outside of these two files:

**Step 1 — Add a `_col_<name>` method to `SimulationCase` in `xplt_core.py`**

The method must return a 1-D NumPy array of length `n_timesteps × n_facets`.
Use `np.tile` for per-facet quantities (same value at every timestep) and
`np.repeat` for per-timestep quantities (same value across all facets).

```python
# Example: expose the Z-component of the surface traction at each facet/timestep
def _col_surface_traction_z(self) -> np.ndarray:
    # self.traction_matrix is shape (n_timesteps, n_facets) — add parsing if needed
    return self.traction_matrix[:, :, 2].ravel()
```

**Step 2 — Register it in `SimulationCase.SURROGATE_COLUMNS`**

```python
SURROGATE_COLUMNS = {
    ...
    'surface_traction_z': ('Z-component of surface traction [MPa]', '_col_surface_traction_z'),
}
```

**Step 3 — Add the column name to surrogate-lab's `configs/config.yaml`**

```yaml
features:
  inputs:
    - centroid_x
    - centroid_y
    - centroid_z
    - facet_area
    - insertion_depth
    - surface_traction_z   # ← new
```

Re-export the CSV and re-run training. No other changes are needed.

---

## CSAR Analysis

### Single-Region CSAR

```python
# Returns (timesteps_array, csar_array, total_region_area_mm2)
ts, csar, area = case.compute_csar(zmin=0.0, zmax=50.0)

# Two-panel plot: CSAR vs time + facet contact map at final timestep
fig = xc.plot_csar(case, zmin=0.0, zmax=50.0)

# Overlay multiple cases on one plot
fig = xc.compare_csar([case_a, case_b, case_c], zmin=0.0, zmax=50.0)

# Export numeric table (columns: time_s, <case_label>, ...)
df = xc.csar_table([case_a, case_b], zmin=0.0, zmax=50.0)
df.to_csv("csar_comparison.csv", index=False)
```

### Multi-Region CSAR

Define Z bands and, optionally, override the reference area with a known
value (e.g. from CAD) before running.

```python
Z_BANDS     = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
BAND_LABELS = ["proximal", "mid", "distal"]

# Raw accumulation — inspect before computing CSAR
ts, band_stats, accumulated = case.compute_region_accumulation(Z_BANDS)

# Each band_stats[i] contains:
# {
#   'zmin', 'zmax',
#   'n_facets_in_region': int,
#   'total_area_mm2':     float,
#   'contact_area_mm2':   np.ndarray (n_timesteps,),
#   'n_contact_facets':   np.ndarray (n_timesteps,),
# }
# 'accumulated' has the same keys (no zmin/zmax) for the union of all bands.

# CSAR using computed area
csar = accumulated['contact_area_mm2'] / accumulated['total_area_mm2']

# CSAR with manual reference area (e.g. from CAD)
csar = accumulated['contact_area_mm2'] / 95.0

# Per-band + accumulated plot for one case
fig = xc.plot_csar_multi_regions(
    case,
    z_bands             = Z_BANDS,
    band_labels         = BAND_LABELS,
    total_area_override = 95.0,   # omit to use computed area
)

# Multi-case comparison
fig = xc.compare_csar_accumulated(
    cases                = [case_a, case_b],
    z_bands              = Z_BANDS,
    band_labels          = BAND_LABELS,
    total_area_overrides = [None, 95.0],  # per-case; None = use computed
)

# Numeric export
df = xc.region_accumulation_table(
    cases                = [case_a, case_b],
    z_bands              = Z_BANDS,
    total_area_overrides = [None, 95.0],
)
df.to_csv("csar_accumulated.csv", index=False)
```

---

## Plotting Functions

All plotting functions accept `save=True` (default) to write a PNG and return
the `matplotlib.Figure` for further customisation.

| Function | Description |
|---|---|
| `plot_geometry(case)` | 2-D centroid projections with tip zone highlighted |
| `plot_contact_overview(case)` | 4-panel: max/mean cp, facet count, facet map, top-10 tip traces |
| `plot_csar(case, zmin, zmax)` | CSAR vs time + final-timestep contact map |
| `compare_csar(cases, zmin, zmax)` | Single-region CSAR overlay for multiple cases |
| `plot_csar_multi_regions(case, z_bands, ...)` | Per-band + accumulated CSAR for one case |
| `compare_csar_accumulated(cases, z_bands, ...)` | Accumulated CSAR comparison across cases |

---

## VTP Export (ParaView)

```python
pvd_path = case.export_vtp()   # writes <label>_vtp/<label>.pvd
```

Open the `.pvd` file in ParaView to animate the contact pressure field over
time on the 3-D surface mesh.

---

## File Structure

```
xplt-parser/
├── xplt_core.py          # All parsing, analysis, and export logic
├── xplt_explorer.ipynb   # Interactive notebook — start here
├── requirements.txt
├── README.md
└── SETUP.md              # VS Code / environment setup notes
```

`xplt_core.py` is a self-contained single-module library. Import it directly:

```python
import xplt_core as xc
```

---

## Typical Workflow

```python
import xplt_core as xc

# ── 1. Define cases ───────────────────────────────────────────────────────────
CASES = [
    ("sim_a/run.feb", "sim_a/run.xplt", "Sim A"),
    ("sim_b/run.feb", "sim_b/run.xplt", "Sim B"),
]

Z_BANDS              = [(0.0, 25.0), (25.0, 50.0)]
BAND_LABELS          = ["distal", "proximal"]
TOTAL_AREA_OVERRIDES = [None, 88.5]   # None = use computed area

# ── 2. Load ───────────────────────────────────────────────────────────────────
cases = [xc.SimulationCase(feb, xplt, label=lbl) for feb, xplt, lbl in CASES]

# ── 3. Inspect ────────────────────────────────────────────────────────────────
for c in cases:
    print(c.summary())
    xc.plot_geometry(c)
    xc.plot_contact_overview(c)

# ── 4. CSAR ───────────────────────────────────────────────────────────────────
xc.compare_csar_accumulated(cases, Z_BANDS, BAND_LABELS, TOTAL_AREA_OVERRIDES)

df_csar = xc.region_accumulation_table(cases, Z_BANDS, TOTAL_AREA_OVERRIDES)
df_csar.to_csv("results_csar.csv", index=False)

# ── 5. Surrogate export ───────────────────────────────────────────────────────
# insertion_depth is computed automatically from each .feb file
for c in cases:
    df_sl = c.df_surrogate()
    df_sl.to_csv(f"{c.label}_surrogate.csv", index=False)
    print(f"Saved {c.label}_surrogate.csv  "
          f"({len(df_sl):,} rows, "
          f"depth {df_sl['insertion_depth'].min():.1f}–"
          f"{df_sl['insertion_depth'].max():.1f} mm)")

# Copy *_surrogate.csv files to surrogate-lab/data/simulations/ to train
```
