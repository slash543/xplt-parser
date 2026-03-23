# xplt-parser — FEBio Post-Processing Toolkit

Parse FEBio `.xplt` binary output and `.feb` mesh files, extract contact surface
pressure data, compute **Contact Surface Area Ratio (CSAR)**, and compare multiple
simulation cases.

---

## Requirements

```
Python >= 3.10
numpy, pandas, matplotlib
```

Install dependencies (virtualenv recommended):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

```python
import xplt_core as xc

# Load one simulation
case = xc.SimulationCase("my_run.feb", "my_run.xplt", label="Run A")
print(case.summary())

# Geometry overview
xc.plot_geometry(case)
xc.plot_contact_overview(case)

# Single-region CSAR (uses auto-detected tip Z range by default)
fig = xc.plot_csar(case, zmin=0.0, zmax=50.0)
```

---

## Core Class: `SimulationCase`

```python
case = xc.SimulationCase(
    feb_path             = "run.feb",
    xplt_path            = "run.xplt",
    label                = "Run A",              # display name
    contact_surface_name = "contact_slave",      # substring match; None = largest surface
    tip_material_names   = {"catheter_tip"},     # materials defining the tip region
    tip_z_cutoff         = 50.0,                 # fallback tip boundary [mm]
    cp_var_id            = 1,                    # xplt variable ID for contact pressure
)
```

**Key attributes after loading:**

| Attribute | Shape | Description |
|---|---|---|
| `case.timesteps` | `(T,)` float64 | Time values [s] |
| `case.cp_matrix` | `(T, N)` float32 | Contact pressure per facet per timestep [MPa] |
| `case.areas` | `(N,)` float64 | Facet areas [mm²] |
| `case.centroids` | `(N, 3)` float64 | Facet centroid XYZ [mm] |
| `case.df_facets` | DataFrame | Per-facet geometry + peak cp |

---

## Single-Region CSAR

```python
# Returns (timesteps, csar_array, total_region_area_mm2)
ts, csar, A = case.compute_csar(zmin=0.0, zmax=50.0)

# Two-panel plot: CSAR vs time + facet contact map
fig = xc.plot_csar(case, zmin=0.0, zmax=50.0)

# Compare multiple cases on one plot
fig = xc.compare_csar([case_a, case_b, case_c], zmin=0.0, zmax=50.0)

# Export numeric table
df = xc.csar_table([case_a, case_b], zmin=0.0, zmax=50.0)
df.to_csv("csar_comparison.csv", index=False)
```

---

## Multi-Region CSAR (Manual Area Adjustment)

Define two or more Z bands **before running the script**.  The function accumulates
facet counts and contact areas across all bands at every timestep, returning raw
numbers you can inspect and normalise manually.

### Step 1 — Define your bands

```python
# Coordinates must be set before running
Z_BANDS = [
    (0.0,  20.0),   # band 0: proximal
    (20.0, 40.0),   # band 1: mid
    (40.0, 60.0),   # band 2: distal
]
BAND_LABELS = ["proximal", "mid", "distal"]
```

### Step 2 — Compute raw accumulation

```python
ts, band_stats, accumulated = case.compute_region_accumulation(Z_BANDS)

# Each band_stats[i] is a dict:
# {
#   'zmin'               : float,
#   'zmax'               : float,
#   'n_facets_in_region' : int,
#   'total_area_mm2'     : float,
#   'contact_area_mm2'   : np.ndarray (n_timesteps,),   ← raw contact area
#   'n_contact_facets'   : np.ndarray (n_timesteps,),
# }
#
# accumulated = same keys (no zmin/zmax), union of all bands
```

### Step 3 — Adjust area and compute CSAR manually

```python
# Use the computed total area
csar = accumulated['contact_area_mm2'] / accumulated['total_area_mm2']

# OR override with your own reference area (e.g. from CAD)
my_ref_area = 95.0   # mm²
csar = accumulated['contact_area_mm2'] / my_ref_area
```

### Step 4 — Plot

```python
# Plots each band (dashed) + accumulated (solid black)
fig = xc.plot_csar_multi_regions(
    case,
    z_bands             = Z_BANDS,
    band_labels         = BAND_LABELS,
    total_area_override = 95.0,   # optional; omit to use computed area
)
```

### Step 5 — Compare multiple simulations

```python
# Same bands applied to all cases; per-case area override is optional
fig = xc.compare_csar_accumulated(
    cases                = [case_a, case_b, case_c],
    z_bands              = Z_BANDS,
    band_labels          = BAND_LABELS,
    total_area_overrides = [None, 95.0, None],   # only case_b uses override
)

# Numeric table
df = xc.region_accumulation_table(
    cases                = [case_a, case_b],
    z_bands              = Z_BANDS,
    total_area_overrides = [None, 95.0],
)
df.to_csv("csar_accumulated.csv", index=False)
```

---

## Plotting Functions Summary

| Function | Description |
|---|---|
| `plot_geometry(case)` | 2-D centroid projections, tip zone highlighted |
| `plot_contact_overview(case)` | 4-panel: max/mean cp, facet count, facet map, top-10 tip traces |
| `plot_csar(case, zmin, zmax)` | CSAR vs time + final-timestep contact map |
| `compare_csar(cases, zmin, zmax)` | Single-region CSAR overlay for multiple cases |
| `plot_csar_multi_regions(case, z_bands, ...)` | Per-band + accumulated CSAR for one case |
| `compare_csar_accumulated(cases, z_bands, ...)` | Accumulated CSAR comparison across cases |

All plotting functions accept `save=True` (default) to write a PNG, and return
the `matplotlib.Figure` for further customisation.

---

## VTP Export (ParaView)

```python
pvd_path = case.export_vtp()   # writes <label>_vtp/<label>.pvd
```

Open the `.pvd` file in ParaView to step through the contact pressure time series
on the 3-D surface mesh.

---

## File Structure

```
xplt_core.py          ← main module (import this)
xplt_explorer.ipynb   ← interactive notebook
requirements.txt
README.md
SETUP.md              ← VS Code / environment setup notes
```

---

## Typical Workflow

```python
import xplt_core as xc

# ── 1. Configuration ──────────────────────────────────────────────────────────
CASES = [
    ("sim_a/run.feb", "sim_a/run.xplt", "Sim A"),
    ("sim_b/run.feb", "sim_b/run.xplt", "Sim B"),
]

# Multi-region bands — set coordinates here before running
Z_BANDS = [(0.0, 25.0), (25.0, 50.0)]
BAND_LABELS = ["distal", "proximal"]
TOTAL_AREA_OVERRIDES = [None, 88.5]   # None = use computed; float = manual override

# ── 2. Load cases ─────────────────────────────────────────────────────────────
cases = [xc.SimulationCase(feb, xplt, label=lbl) for feb, xplt, lbl in CASES]

# ── 3. Inspect ────────────────────────────────────────────────────────────────
for c in cases:
    print(c.summary())
    xc.plot_geometry(c)
    xc.plot_contact_overview(c)

# ── 4. Multi-region CSAR ──────────────────────────────────────────────────────
for c in cases:
    xc.plot_csar_multi_regions(c, Z_BANDS, BAND_LABELS)

xc.compare_csar_accumulated(cases, Z_BANDS, BAND_LABELS, TOTAL_AREA_OVERRIDES)

df = xc.region_accumulation_table(cases, Z_BANDS, TOTAL_AREA_OVERRIDES)
df.to_csv("results.csv", index=False)
```
