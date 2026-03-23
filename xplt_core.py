"""
xplt_core.py  —  FEBio xplt/feb post-processing toolkit
=========================================================

Main entry point is SimulationCase.  Everything else is helpers.

Quickstart
----------
    import xplt_core as xc

    case_a = xc.SimulationCase("run_a.feb", "run_a.xplt", label="Run A")
    case_b = xc.SimulationCase("run_b.feb", "run_b.xplt", label="Run B")

    print(case_a.summary())
    xc.plot_geometry(case_a)
    xc.plot_contact_overview(case_a)
    fig = xc.plot_csar(case_a, zmin=0.0, zmax=50.0)
    fig = xc.compare_csar([case_a, case_b], zmin=0.0, zmax=50.0)
    case_a.export_vtp()

Multi-region CSAR (manual area adjustment)
------------------------------------------
Define several Z bands upfront to accumulate facet counts and contact areas
across all bands at each timestep.  The raw contact-area time series lets you
override the denominator before plotting.

    z_bands = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    band_labels = ["proximal", "mid", "distal"]

    # Raw numbers — inspect or adjust before plotting
    ts, band_stats, accumulated = case_a.compute_region_accumulation(z_bands)

    # Plot all bands + accumulated CSAR; override denominator if needed
    fig = xc.plot_csar_multi_regions(case_a, z_bands, band_labels,
                                     total_area_override=120.0)

    # Compare accumulated CSAR across cases
    fig = xc.compare_csar_accumulated([case_a, case_b], z_bands,
                                      total_area_overrides=[None, 120.0])

Adding a new analysis function
-------------------------------
    def my_metric(case: xc.SimulationCase, **kwargs):
        # case.cp_matrix  : shape (n_timesteps, n_facets), float32
        # case.areas       : shape (n_facets,),             float64  [mm²]
        # case.centroids   : shape (n_facets, 3),           float64  [mm]
        # case.timesteps   : shape (n_timesteps,),          float64  [s]
        ...
"""

from __future__ import annotations

import base64
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  FEBio xplt v53 — block tag constants
# ─────────────────────────────────────────────────────────────────────────────

_TAG: Dict[str, int] = dict(
    ROOT           = 0x01000000,
    MESH           = 0x01040000,
    NODE_SECTION   = 0x01041000,
    NODE_COORDS    = 0x01041200,
    DOMAIN_SECTION = 0x01042000,
    DOMAIN         = 0x01042100,
    DOMAIN_HEADER  = 0x01042101,
    DOM_ELEM_TYPE  = 0x01042102,
    DOM_PART_ID    = 0x01042103,
    DOM_N_ELEMS    = 0x01032104,   # note: 0x0103 prefix is what FEBio actually uses
    DOM_NAME       = 0x01032105,
    DOM_ELEM_LIST  = 0x01042200,
    ELEMENT        = 0x01042201,
    SURF_SECTION   = 0x01043000,
    SURFACE        = 0x01043100,
    SURF_HEADER    = 0x01043101,
    SURF_ID        = 0x01043102,
    SURF_N_FACETS  = 0x01043103,
    SURF_NAME      = 0x01043104,
    FACET_LIST     = 0x01043200,
    FACET          = 0x01043201,
    STATE          = 0x02000000,
    STATE_HEADER   = 0x02010000,
    STATE_TIME     = 0x02010002,
    STATE_DATA     = 0x02020000,
    STATE_VARIABLE = 0x02020001,
    STATE_VAR_ID   = 0x02020002,
    STATE_VAR_DATA = 0x02020003,
    SURFACE_DATA   = 0x02020500,
)

# Variable IDs in STATE surface data (FEBio v3 defaults)
VAR_CONTACT_PRESSURE = 1   # contact pressure [MPa]
VAR_VECTOR_GAP       = 2   # vector gap
VAR_SURFACE_TRACTION = 3   # surface traction
VAR_SLIP_TRACTION    = 4   # slip traction


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level binary helpers  (all private)
# ─────────────────────────────────────────────────────────────────────────────

def _iter_blocks(data: bytes):
    """Iterate (tag, size, chunk) tuples over a TLV (tag-length-value) stream."""
    pos = 0
    while pos + 8 <= len(data):
        tag  = struct.unpack_from('<I', data, pos)[0]
        size = struct.unpack_from('<I', data, pos + 4)[0]
        pos += 8
        if pos + size > len(data):
            break
        yield tag, size, data[pos : pos + size]
        pos += size


def _find_block(data: bytes, target_tag: int) -> Optional[bytes]:
    for tag, _, chunk in _iter_blocks(data):
        if tag == target_tag:
            return chunk
    return None


def _parse_name(c3: bytes) -> str:
    """
    Decode a name field from xplt.  Two formats appear in the wild:
      • length-prefixed: [uint32 len][chars len]
      • null-padded:     [chars][\\x00 ...]
    """
    if len(c3) >= 4:
        length = struct.unpack_from('<I', c3)[0]
        if 0 < length <= len(c3) - 4:
            candidate = c3[4 : 4 + length].decode('latin-1', errors='replace').strip()
            if candidate and all(ch.isprintable() or ch in '\t\n\r' for ch in candidate):
                return candidate
    return c3.rstrip(b'\x00').decode('latin-1', errors='replace').strip()


def _find_surface_data(flat_arr: np.ndarray, target_id: int) -> Optional[np.ndarray]:
    """
    Dynamically scan the flat float32 array from STATE_VAR_DATA to locate the
    sub-array belonging to surface `target_id`.

    Layout in the flat array:
      [surf_id_bits, byte_count_bits, val_0, val_1, ..., val_n, ...]

    IMPORTANT: surf_id and byte_count are uint32 integers stored as raw float32 bit
    patterns (i.e. reinterpret_cast<float>(uint32_value)).  The value 3 becomes
    the denormalised float 4.2e-45, NOT 3.0.  We must reinterpret the bits back
    to uint32 before comparing.
    """
    # View the float32 array as uint32 bytes for header parsing
    raw_u32 = flat_arr.view(np.uint32)  # same buffer, same length, read as uint32
    p = 0
    while p + 1 < len(raw_u32):
        sid        = int(raw_u32[p])
        byte_count = int(raw_u32[p + 1])
        n_vals     = byte_count // 4
        p += 2
        if sid == target_id:
            return flat_arr[p : p + n_vals]
        if n_vals <= 0:
            break
        p += n_vals
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  FEB file parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_feb(feb_path: Path) -> Tuple[Dict[int, str], Dict[str, str]]:
    """
    Returns:
        mat_id_to_name : {material_id (int) → material_name (str)}
        part_to_mat    : {part_name (str)   → material_name (str)}
    """
    root = ET.parse(feb_path).getroot()
    mat_id_to_name: Dict[int, str] = {}
    for m in root.findall('./Material/material'):
        mat_id_to_name[int(m.get('id'))] = m.get('name', '')

    part_to_mat: Dict[str, str] = {}
    for d in root.findall('./MeshDomains/SolidDomain'):
        part_to_mat[d.get('name', '')] = d.get('mat', '')

    return mat_id_to_name, part_to_mat


# ─────────────────────────────────────────────────────────────────────────────
#  XPLT binary reader  (private class — use SimulationCase instead)
# ─────────────────────────────────────────────────────────────────────────────

class _XpltReader:
    """
    Reads and parses a FEBio .xplt v53 binary file.

    The file is scanned once at construction time to locate the MESH block and
    all STATE block positions.  Mesh parsing and state iteration are done on
    demand by the caller (SimulationCase).
    """

    def __init__(self, path: Path):
        with open(path, 'rb') as f:
            self._raw = f.read()

        self._top:    Dict[int, Tuple[int, int]] = {}   # tag → (byte_offset, byte_size)
        self._states: List[Tuple[int, int]]       = []   # [(byte_offset, byte_size), ...]
        self._scan_top_level()

        pos, size = self._top[_TAG['MESH']]
        self._mesh = self._raw[pos + 8 : pos + 8 + size]

    # ── File scan ─────────────────────────────────────────────────────────────

    def _scan_top_level(self):
        # The first 4 bytes are an xplt file signature; top-level TLV blocks
        # begin at offset 4.
        pos = 4
        while pos + 8 <= len(self._raw):
            tag  = struct.unpack_from('<I', self._raw, pos)[0]
            size = struct.unpack_from('<I', self._raw, pos + 4)[0]
            if tag == _TAG['STATE']:
                self._states.append((pos, size))
            else:
                self._top[tag] = (pos, size)
            pos += 8 + size

    # ── Mesh ──────────────────────────────────────────────────────────────────

    def nodes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (global_node_ids: uint32, coords: float64 [N,3])."""
        ns  = _find_block(self._mesh, _TAG['NODE_SECTION'])
        raw = _find_block(ns, _TAG['NODE_COORDS'])
        # Layout per node: [global_id_as_f32_bits, x, y, z]  all float32
        arr = np.frombuffer(raw, dtype='<f4').reshape(-1, 4)
        ids = np.frombuffer(arr[:, 0].tobytes(), dtype='<u4')  # reinterpret f32→u32
        return ids, arr[:, 1:].astype(np.float64)

    def domains(self, mat_id_to_name: Dict[int, str]) -> List[dict]:
        """Returns list of domain dicts with fields: part_id, mat_name, n_elems, elements."""
        ds  = _find_block(self._mesh, _TAG['DOMAIN_SECTION'])
        out = []
        for tag, _, chunk in _iter_blocks(ds):
            if tag != _TAG['DOMAIN']:
                continue
            d: dict = {}
            for t2, _, c2 in _iter_blocks(chunk):
                if t2 == _TAG['DOMAIN_HEADER']:
                    for t3, _, c3 in _iter_blocks(c2):
                        if   t3 == _TAG['DOM_ELEM_TYPE']: d['elem_type'] = struct.unpack_from('<I', c3)[0]
                        elif t3 == _TAG['DOM_PART_ID']:   d['part_id']   = struct.unpack_from('<I', c3)[0]
                        elif t3 == _TAG['DOM_N_ELEMS']:   d['n_elems']   = struct.unpack_from('<I', c3)[0]
                        elif t3 == _TAG['DOM_NAME']:      d['name']      = _parse_name(c3)
                elif t2 == _TAG['DOM_ELEM_LIST']:
                    elems = [
                        struct.unpack_from(f'<{s3 // 4}I', c3)
                        for t3, s3, c3 in _iter_blocks(c2)
                        if t3 == _TAG['ELEMENT']
                    ]
                    d['elements'] = np.array(elems, dtype=np.int32)
            d['mat_name'] = mat_id_to_name.get(d.get('part_id', -1), 'unknown')
            out.append(d)
        return out

    def surfaces(self) -> Dict[int, dict]:
        """Returns {surf_id: surf_dict} with fields: id, name, n_facets, facets (int32 [N,n_nodes])."""
        ss  = _find_block(self._mesh, _TAG['SURF_SECTION'])
        out: Dict[int, dict] = {}
        for tag, _, chunk in _iter_blocks(ss):
            if tag != _TAG['SURFACE']:
                continue
            s: dict = {}
            for t2, _, c2 in _iter_blocks(chunk):
                if t2 == _TAG['SURF_HEADER']:
                    for t3, _, c3 in _iter_blocks(c2):
                        if   t3 == _TAG['SURF_ID']:       s['id']       = struct.unpack_from('<I', c3)[0]
                        elif t3 == _TAG['SURF_N_FACETS']: s['n_facets'] = struct.unpack_from('<I', c3)[0]
                        elif t3 == _TAG['SURF_NAME']:     s['name']     = _parse_name(c3)
                elif t2 == _TAG['FACET_LIST']:
                    facets = []
                    for t3, s3, c3 in _iter_blocks(c2):
                        if t3 == _TAG['FACET']:
                            v       = struct.unpack_from(f'<{s3 // 4}I', c3)
                            # v[0]=facet_id, v[1]=face_type, v[2:]=node_ids (0-based)
                            n_nodes = (s3 // 4) - 2
                            facets.append(v[2 : 2 + n_nodes])
                    # Homogeneous surfaces: all facets have the same node count
                    if facets and all(len(f) == len(facets[0]) for f in facets):
                        s['facets'] = np.array(facets, dtype=np.int32)
                    else:
                        s['facets'] = facets  # mixed — user must handle
            out[s['id']] = s
        return out

    # ── States ────────────────────────────────────────────────────────────────

    def n_states(self) -> int:
        return len(self._states)

    def _state_chunk(self, i: int) -> bytes:
        pos, size = self._states[i]
        return self._raw[pos + 8 : pos + 8 + size]

    def _timestep(self, sc: bytes) -> float:
        h = _find_block(sc, _TAG['STATE_HEADER'])
        return float(struct.unpack_from('<f', _find_block(h, _TAG['STATE_TIME']))[0])

    def _surface_var_flat(self, sc: bytes, var_id: int) -> Optional[np.ndarray]:
        """Return the flat float32 array for the given surface variable."""
        sd = _find_block(sc, _TAG['STATE_DATA'])
        if sd is None:
            return None
        svd = _find_block(sd, _TAG['SURFACE_DATA'])
        if svd is None:
            return None
        for tag, _, chunk in _iter_blocks(svd):
            if tag != _TAG['STATE_VARIABLE']:
                continue
            vid = arr = None
            for t2, _, c2 in _iter_blocks(chunk):
                if   t2 == _TAG['STATE_VAR_ID']:   vid = struct.unpack_from('<I', c2)[0]
                elif t2 == _TAG['STATE_VAR_DATA']: arr = np.frombuffer(c2, dtype='<f4')
            if vid == var_id:
                return arr
        return None

    def parse_states(
        self, surface_id: int, n_facets: int, var_id: int = VAR_CONTACT_PRESSURE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iterate all STATE blocks and extract the per-facet variable for one surface.

        The surface data offset is determined dynamically from the embedded
        [surf_id, byte_count] headers — no hardcoded contact ordering required.

        Returns
        -------
        timesteps  : float64 (n_states,)
        cp_matrix  : float32 (n_states, n_facets)
        """
        timesteps: List[float]      = []
        rows:      List[np.ndarray] = []
        zeros = np.zeros(n_facets, dtype='f4')

        for i in range(self.n_states()):
            sc = self._state_chunk(i)
            timesteps.append(self._timestep(sc))
            flat = self._surface_var_flat(sc, var_id)
            if flat is not None:
                surf_data = _find_surface_data(flat, surface_id)
                if surf_data is not None and len(surf_data) == n_facets:
                    rows.append(surf_data.copy())
                    continue
            rows.append(zeros.copy())

        return (
            np.array(timesteps, dtype=np.float64),
            np.stack(rows),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def _facet_geometry(
    facets: np.ndarray, coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute centroids and areas for an array of tri3 facets.

    Parameters
    ----------
    facets : int (N, 3)   — 0-based node indices into coords
    coords : float (M, 3) — node XYZ coordinates

    Returns
    -------
    centroids : float64 (N, 3)
    areas     : float64 (N,)   in the same units² as coords
    """
    f_coords  = coords[facets]                            # (N, 3, 3)
    centroids = f_coords.mean(axis=1)                     # (N, 3)
    AB        = f_coords[:, 1, :] - f_coords[:, 0, :]
    AC        = f_coords[:, 2, :] - f_coords[:, 0, :]
    areas     = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=1)
    return centroids, areas


# ─────────────────────────────────────────────────────────────────────────────
#  VTP / PVD export helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b64(arr: np.ndarray) -> str:
    """Encode numpy array as VTK binary base64: [uint32 nbytes][data bytes]."""
    data = arr.tobytes()
    return base64.b64encode(struct.pack('<I', len(data)) + data).decode('ascii')


def _write_vtp(
    filepath: Path,
    local_coords:  np.ndarray,
    connectivity:  np.ndarray,
    offsets:       np.ndarray,
    face_ids:      np.ndarray,
    areas:         np.ndarray,
    cp_arr:        np.ndarray,
):
    n_pts    = len(local_coords)
    n_facets = len(face_ids)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt32">',
        '  <PolyData>',
        f'  <Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" NumberOfLines="0"',
        f'         NumberOfStrips="0" NumberOfPolys="{n_facets}">',
        '    <Points>',
        '      <DataArray type="Float32" NumberOfComponents="3" format="binary">',
        f'        {_b64(local_coords.astype(np.float32))}',
        '      </DataArray>',
        '    </Points>',
        '    <Polys>',
        '      <DataArray type="Int32" Name="connectivity" format="binary">',
        f'        {_b64(connectivity.astype(np.int32))}',
        '      </DataArray>',
        '      <DataArray type="Int32" Name="offsets" format="binary">',
        f'        {_b64(offsets.astype(np.int32))}',
        '      </DataArray>',
        '    </Polys>',
        '    <CellData Scalars="contact_pressure_MPa">',
        '      <DataArray type="Int32" Name="face_id" format="binary">',
        f'        {_b64(face_ids.astype(np.int32))}',
        '      </DataArray>',
        '      <DataArray type="Float32" Name="area_mm2" format="binary">',
        f'        {_b64(areas.astype(np.float32))}',
        '      </DataArray>',
        '      <DataArray type="Float32" Name="contact_pressure_MPa" format="binary">',
        f'        {_b64(cp_arr.astype(np.float32))}',
        '      </DataArray>',
        '    </CellData>',
        '  </Piece>',
        '  </PolyData>',
        '</VTKFile>',
    ]
    filepath.write_text('\n'.join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  SimulationCase  —  main user-facing class
# ─────────────────────────────────────────────────────────────────────────────

class SimulationCase:
    """
    Encapsulates one FEBio simulation: parses the .feb and .xplt files,
    extracts contact surface geometry and pressure time-series.

    Parameters
    ----------
    feb_path              : path to the .feb file
    xplt_path             : path to the .xplt file
    label                 : display name (defaults to .feb stem)
    contact_surface_name  : substring to match against xplt surface names.
                            If None, the surface with the most facets is used.
    tip_material_names    : material names defining the 'tip' region.
                            Used to derive tip_z_min / tip_z_max automatically.
    tip_z_cutoff          : fallback tip cutoff [mm] when tip materials are absent.
    cp_var_id             : variable ID for contact pressure in xplt STATE blocks
                            (default = 1 for FEBio v3 contact pressure).
    """

    def __init__(
        self,
        feb_path:             str | Path,
        xplt_path:            str | Path,
        label:                Optional[str] = None,
        contact_surface_name: Optional[str] = None,
        tip_material_names:   Optional[set]  = None,
        tip_z_cutoff:         float          = 50.0,
        cp_var_id:            int            = VAR_CONTACT_PRESSURE,
    ):
        self.feb_path             = Path(feb_path)
        self.xplt_path            = Path(xplt_path)
        self.label                = label or self.feb_path.stem
        self.tip_z_cutoff         = tip_z_cutoff
        self.cp_var_id            = cp_var_id
        self._contact_surface_name = contact_surface_name
        self._tip_material_names   = tip_material_names or {'catheter_tip'}
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        print(f'[{self.label}] Parsing {self.feb_path.name} …')
        self._mat_id_to_name, self._part_to_mat = _parse_feb(self.feb_path)

        print(f'[{self.label}] Reading {self.xplt_path.name} …')
        reader = _XpltReader(self.xplt_path)

        self._node_ids, self.coords = reader.nodes()
        self._domains  = reader.domains(self._mat_id_to_name)
        self._surfaces = reader.surfaces()

        surf = self._pick_contact_surface()
        self._contact_surf_id   = surf['id']
        self._contact_surf_name = surf['name']
        self.facets             = surf['facets']          # (N_facets, 3)  int32
        self.n_facets           = len(self.facets)

        self.centroids, self.areas = _facet_geometry(self.facets, self.coords)
        self.tip_z_min, self.tip_z_max = self._compute_tip_z_range()

        print(f'[{self.label}] Parsing {reader.n_states()} timesteps …')
        self.timesteps, self.cp_matrix = reader.parse_states(
            self._contact_surf_id, self.n_facets, self.cp_var_id
        )
        self.n_timesteps = len(self.timesteps)
        self._df = self._build_dataframe()

        print(
            f'[{self.label}] Ready  |  '
            f'{self.n_facets} facets  |  '
            f'{self.n_timesteps} timesteps  |  '
            f't = [{self.timesteps[0]:.3f} … {self.timesteps[-1]:.3f}] s  |  '
            f'max cp = {self.cp_matrix.max():.4f} MPa'
        )

    def _pick_contact_surface(self) -> dict:
        if self._contact_surface_name:
            needle = self._contact_surface_name.lower()
            for s in self._surfaces.values():
                if needle in s.get('name', '').lower():
                    return s
            warnings.warn(
                f"[{self.label}] No surface matching '{self._contact_surface_name}'; "
                "falling back to the surface with the most facets."
            )
        # Fallback: surface with most facets (usually the primary contact surface)
        return max(self._surfaces.values(), key=lambda s: s.get('n_facets', 0))

    def _compute_tip_z_range(self) -> Tuple[float, float]:
        tip_domains = [
            d for d in self._domains
            if d.get('mat_name') in self._tip_material_names
        ]
        if tip_domains:
            all_node_idx = np.unique(
                np.concatenate([d['elements'][:, 1:].ravel() for d in tip_domains])
            )
            z = self.coords[all_node_idx, 2]
            return float(z.min()), float(z.max())
        warnings.warn(
            f"[{self.label}] Tip material(s) {self._tip_material_names} not found. "
            f"Using z < {self.tip_z_cutoff} mm as the tip region."
        )
        return 0.0, self.tip_z_cutoff

    def _build_dataframe(self) -> pd.DataFrame:
        tip_mask = (
            (self.centroids[:, 2] >= self.tip_z_min) &
            (self.centroids[:, 2] <= self.tip_z_max)
        )
        return pd.DataFrame({
            'face_id'     : np.arange(1, self.n_facets + 1),
            'cx_mm'       : self.centroids[:, 0],
            'cy_mm'       : self.centroids[:, 1],
            'cz_mm'       : self.centroids[:, 2],
            'area_mm2'    : self.areas,
            'in_tip_zone' : tip_mask,
            'cp_peak_MPa' : self.cp_matrix.max(axis=0),
        })

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def df_facets(self) -> pd.DataFrame:
        """Per-facet geometry + peak contact pressure."""
        return self._df

    @property
    def total_area_mm2(self) -> float:
        return float(self.areas.sum())

    @property
    def materials(self) -> Dict[int, str]:
        return dict(self._mat_id_to_name)

    @property
    def surface_names(self) -> Dict[int, str]:
        return {sid: s['name'] for sid, s in self._surfaces.items()}

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        mats = ', '.join(f'"{v}"' for v in sorted(self._mat_id_to_name.values()))
        lines = [
            f"SimulationCase: {self.label}",
            f"  FEB  : {self.feb_path}",
            f"  XPLT : {self.xplt_path}",
            f"  Contact surface  : '{self._contact_surf_name}' (id={self._contact_surf_id})",
            f"  Facets           : {self.n_facets}",
            f"  Total area       : {self.total_area_mm2:.2f} mm²",
            f"  Tip Z range      : [{self.tip_z_min:.2f}, {self.tip_z_max:.2f}] mm",
            f"  Timesteps        : {self.n_timesteps}  "
            f"({self.timesteps[0]:.3f} → {self.timesteps[-1]:.3f} s)",
            f"  Max cp overall   : {self.cp_matrix.max():.4f} MPa",
            f"  Materials        : {mats}",
        ]
        return '\n'.join(lines)

    # ── CSAR analysis ─────────────────────────────────────────────────────────

    def compute_csar(
        self,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute Contact Surface Area Ratio over time.

        CSAR(t) = Σ A_i  [facets i in region AND cp_i(t) > 0]
                  ──────────────────────────────────────────────
                  Σ A_i  [all facets i in region]

        Parameters
        ----------
        zmin : lower Z bound [mm]; defaults to tip_z_min
        zmax : upper Z bound [mm]; defaults to tip_z_max

        Returns
        -------
        timesteps         : float64 (n_timesteps,)
        csar              : float64 (n_timesteps,)  in [0, 1]
        total_region_area : float  [mm²]
        """
        if zmin is None:
            zmin = self.tip_z_min
        if zmax is None:
            zmax = self.tip_z_max

        region_mask    = (self.centroids[:, 2] >= zmin) & (self.centroids[:, 2] <= zmax)
        A_region       = self.areas[region_mask]
        A_total_region = float(A_region.sum())

        cp_region  = self.cp_matrix[:, region_mask]       # (n_timesteps, n_region)
        A_contact  = (cp_region > 0).astype(np.float64) @ A_region  # dot product per timestep
        csar       = np.where(A_total_region > 0, A_contact / A_total_region, 0.0)

        return self.timesteps, csar, A_total_region

    def compute_region_accumulation(
        self,
        z_bands: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, List[dict], dict]:
        """
        Compute per-timestep facet counts and contact areas for multiple Z bands,
        plus an accumulated total across all bands (union — no double-counting).

        Define the bands upfront before running the script; this gives you the raw
        numbers so you can override the denominator manually when computing CSAR.

        Parameters
        ----------
        z_bands : list of (zmin, zmax) tuples defining each Z region [mm]

        Returns
        -------
        timesteps  : float64 (n_timesteps,)
        band_stats : list of dicts, one per band:
                     {
                       'zmin'               : float,
                       'zmax'               : float,
                       'n_facets_in_region' : int,
                       'total_area_mm2'     : float,
                       'contact_area_mm2'   : float64 (n_timesteps,),
                       'n_contact_facets'   : int     (n_timesteps,),
                     }
        accumulated : dict with the same keys (except zmin/zmax) for the union
                      of all bands combined.

        Usage example
        -------------
            ts, bands, acc = case.compute_region_accumulation(
                [(0.0, 20.0), (20.0, 40.0)]
            )
            # Manual CSAR with a custom denominator
            my_total_area = 95.0   # e.g. from CAD or another reference
            csar = acc['contact_area_mm2'] / my_total_area
        """
        band_stats: List[dict] = []
        union_mask = np.zeros(self.n_facets, dtype=bool)

        for zmin, zmax in z_bands:
            mask = (self.centroids[:, 2] >= zmin) & (self.centroids[:, 2] <= zmax)
            union_mask |= mask

            A_region    = self.areas[mask]
            cp_region   = self.cp_matrix[:, mask]
            in_contact  = cp_region > 0
            band_stats.append({
                'zmin'               : float(zmin),
                'zmax'               : float(zmax),
                'n_facets_in_region' : int(mask.sum()),
                'total_area_mm2'     : float(A_region.sum()),
                'contact_area_mm2'   : (in_contact.astype(np.float64) @ A_region),
                'n_contact_facets'   : in_contact.sum(axis=1).astype(int),
            })

        A_acc          = self.areas[union_mask]
        cp_acc         = self.cp_matrix[:, union_mask]
        in_contact_acc = cp_acc > 0
        accumulated = {
            'n_facets_in_region' : int(union_mask.sum()),
            'total_area_mm2'     : float(A_acc.sum()),
            'contact_area_mm2'   : (in_contact_acc.astype(np.float64) @ A_acc),
            'n_contact_facets'   : in_contact_acc.sum(axis=1).astype(int),
        }

        return self.timesteps, band_stats, accumulated

    # ── VTP export ────────────────────────────────────────────────────────────

    def export_vtp(self, output_dir: Optional[Path] = None) -> Path:
        """
        Write a VTP+PVD time series for ParaView visualisation.

        Files are written to: output_dir / {label}_vtp/
        Returns the path to the .pvd collection file.
        """
        if output_dir is None:
            output_dir = Path('.')
        vtp_dir = output_dir / f'{self.label}_vtp'
        vtp_dir.mkdir(parents=True, exist_ok=True)

        # Build local (surface-only) coordinate system for the VTP mesh
        unique_idx, inv   = np.unique(self.facets.ravel(), return_inverse=True)
        local_coords      = self.coords[unique_idx]
        local_facets      = inv.reshape(-1, 3)
        connectivity      = local_facets.ravel()
        offsets           = np.arange(3, self.n_facets * 3 + 1, 3, dtype=np.int32)
        face_ids          = np.arange(1, self.n_facets + 1, dtype=np.int32)

        vtp_entries: List[Tuple[float, str]] = []
        for i, (t, cp_row) in enumerate(zip(self.timesteps, self.cp_matrix)):
            fname = f'{self.label}_t{i:04d}.vtp'
            _write_vtp(vtp_dir / fname, local_coords, connectivity, offsets,
                       face_ids, self.areas, cp_row)
            vtp_entries.append((t, fname))
            if i % 30 == 0:
                print(f'  [{self.label}] VTP {i + 1}/{self.n_timesteps}: {fname}')

        pvd_lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            '  <Collection>',
        ]
        for t, fname in vtp_entries:
            pvd_lines.append(
                f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{fname}"/>'
            )
        pvd_lines += ['  </Collection>', '</VTKFile>']

        pvd_path = vtp_dir / f'{self.label}.pvd'
        pvd_path.write_text('\n'.join(pvd_lines))
        print(f'[{self.label}] VTP export complete → {pvd_path}')
        return pvd_path


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting functions  (all return the figure for further customisation)
# ─────────────────────────────────────────────────────────────────────────────

def plot_geometry(case: SimulationCase, save: bool = True) -> plt.Figure:
    """
    Three 2-D projections of facet centroids coloured by region.
    Body facets = blue, tip zone facets = orange.
    """
    c   = case.centroids
    tip = case.df_facets['in_tip_zone'].values
    cx, cy, cz = c[:, 0], c[:, 1], c[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'{case.label} — Facet Centroids\n'
        f'orange = tip zone  z ∈ [{case.tip_z_min:.1f}, {case.tip_z_max:.1f}] mm',
        fontsize=12,
    )
    for ax, (hx, hy, xl, yl) in zip(axes, [
        (cz, cy, 'Z (mm)', 'Y (mm)'),
        (cz, cx, 'Z (mm)', 'X (mm)'),
        (cx, cy, 'X (mm)', 'Y (mm)'),
    ]):
        ax.scatter(hx[~tip], hy[~tip], s=1, color='steelblue',  label='body')
        ax.scatter(hx[ tip], hy[ tip], s=2, color='darkorange', label='tip zone')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.3)

    for ax in axes[:2]:
        ax.axvline(case.tip_z_min, color='red', lw=0.8, ls='--')
        ax.axvline(case.tip_z_max, color='red', lw=0.8, ls='--')
    axes[0].legend(fontsize=8, markerscale=4)

    plt.tight_layout()
    if save:
        p = Path(f'{case.label}_geometry.png')
        fig.savefig(p, dpi=150)
        print(f'Saved: {p}')
    return fig


def plot_contact_overview(case: SimulationCase, save: bool = True) -> plt.Figure:
    """
    4-panel overview:
      • max / mean cp vs time
      • number of facets in contact vs time
      • facet map coloured by peak cp
      • top-10 tip-zone facet traces
    """
    ts  = case.timesteps
    cp  = case.cp_matrix
    df  = case.df_facets
    tip = df['in_tip_zone'].values

    cp_max      = cp.max(axis=1)
    cp_mean_nz  = np.array([r[r > 0].mean() if (r > 0).any() else 0.0 for r in cp])
    n_contact   = (cp > 0).sum(axis=1)
    cp_tip_max  = cp[:, tip].max(axis=1) if tip.any() else np.zeros_like(cp_max)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'{case.label} — Contact Pressure Overview', fontsize=13)

    # Top-left: max / mean cp
    ax = axes[0, 0]
    ax.plot(ts, cp_max,     color='crimson',    lw=1.5, label='Max (all)')
    ax.plot(ts, cp_mean_nz, color='steelblue',  lw=1.5, ls='--', label='Mean (in-contact)')
    ax.plot(ts, cp_tip_max, color='darkorange', lw=1.2, ls=':',  label='Max (tip zone)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Contact Pressure (MPa)')
    ax.set_title('Max / Mean Contact Pressure')
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    # Top-right: facets in contact
    ax = axes[0, 1]
    ax.fill_between(ts, n_contact, alpha=0.4, color='teal')
    ax.plot(ts, n_contact, color='teal', lw=1.5)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Count')
    ax.set_title('Facets in Contact (cp > 0)')
    ax.grid(alpha=0.4)

    # Bottom-left: facet map coloured by peak cp
    ax = axes[1, 0]
    vmax = np.percentile(df['cp_peak_MPa'], 99) or df['cp_peak_MPa'].max() or 1.0
    sc = ax.scatter(df['cz_mm'], df['cy_mm'],
                    c=df['cp_peak_MPa'], cmap='hot_r', s=1, vmin=0, vmax=vmax)
    plt.colorbar(sc, ax=ax, label='Peak cp (MPa)')
    ax.axvline(case.tip_z_min, color='cyan', lw=0.8, ls='--', label='tip zone')
    ax.axvline(case.tip_z_max, color='cyan', lw=0.8, ls='--')
    ax.set_xlabel('Z (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title('Facet Map — Peak cp')
    ax.legend(fontsize=8)

    # Bottom-right: top-10 tip-zone facet traces
    ax = axes[1, 1]
    top10 = df[df['in_tip_zone']].nlargest(10, 'cp_peak_MPa')
    colors10 = cm.get_cmap('tab10', 10)
    for k, (_, row) in enumerate(top10.iterrows()):
        idx = int(row['face_id']) - 1  # 0-based
        ax.plot(ts, cp[:, idx], color=colors10(k), lw=0.8,
                label=f'f{int(row["face_id"])}  z={row["cz_mm"]:.1f}')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Contact Pressure (MPa)')
    ax.set_title('Top-10 Tip-Zone Facets — cp vs Time')
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.4)

    plt.tight_layout()
    if save:
        p = Path(f'{case.label}_contact_overview.png')
        fig.savefig(p, dpi=150)
        print(f'Saved: {p}')
    return fig


def plot_csar(
    case: SimulationCase,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    save: bool = True,
) -> plt.Figure:
    """
    Two-panel CSAR plot:
      • CSAR vs time
      • facet contact state map at the final timestep
    """
    if zmin is None:
        zmin = case.tip_z_min
    if zmax is None:
        zmax = case.tip_z_max

    ts, csar, A_region = case.compute_csar(zmin, zmax)
    region_mask = (case.centroids[:, 2] >= zmin) & (case.centroids[:, 2] <= zmax)
    cp_final    = case.cp_matrix[-1]
    in_contact  = (cp_final > 0) & region_mask
    no_contact  = (cp_final == 0) & region_mask
    outside     = ~region_mask

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'{case.label} — Contact Surface Area Ratio\n'
        f'z ∈ [{zmin:.1f}, {zmax:.1f}] mm  |  '
        f'{region_mask.sum()} facets  |  '
        f'A_region = {A_region:.2f} mm²',
        fontsize=12,
    )

    ax = axes[0]
    ax.plot(ts, csar, color='mediumseagreen', lw=1.5)
    ax.fill_between(ts, csar, alpha=0.25, color='mediumseagreen')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('CSAR  (A_contact / A_region)')
    ax.set_title('Contact Surface Area Ratio vs Time')
    ax.set_ylim(0, max(1.0, csar.max() * 1.1))
    ax.grid(alpha=0.4)

    ax = axes[1]
    c = case.centroids
    ax.scatter(c[outside,   2], c[outside,   1], s=1, color='lightgrey', label='outside window')
    ax.scatter(c[no_contact,2], c[no_contact,1], s=2, color='steelblue', label='no contact')
    ax.scatter(c[in_contact,2], c[in_contact,1], s=4, color='crimson',   label='in contact')
    ax.axvline(zmin, color='black', lw=0.8, ls='--', label=f'z={zmin:.1f}')
    ax.axvline(zmax, color='black', lw=0.8, ls=':',  label=f'z={zmax:.1f}')
    ax.set_xlabel('Z (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title(f'Facet State at t={ts[-1]:.3f} s   (CSAR = {csar[-1]:.4f})')
    ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        p = Path(f'{case.label}_csar.png')
        fig.savefig(p, dpi=150)
        print(f'Saved: {p}')
    return fig


def compare_csar(
    cases:  List[SimulationCase],
    zmin:   Optional[float] = None,
    zmax:   Optional[float] = None,
    save:   bool = True,
) -> plt.Figure:
    """
    Overlay CSAR vs time for multiple simulation cases on one axes.
    zmin/zmax default to the tip Z range of the first case.
    """
    if not cases:
        raise ValueError("No cases provided.")
    if zmin is None:
        zmin = cases[0].tip_z_min
    if zmax is None:
        zmax = cases[0].tip_z_max

    fig, ax = plt.subplots(figsize=(10, 6))
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, case in enumerate(cases):
        ts, csar, A_reg = case.compute_csar(zmin, zmax)
        c = prop_cycle[i % len(prop_cycle)]
        ax.plot(ts, csar, color=c, lw=1.8, label=f'{case.label}   (A = {A_reg:.1f} mm²)')
        ax.fill_between(ts, csar, alpha=0.08, color=c)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CSAR  (A_contact / A_region)')
    ax.set_title(f'Contact Surface Area Ratio — Comparison\nz ∈ [{zmin:.1f}, {zmax:.1f}] mm')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10); ax.grid(alpha=0.4)

    plt.tight_layout()
    if save:
        p = Path('csar_comparison.png')
        fig.savefig(p, dpi=150)
        print(f'Saved: {p}')
    return fig


def csar_table(
    cases: List[SimulationCase],
    zmin:  Optional[float] = None,
    zmax:  Optional[float] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns [time, label_1, label_2, ...] for CSAR values.
    Useful for exporting numeric comparison results.
    """
    if not cases:
        raise ValueError("No cases provided.")
    if zmin is None:
        zmin = cases[0].tip_z_min
    if zmax is None:
        zmax = cases[0].tip_z_max

    # Use the first case's timesteps as the reference time axis
    ref_ts = cases[0].timesteps
    data   = {'time_s': ref_ts}
    for case in cases:
        ts, csar, _ = case.compute_csar(zmin, zmax)
        if len(ts) == len(ref_ts):
            data[case.label] = csar
        else:
            # Interpolate onto reference time axis if timestep counts differ
            data[case.label] = np.interp(ref_ts, ts, csar)

    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-region CSAR helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_csar_multi_regions(
    case:                SimulationCase,
    z_bands:             List[Tuple[float, float]],
    band_labels:         Optional[List[str]] = None,
    total_area_override: Optional[float] = None,
    save:                bool = True,
) -> plt.Figure:
    """
    Plot CSAR vs time for each Z band individually, plus the accumulated total.

    Parameters
    ----------
    case                : SimulationCase to analyse
    z_bands             : list of (zmin, zmax) region definitions [mm]
    band_labels         : display name for each band; auto-generated if None
    total_area_override : use this value [mm²] as the accumulated-CSAR denominator
                          instead of the sum of band areas.  Pass a float to
                          normalise against a reference area (e.g. from CAD).
    save                : write PNG to disk

    Returns
    -------
    matplotlib Figure
    """
    ts, band_stats, accumulated = case.compute_region_accumulation(z_bands)

    if band_labels is None:
        band_labels = [
            f'z=[{bs["zmin"]:.1f}, {bs["zmax"]:.1f}] mm  ({bs["n_facets_in_region"]} facets)'
            for bs in band_stats
        ]

    denom    = total_area_override if total_area_override is not None else accumulated['total_area_mm2']
    acc_csar = np.where(denom > 0, accumulated['contact_area_mm2'] / denom, 0.0)

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    denom_note = f'A_ref={denom:.1f} mm² (override)' if total_area_override is not None \
                 else f'A_acc={denom:.1f} mm²'
    fig.suptitle(
        f'{case.label} — Multi-Region CSAR  |  {len(z_bands)} bands  |  {denom_note}',
        fontsize=12,
    )

    # Left: CSAR per band + accumulated
    ax = axes[0]
    for i, (bs, lbl) in enumerate(zip(band_stats, band_labels)):
        band_denom = bs['total_area_mm2']
        band_csar  = np.where(band_denom > 0, bs['contact_area_mm2'] / band_denom, 0.0)
        c = prop_cycle[i % len(prop_cycle)]
        ax.plot(ts, band_csar, color=c, lw=1.2, ls='--',
                label=f'{lbl}  (A={band_denom:.1f} mm²)')
    ax.plot(ts, acc_csar, color='black', lw=2.0,
            label=f'Accumulated  (A={denom:.1f} mm²)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CSAR  (A_contact / A_region)')
    ax.set_title('CSAR per Band + Accumulated')
    ax.set_ylim(0, max(1.0, acc_csar.max() * 1.1))
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    # Right: facets in contact per band + accumulated
    ax = axes[1]
    for i, (bs, lbl) in enumerate(zip(band_stats, band_labels)):
        c = prop_cycle[i % len(prop_cycle)]
        ax.plot(ts, bs['n_contact_facets'], color=c, lw=1.2, ls='--', label=lbl)
    ax.plot(ts, accumulated['n_contact_facets'], color='black', lw=2.0, label='Accumulated')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Facets in contact (cp > 0)')
    ax.set_title('Facets in Contact per Band')
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    plt.tight_layout()
    if save:
        p = Path(f'{case.label}_csar_multi_regions.png')
        fig.savefig(p, dpi=150)
        print(f'Saved: {p}')
    return fig


def compare_csar_accumulated(
    cases:                List[SimulationCase],
    z_bands:              List[Tuple[float, float]],
    band_labels:          Optional[List[str]] = None,
    total_area_overrides: Optional[List[Optional[float]]] = None,
    save:                 bool = True,
) -> plt.Figure:
    """
    Compare the accumulated (multi-band union) CSAR across multiple simulation
    cases on a single axes.

    The same z_bands definition is applied to every case.  Per-case denominator
    overrides allow manual area adjustment for each simulation independently.

    Parameters
    ----------
    cases                : list of SimulationCase
    z_bands              : list of (zmin, zmax) band definitions applied to all cases
    band_labels          : optional names for each band (shown in subtitle only)
    total_area_overrides : per-case denominator override list, same length as cases.
                           Use None for a specific case to use its computed area.
                           Example: [None, 95.0, None]  — only case[1] is overridden.
    save                 : write PNG to disk

    Returns
    -------
    matplotlib Figure
    """
    if not cases:
        raise ValueError("No cases provided.")
    if total_area_overrides is None:
        total_area_overrides = [None] * len(cases)
    if len(total_area_overrides) != len(cases):
        raise ValueError("total_area_overrides must have the same length as cases.")

    if band_labels is None:
        band_labels = [f'z=[{z0:.1f},{z1:.1f}]' for z0, z1 in z_bands]

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax    = plt.subplots(figsize=(11, 6))
    ax.set_title(
        f'Accumulated CSAR Comparison — {len(z_bands)} bands:\n'
        + ',  '.join(band_labels),
        fontsize=11,
    )

    for i, (case, override) in enumerate(zip(cases, total_area_overrides)):
        ts, _, accumulated = case.compute_region_accumulation(z_bands)
        denom  = override if override is not None else accumulated['total_area_mm2']
        csar   = np.where(denom > 0, accumulated['contact_area_mm2'] / denom, 0.0)
        c      = prop_cycle[i % len(prop_cycle)]
        label  = (
            f'{case.label}  (A={denom:.1f} mm²'
            + (' — override' if override is not None else '') + ')'
        )
        ax.plot(ts, csar, color=c, lw=1.8, label=label)
        ax.fill_between(ts, csar, alpha=0.08, color=c)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CSAR  (A_contact / A_accumulated)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9); ax.grid(alpha=0.4)

    plt.tight_layout()
    if save:
        p = Path('csar_accumulated_comparison.png')
        fig.savefig(p, dpi=150)
        print(f'Saved: {p}')
    return fig


def region_accumulation_table(
    cases:                List[SimulationCase],
    z_bands:              List[Tuple[float, float]],
    total_area_overrides: Optional[List[Optional[float]]] = None,
) -> pd.DataFrame:
    """
    Return a wide DataFrame with per-case accumulated CSAR values over time.

    Columns: time_s, <case_label_1>, <case_label_2>, ...

    Parameters
    ----------
    cases                : list of SimulationCase
    z_bands              : band definitions applied to all cases
    total_area_overrides : per-case denominator override (None = use computed area)
    """
    if not cases:
        raise ValueError("No cases provided.")
    if total_area_overrides is None:
        total_area_overrides = [None] * len(cases)

    ref_ts = cases[0].timesteps
    data: dict = {'time_s': ref_ts}

    for case, override in zip(cases, total_area_overrides):
        ts, _, accumulated = case.compute_region_accumulation(z_bands)
        denom = override if override is not None else accumulated['total_area_mm2']
        csar  = np.where(denom > 0, accumulated['contact_area_mm2'] / denom, 0.0)
        if len(ts) != len(ref_ts):
            csar = np.interp(ref_ts, ts, csar)
        data[case.label] = csar

    return pd.DataFrame(data)
