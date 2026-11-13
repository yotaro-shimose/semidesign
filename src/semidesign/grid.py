"""Rectilinear grid for 2D/3D process and device simulation.

Each axis is cut by its own set of lines (cell edges), so cells can be small
where the structure needs it (a gate oxide, a junction, a gate edge) and large
elsewhere, the way a commercial process simulator starts from 'line'
statements and refinement boxes. A uniform grid is the special case with no
refinements.

Array layout: lateral axes first, depth last; 2D arrays are (nx, nz), 3D
arrays (nx, ny, nz). Depth z is 0 at the initial silicon surface and grows
downward; ``headroom_nm`` above the surface is kept for oxide and deposits.
All lengths are in nm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .fv import face_pairs, series_conductance

DEPTH = -1


@dataclass(frozen=True)
class Refinement:
    """Cells no larger than `spacing_nm` between lo_nm and hi_nm on one axis."""

    axis: int  # 0, 1 for lateral axes; -1 (DEPTH) for depth
    lo_nm: float
    hi_nm: float
    spacing_nm: float


def graded_edges(lo: float, hi: float, base: float, zones: list[Refinement],
                 growth: float) -> np.ndarray:
    """Cell edges from lo to hi: `base` spacing, at most `spacing_nm` inside the
    zones, and changing by at most a factor `growth` from one cell to the next."""

    def allowed(x: float) -> float:
        """Largest cell starting at x that is no larger than the local target anywhere."""
        s = float(base)
        for z in zones:
            if x < z.lo_nm:  # zone ahead: the cell must not end too big near it
                s = min(s, (z.spacing_nm + (growth - 1) * (z.lo_nm - x)) / growth)
            elif x < z.hi_nm:  # inside
                s = min(s, z.spacing_nm)
            else:  # zone behind: grow away from it
                s = min(s, z.spacing_nm + (growth - 1) * (x - z.hi_nm))
        return max(s, 1e-6)

    edges = [lo]
    x = lo
    while x < hi - 1e-9:
        x = min(x + allowed(x), hi)
        edges.append(x)
    edges = np.array(edges)
    w = np.diff(edges)
    if w.size > 2 and w[-1] < w[-2] / growth:
        # a sliver at the end: share the last few cells' length evenly among them
        k = min(6, w.size)
        w[-k:] = w[-k:].mean()
        edges = lo + np.concatenate([[0.0], np.cumsum(w)])
        edges[-1] = hi
    return edges


def _mirror(zones: list[Refinement]) -> list[Refinement]:
    return [Refinement(z.axis, -z.hi_nm, -z.lo_nm, z.spacing_nm) for z in zones]


class Grid:
    def __init__(self, lateral_nm: tuple[float, ...], depth_nm: float, spacing_nm: float = 10.0,
                 headroom_nm: float = 200.0, periodic: bool = False,
                 refine: tuple[Refinement, ...] = (), growth: float = 1.3):
        if not lateral_nm:
            raise ValueError("need at least one lateral axis (use 2D or 3D)")
        if headroom_nm < spacing_nm:
            raise ValueError("headroom_nm must be at least one grid spacing (the wafer needs a surface)")
        if growth < 1:
            raise ValueError("growth must be at least 1")
        self.lateral_nm = tuple(float(w) for w in lateral_nm)
        self.depth_nm = float(depth_nm)
        self.headroom_nm = float(headroom_nm)
        self.base_spacing_nm = float(spacing_nm)
        self.periodic = periodic
        self.refine = tuple(refine)
        self.growth = growth
        ndim = len(self.lateral_nm) + 1
        zones = lambda ax: [r for r in self.refine if r.axis % ndim == ax]
        lateral = [graded_edges(0.0, w, spacing_nm, zones(ax), growth)
                   for ax, w in enumerate(self.lateral_nm)]
        # the initial silicon surface (z = 0) is always a cell edge
        above = -graded_edges(0.0, self.headroom_nm, spacing_nm, _mirror(zones(ndim - 1)), growth)[::-1]
        below = graded_edges(0.0, self.depth_nm, spacing_nm, zones(ndim - 1), growth)
        self.edges = tuple(lateral) + (np.concatenate([above[:-1], below]),)
        self.shape = tuple(e.size - 1 for e in self.edges)

    # ------------------------------------------------------------------ axes

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def lateral_shape(self) -> tuple[int, ...]:
        return self.shape[:-1]

    @property
    def periodic_axes(self) -> tuple[int, ...]:
        return tuple(range(self.ndim - 1)) if self.periodic else ()

    def widths(self, axis: int) -> np.ndarray:
        return np.diff(self.edges[axis])

    def centers(self, axis: int) -> np.ndarray:
        e = self.edges[axis]
        return 0.5 * (e[:-1] + e[1:])

    def lateral_centers(self, axis: int) -> np.ndarray:
        return self.centers(axis)

    @property
    def z_edges(self) -> np.ndarray:
        return self.edges[DEPTH]

    @property
    def z_centers(self) -> np.ndarray:
        return self.centers(DEPTH)

    @property
    def is_uniform(self) -> bool:
        return all(np.allclose(self.widths(a), self.widths(a)[0]) for a in range(self.ndim)) and \
            len({round(float(self.widths(a)[0]), 9) for a in range(self.ndim)}) == 1

    @property
    def spacing_nm(self) -> float:
        """The cell size of a uniform grid."""
        if not self.is_uniform:
            raise ValueError("the grid is not uniform; use widths(axis)")
        return float(self.widths(0)[0])

    def lateral_index(self, pos_nm: float, axis: int = 0) -> int:
        i = int(np.searchsorted(self.edges[axis], pos_nm, side="right")) - 1
        return min(max(i, 0), self.shape[axis] - 1)

    # ----------------------------------------------------------- cells/faces

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    @cached_property
    def volumes(self) -> np.ndarray:
        """Cell sizes in nm^ndim (area in 2D, volume in 3D), grid-shaped."""
        v = np.ones(())
        for a in range(self.ndim):
            v = np.multiply.outer(v, self.widths(a))
        return v

    @property
    def volumes_cm(self) -> np.ndarray:
        return self.volumes * 1e-7**self.ndim

    @cached_property
    def faces(self) -> tuple[np.ndarray, np.ndarray]:
        """Flat indices (i, j) of all pairs of cells sharing a face."""
        i, j, _ = self._faces
        return i, j

    @cached_property
    def _faces(self):
        return face_pairs(self.shape, self.periodic_axes)

    @cached_property
    def face_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per face: half width of cell i and of cell j along the face normal, and face area."""
        i, j, axis = self._faces
        idx_i = np.unravel_index(i, self.shape)
        idx_j = np.unravel_index(j, self.shape)
        half_i = np.empty(i.size)
        half_j = np.empty(i.size)
        area = np.ones(i.size)
        for a in range(self.ndim):
            w = self.widths(a)
            along = axis == a
            half_i[along] = 0.5 * w[idx_i[a][along]]
            half_j[along] = 0.5 * w[idx_j[a][along]]
            area[~along] *= w[idx_i[a][~along]]
        return half_i, half_j, area

    def conductance(self, k_cell: np.ndarray) -> np.ndarray:
        """Face conductances for a per-cell coefficient (series of the two half cells)."""
        k = np.asarray(k_cell, dtype=float).ravel()
        i, j = self.faces
        half_i, half_j, area = self.face_geometry
        return series_conductance(k[i], k[j], half_i, half_j, area)

    # ------------------------------------------------------------------ misc

    def mask(self, openings: list[list[float]] | None) -> np.ndarray:
        """Boolean lateral mask, True where the mask is open.

        2D: openings are [x0, x1] intervals. 3D: [x0, x1, y0, y1] rectangles.
        ``None`` means fully open.
        """
        lat = self.lateral_shape
        if openings is None:
            return np.ones(lat, dtype=bool)
        m = np.zeros(lat, dtype=bool)
        centers = [self.centers(a) for a in range(len(lat))]
        for op in openings:
            if len(op) != 2 * len(lat):
                raise ValueError(f"opening {op} needs {2 * len(lat)} numbers for a {self.ndim}D grid")
            sel = np.ones(lat, dtype=bool)
            for a, c in enumerate(centers):
                lo, hi = sorted(op[2 * a: 2 * a + 2])
                shape = [1] * len(lat)
                shape[a] = -1
                sel &= ((c >= lo) & (c < hi)).reshape(shape)
            m |= sel
        return m

    def open_fraction(self, openings: list[list[float]] | None) -> np.ndarray:
        """Fraction of each lateral cell that lies inside the openings (0..1)."""
        lat = self.lateral_shape
        if openings is None:
            return np.ones(lat)
        frac = np.zeros(lat)
        for op in openings:
            if len(op) != 2 * len(lat):
                raise ValueError(f"opening {op} needs {2 * len(lat)} numbers for a {self.ndim}D grid")
            part = np.ones(lat)
            for a in range(len(lat)):
                lo, hi = sorted(op[2 * a: 2 * a + 2])
                e = self.edges[a]
                overlap = np.clip(np.minimum(e[1:], hi) - np.maximum(e[:-1], lo), 0, None) / np.diff(e)
                shape = [1] * len(lat)
                shape[a] = -1
                part = part * overlap.reshape(shape)
            frac += part
        return np.minimum(frac, 1.0)

    def describe(self) -> dict:
        return {
            "ndim": self.ndim,
            "shape": list(self.shape),
            "n_cells": self.n_cells,
            "lateral_nm": list(self.lateral_nm),
            "depth_nm": self.depth_nm,
            "headroom_nm": self.headroom_nm,
            "spacing_nm": {"base": self.base_spacing_nm,
                           "min": float(min(self.widths(a).min() for a in range(self.ndim)))},
            "refine": [vars(r) for r in self.refine],
            "periodic": self.periodic,
        }
