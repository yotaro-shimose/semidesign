"""Uniform structured grid for 1D/2D/3D process simulation.

Array layout: lateral axes first, depth (pointing down into the wafer) last.
2D arrays have shape (nx, nz); 3D arrays (nx, ny, nz).

The depth coordinate is 0 at the initial silicon surface and negative above
it. ``headroom_nm`` of space above the surface is kept for oxide and deposits.
All lengths are in nm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Grid:
    lateral_nm: tuple[float, ...]  # box width along each lateral axis
    depth_nm: float  # silicon depth below the initial surface
    spacing_nm: float = 10.0
    headroom_nm: float = 200.0
    periodic: bool = False  # lateral edges wrap (repeating pixels) instead of mirror
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not self.lateral_nm:
            raise ValueError("need at least one lateral axis (use 2D or 3D)")
        h = self.spacing_nm
        lat = tuple(max(1, round(w / h)) for w in self.lateral_nm)
        nz = max(2, round((self.depth_nm + self.headroom_nm) / h))
        object.__setattr__(self, "shape", lat + (nz,))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def lateral_shape(self) -> tuple[int, ...]:
        return self.shape[:-1]

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def cell_volume_cm(self) -> float:
        """Cell size in cm^ndim (area in 2D, volume in 3D)."""
        return (self.spacing_nm * 1e-7) ** self.ndim

    def lateral_centers(self, axis: int) -> np.ndarray:
        n = self.shape[axis]
        return (np.arange(n) + 0.5) * self.spacing_nm

    @property
    def z_centers(self) -> np.ndarray:
        return (np.arange(self.shape[-1]) + 0.5) * self.spacing_nm - self.headroom_nm

    def lateral_index(self, pos_nm: float, axis: int = 0) -> int:
        i = int(pos_nm // self.spacing_nm)
        return min(max(i, 0), self.shape[axis] - 1)

    def mask(self, openings: list[list[float]] | None) -> np.ndarray:
        """Boolean lateral mask, True where the mask is open.

        2D: openings are [x0, x1] intervals. 3D: [x0, x1, y0, y1] rectangles.
        ``None`` means fully open.
        """
        lat = self.lateral_shape
        if openings is None:
            return np.ones(lat, dtype=bool)
        m = np.zeros(lat, dtype=bool)
        centers = [self.lateral_centers(a) for a in range(len(lat))]
        for op in openings:
            if len(op) != 2 * len(lat):
                raise ValueError(
                    f"opening {op} needs {2 * len(lat)} numbers for a {self.ndim}D grid"
                )
            sel = np.ones(lat, dtype=bool)
            for a, c in enumerate(centers):
                lo, hi = sorted(op[2 * a: 2 * a + 2])
                inside = (c >= lo) & (c < hi)
                shape = [1] * len(lat)
                shape[a] = -1
                sel &= inside.reshape(shape)
            m |= sel
        return m

    def describe(self) -> dict:
        return {
            "ndim": self.ndim,
            "shape": list(self.shape),
            "lateral_nm": list(self.lateral_nm),
            "depth_nm": self.depth_nm,
            "spacing_nm": self.spacing_nm,
            "headroom_nm": self.headroom_nm,
            "periodic": self.periodic,
            "n_cells": self.n_cells,
        }
