"""Cell-based geometry operations: etch, deposit, planarize.

The structure is a material code per cell (see materials.py). Surfaces are
cell faces, so features are resolved to the local cell size.

Every operation works from the top of the arrays it is given. The back side
of the wafer (after it is bonded and thinned) is the top of the flipped view,
``from_side(grid, mat, "back")``: the same code then etches, deposits and
polishes from below, writing through the view into the real arrays.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .grid import Grid
from .materials import GAS


SIDES = ("front", "back")


def from_side(grid: Grid, mat: np.ndarray, side: str) -> tuple[Grid, np.ndarray]:
    """Grid and material view with `side` on top (views share memory with `mat`)."""
    if side == "front":
        return grid, mat
    if side == "back":
        return grid.flipped(), mat[..., ::-1]
    raise ValueError(f"side must be one of {SIDES}, not {side!r}")


def unflip(arr: np.ndarray, side: str) -> np.ndarray:
    """An array computed on the `side` view, back in front-side orientation."""
    return arr[..., ::-1] if side == "back" else arr


def ambient(mat: np.ndarray) -> np.ndarray:
    """Gas open to the top of the box: what a process on this side can reach.

    Voids sealed by an earlier film and the gas on the other side of a
    thinned wafer are left out.
    """
    labels, _ = ndimage.label(mat == GAS)
    open_labels = np.unique(labels[..., 0])
    return np.isin(labels, open_labels[open_labels > 0])


def top_solid_index(mat: np.ndarray) -> np.ndarray:
    """Index along depth of the first non-gas cell in each lateral column.

    Columns with no solid get nz.
    """
    solid = mat != GAS
    idx = np.argmax(solid, axis=-1)
    idx[~solid.any(axis=-1)] = mat.shape[-1]
    return idx


def depth_below_top(grid: Grid, mat: np.ndarray) -> np.ndarray:
    """Depth (nm) of each cell's upper face below the top solid surface of its column."""
    z = grid.z_edges
    top = top_solid_index(mat)
    surface = np.append(z, z[-1])[top]
    return z[:-1] - surface[..., None]


def etch(grid: Grid, mat: np.ndarray, openings, depth_nm: float,
         materials: list[int] | None = None) -> np.ndarray:
    """Anisotropic (straight-down) etch through mask openings.

    Removes solid cells whose centre lies within ``depth_nm`` of the local top
    surface. ``materials`` limits which materials are removed (default: all).
    Returns the boolean array of removed cells.
    """
    d = depth_below_top(grid, mat) + 0.5 * grid.widths(-1)
    region = (d >= 0) & (d < depth_nm) & grid.mask(openings)[..., None]
    removable = mat != GAS
    if materials is not None:
        removable &= np.isin(mat, materials)
    removed = region & removable
    mat[removed] = GAS
    return removed


def _cell_centers(grid: Grid) -> np.ndarray:
    axes = np.meshgrid(*(grid.centers(a) for a in range(grid.ndim)), indexing="ij")
    return np.stack([a.ravel() for a in axes], axis=1)


def deposit(grid: Grid, mat: np.ndarray, material: int, thickness_nm: float,
            conformal: bool = True) -> np.ndarray:
    """Deposit a film. Returns the boolean array of new cells.

    Only gas open to the top is filled (see ambient). conformal: fills every
    such gas cell whose centre is within ``thickness_nm`` of the centre of a solid cell (side walls included, so narrow trenches close
    up). Otherwise the film is stacked straight up on top of each column.
    """
    gas = ambient(mat)
    if conformal:
        solid = mat != GAS
        if not gas.any() or not solid.any():
            return np.zeros_like(gas)
        centers = _cell_centers(grid)
        tree = cKDTree(centers[solid.ravel()])
        dist, _ = tree.query(centers[gas.ravel()], distance_upper_bound=thickness_nm + 1e-9)
        new = np.zeros(gas.size, bool)
        new[np.flatnonzero(gas.ravel())[np.isfinite(dist)]] = True
        new = new.reshape(gas.shape)
    else:
        z = grid.z_centers
        top = top_solid_index(mat)
        surface = np.append(grid.z_edges, grid.z_edges[-1])[top]
        new = gas & (z < surface[..., None]) & (z > surface[..., None] - thickness_nm)
    mat[new] = material
    return new


def planarize(grid: Grid, mat: np.ndarray, level_nm: float) -> np.ndarray:
    """Remove everything above ``level_nm`` (polishing flat). Returns removed cells."""
    removed = (mat != GAS) & (grid.z_centers < level_nm)
    mat[removed] = GAS
    return removed
