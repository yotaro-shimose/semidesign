"""Cell-based geometry operations: etch, deposit, planarize.

The structure is a material code per cell (see physics.GAS/SI/OXIDE/POLY).
Surfaces are cell faces, so features are resolved to one grid spacing.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .grid import Grid
from .physics import GAS


def top_solid_index(mat: np.ndarray) -> np.ndarray:
    """Index along depth of the first non-gas cell in each lateral column.

    Columns with no solid get nz.
    """
    solid = mat != GAS
    idx = np.argmax(solid, axis=-1)
    idx[~solid.any(axis=-1)] = mat.shape[-1]
    return idx


def etch(grid: Grid, mat: np.ndarray, openings, depth_nm: float,
         materials: list[int] | None = None) -> np.ndarray:
    """Anisotropic (straight-down) etch through mask openings.

    Removes solid cells from the local top surface down to ``depth_nm`` below it.
    ``materials`` limits which materials are removed (default: all solids).
    Returns the boolean array of removed cells.
    """
    open_ = grid.mask(openings)
    top = top_solid_index(mat)
    n_cells = int(round(depth_nm / grid.spacing_nm))
    k = np.arange(mat.shape[-1])
    region = (k >= top[..., None]) & (k < top[..., None] + n_cells) & open_[..., None]
    removable = mat != GAS
    if materials is not None:
        removable &= np.isin(mat, materials)
    removed = region & removable
    mat[removed] = GAS
    return removed


def deposit(grid: Grid, mat: np.ndarray, material: int, thickness_nm: float,
            conformal: bool = True) -> np.ndarray:
    """Deposit a film. Returns the boolean array of new cells.

    conformal: fills every gas cell within ``thickness_nm`` of an existing solid
    (side walls included, so narrow trenches close up). Otherwise the film is
    stacked straight up on top of each column.
    """
    h = grid.spacing_nm
    if conformal:
        gas = mat == GAS
        if not gas.any():
            return np.zeros_like(gas)
        dist = ndimage.distance_transform_edt(gas) * h  # distance to nearest solid
        # Solid only exists below; without a solid neighbour EDT returns a large value
        new = gas & (dist <= thickness_nm + 1e-9)
    else:
        top = top_solid_index(mat)
        n = int(round(thickness_nm / h))
        k = np.arange(mat.shape[-1])
        new = (k < top[..., None]) & (k >= top[..., None] - n) & (mat == GAS)
    mat[new] = material
    return new


def planarize(grid: Grid, mat: np.ndarray, level_nm: float) -> np.ndarray:
    """Remove everything above ``level_nm`` (polishing flat). Returns removed cells."""
    above = grid.z_centers < level_nm
    removed = (mat != GAS) & above
    mat[removed] = GAS
    return removed
