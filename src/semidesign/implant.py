"""Ion implantation on a 2D/3D grid.

Depth: the Monte Carlo range distribution from bca.py, stored as depth
quantiles (data/implant_ranges.json) and interpolated between tabulated
energies quantile by quantile, linearly in log energy. The profile shape,
including the long shallow tail of MeV implants, therefore comes from the
collision physics rather than from a fitted distribution function.

Lateral: Gaussian with the Monte Carlo lateral straggle.

Every solid stops ions as if it were silicon, so depth counts from the local
top surface. Masked columns receive nothing (photoresist stops the beam).

Calibration: rp_nm, drp_nm and lateral_sigma_nm shift and stretch the
tabulated distribution, keeping its shape, to match measured values (SIMS).
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.special import erf

from .bca import RANGE_TABLE, RangeDistribution
from .constants import NM_PER_CM
from .geometry import depth_below_top
from .grid import Grid
from .materials import GAS


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(RANGE_TABLE.read_text())


def range_at(dopant: str, energy_kev: float) -> RangeDistribution:
    """Range distribution at any energy inside the tabulated span."""
    table = _table()
    if dopant not in table["ions"]:
        raise ValueError(f"no range table for {dopant!r}; available: {sorted(table['ions'])}")
    t = table["ions"][dopant]
    energies = np.asarray(t["energy_kev"])
    if not energies[0] <= energy_kev <= energies[-1]:
        raise ValueError(
            f"{dopant} at {energy_kev:g} keV is outside the range table "
            f"({energies[0]:g}-{energies[-1]:g} keV); extend TABLE_ENERGIES_KEV in bca.py "
            "and regenerate, or give rp_nm and drp_nm explicitly")
    k = int(np.clip(np.searchsorted(energies, energy_kev) - 1, 0, energies.size - 2))
    w = math.log(energy_kev / energies[k]) / math.log(energies[k + 1] / energies[k])

    def mix(key):
        a, b = np.asarray(t[key][k]), np.asarray(t[key][k + 1])
        return (1 - w) * a + w * b

    return RangeDistribution(
        rp_nm=float(mix("rp_nm")), drp_nm=float(mix("drp_nm")),
        skewness=float(mix("skewness")), kurtosis=float(mix("kurtosis")),
        lateral_nm=float(mix("lateral_nm")), retained=float(mix("retained")),
        depth_quantiles_nm=tuple(mix("depth_quantiles_nm")),
    )


def calibrated(dist: RangeDistribution, rp_nm: float | None = None, drp_nm: float | None = None,
               lateral_nm: float | None = None) -> RangeDistribution:
    """Shift/stretch a distribution to a given mean and spread, keeping its shape."""
    rp = dist.rp_nm if rp_nm is None else rp_nm
    scale = 1.0 if drp_nm is None else drp_nm / dist.drp_nm
    q = rp + (np.asarray(dist.depth_quantiles_nm) - dist.rp_nm) * scale
    return replace(dist, rp_nm=rp, drp_nm=dist.drp_nm * scale,
                   lateral_nm=dist.lateral_nm if lateral_nm is None else lateral_nm,
                   depth_quantiles_nm=tuple(q))


def depth_cdf(dist: RangeDistribution, depth_nm: np.ndarray) -> np.ndarray:
    """Fraction of the retained ions that stop shallower than depth_nm."""
    q = np.maximum.accumulate(np.asarray(dist.depth_quantiles_nm))
    q = q + np.arange(q.size) * 1e-9  # strictly increasing for interpolation
    levels = np.asarray(_table()["quantile_levels"])
    return np.interp(depth_nm, q, levels, left=0.0, right=1.0)


def lateral_spread(edges: np.ndarray, sigma_nm: float, periodic: bool) -> np.ndarray:
    """Matrix M with M[a, b] = share of the ions entering cell b that end up in cell a.

    A Gaussian of width sigma around each cell centre, integrated over every
    cell; the box edges reflect it (mirror symmetry) or wrap it (periodic).
    """
    c = 0.5 * (edges[:-1] + edges[1:])
    length = edges[-1] - edges[0]
    if periodic:
        images = [(c + k * length, 1.0) for k in range(-2, 3)]
    else:
        images = [(c + 2 * k * length, 1.0) for k in (-1, 0, 1)] + \
                 [(2 * edges[0] - c + 2 * k * length, 1.0) for k in (-1, 0, 1)]
    cdf = lambda x: 0.5 * (1 + erf(x / (math.sqrt(2) * sigma_nm)))
    m = np.zeros((c.size, c.size))
    for centre, weight in images:
        m += weight * (cdf(edges[1:, None] - centre[None, :]) - cdf(edges[:-1, None] - centre[None, :]))
    return m


def implant_profile(grid: Grid, mat: np.ndarray, dose: float, dist: RangeDistribution,
                    openings=None) -> np.ndarray:
    """Concentration (cm^-3) added by an implant of `dose` (cm^-2)."""
    depth_top = depth_below_top(grid, mat)  # upper face of each cell below the surface
    height = grid.widths(-1)
    frac = depth_cdf(dist, depth_top + height) - depth_cdf(dist, depth_top)
    frac[depth_top < 0] = 0.0
    conc = dose * dist.retained * frac / (height / NM_PER_CM)
    conc *= grid.open_fraction(openings)[..., None]
    for ax in range(grid.ndim - 1):
        if dist.lateral_nm > 0.05 * grid.widths(ax).min():
            w = grid.widths(ax)
            spread = lateral_spread(grid.edges[ax], dist.lateral_nm, grid.periodic)
            amount = np.moveaxis(conc, ax, -1) * w  # per unit length along ax
            moved = amount @ spread.T / w
            conc = np.moveaxis(moved, -1, ax)
    conc[mat == GAS] = 0.0
    return conc
