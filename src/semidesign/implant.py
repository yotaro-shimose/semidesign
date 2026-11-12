"""Ion implantation on a 2D/3D grid.

Vertical profile: Gaussian in depth below the local top surface, integrated over
each cell so the dose is conserved exactly. Lateral spread: Gaussian blur along
the lateral axes. All solids stop ions alike (oxide, poly and Si share one range
table); masked columns receive nothing (photoresist stops the beam).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.special import erf

from .geometry import top_solid_index
from .grid import Grid
from .physics import GAS, get_dopant


def implant_profile(grid: Grid, mat: np.ndarray, dopant: str, dose: float,
                    energy_kev: float, openings=None, rp_nm: float | None = None,
                    drp_nm: float | None = None,
                    lateral_sigma_nm: float | None = None) -> tuple[np.ndarray, dict]:
    """Concentration (cm^-3) added by one implant, plus the parameters used."""
    d = get_dopant(dopant)
    rp_tab, drp_tab = d.projected_range(energy_kev)
    rp = rp_nm if rp_nm is not None else rp_tab
    drp = drp_nm if drp_nm is not None else drp_tab
    sig_lat = lateral_sigma_nm if lateral_sigma_nm is not None else drp

    h = grid.spacing_nm
    top = top_solid_index(mat)  # (lateral...)
    k = np.arange(mat.shape[-1])
    depth_lo = (k - top[..., None]) * h  # depth of each cell's upper face below surface
    s2 = np.sqrt(2) * drp
    frac = 0.5 * (erf((depth_lo + h - rp) / s2) - erf((depth_lo - rp) / s2))
    frac[depth_lo < 0] = 0.0
    conc = dose * frac / (h * 1e-7)  # cm^-2 spread over a cell height in cm
    conc *= grid.mask(openings)[..., None]

    sigma_cells = sig_lat / h
    if sigma_cells > 0.25:
        mode = "wrap" if grid.periodic else "reflect"
        for ax in range(grid.ndim - 1):
            conc = ndimage.gaussian_filter1d(conc, sigma_cells, axis=ax, mode=mode)
    conc[mat == GAS] = 0.0
    params = {"rp_nm": round(rp, 2), "drp_nm": round(drp, 2),
              "lateral_sigma_nm": round(sig_lat, 2),
              "range_from_table": rp_nm is None and drp_nm is None,
              "range_extrapolated": not (d.range_table[0][0] <= energy_kev <= d.range_table[-1][0])}
    return conc, params
