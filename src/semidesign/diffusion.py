"""Dopant transport: diffusion in every material and exchange across interfaces.

Each dopant obeys  w dU/dt = div(kappa grad U)  with  C = w U  and  kappa = D w,
where w is the material's capacity (dopants.Dopant.capacity_by_material).
U is continuous across interfaces in equilibrium, so segregation
(C_Si / C_SiO2 = m) emerges from the equations instead of a special rule.
Face conductances are harmonic means of kappa; outer edges are closed,
lateral edges optionally periodic. Time stepping is backward Euler.

In silicon the diffusivity is multiplied by the point-defect factor when an
interstitial supersaturation field is given (transient enhanced diffusion).

A moving interface (silicon turning into oxide) is handled by
segregate_at_new_oxide, which applies the same equilibrium locally.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import fv
from .constants import NM2_PER_CM2
from .dopants import get_dopant
from .grid import Grid
from .materials import OXIDE, SI


class DopantTransport:
    """Backward-Euler diffusion of all dopants at one temperature.

    The factorized system is reused across steps while the geometry, the time
    step and the defect field are unchanged.
    """

    def __init__(self, grid: Grid, temp_c: float):
        self.grid = grid
        self.temp_c = temp_c
        self.i, self.j = grid.faces
        self._cache: dict[str, tuple[np.ndarray, float, fv.SPDSolver, np.ndarray]] = {}

    def step(self, conc: dict[str, np.ndarray], mat: np.ndarray, dt_s: float,
             supersaturation: np.ndarray | None = None) -> str:
        """Advance every dopant by dt in place. Returns the solver method used."""
        method = "none"
        for name, c in conc.items():
            solver, w = self._system(name, mat, dt_s, supersaturation)
            u = solver.solve(c.ravel())
            conc[name] = np.maximum(w * u, 0.0).reshape(c.shape)
            method = solver.method
        return method

    def _system(self, name, mat, dt_s, supersaturation):
        cached = self._cache.get(name)
        if (supersaturation is None and cached is not None and cached[1] == dt_s
                and np.array_equal(cached[0], mat)):
            return cached[2], cached[3]
        d = get_dopant(name)
        m = mat.ravel()
        w = d.capacity_by_material()[m]
        diff = d.diffusivity_by_material(self.temp_c)[m] * NM2_PER_CM2  # nm^2/s
        if supersaturation is not None:
            si = m == SI
            diff[si] *= d.point_defect_factor(supersaturation.ravel()[si])
        kappa = diff * w
        g = fv.harmonic_mean(kappa[self.i], kappa[self.j]) / self.grid.spacing_nm**2
        lap = fv.laplacian(m.size, self.i, self.j, g)
        solver = fv.SPDSolver(sp.diags(w) - dt_s * lap)
        if supersaturation is None:
            self._cache[name] = (mat.copy(), dt_s, solver, w)
        return solver, w


def segregate_at_new_oxide(conc: dict[str, np.ndarray], mat: np.ndarray,
                           converted: np.ndarray, i: np.ndarray, j: np.ndarray) -> None:
    """Redistribute dopant where silicon has just turned into oxide.

    The new oxide cell and the silicon cells it touches come to local
    segregation equilibrium (equal U = C / w), conserving the dopant. This is
    the snowplow effect of a moving interface: donors (m > 1) are pushed into
    the silicon ahead of the front, boron (m < 1) is drawn into the oxide.
    """
    conv = converted.ravel()
    if not conv.any():
        return
    m = mat.ravel()
    si = m == SI
    pairs = np.concatenate([np.stack([i, j], 1), np.stack([j, i], 1)])
    pairs = pairs[conv[pairs[:, 0]] & si[pairs[:, 1]]]  # (new oxide cell, touching Si cell)
    if pairs.size == 0:
        return
    order = np.argsort(pairs[:, 0], kind="stable")
    pairs = pairs[order]
    cells, start = np.unique(pairs[:, 0], return_index=True)
    groups = np.split(pairs[:, 1], start[1:])
    for name, c in conc.items():
        w_ox = get_dopant(name).capacity_by_material()[OXIDE]
        flat = c.reshape(-1)
        for cell, nbrs in zip(cells, groups):
            total = flat[cell] + flat[nbrs].sum()
            u = total / (w_ox + nbrs.size)
            flat[cell] = w_ox * u
            flat[nbrs] = u
