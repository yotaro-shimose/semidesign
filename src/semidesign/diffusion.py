"""Dopant transport: diffusion in every material and exchange across interfaces.

Each dopant obeys  w dU/dt = div(kappa grad U)  with  C = w U  and  kappa = D w,
where w is the material's capacity (dopants.Dopant.capacity_by_material).
U is continuous across interfaces in equilibrium, so segregation
(C_Si / C_SiO2 = m) emerges from the equations instead of a special rule.
Finite volumes: each face conducts like the two half cells in series (any
cell sizes); outer edges are closed, lateral edges optionally periodic. Time
stepping is backward Euler.

In silicon the diffusivity is multiplied by the point-defect factor when an
interstitial supersaturation field is given (transient enhanced diffusion).

A moving interface (silicon turning into oxide) is handled by
segregate_at_new_oxide, which applies the same equilibrium locally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import fv
from .constants import NM2_PER_CM2
from .dopants import get_dopant
from .grid import Grid
from .materials import OXIDE, SI


class DopantTransport:
    """Backward-Euler diffusion of all dopants at one temperature.

    A new system (geometry, time step or defect field changed) is solved
    iteratively; once the same system comes back, it is factorized and the
    factorization reused.
    """

    def __init__(self, grid: Grid, temp_c: float):
        self.grid = grid
        self.temp_c = temp_c
        self.i, self.j = grid.faces
        self._cache: dict[str, _CachedSystem] = {}

    def step(self, conc: dict[str, np.ndarray], mat: np.ndarray, dt_s: float,
             supersaturation: np.ndarray | None = None) -> str:
        """Advance every dopant by dt in place. Returns the solver method used."""
        method = "none"
        volume = self.grid.volumes.ravel()
        for name, c in conc.items():
            solver, w = self._system(name, mat, dt_s, supersaturation)
            u = solver.solve(c.ravel() * volume, x0=c.ravel() / w)
            conc[name] = np.maximum(w * u, 0.0).reshape(c.shape)
            method = solver.method
        return method

    def _system(self, name, mat, dt_s, supersaturation) -> tuple[fv.SPDSolver, np.ndarray]:
        cached = self._cache.get(name)
        if supersaturation is None and cached is not None and cached.matches(mat, dt_s):
            if not cached.factorized:
                cached.solver = fv.SPDSolver(cached.matrix, factorize=True)
                cached.factorized = True
            return cached.solver, cached.w
        matrix, w = self._matrix(name, mat, dt_s, supersaturation)
        solver = fv.SPDSolver(matrix, factorize=False)
        if supersaturation is None:
            self._cache[name] = _CachedSystem(mat.copy(), dt_s, matrix, w, solver)
        else:
            self._cache.pop(name, None)
        return solver, w

    def _matrix(self, name, mat, dt_s, supersaturation):
        d = get_dopant(name)
        m = mat.ravel()
        w = d.capacity_by_material()[m]
        diff = d.diffusivity_by_material(self.temp_c)[m] * NM2_PER_CM2  # nm^2/s
        if supersaturation is not None:
            si = m == SI
            diff[si] *= d.point_defect_factor(supersaturation.ravel()[si])
        lap = fv.laplacian(m.size, self.i, self.j, self.grid.conductance(diff * w))
        return sp.diags(w * self.grid.volumes.ravel()) - dt_s * lap, w


@dataclass
class _CachedSystem:
    mat: np.ndarray
    dt_s: float
    matrix: sp.spmatrix
    w: np.ndarray
    solver: fv.SPDSolver
    factorized: bool = False

    def matches(self, mat: np.ndarray, dt_s: float) -> bool:
        return self.dt_s == dt_s and np.array_equal(self.mat, mat)


def segregate_at_new_oxide(grid: Grid, conc: dict[str, np.ndarray], mat: np.ndarray,
                           converted: np.ndarray) -> None:
    """Redistribute dopant where silicon has just turned into oxide.

    The new oxide cell and the silicon cells it touches come to local
    segregation equilibrium (equal U = C / w), conserving the dopant. This is
    the snowplow effect of a moving interface: donors (m > 1) are pushed into
    the silicon ahead of the front, boron (m < 1) is drawn into the oxide.
    """
    conv = converted.ravel()
    if not conv.any():
        return
    i, j = grid.faces
    si = mat.ravel() == SI
    pairs = np.concatenate([np.stack([i, j], 1), np.stack([j, i], 1)])
    pairs = pairs[conv[pairs[:, 0]] & si[pairs[:, 1]]]  # (new oxide cell, touching Si cell)
    if pairs.size == 0:
        return
    order = np.argsort(pairs[:, 0], kind="stable")
    pairs = pairs[order]
    cells, start = np.unique(pairs[:, 0], return_index=True)
    groups = np.split(pairs[:, 1], start[1:])
    volume = grid.volumes.ravel()
    for name, c in conc.items():
        w_ox = get_dopant(name).capacity_by_material()[OXIDE]
        flat = c.reshape(-1)
        for cell, nbrs in zip(cells, groups):
            total = flat[cell] * volume[cell] + (flat[nbrs] * volume[nbrs]).sum()
            u = total / (w_ox * volume[cell] + volume[nbrs].sum())
            flat[cell] = w_ox * u
            flat[nbrs] = u
